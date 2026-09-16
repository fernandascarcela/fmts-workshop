"""
Common contract for every model adapter in the probe harness.

Every time-series foundation model in the roster is wrapped in a small
adapter that implements the `ModelAdapter` contract defined here. The rest
of the harness (interfaces, metrics, runner) only ever talks to a
`ModelAdapter` -- never to a specific model's SDK.

Consequence: adding a new model to the roster means writing one new adapter
file. Nothing else in the codebase has to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Interface(str, Enum):
    """How a model can receive the intervention (the do-value).

    P1       -- native known-future covariate channel (the clean case).
    P1*      -- TimesFM 2.5 limited future covariate channel using Xreg
    P2       -- treatment supplied as an ordinary series and clamped during
                decoding (for multivariate models without a P1 channel).
    LIMITED  -- no channel for an action at all; this is itself a finding
                (the "Finding 0" capability rows).
    """

    P1 = "P1"
    P1_STAR = "P1*"
    P2 = "P2"
    LIMITED = "interface-limited"


@dataclass
class Forecast:
    """A model's prediction for the OUTCOME variable over the horizon.

    point     -- point forecast, shape (horizon,). Defined as the median
                 forecast (q=0.5) when available.
    quantiles -- optional {level: array(horizon,)}, used later for the
                 calibration metric (R3). Left as None for models that do
                 not emit predictive quantiles.
    """

    point: np.ndarray
    quantiles: dict[float, np.ndarray] | None = None


class UnsupportedInterventionError(RuntimeError):
    """Raised when a do-value is handed to an interface-limited model."""


class ModelAdapter(ABC):
    """Base contract every roster model must implement.

    A concrete adapter (e.g. `ChronosAdapter`) provides:
      * `name`       -- short identifier used in results tables.
      * `interface`  -- which do-value channel the model supports.
      * `_forecast`  -- the actual call into the model's SDK.

    The harness never calls `_forecast` directly. It calls `predict`, which
    holds the P0-vs-P1/P2 branching in ONE place so every model is treated
    identically.
    """

    #: short identifier, e.g. "chronos-2". Set by the subclass.
    name: str
    #: which intervention channel this model supports. Set by the subclass.
    interface: Interface

    @abstractmethod
    def _forecast(
        self,
        context: np.ndarray,
        horizon: int,
        outcome_index: int,
        known_future: dict[int, np.ndarray] | None = None,
    ) -> Forecast:
        """Run the underlying model once and return the OUTCOME forecast.

        context       -- past observations, shape (context_len, n_vars).
        horizon       -- number of future steps to predict.
        outcome_index -- column of `context` that holds the outcome Y.
        known_future  -- {var_index: array(horizon,)} of future values the
                         model is told in advance (the treatment trajectory
                         for P1/P2), or None for an unconditioned forecast.

        Subclasses implement this per model. It is the ONLY method that
        touches a specific model's SDK.
        """
        raise NotImplementedError

    def predict(
        self,
        context: np.ndarray,
        horizon: int,
        outcome_index: int,
        treatment: np.ndarray | None = None,
        treatment_index: int | None = None,
    ) -> Forecast:
        """Single entry point used by the rest of the harness.

        treatment is None   -> P0: unconditioned forecast (the anchor).
        treatment is given  -> P1/P2: the intervention is supplied.

        Handing a treatment to an interface-limited model is a bug, so we
        fail loudly rather than silently ignore the do-value.
        """
        conditioned = treatment is not None

        if conditioned and self.interface is Interface.LIMITED:
            raise UnsupportedInterventionError(
                f"{self.name} is interface-limited; it cannot receive a do-value."
            )

        known_future: dict[int, np.ndarray] | None = None
        if conditioned:
            if treatment_index is None:
                raise ValueError("treatment given but treatment_index is None.")
            known_future = {treatment_index: np.asarray(treatment, dtype=float)}

        return self._forecast(
            context=context,
            horizon=horizon,
            outcome_index=outcome_index,
            known_future=known_future,
        )
