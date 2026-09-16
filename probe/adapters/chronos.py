"""
Chronos-2 adapter for the P1 intervention interface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecast, Interface, ModelAdapter


class ChronosAdapter(ModelAdapter):
    """Wrap a pretrained Chronos-2 checkpoint behind the probe contract."""

    name = "chronos-2"
    interface = Interface.P1

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        device: str = "cuda",
        quantile_levels: tuple[float, ...] = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.quantile_levels = tuple(quantile_levels)
        self._pipeline = None

    def _load(self):
        """Load the Chronos-2 checkpoint once, then reuse it."""

        if self._pipeline is None:
            from chronos import Chronos2Pipeline

            self._pipeline = Chronos2Pipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
            )

        return self._pipeline

    def _forecast(
        self,
        context: np.ndarray,
        horizon: int,
        outcome_index: int,
        known_future: dict[int, np.ndarray] | None = None,
    ) -> Forecast:

        context = np.asarray(context, dtype=float)
        context_len, n_vars = context.shape

        # --------------------------------------------------------------
        # Synthetic time axis
        #
        # DoTime does not provide meaningful calendar dates.
        # Chronos-2's DataFrame API requires timestamps: 
        # "At a minimum, the input must contain three columns: an item ID, a timestamp, and the target value to forecast"
        # https://auto.gluon.ai/cloud/stable/tutorials/foundation-model-timeseries.html
        # so we create a regular synthetic index.
        # --------------------------------------------------------------
        freq = "D"

        timestamps = pd.date_range(
            "2000-01-01",
            periods=context_len,
            freq=freq,
        )

        # --------------------------------------------------------------
        # Historical context
        #
        # target = queried outcome
        # cov_*  = EVERY other observed variable
        #
        # This DataFrame is identical for P0 and P1.
        # --------------------------------------------------------------
        context_df = pd.DataFrame(
            {
                "id": 0,
                "timestamp": timestamps,
                "target": context[:, outcome_index],
            }
        )

        for c in range(n_vars):
            if c != outcome_index:
                context_df[f"cov_{c}"] = context[:, c]

        # --------------------------------------------------------------
        # P0 / P1 distinction
        #
        # P0:
        #   future_df = None
        #
        #   All non-target variables remain available as past-only
        #   covariates.
        #
        # P1:
        #   future_df contains ONLY variables whose future is explicitly
        #   known because of the intervention.
        #
        #   Other variables remain past-only covariates.
        # --------------------------------------------------------------
        future_df = None

        if known_future is not None:

            # Future timestamps must be the direct continuation of the
            # context using exactly the same frequency.
            future_timestamps = pd.date_range(
                start=timestamps[-1],
                periods=horizon + 1,
                freq=freq,
            )[1:]

            future_df = pd.DataFrame(
                {
                    "id": 0,
                    "timestamp": future_timestamps,
                }
            )

            # ----------------------------------------------------------
            # Add ONLY known-future treatment variable(s).
            #
            # For the current DoTime episodes this is normally one
            # treatment variable.
            # ----------------------------------------------------------
            for treat_index, treat_future in known_future.items():

                treat_future = np.asarray(
                    treat_future,
                    dtype=float,
                ).reshape(-1)

                future_df[f"cov_{treat_index}"] = treat_future

        # --------------------------------------------------------------
        # Forecast
        #
        # Both conditions use exactly the same context_df.
        #
        # P0 -> future_df=None
        # P1 -> future_df carries the do-value.
        # --------------------------------------------------------------
        pred_df = self._load().predict_df(
            context_df,
            future_df=future_df,
            prediction_length=horizon,
            quantile_levels=list(self.quantile_levels),
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

        # --------------------------------------------------------------
        # Parse probabilistic output.
        #
        # Chronos-2 returns quantile columns named as strings:
        # "0.1", "0.5", "0.9", ...
        # --------------------------------------------------------------
        quantiles = {
            level: pred_df[str(level)]
            .to_numpy(dtype=float)
            .reshape(-1)
            for level in self.quantile_levels
        }

        point = quantiles[0.5]

        if len(point) != horizon:
            raise RuntimeError(
                f"Chronos-2 returned {len(point)} forecast steps, "
                f"expected {horizon}."
            )

        return Forecast(
            point=np.asarray(point, dtype=float),
            quantiles=quantiles,
        )