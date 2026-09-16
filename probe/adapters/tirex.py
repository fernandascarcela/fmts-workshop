"""
TiRex-2 adapter for the P1 known-future-covariate interface.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import Forecast, Interface, ModelAdapter


class TiRexAdapter(ModelAdapter):

    name = "tirex-2"
    interface = Interface.P1

    def __init__(
        self,
        model_id: str = "NX-AI/TiRex-2",
        device: str | None = None,
    ) -> None:

        self.model_id = model_id

        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._model = None

    def _load(self):
        """Load the pretrained TiRex-2 model once."""

        if self._model is None:

            from tirex2 import load_model

            self._model = load_model(
                self.model_id,
                device=self.device,
            )

        return self._model

    def _forecast(
        self,
        context: np.ndarray,
        horizon: int,
        outcome_index: int,
        known_future: dict[int, np.ndarray] | None = None,
    ) -> Forecast:
        """
        Translate the generic probe representation into TiRex-2 inputs.

        known_future is generic:

            {
                variable_index: future_trajectory
            }

        The adapter does not know why those future values are known.
        """

        from tirex2 import TimeseriesType

        model = self._load()

        context = np.asarray(
            context,
            dtype=np.float32,
        )

        _, n_vars = context.shape

        known_future = known_future or {}

        # --------------------------------------------------------------
        # Split non-target variables according to whether their future
        # values are known.
        # --------------------------------------------------------------
        future_indices = [
            idx
            for idx in known_future
            if idx != outcome_index
        ]

        past_only_indices = [
            idx
            for idx in range(n_vars)
            if idx != outcome_index
            and idx not in future_indices
        ]

        # --------------------------------------------------------------
        # Queried outcome -> target.
        #
        # TiRex-2 expects:
        #
        #     n_target_variates x context_length
        #
        # We forecast one queried outcome, therefore:
        #
        #     1 x context_length
        # --------------------------------------------------------------
        target = torch.as_tensor(
            context[:, outcome_index],
            dtype=torch.float32,
        )[None, :]

        # --------------------------------------------------------------
        # Variables with only observed history -> past_covariates.
        #
        # TiRex-2 expects:
        #
        #     n_past_covariates x context_length
        #
        # Our context uses:
        #
        #     context_length x n_variables
        #
        # so the selected columns are transposed.
        # --------------------------------------------------------------
        if past_only_indices:

            past_covariates = torch.as_tensor(
                context[:, past_only_indices].T,
                dtype=torch.float32,
            )

        else:

            past_covariates = None

        # --------------------------------------------------------------
        # Variables with known futures -> future_covariates.
        #
        # Each variable is represented as:
        #
        #     [observed history | known future trajectory]
        #
        # TiRex-2 expects:
        #
        #     n_future_covariates
        #         x
        #     (context_length + prediction_length)
        # --------------------------------------------------------------
        if future_indices:

            future_rows = []

            for idx in future_indices:

                history = context[:, idx]

                future = np.asarray(
                    known_future[idx],
                    dtype=np.float32,
                ).reshape(-1)

                full_covariate = np.concatenate(
                    [
                        history,
                        future,
                    ]
                )

                future_rows.append(full_covariate)

            future_covariates = torch.as_tensor(
                np.stack(
                    future_rows,
                    axis=0,
                ),
                dtype=torch.float32,
            )

        else:

            future_covariates = None

        # --------------------------------------------------------------
        # Native TiRex-2 representation.
        # --------------------------------------------------------------
        timeseries = TimeseriesType(
            target=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        # --------------------------------------------------------------
        # Forecast.
        #
        # Returned shape for one target:
        #
        #     1 x n_quantiles x horizon
        # --------------------------------------------------------------
        prediction = model.forecast(
            timeseries=[timeseries],
            prediction_length=horizon,
            output_type="numpy",
            batch_size=1,
        )[0]

        prediction = np.asarray(
            prediction,
            dtype=float,
        )

        # Remove the target dimension.
        #
        # Before:
        #     1 x n_quantiles x horizon
        #
        # After:
        #     n_quantiles x horizon
        prediction = prediction[0]

        # --------------------------------------------------------------
        # Read native quantile levels from the loaded model rather than
        # assuming their positions.
        # --------------------------------------------------------------
        quantile_levels = [
            round(float(q), 3)
            for q in model.quantiles.detach().cpu().numpy()
        ]

        quantiles = {
            level: prediction[i].reshape(-1)
            for i, level in enumerate(quantile_levels)
        }

        return Forecast(
            point=quantiles[0.5].copy(),
            quantiles=quantiles,
        )