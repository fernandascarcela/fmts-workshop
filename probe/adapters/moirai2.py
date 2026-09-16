"""
Moirai-2.0 adapter for the P1 known-future-covariate interface.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import Forecast, Interface, ModelAdapter


class Moirai2Adapter(ModelAdapter):

    name = "moirai-2.0"
    interface = Interface.P1

    def __init__(
        self,
        model_id: str = "Salesforce/moirai-2.0-R-small",
        device: str | None = None,
    ) -> None:

        self.model_id = model_id

        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._module = None

    def _load(self):
        """Load the pretrained Moirai-2.0 module once."""

        if self._module is None:

            from uni2ts.model.moirai2 import Moirai2Module

            self._module = Moirai2Module.from_pretrained(
                self.model_id
            )

            self._module.to(self.device)
            self._module.eval()

        return self._module

    def _forecast(
        self,
        context: np.ndarray,
        horizon: int,
        outcome_index: int,
        known_future: dict[int, np.ndarray] | None = None,
    ) -> Forecast:
        """
        Translate the generic probe representation into Moirai-2.0 inputs.

        known_future is generic:

            {
                variable_index: future_trajectory
            }

        The adapter does not know why those values are known.
        """

        from uni2ts.model.moirai2 import Moirai2Forecast

        module = self._load()

        context = np.asarray(
            context,
            dtype=np.float32,
        )

        context_length, n_vars = context.shape

        # --------------------------------------------------------------
        # Queried outcome -> target.
        #
        # Shape:
        #     batch x context_time x target_dim
        # --------------------------------------------------------------
        past_target = torch.as_tensor(
            context[:, outcome_index],
            dtype=torch.float32,
            device=self.device,
        )[None, :, None]

        # Every value in the DoTime context is observed.
        past_observed_target = torch.ones_like(
            past_target,
            dtype=torch.bool,
        )

        # No artificial padding exists in the DoTime context.
        past_is_pad = torch.zeros(
            (1, context_length),
            dtype=torch.bool,
            device=self.device,
        )

        # --------------------------------------------------------------
        # Generic known-future representation.
        # --------------------------------------------------------------
        known_future = known_future or {}

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
        # Variables with known futures -> feat_dynamic_real.
        #
        # Each column becomes:
        #
        #     historical values
        #           +
        #     known future trajectory
        #
        # Shape:
        #     batch x (context + horizon) x n_future_features
        # --------------------------------------------------------------
        if future_indices:

            future_matrix = np.column_stack(
                [
                    np.asarray(
                        known_future[idx],
                        dtype=np.float32,
                    ).reshape(-1)
                    for idx in future_indices
                ]
            )

            known_history = context[
                :,
                future_indices,
            ]

            full_dynamic = np.concatenate(
                [
                    known_history,
                    future_matrix,
                ],
                axis=0,
            )

            feat_dynamic_real = torch.as_tensor(
                full_dynamic,
                dtype=torch.float32,
                device=self.device,
            )[None, :, :]

            observed_feat_dynamic_real = torch.ones_like(
                feat_dynamic_real,
                dtype=torch.bool,
            )

        else:

            feat_dynamic_real = None
            observed_feat_dynamic_real = None

        # --------------------------------------------------------------
        # Variables without known futures -> past_feat_dynamic_real.
        #
        # Their complete observed histories are still retained.
        # --------------------------------------------------------------
        if past_only_indices:

            past_feat_dynamic_real = torch.as_tensor(
                context[:, past_only_indices],
                dtype=torch.float32,
                device=self.device,
            )[None, :, :]

            past_observed_feat_dynamic_real = torch.ones_like(
                past_feat_dynamic_real,
                dtype=torch.bool,
            )

        else:

            past_feat_dynamic_real = None
            past_observed_feat_dynamic_real = None

        # --------------------------------------------------------------
        # Forecast wrapper.
        #
        # Moirai-2.0 obtains patch size and quantile levels directly
        # from the pretrained module.
        # --------------------------------------------------------------
        model = Moirai2Forecast(
            module=module,
            prediction_length=horizon,
            context_length=context_length,
            target_dim=1,
            feat_dynamic_real_dim=len(future_indices),
            past_feat_dynamic_real_dim=len(past_only_indices),
        )

        model.to(self.device)
        model.eval()

        with torch.inference_mode():

            prediction = model(
                past_target=past_target,
                past_observed_target=past_observed_target,
                past_is_pad=past_is_pad,
                feat_dynamic_real=feat_dynamic_real,
                observed_feat_dynamic_real=(
                    observed_feat_dynamic_real
                ),
                past_feat_dynamic_real=past_feat_dynamic_real,
                past_observed_feat_dynamic_real=(
                    past_observed_feat_dynamic_real
                ),
            )

        # --------------------------------------------------------------
        # Official output:
        #
        #     batch x num_quantiles x horizon
        #
        # for target_dim = 1.
        #
        # Moirai-2.0 natively uses:
        #
        #     0.1, 0.2, ..., 0.9
        # --------------------------------------------------------------
        prediction = (
            prediction[0]
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )

        quantiles = {
            float(level): prediction[i].reshape(-1)
            for i, level in enumerate(module.quantile_levels)
        }

        return Forecast(
            point=quantiles[0.5].copy(),
            quantiles=quantiles,
        )