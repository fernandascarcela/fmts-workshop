"""
Moirai-1.1 adapter for the P1 known-future-covariate interface.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import Forecast, Interface, ModelAdapter


class MoiraiAdapter(ModelAdapter):

    name = "moirai-1.1"
    interface = Interface.P1

    def __init__(
        self,
        model_id: str = "Salesforce/moirai-1.1-R-base",
        quantile_levels: tuple[float, ...] = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
        num_samples: int = 256,
        patch_size: int = 32,
        seed: int = 42,
        device: str | None = None,
    ) -> None:

        self.model_id = model_id
        self.quantile_levels = quantile_levels
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.seed = seed

        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._module = None

    def _load(self):
        """Load the pretrained Moirai backbone once."""

        if self._module is None:

            from uni2ts.model.moirai import MoiraiModule

            self._module = MoiraiModule.from_pretrained(
                self.model_id
            )

            self._module.to(self.device)
            self._module.eval()

        return self._module

    def _reset_seed(self) -> None:
        """Use the same Monte Carlo random numbers across conditions."""

        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _forecast(
        self,
        context: np.ndarray,
        horizon: int,
        outcome_index: int,
        known_future: dict[int, np.ndarray] | None = None,
    ) -> Forecast:
        """
        Translate the generic probe representation into Moirai inputs.

        known_future is completely generic:

        {variable_index: future_trajectory}

        The adapter does not interpret why those future values are known.
        """

        from uni2ts.model.moirai import MoiraiForecast

        module = self._load()

        context = np.asarray(
            context,
            dtype=np.float32,
        )

        context_length, n_vars = context.shape

        # --------------------------------------------------------------
        # Outcome history -> forecasting target.
        #
        # Shape:
        #     batch x time x target_dim
        # --------------------------------------------------------------
        past_target = torch.as_tensor(
            context[:, outcome_index],
            dtype=torch.float32,
            device=self.device,
        )[None, :, None]

        past_observed_target = torch.ones_like(
            past_target,
            dtype=torch.bool,
        )

        # [False,...,False] so Moirai know no value is padding, all of them are important
        past_is_pad = torch.zeros(
            (1, context_length),
            dtype=torch.bool,
            device=self.device,
        )

        # --------------------------------------------------------------
        # Which non-target variables have known future trajectories?
        #
        # P0:
        #     known_future = None
        #
        # P1:
        #     known_future = {variable_index: future_values}
        #
        # Continuous works identically; future_values may contain any
        # number of steps, provided by the harness.
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
        # Known-future variables -> feat_dynamic_real.
        #
        # Each variable contributes:
        #
        #     observed history
        #          +
        #     supplied future trajectory
        #
        # Shape:
        #     batch x (context + horizon) x n_known_future
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
        # Their complete observed histories are retained.
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
        # Configure Moirai for this forecast request.
        #
        # prediction_length comes directly from the harness.
        # Nothing here assumes horizon=1.
        # --------------------------------------------------------------
        model = MoiraiForecast(
            module=module,
            prediction_length=horizon,
            context_length=context_length,
            patch_size=self.patch_size,
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=len(future_indices),
            past_feat_dynamic_real_dim=len(past_only_indices),
        )

        model.to(self.device)
        model.eval()

        # Same random draws for matched forecast conditions.
        self._reset_seed()

        with torch.inference_mode():

            samples = model(
                past_target=past_target,
                past_observed_target=past_observed_target,
                past_is_pad=past_is_pad,
                feat_dynamic_real=feat_dynamic_real,
                observed_feat_dynamic_real=observed_feat_dynamic_real,
                past_feat_dynamic_real=past_feat_dynamic_real,
                past_observed_feat_dynamic_real=(
                    past_observed_feat_dynamic_real
                ),
                num_samples=self.num_samples,
            )

        # --------------------------------------------------------------
        # target_dim = 1:
        #
        #     batch x samples x horizon
        # --------------------------------------------------------------
        sample_values = (
            samples[0]
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )

        quantiles = {
            q: np.quantile(sample_values,q,axis=0,).reshape(-1)
            for q in self.quantile_levels
        }

        return Forecast(
            point=quantiles[0.5].copy(),
            quantiles=quantiles,
        )