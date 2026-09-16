"""
TimesFM 2.5 adapter for the P1* intervention interface.
"""

from __future__ import annotations

import numpy as np

from .base import Forecast, Interface, ModelAdapter


class TimesFMAdapter(ModelAdapter):
    """Wrap TimesFM 2.5 + its deployed XReg path behind the probe contract."""

    name = "timesfm-2.5-xreg"
    interface = Interface.P1_STAR

    def __init__(
        self,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        quantile_levels: tuple[float, ...] = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
        max_context: int = 1024,
        max_horizon: int = 128,
        xreg_mode: str = "xreg + timesfm",
        normalize_xreg_target_per_input: bool = True,
        ridge: float = 0.0,
        force_xreg_cpu: bool = True,
        torch_compile: bool = True,
    ) -> None:

        self.model_id = model_id
        self.quantile_levels = tuple(quantile_levels)

        self.max_context = max_context
        self.max_horizon = max_horizon

        self.xreg_mode = xreg_mode
        self.normalize_xreg_target_per_input = (
            normalize_xreg_target_per_input
        )

        self.ridge = ridge
        self.force_xreg_cpu = force_xreg_cpu
        self.torch_compile = torch_compile

        self._model = None

    def _load(self):
        """Load and compile TimesFM 2.5 once, then reuse it."""

        if self._model is None:

            import torch
            import timesfm

            # Recommended by the official TimesFM PyTorch example.
            torch.set_float32_matmul_precision("high")

            model = (
                timesfm.TimesFM_2p5_200M_torch
                .from_pretrained(
                    self.model_id,
                    torch_compile=self.torch_compile,
                )
            )

            model.compile(
                timesfm.ForecastConfig(
                    max_context=self.max_context,
                    max_horizon=self.max_horizon,

                    # Recommended TimesFM preprocessing.
                    normalize_inputs=True,

                    # One DoTime episode is evaluated at a time.
                    per_core_batch_size=1,

                    # Keep probabilistic forecasts.
                    use_continuous_quantile_head=True,

                    # Official TimesFM forecasting options.
                    force_flip_invariance=True,

                    # DoTime variables can be negative.
                    infer_is_positive=False,

                    # Keep q0.1 <= ... <= q0.9.
                    fix_quantile_crossing=True,

                    # REQUIRED by forecast_with_covariates().
                    return_backcast=True,
                )
            )

            self._model = model

        return self._model

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
        # TimesFM core input
        #
        # The foundation-model core is univariate.
        # Only the queried outcome is used as the target series.
        # --------------------------------------------------------------
        target_history = np.asarray(
            context[:, outcome_index],
            dtype=np.float32,
        )

        model = self._load()

        # ==============================================================
        # P0: observational anchor
        #
        # No covariate / intervention path is activated.
        # ==============================================================
        if known_future is None:

            point_raw, quantile_raw = model.forecast(
                horizon=horizon,
                inputs=[target_history],
            )

            point_raw = np.asarray(point_raw, dtype=float)
            quantile_raw = np.asarray(quantile_raw, dtype=float)

            point = point_raw[0, -horizon:]
            q_values = quantile_raw[0, -horizon:, :]

        # ==============================================================
        # P1*: deployed TimesFM + XReg intervention path
        # ==============================================================
        else:

            dynamic_covariates: dict[
                str,
                list[np.ndarray],
            ] = {}

            for treat_index, treat_future in known_future.items():
                treat_future = np.asarray(
                    treat_future,
                    dtype=np.float32,
                ).reshape(-1)

                treatment_history = np.asarray(
                    context[:, treat_index],
                    dtype=np.float32,
                )

                # ------------------------------------------------------
                # TimesFM requires every dynamic XReg covariate to span:
                #
                #       context + forecast horizon
                #
                # The past is observational treatment history.
                # The future is the supplied do-trajectory.
                # ------------------------------------------------------
                treatment_full = np.concatenate(
                    [
                        treatment_history,
                        treat_future,
                    ]
                )

                dynamic_covariates[
                    f"treatment_{treat_index}"
                ] = [treatment_full]

            # ----------------------------------------------------------
            # forecast_with_covariates() infers the forecast horizon
            # from:
            #
            # len(dynamic_covariate) - len(target_history)
            #
            # Therefore there is no explicit `horizon=` argument here.
            # ----------------------------------------------------------
            point_raw, quantile_raw = (
                model.forecast_with_covariates(
                    inputs=[target_history],
                    dynamic_numerical_covariates=(dynamic_covariates),
                    xreg_mode=self.xreg_mode,
                    normalize_xreg_target_per_input=(self.normalize_xreg_target_per_input),
                    ridge=self.ridge,
                    force_on_cpu=self.force_xreg_cpu,
                )
            )

            point_raw = np.asarray(
                point_raw,
                dtype=float,
            )

            quantile_raw = np.asarray(
                quantile_raw,
                dtype=float,
            )

            # XReg returns only the inferred future test window.
            if point_raw.shape[1] != horizon:
                raise RuntimeError(
                    "TimesFM XReg returned "
                    f"{point_raw.shape[1]} forecast steps, "
                    f"expected {horizon}."
                )

            point = point_raw[0]
            q_values = quantile_raw[0]

        # --------------------------------------------------------------
        # Validate common output shape
        # --------------------------------------------------------------
        point = np.asarray(
            point,
            dtype=float,
        ).reshape(-1)

        q_values = np.asarray(
            q_values,
            dtype=float,
        )

        if len(point) != horizon:
            raise RuntimeError(
                f"TimesFM returned {len(point)} forecast steps, "
                f"expected {horizon}."
            )

        if q_values.ndim != 2:
            raise RuntimeError(
                "TimesFM quantile output must have shape "
                f"(horizon, channels); got {q_values.shape}."
            )

        # --------------------------------------------------------------
        # TimesFM quantile channels:
        #
        # index 0 = mean
        # index 1 = q0.1
        # index 2 = q0.2
        # ...
        # index 5 = q0.5
        # ...
        # index 9 = q0.9
        # --------------------------------------------------------------
        quantiles = {}

        for level in self.quantile_levels:

            channel = int(round(level * 10))

            quantiles[level] = (
                q_values[:, channel]
                .astype(float)
                .reshape(-1)
            )

        # TimesFM's public point forecast is the median.
        # We keep the model-returned point rather than recomputing it.
        return Forecast(
            point=point,
            quantiles=quantiles,
        )