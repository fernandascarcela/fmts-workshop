"""
TabPFN-TS adapter for the P1 intervention interface.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .base import Forecast, Interface, ModelAdapter


class TabPFNTSAdapter(ModelAdapter):
    """Wrap TabPFN-TS behind the common probe-harness contract."""

    name = "tabpfn-ts"
    interface = Interface.P1

    def __init__(
        self,
        device: str = "cuda",
        quantile_levels: tuple[float, ...] = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
        max_context_length: int = 32768,
    ) -> None:
        self.device = device
        self.quantile_levels = tuple(quantile_levels)
        self.max_context_length = max_context_length
        self._pipeline = None

    def _load(self):
        """Create the TabPFN-TS pipeline once, then reuse it."""

        if self._pipeline is None:
            from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline
            from tabpfn_time_series.features import RunningIndexFeature

            self._pipeline = TabPFNTSPipeline(
                tabpfn_mode=TabPFNMode.LOCAL,
                max_context_length=self.max_context_length,
                tabpfn_model_config={"device": self.device},

                # DoTime timestamps are synthetic.
                # Keep temporal order, but remove arbitrary calendar semantics.
                temporal_features=[RunningIndexFeature()],
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
        # The actual calendar date has no semantic meaning for DoTime.
        # It exists only because TabPFN-TS expects timestamps.
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
        # This context is identical for P0 and P1.
        # --------------------------------------------------------------
        context_df = pd.DataFrame(
            {
                "item_id": 0,
                "timestamp": timestamps,
                "target": context[:, outcome_index],
            }
        )

        for c in range(n_vars):
            if c != outcome_index:
                context_df[f"cov_{c}"] = context[:, c]

        # --------------------------------------------------------------
        # Future timestamps
        # --------------------------------------------------------------
        future_timestamps = pd.date_range(
            start=timestamps[-1],
            periods=horizon + 1,
            freq=freq,
        )[1:]

        # --------------------------------------------------------------
        # Future covariates
        #
        # TabPFN-TS only uses covariates that are present in both
        # context_df and future_df.
        #
        # 'You can pass multiple item_ids here as long as future_df contains each item's horizon covariates." - quickstart tutorial
        #
        # Therefore every non-target variable is included here.
        #
        # Initially all future values are unknown (NaN). This is P0.
        # P1 will overwrite only the treatment variable with its
        # intervention trajectory.
        # --------------------------------------------------------------
        future_df = pd.DataFrame(
            {
                "item_id": 0,
                "timestamp": future_timestamps,
            }
        )

        for c in range(n_vars):
            if c != outcome_index:
                future_df[f"cov_{c}"] = np.nan

        # --------------------------------------------------------------
        # P1 intervention
        #
        # known_future is None:
        #     P0 -> every future covariate remains unknown.
        #
        # known_future is provided:
        #     P1 -> overwrite only the intervened treatment variable.
        #
        # Importantly, historical context does NOT change.
        # --------------------------------------------------------------
        if known_future is not None:

            for treat_index, treat_future in known_future.items():

                treat_future = np.asarray(
                    treat_future,
                    dtype=float,
                ).reshape(-1)

                future_df[f"cov_{treat_index}"] = treat_future

        # --------------------------------------------------------------
        # Forecast
        #
        # We deliberately use future_df for BOTH P0 and P1 so that
        # both conditions follow the same TabPFN-TS API path.
        #
        # TabPFN-TS warns when future covariates contain NaN. Here those
        # NaNs are intentional: they mean "future value unknown".
        # --------------------------------------------------------------
        pipeline = self._load()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Some covariate values in future_tsdf are missing.*",
                category=UserWarning,
            )

            pred_df = pipeline.predict_df(
                context_df=context_df,
                future_df=future_df,
                quantiles=list(self.quantile_levels),
            )

        # --------------------------------------------------------------
        # Parse probabilistic output
        #
        # Quantile columns in the installed TabPFN-TS version use
        # float column names, e.g. 0.1, 0.5, 0.9.
        # --------------------------------------------------------------
        quantiles = {
            level: pred_df[level]
            .to_numpy(dtype=float)
            .reshape(-1)
            for level in self.quantile_levels
        }

        # TabPFN-TS uses the median as its point prediction by default.
        point_forecast = (
            pred_df["target"]
            .to_numpy(dtype=float)
            .reshape(-1)
        )

        return Forecast(
            point=point_forecast,
            quantiles=quantiles,
        )