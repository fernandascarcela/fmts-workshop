"""
Toto-1.0 adapter for the P2† intervention interface.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import Forecast, Interface, ModelAdapter


class TotoAdapter(ModelAdapter):

    name = "toto-1.0"
    interface = Interface.P2

    def __init__(
        self,
        model_id: str = "Datadog/Toto-Open-Base-1.0",
        quantile_levels: tuple[float, ...] = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
        num_samples: int = 256,
        samples_per_batch: int = 32,
        seed: int = 42,
        device: str | None = None,
    ) -> None:

        self.model_id = model_id
        self.quantile_levels = quantile_levels

        self.num_samples = num_samples
        self.samples_per_batch = samples_per_batch
        self.seed = seed

        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._model = None
        self._forecaster = None

    def _load(self):
        """Load Toto-1.0 and its official forecaster once."""

        if self._model is None:

            from toto.inference.forecaster import TotoForecaster
            from toto.model.toto import Toto

            self._model = Toto.from_pretrained(
                self.model_id
            )

            self._model.to(self.device)
            self._model.eval()

            self._forecaster = TotoForecaster(
                self._model.model
            )

        return self._forecaster

    def _reset_seed(self) -> None:
        """
        Use common random numbers for P0 and P1.

        The runner calls P0 and P1 separately. Resetting Toto's sampling
        seed before each forecast ensures that differences are not caused
        merely by different Monte Carlo draws.
        """

        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def predict(
        self,
        context: np.ndarray,
        horizon: int,
        outcome_index: int,
        treatment: np.ndarray | None = None,
        treatment_index: int | None = None,
    ) -> Forecast:
        """
        Toto needs treatment_index in both P0 and P1 so that the treatment
        occupies the same final channel in both conditions.
        """

        if treatment_index is None:
            raise ValueError(
                "Toto requires treatment_index for both P0 and P1."
            )

        context = np.asarray(
            context,
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Toto requires exogenous variables to be the LAST channels.
        # Keep all observed variables, but move the treatment to the end.
        # --------------------------------------------------------------
        order = [
            i
            for i in range(context.shape[1])
            if i != treatment_index
        ]

        order.append(treatment_index)

        toto_context = context[:, order]

        # Toto expects:
        #
        #     variates x time
        #
        # whereas our probe uses:
        #
        #     time x variates
        #
        series_np = toto_context.T

        # Where did the queried outcome move after reordering?
        outcome_position = order.index(outcome_index)

        return self._run_toto(
            series_np=series_np,
            horizon=horizon,
            outcome_position=outcome_position,
            treatment=treatment,
        )

    def _run_toto(
        self,
        series_np: np.ndarray,
        horizon: int,
        outcome_position: int,
        treatment: np.ndarray | None,
    ) -> Forecast:

        from toto.data.util.dataset import MaskedTimeseries

        forecaster = self._load()

        series = torch.as_tensor(
            series_np,
            dtype=torch.float32,
            device=self.device,
        )

        n_vars, context_len = series.shape

        # --------------------------------------------------------------
        # Every DoTime variable belongs to the same multivariate series.
        #
        # Official Toto semantics:
        # variates sharing the same id_mask ID can attend to each other.
        # --------------------------------------------------------------
        id_mask = torch.zeros(
            (n_vars, context_len),
            dtype=torch.long,
            device=self.device,
        )

        padding_mask = torch.ones(
            (n_vars, context_len),
            dtype=torch.bool,
            device=self.device,
        )

        # Toto's API currently expects timestamp information even though
        # the current Toto-1.0 release does not use it for prediction.
        timestamps = (
            torch.arange(
                context_len,
                dtype=torch.long,
                device=self.device,
            )
            .unsqueeze(0)
            .repeat(n_vars, 1)
        )

        intervals = torch.ones(
            n_vars,
            dtype=torch.long,
            device=self.device,
        )

        inputs = MaskedTimeseries(
            series=series,
            padding_mask=padding_mask,
            id_mask=id_mask,
            timestamp_seconds=timestamps,
            time_interval_seconds=intervals,

            # Treatment is always the final channel.
            num_exogenous_variables=1,
        )

        # --------------------------------------------------------------
        # P0
        #
        # Same multivariate history, treatment marked in the same place,
        # but no known future intervention is supplied.
        # --------------------------------------------------------------
        if treatment is None:

            future_exogenous = None

        # --------------------------------------------------------------
        # P1 / P2†
        #
        # Supply the intervention trajectory for the final treatment
        # channel.
        # --------------------------------------------------------------
        else:

            treatment = np.asarray(
                treatment,
                dtype=np.float32,
            ).reshape(-1)

            future_exogenous = torch.as_tensor(
                treatment,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            # Shape:
            #
            #     (num_exogenous_variables, horizon)
            #
            # The forecaster automatically adds the batch dimension.

        # Same Monte Carlo random numbers for P0 and P1.
        self._reset_seed()

        toto_forecast = forecaster.forecast(
            inputs=inputs,
            prediction_length=horizon,
            num_samples=self.num_samples,
            samples_per_batch=self.samples_per_batch,
            use_kv_cache=True,
            future_exogenous_variables=future_exogenous,
        )

        # --------------------------------------------------------------
        # Toto output:
        #
        # quantile(q):
        #     batch x variates x horizon
        #
        # We have one batch item and want only the queried outcome.
        # --------------------------------------------------------------
        quantiles = {
            q: (
                toto_forecast
                .quantile(q)[0, outcome_position]
                .detach()
                .cpu()
                .numpy()
                .astype(float)
                .reshape(-1)
            )
            for q in self.quantile_levels
        }

        # Common probe convention:
        #
        #     point = median = q0.5
        #
        point = quantiles[0.5].copy()

        return Forecast(
            point=point,
            quantiles=quantiles,
        )

    # The Toto adapter overrides predict() because it needs the treatment
    # index even for P0 in order to preserve the same channel ordering.
    # This method exists only to satisfy the common adapter interface.
    def _forecast(
        self,
        context: np.ndarray,
        horizon: int,
        outcome_index: int,
        known_future: dict[int, np.ndarray] | None = None,
    ) -> Forecast:
        raise NotImplementedError(
            "TotoAdapter uses its custom predict() method."
        )