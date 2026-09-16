"""
Do-Over-Time-PFN adapter.

Uses the official DoTime PFN evaluation path.
"""

from __future__ import annotations

import numpy as np
import torch

from huggingface_hub import hf_hub_download

from .base import Forecast, Interface, ModelAdapter


STRUCTURE_CODES = {
    "back_door": "bd",
    "front_door": "fd",
    "instrumental_variable": "iv",
}


class DoTimePFNAdapter(ModelAdapter):

    interface = Interface.P1

    def __init__(
        self,
        seed: int = 42,
        observational: bool = False,
        repo_id: str = "thummd/do-over-time-pfn",
        device: str | None = None,
        use_all: bool = False,
    ) -> None:

        if seed not in (42, 43, 44):
            raise ValueError(
                "DoTime-PFN seed must be 42, 43, or 44."
            )

        self.seed = seed
        self.observational = observational
        self.repo_id = repo_id
        self.use_all = use_all

        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        mode = "obs" if observational else "causal"

        self.name = f"dotime-pfn-{mode}-seed{seed}"

        self._p0_models = {}
        self._p1_models = {}

    def _checkpoint_folder(
        self,
        structure: str,
        observational: bool,
    ) -> str:

        mode = "obs" if observational else "causal"

        # General checkpoint trained jointly.
        if self.use_all:
            if self.seed != 42:
                raise ValueError(
                    "The public 'all' checkpoint is only available for the main seed."
                )

            return f"s9ho_all_{mode}"

        # Structure-matched checkpoints.
        if structure not in STRUCTURE_CODES:
            raise ValueError(
                f"Unsupported DoTime-PFN structure: {structure}"
            )

        code = STRUCTURE_CODES[structure]
        suffix = "" if self.seed == 42 else f"_seed{self.seed}"

        return f"s9ho_{code}_{mode}{suffix}"

    def _download_checkpoint(
        self,
        structure: str,
        observational: bool,
    ) -> str:

        folder = self._checkpoint_folder(
            structure,
            observational,
        )

        return hf_hub_download(
            repo_id=self.repo_id,
            filename=f"{folder}/do_over_time_pfn_best.pt",
        )

    def _load_models(self, structure):
        key = "__all__" if self.use_all else structure

        if self.observational:
            if key in self._p0_models:
                return

            from dotime.reference.pfn import PFNRef

            checkpoint = self._download_checkpoint(
                structure,
                observational=True,
            )

            self._p0_models[key] = PFNRef(
                checkpoint=checkpoint,
                device=self.device,
                observational=True,
            )

        else:
            if key in self._p1_models:
                return

            from dotime.reference.pfn import PFNRef

            checkpoint = self._download_checkpoint(
                structure,
                observational=False,
            )

            self._p1_models[key] = PFNRef(
                checkpoint=checkpoint,
                device=self.device,
                observational=False,
            )

    def predict_episode(self, episode) -> Forecast:
        """
        Predict the queried outcome for one complete DoTime episode.

        observational=False -> causal PFN
        observational=True  -> observational PFN
        """
        structure = episode.structure
        self._load_models(structure)

        key = "__all__" if self.use_all else structure

        if self.observational:
            model = self._p0_models[key]
        else:
            model = self._p1_models[key]

        pred = (
            model.predict(episode)
            .detach()
            .cpu()
            .numpy()
            .astype(float)
            .reshape(-1)
        )

        return Forecast(
            point=pred,
            quantiles={},
        )

    def predict(
        self,
        context: np.ndarray,
        horizon: int,
        outcome_index: int,
        treatment: np.ndarray | None = None,
        treatment_index: int | None = None,
    ) -> Forecast:

        raise NotImplementedError(
            "DoTimePFNAdapter requires the complete DoTime Episode. "
            "Use predict_episode()."
        )

    def _forecast(
        self,
        context: np.ndarray,
        horizon: int,
        outcome_index: int,
        known_future: dict[int, np.ndarray] | None = None,
    ) -> Forecast:

        raise NotImplementedError(
            "DoTimePFNAdapter uses predict_episode()."
        )