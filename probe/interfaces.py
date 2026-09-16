"""
Applies the P0 / P1 (P2) conditioning interfaces to a single DoTime episode,
using any ModelAdapter.

This layer speaks ONLY the neutral array format. It never knows how a specific
model consumes covariates -- that translation lives inside each adapter. Its
whole job is:

  1. slice the episode into (context, horizon, treatment trajectory);
  2. call the adapter twice -- once WITHOUT the treatment (P0, the anchor) and
     once WITH it (P1/P2, the intervention);
  3. hand back both forecasts plus the exact ground-truth effect, ready for
     the metrics layer.

Interface-limited models (Finding 0) have no do-value channel, so only P0 is
produced and P1 is left as None.

DoTime-PFN is handled separately because it requires the complete DoTime
episode rather than the neutral context/horizon interface. Its causal and
observational checkpoints are separate trained models, so they must not be
subtracted as if they were matched P0/P1 runs of the same frozen model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adapters.base import Forecast, Interface, ModelAdapter


def to_numpy(x) -> np.ndarray:
    """
    DoTime episode fields are torch Tensors. So we convert to numpy safely,
    handling tensors that live on the GPU or carry gradients.

    Public because the reference/baseline modules must convert values the
    same way -- a second local copy would risk diverging on dtype.
    """
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(float)

    return np.asarray(x, dtype=float)


# Kept so existing internal call sites keep working.
_to_numpy = to_numpy


@dataclass
class EpisodeInputs:
    """The neutral, model-agnostic slice of one episode."""

    context: np.ndarray           # observational history before the intervention, shape (t_start, n_vars)
    horizon: int                  # number of forecast steps from t_start up to and including query_step
    outcome_index: int            # index of the variable whose value is queried/evaluated
    treatment_index: int          # index of the variable receiving the intervention
    treatment_value: float        # scalar do-value imposed on the treatment variable

    # Outcome levels at the query point. Carried here so that every consumer
    # (models, baselines, metrics) reads one definition instead of slicing
    # x_obs / x_int again on its own.
    y_obs: float                  # factual outcome level
    y_true: float                 # interventional (counterfactual) outcome level
    true_effect: float            # ground-truth causal effect, y_true - y_obs

    structure: str                # DoTime causal/structural label for the episode

    t_start: int                  # discrete step at which the intervention begins
    t_end: int                    # first discrete step after the intervention window; window is [t_start, t_end)
    query_step: int               # discrete time step at which the requested outcome is evaluated


@dataclass
class ProbeResult:
    """Everything the metrics layer needs for one (model, episode) pair."""

    model: str
    structure: str
    interface: Interface

    # DoTime-PFN causal has only pred_p1, while the observational
    # checkpoint has only pred_p0. Standard TSFMs normally have pred_p0
    # and, when supported, pred_p1.
    pred_p0: Forecast | None
    pred_p1: Forecast | None

    true_effect: float


def get_query_step(ep, suite_name: str) -> int:
    """Return the discrete query step for the episode."""

    query_time = float(to_numpy(ep.query_time).reshape(-1)[0])
    length = int(ep.length)

    if suite_name == "dot-Identifiability-v1":
        return int(np.rint(query_time * length))

    if suite_name == "dot-Continuous-v1":
        return int(np.rint(query_time * (length - 1)))

    raise ValueError(f"Unsupported suite: {suite_name}")


def get_t_end(ep, suite_name: str) -> int:
    """Return the first step after the intervention window."""

    t_start = int(ep.intervention.times[0])

    if suite_name == "dot-Identifiability-v1":
        return t_start + 1

    if suite_name == "dot-Continuous-v1":
        _, t_end = reconstruct_window(ep)
        return t_end

    raise ValueError(f"Unsupported suite: {suite_name}")


def parse_episode(ep, suite_name: str) -> EpisodeInputs:
    """Extract the neutral, model-agnostic inputs from one DoTime episode."""

    x_obs = to_numpy(ep.x_obs)
    x_int = to_numpy(ep.x_int)

    treatment_index = int(ep.intervention.targets[0])
    outcome_index = int(ep.query_target[0])
    treatment_value = float(
        np.asarray(ep.intervention.values).reshape(-1)[0]
    )

    t_start = int(ep.intervention.times[0])
    t_end = get_t_end(ep, suite_name)
    query_step = get_query_step(ep, suite_name)

    # Factual observational history strictly before the intervention.
    # P0 and P1 receive exactly the same pre-intervention context.
    context = x_obs[:t_start]

    # Forecast from the intervention onset through the query step,
    # including the query itself.
    horizon = query_step - t_start + 1

    if horizon < 1:
        raise ValueError(
            f"Invalid horizon: t_start={t_start}, query_step={query_step}"
        )

    # Outcome levels at the query point, from the released paired
    # trajectories. The counterfactual is exact because both runs share
    # the same noise realisation.
    y_obs = float(x_obs[query_step, outcome_index])
    y_true = float(x_int[query_step, outcome_index])

    # Ground-truth causal effect at the query point.
    #
    # NOTE ON THE STORED FIELD:
    # metadata["y_causal_effect"] is not the same object across suites. In
    # dot-Continuous-v1 it is the paired contrast and agrees with the
    # reconstruction below. In dot-Identifiability-v1 it equals the
    # interventional LEVEL y_true -- consistent with the DoTime paper, whose
    # Algorithm 2 defines the counterfactual target as x_int[t_q, Y], and
    # with the generator computing the difference against a causally masked
    # observational trajectory (zero from onset).
    #
    # Our estimand is a difference of predictions (P1 - P0), so we score it
    # against the paired contrast and reconstruct it here for both suites.
    # Level-scale comparisons against the DoTime leaderboard use y_true.
    true_effect = y_true - y_obs

    return EpisodeInputs(
        context=context,
        horizon=horizon,
        outcome_index=outcome_index,
        treatment_index=treatment_index,
        treatment_value=treatment_value,
        y_obs=y_obs,
        y_true=y_true,
        true_effect=true_effect,
        structure=ep.structure,
        t_start=t_start,
        t_end=t_end,
        query_step=query_step,
    )


def reconstruct_window(ep):
    """
    Recover the discrete intervention window [t_start, t_end)
    from a frozen Continuous episode.
    """

    x_obs = to_numpy(ep.x_obs)
    x_int = to_numpy(ep.x_int)

    target = int(ep.intervention.targets[0])
    t_start = int(ep.intervention.times[0])
    do_value = float(
        np.asarray(ep.intervention.values).reshape(-1)[0]
    )

    # The frozen Continuous benchmark currently contains HARD scalar
    # interventions. Fail explicitly if that assumption is violated.
    if str(ep.intervention.intervention_type) != "InterventionType.HARD":
        raise ValueError("Unsupported intervention type.")

    # Before the intervention begins, factual and interventional
    # trajectories must be exactly identical.
    if not np.array_equal(
        x_obs[:t_start],
        x_int[:t_start],
    ):
        raise ValueError(
            "x_obs and x_int differ before intervention start."
        )

    treatment_int = x_int[:, target]

    # During a HARD intervention, the treatment channel is exactly equal
    # to the imposed do-value. The first different value marks t_end.
    t_end = t_start

    while (
        t_end < len(treatment_int)
        and treatment_int[t_end] == do_value
    ):
        t_end += 1

    if t_end == t_start:
        raise ValueError(
            "Could not reconstruct intervention window."
        )

    return t_start, t_end


def run_episode(
    adapter: ModelAdapter,
    ep,
    suite_name: str,
) -> ProbeResult:
    """Apply P0 and P1/P2 to one episode with one model."""

    inp = parse_episode(
        ep,
        suite_name=suite_name,
    )

    # --------------------------------------------------------------
    # DoTime-PFN
    #
    # Unlike the TSFMs, DoTime-PFN requires the COMPLETE DoTime
    # episode rather than the neutral context/horizon representation.
    #
    # Its causal and observational predictions also come from
    # SEPARATELY TRAINED checkpoints. Therefore:
    #
    # causal checkpoint        -> stored as pred_p1 only
    # observational checkpoint -> stored as pred_p0 only
    #
    # We intentionally never place both predictions in the same
    # ProbeResult, preventing the metrics layer from accidentally
    # interpreting causal - observational as the matched P1 - P0
    # intervention effect used for the TSFMs.
    # --------------------------------------------------------------
    if hasattr(adapter, "predict_episode"):

        pred = adapter.predict_episode(ep)

        if adapter.observational:
            pred_p0 = pred
            pred_p1 = None
        else:
            pred_p0 = None
            pred_p1 = pred

        return ProbeResult(
            model=adapter.name,
            structure=inp.structure,
            interface=adapter.interface,
            pred_p0=pred_p0,
            pred_p1=pred_p1,
            true_effect=inp.true_effect,
        )

    # --------------------------------------------------------------
    # Standard TSFM path
    # --------------------------------------------------------------

    # P0 -- unconditioned anchor: no treatment supplied.
    pred_p0 = adapter.predict(
        context=inp.context,
        horizon=inp.horizon,
        outcome_index=inp.outcome_index,
        treatment_index=inp.treatment_index,  # needed by Toto
    )

    # --------------------------------------------------------------
    # P1/P2 treatment trajectory.
    #
    # The do-value is supplied only while the intervention is active:
    #
    #     [t_start, t_end)
    #
    # Values after the intervention ends remain unknown (NaN).
    # --------------------------------------------------------------

    # Initialize the full treatment forecast as unknown (NaN).
    # Fill only the steps where the intervention is actually active,
    # leaving all post-intervention steps as NaN.
    treatment_future = np.full(
        inp.horizon,
        np.nan,
        dtype=float,
    )

    # Number of forecast steps that overlap with the intervention window
    # [t_start, t_end), truncated if the query occurs before the window ends.
    active_steps = (
        min(inp.t_end, inp.query_step + 1)
        - inp.t_start
    )

    # Supply the scalar do-value only during the active intervention window.
    treatment_future[:active_steps] = inp.treatment_value

    pred_p1: Forecast | None = None

    # Interface-limited models do not expose a channel through which the
    # future intervention trajectory can be supplied, so they stop at P0.
    if adapter.interface is not Interface.LIMITED:

        pred_p1 = adapter.predict(
            context=inp.context,
            horizon=inp.horizon,
            outcome_index=inp.outcome_index,
            treatment=treatment_future,
            treatment_index=inp.treatment_index,
        )

    return ProbeResult(
        model=adapter.name,
        structure=inp.structure,
        interface=adapter.interface,
        pred_p0=pred_p0,
        pred_p1=pred_p1,
        true_effect=inp.true_effect,
    )