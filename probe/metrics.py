"""
Metrics for the intervention probe.

Every metric is evaluated AT THE QUERY POINT only.

Ground truth:
  - y_true: exact realized interventional outcome at the query point.
  - true_effect: paired Y_int - Y_obs effect supplied by the evaluation row.
                 Effect metrics should only be used where that pairing is
                 scientifically valid for the evaluated suite.

Metric families:
  1. outcome level  -- how accurate was the forecast of Y_int?
  2. usage gap      -- did conditioning change the forecast?
  3. effect-delta   -- did the implied effect match the true effect? (PRIMARY)
  4. direction      -- did it get the sign right? (DoTime implementation)
  5. calibration    -- coverage + approximate CRPS on the P1 predictive
                       distribution against the exact interventional outcome.

Effect-delta is currently reported on the raw scale, matching the released
DoTime evaluation path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dotime.evaluation import (
    DIR_ACC_EPS,
    direction_accuracy as dotime_direction_accuracy,
)

# The common decile grid, the roster's intersection.
DECILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quantile_array(quantiles, level):
    """Return one predictive quantile array.

    Quantile keys may be numeric in memory or strings after a JSON round-trip.
    Matching is numeric, rounded to six decimals, so 0.3, "0.3" and
    "0.30000000000000004" all resolve to the same requested decile.

    Raises
    ------
    KeyError
        If the requested quantile is absent.
    ValueError
        If multiple keys collapse to the same quantile level.
    """
    target = round(float(level), 6)
    matches = []

    for key, value in quantiles.items():
        try:
            key_level = round(float(key), 6)
        except (TypeError, ValueError):
            continue
        if key_level == target:
            matches.append(value)

    if len(matches) == 1:
        return np.asarray(matches[0], dtype=float)
    if len(matches) > 1:
        raise ValueError(f"Ambiguous quantile keys matching level {level}.")
    raise KeyError(f"Missing quantile level {level}.")


# ---------------------------------------------------------------------------
# 1. Outcome level  --  did the model predict Y_int accurately?
# ---------------------------------------------------------------------------

def outcome_error(point_pred, y_true):
    """Per-episode signed outcome error: prediction - Y_int."""
    return np.asarray(point_pred, float) - np.asarray(y_true, float)


def outcome_rmse(point_pred, y_true):
    """Outcome-level RMSE against the realized interventional outcome."""
    err = outcome_error(point_pred, y_true)
    return float(np.sqrt(np.mean(err ** 2)))


def outcome_mae(point_pred, y_true):
    """Outcome-level MAE against the realized interventional outcome."""
    err = outcome_error(point_pred, y_true)
    return float(np.mean(np.abs(err)))


# ---------------------------------------------------------------------------
# 2. Usage gap  --  did conditioning change anything?
# ---------------------------------------------------------------------------

def usage_gap(point_p1, point_p0):
    """Primary usage gap: |pred_P1 - pred_P0| at the query point, per episode.

    Large -> the model reacted to the intervention.

    ~0 -> conditioning produced little change in the forecast; this does not
          by itself imply that the intervention was ignored, since some
          interventions have genuinely small effects.
    """
    return np.abs(np.asarray(point_p1, float) - np.asarray(point_p0, float))


def usage_gap_delta_err(point_p1, point_p0, y_true):
    """Did conditioning help? delta_err = |err(P1)| - |err(P0)|, per episode.

    Negative -> conditioning moved the forecast closer to Y_int.
    Positive -> conditioning moved the forecast farther from Y_int.

    This is an absolute-error difference and is reported alongside the
    primary usage gap.
    """
    y = np.asarray(y_true, float)
    err_p1 = np.abs(np.asarray(point_p1, float) - y)
    err_p0 = np.abs(np.asarray(point_p0, float) - y)
    return err_p1 - err_p0


# ---------------------------------------------------------------------------
# 3. Effect-delta  --  PRIMARY. Did the implied effect match the true effect?
# ---------------------------------------------------------------------------

def implied_effect(point_p1, point_p0):
    """What the model implies the intervention does: pred_P1 - pred_P0."""
    return np.asarray(point_p1, float) - np.asarray(point_p0, float)


def effect_delta(point_p1, point_p0, true_effect):
    """Per-episode signed effect error, on the raw scale.

    implied effect = P1 - P0
    true effect    = Y_int - Y_obs at the query point

    Methods with no intervention-sensitive channel have P0 == P1 and
    therefore imply zero effect. Such methods tie on effect metrics even
    when their outcome-level forecast errors differ.
    """
    return implied_effect(point_p1, point_p0) - np.asarray(true_effect, float)


def effect_delta_rmse(point_p1, point_p0, true_effect):
    """PRIMARY aggregate: RMSE of the raw effect error."""
    delta = effect_delta(point_p1, point_p0, true_effect)
    return float(np.sqrt(np.mean(delta ** 2)))


def effect_delta_mae(point_p1, point_p0, true_effect):
    """Secondary aggregate: MAE of the raw effect error."""
    delta = effect_delta(point_p1, point_p0, true_effect)
    return float(np.mean(np.abs(delta)))


# ---------------------------------------------------------------------------
# 4. Direction accuracy  --  reuse the DoTime implementation
# ---------------------------------------------------------------------------

def direction_accuracy(point_p1, point_p0, true_effect):
    """Direction accuracy using DoTime's official implementation.

    Returns
    -------
    dict
        {
            "accuracy":   fraction correct among eligible episodes,
            "n_valid":    number with |true_effect| >= DIR_ACC_EPS,
            "n_excluded": number excluded as near-zero effect,
        }

    Near-zero effects are handled by DoTime using DIR_ACC_EPS.

    Direction is a weak metric: a model can miss the magnitude badly and
    still score the sign correctly. Report it, but let effect-delta decide.
    """
    implied = implied_effect(point_p1, point_p0)
    true = np.asarray(true_effect, float)

    preds = torch.as_tensor(implied, dtype=torch.float32).reshape(-1)
    targets = torch.as_tensor(true, dtype=torch.float32).reshape(-1)

    return dotime_direction_accuracy(preds, targets, eps=DIR_ACC_EPS)


# ---------------------------------------------------------------------------
# 5. Calibration  --  coverage + approximate CRPS on Y_int
# ---------------------------------------------------------------------------

def _pinball(q_level, q_pred, y_true):
    """Pinball loss for one quantile level, per episode."""
    if not 0.0 < q_level < 1.0:
        raise ValueError("q_level must be between 0 and 1.")

    q_pred = np.asarray(q_pred, float)
    y_true = np.asarray(y_true, float)
    diff = y_true - q_pred

    # y above prediction -> weight q ; y below prediction -> weight (1 - q)
    return np.where(diff >= 0, q_level * diff, (q_level - 1.0) * diff)


def coverage_80_per_episode(quantiles, y_true):
    """Per-episode indicator for central 80% predictive-interval coverage.

    Episodes with a missing or non-finite q10, q90, or target return NaN
    rather than 0.0, so bootstrap_ci excludes them and reports the effective
    denominator in `n`. Comparison operators evaluate to False on NaN, so a
    plain `(y >= lo) & (y <= hi)` would score an absent prediction as an
    interval miss.
    """
    lo = _quantile_array(quantiles, 0.1)
    hi = _quantile_array(quantiles, 0.9)
    y = np.asarray(y_true, float)

    valid = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y)

    return np.where(valid, (y >= lo) & (y <= hi), np.nan)


def coverage_80(quantiles, y_true):
    """PRIMARY calibration: empirical central-80% interval coverage.

    The predictive interval is [q10, q90] from the P1 distribution.

    ~0.80 -> nominal coverage
    <0.80 -> undercoverage: intervals are too narrow / overconfident
    >0.80 -> overcoverage: intervals are too wide / underconfident
    """
    per_episode = coverage_80_per_episode(
        quantiles,
        y_true,
    )

    valid = per_episode[np.isfinite(per_episode)]

    if valid.size == 0:
        return float("nan")

    return float(np.mean(valid))


def coverage_ladder(quantiles, y_true, levels=DECILES):
    """Appendix: empirical coverage for each available central interval.

    For the decile grid this reports:

        [q10, q90] -> nominal 0.80
        [q20, q80] -> nominal 0.60
        [q30, q70] -> nominal 0.40
        [q40, q60] -> nominal 0.20

    As in coverage_80_per_episode, episodes with missing or non-finite bounds
    are excluded rather than counted as interval misses. A level with no valid
    episode is reported as NaN, so it stays visible in the table instead of
    disappearing.
    """
    y = np.asarray(y_true, float)
    ladder = {}

    for lo_q in [level for level in levels if level < 0.5]:
        hi_q = round(1.0 - lo_q, 2)
        try:
            lo = _quantile_array(quantiles, lo_q)
            hi = _quantile_array(quantiles, hi_q)
        except KeyError:
            continue

        nominal = round(hi_q - lo_q, 2)
        valid = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y)

        if not valid.any():
            ladder[nominal] = float("nan")
            continue

        covered = (y[valid] >= lo[valid]) & (y[valid] <= hi[valid])
        ladder[nominal] = float(np.mean(covered))

    return ladder


def crps(quantiles, y_true, levels=DECILES):
    """Approximate CRPS: 2 x mean pinball loss over the nine deciles.

    The exact CRPS is 2 * integral of the pinball loss over q in [0, 1].

    Two departures, both stated in the paper:

    1. Discretisation. Nine equally spaced levels are averaged rather than
       integrated. Rectangle-rule integration would multiply this by 0.9,
       a constant, so rankings are unaffected.

    2. Tails. Quantiles below 0.1 and above 0.9 are not available on the
       common grid, so those regions of the integral are unobserved. No
       rescaling fixes this.

    Levels match the released DoTime metric functions for leaderboard
    comparability.

    Returns one approximate-CRPS value per episode. Lower is better.
    """
    y = np.asarray(y_true, float)
    per_level = [_pinball(q, _quantile_array(quantiles, q), y) for q in levels]
    mean_pinball = np.mean(per_level, axis=0)

    return 2.0 * mean_pinball


# ---------------------------------------------------------------------------
# Aggregation with bootstrap CIs over episodes
# ---------------------------------------------------------------------------

@dataclass
class Estimate:
    """Point estimate with percentile-bootstrap CI and effective sample size."""

    estimate: float
    lo: float
    hi: float
    n: int


# Aggregators for bootstrap_ci. Each reduces a bootstrap matrix along the
# episode axis:
#
#   mean usage gap     bootstrap_ci(usage_gap(...), AGG_MEAN)
#   delta-error mean   bootstrap_ci(usage_gap_delta_err(...), AGG_MEAN)
#   P1 outcome RMSE    bootstrap_ci(outcome_error(...), AGG_RMSE)
#   P1 outcome MAE     bootstrap_ci(outcome_error(...), AGG_MAE)
#   effect-delta RMSE  bootstrap_ci(effect_delta(...), AGG_RMSE)
#   effect-delta MAE   bootstrap_ci(effect_delta(...), AGG_MAE)
#   80% coverage       bootstrap_ci(coverage_80_per_episode(...), AGG_MEAN)
#   approximate CRPS   bootstrap_ci(crps(...), AGG_MEAN)

def AGG_MEAN(sample, axis):
    """Mean over the requested axis."""
    return np.mean(sample, axis=axis)


def AGG_MAE(sample, axis):
    """Mean absolute value over the requested axis."""
    return np.mean(np.abs(sample), axis=axis)


def AGG_RMSE(sample, axis):
    """Root mean squared value over the requested axis."""
    return np.sqrt(np.mean(sample ** 2, axis=axis))


def bootstrap_ci(per_episode, agg, n_boot=10_000, alpha=0.05, seed=0, block=500):
    """Bootstrap confidence interval over episodes.

    Resampling is vectorised in blocks rather than materialising the complete
    (n_boot, n_episodes) matrix at once. This keeps peak memory bounded while
    avoiding a 10,000-iteration Python loop.

    `block` controls only how many bootstrap replicates are materialised at
    once; it is a memory/performance parameter, not a statistical one. With
    the default NumPy int64 index draws used here, fixed `seed` and `n_boot`
    produce the same concatenated draws regardless of `block`. Changing the
    integer dtype is therefore a reproducibility-relevant code change.

    Non-finite episode values are excluded before aggregation, and
    `Estimate.n` records how many finite values actually contributed. That
    matters for methods or interfaces where a metric is unavailable on some
    or all episodes.

    Parameters
    ----------
    per_episode:
        One scalar value per episode.
    agg:
        Statistic recomputed on each bootstrap sample. Must accept an array
        and an `axis` argument. Use AGG_MEAN, AGG_MAE, or AGG_RMSE above.
    n_boot:
        Number of bootstrap replicates.
    alpha:
        Two-sided interval error rate. alpha=0.05 gives a 95% percentile CI.
    seed:
        Bootstrap RNG seed, separate from the episode-selection and
        model-inference seeds.
    block:
        Number of bootstrap replicates processed at once.

    Returns
    -------
    Estimate
        Point estimate, percentile-bootstrap bounds, and the effective number
        of finite episodes contributing to the statistic.
    """
    if n_boot <= 0:
        raise ValueError("n_boot must be positive.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    if block <= 0:
        raise ValueError("block must be positive.")

    x = np.asarray(per_episode, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    n = int(x.size)

    if n == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)

    for start in range(0, n_boot, block):
        size = min(block, n_boot - start)
        idx = rng.integers(low=0, high=n, size=(size, n))
        boots[start:start + size] = agg(x[idx], axis=1)

    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point_estimate = agg(x[None, :], axis=1)[0]

    return Estimate(float(point_estimate), float(lo), float(hi), n)