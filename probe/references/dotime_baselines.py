"""Released DoTime outcome-level baselines.

TrajMean, AR1 and Zero do not consume the intervention value. Therefore,
for the probe we represent them as producing the same forecast under P0
and P1, giving an implied intervention effect of zero by construction --
which is the paper's point: a mean predictor ties on level RMSE while
scoring zero on effect-delta.

Oracle is different: DoTime's OracleBaseline predicts the true
interventional outcome level. It is therefore an outcome-level upper bound,
not a matched P0/P1 intervention probe. Effect-level Oracle evaluation lives
in probe/anchors.py.

Rows use the same schema as run_one(), so the metrics layer needs no special
case. Query-point targets come from EpisodeInputs rather than being sliced
again here -- one definition of the effect, in one place.
"""

from __future__ import annotations

from dotime.baselines import (
    MeanBaseline,
    AR1Baseline,
    ZeroBaseline,
    OracleBaseline,
)

from ..interfaces import parse_episode, to_numpy

# No intervention channel -> matched P0/P1, zero implied effect.
BASELINES = {
    "trajmean": MeanBaseline,
    "ar1": AR1Baseline,
    "zero": ZeroBaseline,
}

ORACLE_BASELINE = OracleBaseline


def _predict_at_query(baseline, ep) -> float:
    """Read the baseline prediction at the query.

    Released DoTime baselines currently return a one-element tensor, but
    [-1] stays correct if a future version returns a full trajectory.
    """
    return float(to_numpy(baseline.predict(ep)).reshape(-1)[-1])


def _row(index, ep, inp, *, interface, p0, p1, implied_effect) -> dict:
    """Assemble one result row in the run_one() schema."""
    return {
        "episode": int(index),
        "structure": str(ep.structure),
        "interface": interface,
        "horizon": int(inp.horizon),
        "query_step": int(inp.query_step),
        "p0_query": p0,
        "p1_query": p1,
        "p0_quantiles": None,          # point predictors only
        "p1_quantiles": None,
        "implied_effect": implied_effect,
        "y_obs_query": float(inp.y_obs),
        "y_true_query": float(inp.y_true),
        "true_effect": float(inp.true_effect),
        "status": "OK",
    }


def _run(suite, suite_name, indices, make_row):
    """Shared loop; make_row(index, ep, inp) builds the successful row."""
    rows = []

    for position, index in enumerate(indices, start=1):
        if position % 100 == 0:
            print(f"  {position}/{len(indices)}")

        ep = suite[index]
        try:
            rows.append(make_row(index, ep, parse_episode(ep, suite_name)))
        except Exception as exc:
            rows.append({
                "episode": int(index),
                "structure": str(ep.structure),
                "status": f"FAILED: {type(exc).__name__}: {exc}",
            })

    return rows


def run_baseline(baseline, suite, suite_name, indices):
    """Run one non-interventional DoTime baseline on fixed episode IDs."""

    def make_row(index, ep, inp):
        pred = _predict_at_query(baseline, ep)
        # No intervention channel, so P1 is the same forecast as P0.
        return _row(index, ep, inp, interface="BASELINE",
                    p0=pred, p1=pred, implied_effect=0.0)

    return _run(suite, suite_name, indices, make_row)


def run_oracle(suite, suite_name, indices):
    """Run DoTime's Oracle outcome-level upper bound.

    OracleBaseline returns the true interventional outcome. We deliberately
    do not fabricate a matched P0, so implied_effect stays None: an Oracle
    with a zero implied effect would turn the paper's ceiling into a floor.
    """
    baseline = ORACLE_BASELINE()

    def make_row(index, ep, inp):
        pred = _predict_at_query(baseline, ep)
        return _row(index, ep, inp, interface="ORACLE_OUTCOME",
                    p0=None, p1=pred, implied_effect=None)

    return _run(suite, suite_name, indices, make_row)