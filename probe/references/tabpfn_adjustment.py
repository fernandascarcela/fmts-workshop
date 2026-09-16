"""Static TabPFN causal-adjustment reference for DoTime Identifiability v1.1.0.

This is an FMTS adaptation of the released DoTime Static TabPFN reference.
It fixes the two representation issues that matter for the corrected
dot-Identifiability-v1 v1.1.0 artifact:

1) temporal leakage: every fitted regressor uses STRICTLY pre-intervention
   observational rows only;
2) variable roles: treatment / confounder / mediator / hidden-variable indices
   are explicit per structure. We never infer "the first non-(A,Y) column".

Canonical v1.1.0 layouts used here
---------------------------------
rct_no_confounding / bi_variate : [A, Y]
observed_confounder             : [A, X, Y]
back_door                       : [A, X, Y]
confounder_mediator             : [A, X, M, Y]
mediator                        : [A, M, Y]
front_door                      : [A, U(hidden), M, Y]
instrumental_variable           : [A, U(hidden), X, Y]
unobserved_confounder           : [A, U(hidden), Y]

Applicability
-------------
The released DoTime Static TabPFN reference defines:
- back-door adjustment for:
    observed_confounder, back_door, confounder_mediator
- front-door adjustment for:
    mediator, front_door
- pre-intervention outcome-mean fallback where those assumptions do not hold.

Accordingly, this module returns predictions for ALL eight structures, but it
does NOT invent a TabPFN-IV estimator and it does NOT pretend the hidden-
confounder structure is identifiable.  The following structures use the same
matched fallback in P0 and P1:
    rct_no_confounding / bi_variate
    instrumental_variable
    unobserved_confounder

IV2SLS should remain a separate causal reference if IV identification is to be
evaluated explicitly.

P0/P1
-----
For applicable adjustment structures, P0 and P1 use the SAME fitted TabPFN
regressor(s) and differ only in the treatment value inserted at prediction:
- P0: last observed treatment value before intervention;
- P1: the actual do(A=v) value.

This produces a clean matched reference contrast.  It must still be labelled
as a STRUCTURE-AWARE causal reference, not as an ordinary TSFM interface probe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tabpfn import TabPFNRegressor

from ..interfaces import parse_episode, to_numpy


MIN_HISTORY = 8

BACKDOOR_STRUCTURES = {
    "observed_confounder",
    "back_door",
    "confounder_mediator",
}

FRONTDOOR_STRUCTURES = {
    "mediator",
    "front_door",
}

FALLBACK_STRUCTURES = {
    "rct_no_confounding",
    "bi_variate",  # current code alias; released v1 suite uses rct_no_confounding
    "instrumental_variable",
    "unobserved_confounder",
}

ALL_STRUCTURES = (
    BACKDOOR_STRUCTURES
    | FRONTDOOR_STRUCTURES
    | FALLBACK_STRUCTURES
)


@dataclass(frozen=True)
class RoleSpec:
    n_vars: int
    treatment: int
    outcome: int
    confounder: int | None = None
    mediator: int | None = None
    instrument: int | None = None
    hidden: tuple[int, ...] = ()


# Corrected v1.1.0 canonical column order.
ROLE_SPECS = {
    "rct_no_confounding": RoleSpec(
        n_vars=2, treatment=0, outcome=1
    ),
    "bi_variate": RoleSpec(
        n_vars=2, treatment=0, outcome=1
    ),
    "observed_confounder": RoleSpec(
        n_vars=3, treatment=0, confounder=1, outcome=2
    ),
    "back_door": RoleSpec(
        n_vars=3, treatment=0, confounder=1, outcome=2
    ),
    "confounder_mediator": RoleSpec(
        n_vars=4, treatment=0, confounder=1, mediator=2, outcome=3
    ),
    "mediator": RoleSpec(
        n_vars=3, treatment=0, mediator=1, outcome=2
    ),
    "front_door": RoleSpec(
        n_vars=4, treatment=0, hidden=(1,), mediator=2, outcome=3
    ),
    "instrumental_variable": RoleSpec(
        n_vars=4, treatment=0, hidden=(1,), instrument=2, outcome=3
    ),
    "unobserved_confounder": RoleSpec(
        n_vars=3, treatment=0, hidden=(1,), outcome=2
    ),
}


def _as_numpy(x) -> np.ndarray:
    return np.asarray(to_numpy(x))


def _intervention_value(ep) -> float:
    """Return the scalar do-value."""
    values = _as_numpy(ep.intervention.values).reshape(-1)
    if values.size != 1:
        raise ValueError(
            f"Static TabPFN expects one scalar intervention value, got {values.size}"
        )
    return float(values[0])


def _structure(ep) -> str:
    return str(ep.structure)


def _validate_episode_roles(ep, suite_name: str) -> tuple[RoleSpec, object, np.ndarray]:
    """Validate the corrected-v1.1.0 representation before fitting anything."""
    structure = _structure(ep)

    if structure not in ROLE_SPECS:
        raise ValueError(f"Unknown Identifiability structure: {structure}")

    spec = ROLE_SPECS[structure]
    inp = parse_episode(ep, suite_name)

    x_obs = np.asarray(_as_numpy(ep.x_obs), dtype=float)
    x_int = np.asarray(_as_numpy(ep.x_int), dtype=float)

    if x_obs.ndim != 2 or x_int.ndim != 2:
        raise ValueError(
            f"{structure}: expected 2-D trajectories, "
            f"got x_obs={x_obs.shape}, x_int={x_int.shape}"
        )
    if x_obs.shape != x_int.shape:
        raise ValueError(
            f"{structure}: x_obs/x_int shape mismatch "
            f"{x_obs.shape} vs {x_int.shape}"
        )
    if x_obs.shape[1] != spec.n_vars:
        raise ValueError(
            f"{structure}: expected {spec.n_vars} canonical variables, "
            f"got {x_obs.shape[1]}"
        )

    a = int(_as_numpy(ep.intervention.targets).reshape(-1)[0])
    y = int(_as_numpy(ep.query_target).reshape(-1)[0])

    if a != spec.treatment:
        raise ValueError(
            f"{structure}: expected treatment A at column {spec.treatment}, got {a}"
        )
    if y != spec.outcome:
        raise ValueError(
            f"{structure}: expected outcome Y at column {spec.outcome}, got {y}"
        )

    t_start = int(inp.t_start)
    if not (1 <= t_start < len(x_obs)):
        raise ValueError(
            f"{structure}: invalid intervention onset {t_start} for T={len(x_obs)}"
        )

    # Exact paired trajectories must coincide before intervention.
    if not np.array_equal(x_obs[:t_start], x_int[:t_start]):
        raise ValueError(
            f"{structure}: x_obs and x_int differ before intervention onset"
        )

    # Hidden columns in the corrected released representation must be masked.
    for h in spec.hidden:
        if not np.all(x_obs[:, h] == 0.0):
            raise ValueError(
                f"{structure}: hidden column {h} leaks non-zero values in x_obs"
            )
        if not np.all(x_int[:, h] == 0.0):
            raise ValueError(
                f"{structure}: hidden column {h} leaks non-zero values in x_int"
            )

    # For Identifiability v1.1.0 the intervention is a one-step hard do(A=v).
    # This check catches an incorrect treatment index / stale artifact.
    do_value = _intervention_value(ep)
    got = float(x_int[t_start, spec.treatment])
    if not np.isclose(got, do_value, atol=1e-6, rtol=0.0):
        raise ValueError(
            f"{structure}: intervention alignment failed at t={t_start}: "
            f"x_int[t,A]={got}, do(A)={do_value}"
        )

    # The query target in the harness must be the paired interventional level.
    expected_y_true = float(x_int[int(inp.query_step), spec.outcome])
    if not np.isclose(
        expected_y_true,
        float(inp.y_true),
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(
            f"{structure}: y_true alignment failed: "
            f"x_int[query,Y]={expected_y_true}, parser={float(inp.y_true)}"
        )

    return spec, inp, x_obs


def _new_regressor(*, device: str, random_state: int) -> TabPFNRegressor:
    """Construct the same model class used by the released reference.

    random_state=0 is passed explicitly because it is the current TabPFN
    default and makes the experiment provenance unambiguous.
    """
    return TabPFNRegressor(
        device=device,
        random_state=random_state,
    )


def _mean_fallback_from_validated(
    x_obs: np.ndarray,
    *,
    outcome: int,
    t_start: int,
) -> tuple[float, float]:
    """Matched P0/P1 fallback used off the adjustment's applicable structures."""
    pred = float(np.mean(x_obs[:t_start, outcome]))
    if not np.isfinite(pred):
        raise ValueError("Non-finite trajectory-mean fallback")
    return pred, pred


def _mean_fallback(ep, suite_name: str) -> tuple[float, float]:
    spec, inp, x_obs = _validate_episode_roles(ep, suite_name)
    return _mean_fallback_from_validated(
        x_obs,
        outcome=spec.outcome,
        t_start=int(inp.t_start),
    )


def predict_backdoor_pair(
    ep,
    suite_name: str,
    *,
    device: str = "cuda",
    random_state: int = 0,
) -> tuple[float, float]:
    """Matched (P0, P1) for the three back-door-family structures.

    Regression form follows the released Static TabPFN reference:

        Y_t ~ [A_t, X_t, Y_{t-1}]

    For confounder_mediator, X is column 1 and M is column 2.  M is
    deliberately excluded from the adjustment set when estimating the total
    effect of A on Y.
    """
    structure = _structure(ep)
    if structure not in BACKDOOR_STRUCTURES:
        raise ValueError(f"Not a back-door structure: {structure}")

    spec, inp, x_obs = _validate_episode_roles(ep, suite_name)

    if spec.confounder is None:
        raise RuntimeError(f"{structure}: missing validated confounder index")

    a = spec.treatment
    x_idx = spec.confounder
    y = spec.outcome
    t_start = int(inp.t_start)

    if t_start < MIN_HISTORY:
        return _mean_fallback_from_validated(
            x_obs, outcome=y, t_start=t_start
        )

    # STRICTLY pre-intervention rows.
    a_t = x_obs[1:t_start, a]
    x_t = x_obs[1:t_start, x_idx]
    y_prev = x_obs[: t_start - 1, y]
    y_t = x_obs[1:t_start, y]

    X_train = np.column_stack([a_t, x_t, y_prev])

    if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(y_t)):
        raise ValueError(f"{structure}: non-finite back-door training data")

    model = _new_regressor(device=device, random_state=random_state)
    model.fit(X_train, y_t)

    # Empirical adjustment over the PRE-INTERVENTION distribution of X.
    x_mc = x_t
    y_last = float(x_obs[t_start - 1, y])

    # Released-reference observational convention.
    a_p0 = float(x_obs[t_start - 1, a])
    a_p1 = _intervention_value(ep)

    X_p0 = np.column_stack([
        np.full(len(x_mc), a_p0),
        x_mc,
        np.full(len(x_mc), y_last),
    ])
    X_p1 = np.column_stack([
        np.full(len(x_mc), a_p1),
        x_mc,
        np.full(len(x_mc), y_last),
    ])

    # Same fitted model; one batched call for matched P0/P1.
    pred = np.asarray(
        model.predict(np.vstack([X_p0, X_p1])),
        dtype=float,
    ).reshape(-1)

    n = len(x_mc)
    if pred.size != 2 * n:
        raise RuntimeError(
            f"{structure}: expected {2*n} predictions, got {pred.size}"
        )

    p0 = float(np.mean(pred[:n]))
    p1 = float(np.mean(pred[n:]))

    if not np.isfinite(p0) or not np.isfinite(p1):
        raise ValueError(f"{structure}: non-finite back-door prediction")

    return p0, p1


def predict_frontdoor_pair(
    ep,
    suite_name: str,
    *,
    device: str = "cuda",
    random_state: int = 0,
) -> tuple[float, float]:
    """Matched (P0, P1) for mediator/front-door structures.

    Correct mediator indices:
        mediator   : [A, M, Y]            -> M=1
        front_door : [A, U(hidden), M, Y] -> M=2

    We follow the released DoTime reference form:
        M_t ~ A_t
        Y_t ~ [M_t, A_t]

    For each arm, first predict M under the plugged treatment value, then
    integrate the outcome model over the empirical pre-intervention A_t rows.

    The hidden U column is NEVER used.
    """
    structure = _structure(ep)
    if structure not in FRONTDOOR_STRUCTURES:
        raise ValueError(f"Not a front-door structure: {structure}")

    spec, inp, x_obs = _validate_episode_roles(ep, suite_name)

    if spec.mediator is None:
        raise RuntimeError(f"{structure}: missing validated mediator index")

    a = spec.treatment
    m = spec.mediator
    y = spec.outcome
    t_start = int(inp.t_start)

    if t_start < MIN_HISTORY:
        return _mean_fallback_from_validated(
            x_obs, outcome=y, t_start=t_start
        )

    # STRICTLY pre-intervention rows only.
    a_t = x_obs[1:t_start, a]
    m_t = x_obs[1:t_start, m]
    y_t = x_obs[1:t_start, y]

    X_m = a_t.reshape(-1, 1)
    X_y = np.column_stack([m_t, a_t])

    if (
        not np.all(np.isfinite(X_m))
        or not np.all(np.isfinite(X_y))
        or not np.all(np.isfinite(m_t))
        or not np.all(np.isfinite(y_t))
    ):
        raise ValueError(f"{structure}: non-finite front-door training data")

    # Same two fitted regressors are used for both P0 and P1.
    model_m = _new_regressor(device=device, random_state=random_state)
    model_m.fit(X_m, m_t)

    model_y = _new_regressor(device=device, random_state=random_state)
    model_y.fit(X_y, y_t)

    a_p0 = float(x_obs[t_start - 1, a])
    a_p1 = _intervention_value(ep)

    # One batched mediator call for the two arms.
    m_pair = np.asarray(
        model_m.predict(np.array([[a_p0], [a_p1]], dtype=float)),
        dtype=float,
    ).reshape(-1)

    if m_pair.size != 2:
        raise RuntimeError(
            f"{structure}: mediator model returned {m_pair.size} values"
        )

    m_p0, m_p1 = float(m_pair[0]), float(m_pair[1])

    # Front-door standardization over empirical pre-intervention A.
    X_p0 = np.column_stack([
        np.full(len(a_t), m_p0),
        a_t,
    ])
    X_p1 = np.column_stack([
        np.full(len(a_t), m_p1),
        a_t,
    ])

    pred = np.asarray(
        model_y.predict(np.vstack([X_p0, X_p1])),
        dtype=float,
    ).reshape(-1)

    n = len(a_t)
    if pred.size != 2 * n:
        raise RuntimeError(
            f"{structure}: expected {2*n} outcome predictions, got {pred.size}"
        )

    p0 = float(np.mean(pred[:n]))
    p1 = float(np.mean(pred[n:]))

    if not np.isfinite(p0) or not np.isfinite(p1):
        raise ValueError(f"{structure}: non-finite front-door prediction")

    return p0, p1


def predict_pair(
    ep,
    suite_name: str,
    *,
    device: str = "cuda",
    random_state: int = 0,
) -> tuple[float, float, str, bool]:
    """Return (P0, P1, adjustment_mode, applicable) for ANY canonical structure.

    "applicable=False" is intentional: the released Static TabPFN reference
    falls back to the pre-intervention outcome mean when back-/front-door
    assumptions do not hold.
    """
    structure = _structure(ep)

    if structure in BACKDOOR_STRUCTURES:
        p0, p1 = predict_backdoor_pair(
            ep,
            suite_name,
            device=device,
            random_state=random_state,
        )
        return p0, p1, "backdoor_tabpfn", True

    if structure in FRONTDOOR_STRUCTURES:
        p0, p1 = predict_frontdoor_pair(
            ep,
            suite_name,
            device=device,
            random_state=random_state,
        )
        return p0, p1, "frontdoor_tabpfn", True

    if structure in FALLBACK_STRUCTURES:
        p0, p1 = _mean_fallback(ep, suite_name)

        if structure in {"rct_no_confounding", "bi_variate"}:
            mode = "fallback_trajmean_rct"
        elif structure == "instrumental_variable":
            mode = "fallback_trajmean_iv"
        else:
            mode = "fallback_trajmean_nonidentifiable"

        return p0, p1, mode, False

    raise ValueError(f"Unknown Identifiability structure: {structure}")


def run_adjustment(
    suite,
    suite_name: str,
    indices,
    *,
    device: str = "cuda",
    random_state: int = 0,
) -> list[dict]:
    """Run the corrected Static TabPFN reference on fixed episode IDs.

    Every canonical Identifiability structure gets a row.
    Raw JSON operands are stored; implied/true effects are intentionally NOT
    duplicated because the aggregation layer can reconstruct:
        implied_effect = p1_query - p0_query
        true_effect    = y_true_query - y_obs_query
    """
    rows: list[dict] = []

    for position, index in enumerate(indices, start=1):
        if position % 20 == 0 or position == len(indices):
            print(f"  {position}/{len(indices)}")

        ep = suite[index]

        try:
            spec, inp, _ = _validate_episode_roles(ep, suite_name)

            p0, p1, mode, applicable = predict_pair(
                ep,
                suite_name,
                device=device,
                random_state=random_state,
            )

            rows.append({
                "episode": int(index),
                "structure": _structure(ep),
                "interface": "CAUSAL_REFERENCE",
                "adjustment_mode": mode,
                "adjustment_applicable": bool(applicable),

                "horizon": int(inp.horizon),
                "query_step": int(inp.query_step),

                "p0_query": float(p0),
                "p1_query": float(p1),
                "p0_quantiles": None,
                "p1_quantiles": None,

                "y_obs_query": float(inp.y_obs),
                "y_true_query": float(inp.y_true),

                "status": "OK",
            })

        except Exception as exc:
            rows.append({
                "episode": int(index),
                "structure": _structure(ep),
                "status": f"FAILED: {type(exc).__name__}: {exc}",
            })

    return rows


# Backward-compatible alias if an older runner imports run_backdoor.
run_backdoor = run_adjustment
