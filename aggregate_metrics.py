"""
Aggregate FMTS run JSONs into paper-ready metrics.

Causal references are kept separate from the ordinary TSFM P0/P1 estimand:
- DoT-PFN: single-arm effect = P1 - observed factual outcome.
- Static TabPFN: its structure-aware effect = P1 - P0.
Neither contributes to TSFM usage-gap/effect-delta columns or defines the
common TSFM population.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from dotime.benchmarks import load_benchmark, _SUITE_REGISTRY, _parse_suite_dir
from probe.interfaces import reconstruct_window
from probe import metrics


RESULTS = Path("results/runs")
OUT = Path("results/metrics")
DATA = Path("data")

IDENT = "dot-Identifiability-v1"
CONT = "dot-Continuous-v1"
IDENT_VERSION = "1.1.0"
IDENT_COMMIT = "03e330504eeac298a0063c7c694a902a9fd0180d"
CONT_VERSION = "1.0.0"

N_BOOT = 10_000
BOOT_SEED = 42
ALPHA = 0.05

FOLDERS = {
    "identifiability": (IDENT, "identifiability"),
    "continuous_in_window": (CONT, "in_window"),
    "continuous_post_window": (CONT, "post_window"),
}
SEED_RE = re.compile(r"_inf(\d+)$")

# Simple forecasting anchors are reported, but they must not define the
# common TSFM population or enter the duration-accrual figure.
ANCHORS = {"ar1", "trajmean", "zero"}
ORACLE_TOLERANCE = 1e-6

def npy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def load_suites():
    suite_dir = DATA / f"{IDENT}-{IDENT_COMMIT[:8]}"
    snapshot_download(
        repo_id="thummd/dot-Identifiability-v1",
        repo_type="dataset",
        revision=IDENT_COMMIT,
        local_dir=suite_dir,
    )
    meta = replace(_SUITE_REGISTRY[IDENT], version=IDENT_VERSION)
    return {
        IDENT: _parse_suite_dir(meta, suite_dir),
        CONT: load_benchmark(CONT, version=CONT_VERSION),
    }


def parse_path(path):
    if path.parent.name not in FOLDERS:
        return None
    suite, regime = FOLDERS[path.parent.name]
    match = SEED_RE.search(path.stem)
    model = SEED_RE.sub("", path.stem)
    seed = int(match.group(1)) if match else None
    return model, seed, suite, regime


def point(quantiles, native):
    if isinstance(quantiles, dict):
        for key in ("0.5", 0.5):
            if key in quantiles:
                return float(quantiles[key])
    return float(native) if native is not None else np.nan


def role_from_payload(model, payload):
    model_type = payload.get("model_type")

    # DoT-PFN and Oracle are single-arm references: neither has a comparable
    # same-checkpoint observational P0 arm. Oracle is kept only as a level
    # sanity check; its effect columns remain empty below.
    if model in {"dotime-pfn", "oracle"} or model_type == "causal_reference_single_arm":
        return "reference_single"

    if model_type == "structure_aware_causal_reference":
        return "reference_paired"

    return "standard"


def load_result(path, suite):
    parsed = parse_path(path)
    if parsed is None:
        return pd.DataFrame(), 0

    parsed_model, seed, suite_name, regime = parsed
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["results"] if isinstance(payload, dict) else payload

    # Prefer the identity recorded by the runner. This matters for copied or
    # renamed files (e.g. tabpfn-static.json) and keeps the summary identity
    # stable as tabpfn-static-adjustment.
    model = (
        str(payload.get("model", parsed_model))
        if isinstance(payload, dict)
        else parsed_model
    )
    role = role_from_payload(model, payload if isinstance(payload, dict) else {})
    n_failed = sum(r.get("status") != "OK" for r in rows)

    out = []
    for row in rows:
        if row.get("status") != "OK":
            continue

        i = int(row["episode"])
        ep = suite[i]
        if str(row["structure"]) != str(ep.structure):
            raise ValueError(f"{path.name}, episode {i}: structure mismatch")

        q = int(row["query_step"])
        y_idx = int(npy(ep.query_target).reshape(-1)[0])
        x_obs, x_int = npy(ep.x_obs), npy(ep.x_int)

        y_obs = float(x_obs[q, y_idx])
        y_int = float(x_int[q, y_idx])
        y_true = float(npy(ep.y_true).reshape(-1)[0])

        if not np.isclose(y_true, y_int, atol=1e-6):
            raise ValueError(f"{path.name}, episode {i}: y_true/query mismatch")

        p0 = point(row.get("p0_quantiles"), row.get("p0_query"))
        p1 = point(row.get("p1_quantiles"), row.get("p1_query"))
        true_effect = y_int - y_obs

        # Oracle is an alignment sanity check, not an effect estimator. Its
        # interventional prediction must equal the exact query target. Fail
        # loudly here rather than letting a bad alignment become a NaN row.
        oracle_alignment_abs_error = np.nan
        if model == "oracle":
            oracle_alignment_abs_error = abs(p1 - y_true)
            if (
                not np.isfinite(oracle_alignment_abs_error)
                or oracle_alignment_abs_error > ORACLE_TOLERANCE
            ):
                raise ValueError(
                    f"{path.name}, episode {i}: Oracle alignment failed: "
                    f"p1={p1:.9g}, y_true={y_true:.9g}, "
                    f"abs_error={oracle_alignment_abs_error:.3g}"
                )

        if role == "reference_single":
            p0 = np.nan

        standard_effect = p1 - p0 if role == "standard" else np.nan
        reference_effect = (
            p1 - y_obs if role == "reference_single" and model == "dotime-pfn"
            else p1 - p0 if role == "reference_paired"
            else np.nan
        )

        item = dict(row)
        item.update(
            model=model,
            model_type=payload.get("model_type"),
            evaluation_role=role,
            inference_seed=seed,
            suite=suite_name,
            regime=regime,
            n_failed_source=n_failed,
            p0_metric=p0,
            p1_metric=p1,
            y_obs_query=y_obs,
            y_int_query=y_int,
            y_true=y_true,
            true_effect=true_effect,
            implied_effect_metric=standard_effect,
            reference_implied_effect_metric=reference_effect,
            oracle_alignment_abs_error=oracle_alignment_abs_error,
        )

        if suite_name == CONT:
            t0, t1 = reconstruct_window(ep)
            item.update(
                intervention_start=int(t0),
                intervention_end=int(t1),
                intervention_length=int(t1 - t0),
                query_offset_from_start=int(q - t0),
            )
        else:
            for key in (
                "intervention_start", "intervention_end",
                "intervention_length", "query_offset_from_start",
            ):
                item[key] = np.nan

        out.append(item)

    return pd.DataFrame(out), n_failed


def restrict_common(df):
    kept = []
    for regime, block in df.groupby("regime", sort=False):
        # Only ordinary TSFMs define the common comparison population.
        # Anchors, Oracle, and causal references are evaluated on that same
        # population afterwards, but cannot shrink it.
        main = block[
            (block["evaluation_role"] == "standard")
            & (~block["model"].isin(ANCHORS))
        ]
        populations = [
            set(g["episode"])
            for _, g in main.groupby(["model", "inference_seed"], dropna=False)
        ]
        shared = set.intersection(*populations) if populations else set(block["episode"])
        print(f"{regime}: {len(shared)} common main-comparison episodes")
        kept.append(block[block["episode"].isin(shared)])
    return pd.concat(kept, ignore_index=True)


def mean(x):
    return float(np.mean(x))


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def mae(x):
    return float(np.mean(np.abs(x)))


def max_value(x):
    return float(np.max(x))


def safe_nanmean(x):
    """Mean over finite values without emitting RuntimeWarning on all-NaN input."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else np.nan


def apply(values, agg, idx=None):
    x = values if idx is None else values[idx]
    x = x[np.isfinite(x)]
    return agg(x) if x.size else np.nan


def quantiles_from(group):
    levels = tuple(float(q) for q in metrics.DECILES)
    out = {q: [] for q in levels}

    for value in group["p1_quantiles"]:
        if not isinstance(value, dict):
            return None
        for q in levels:
            if str(q) in value:
                out[q].append(float(value[str(q)]))
            elif q in value:
                out[q].append(float(value[q]))
            else:
                return None

    return {q: np.asarray(v, float) for q, v in out.items()}


def episode_metrics(group):
    role = group["evaluation_role"].iloc[0]
    p0 = group["p0_metric"].to_numpy(float)
    p1 = group["p1_metric"].to_numpy(float)
    y = group["y_true"].to_numpy(float)
    y_obs = group["y_obs_query"].to_numpy(float)
    tau = group["true_effect"].to_numpy(float)

    # P0 and P1 are both scored against the same interventional target so
    # their outcome RMSEs are directly comparable. y_obs is retained only to
    # construct the exact causal effect and the single-arm DoT-PFN reference.
    out = {
        "p0_outcome_rmse": (metrics.outcome_error(p0, y), rmse),
        "p1_outcome_rmse": (metrics.outcome_error(p1, y), rmse),
        "true_effect_scale": (np.abs(tau), mean),
    }

    if role == "standard":
        err = metrics.effect_delta(p1, p0, tau)
        out.update(
            usage_gap=(metrics.usage_gap(p1, p0), mean),
            delta_err=(metrics.usage_gap_delta_err(p1, p0, y), mean),
            effect_rmse=(err, rmse),
            effect_mae=(err, mae),
        )
    else:
        # Only references with an actual reference effect get effect metrics.
        # DoT-PFN: p1 - y_obs. Static TabPFN: p1 - p0. Oracle intentionally
        # has no effect estimand; it is retained solely as an alignment check.
        implied = group["reference_implied_effect_metric"].to_numpy(float)
        if np.isfinite(implied).any():
            err = implied - tau
            out.update(
                reference_effect_rmse=(err, rmse),
                reference_effect_mae=(err, mae),
            )

    if group["model"].iloc[0] == "oracle":
        alignment = group["oracle_alignment_abs_error"].to_numpy(float)
        out["oracle_alignment_max_abs_error"] = (alignment, max_value)

    qs = quantiles_from(group)
    if qs is not None:
        inside = ((y >= qs[0.1]) & (y <= qs[0.9])).astype(float)
        out["coverage_80"] = (inside, mean)
        out["crps_approx"] = (metrics.crps(qs, y), mean)

    return out, qs


def direction_for(group, reference=False):
    true = group["true_effect"].to_numpy(float)

    if reference:
        effect = group["reference_implied_effect_metric"].to_numpy(float)
        # direction_accuracy only needs p1-p0; zero is a neutral baseline here.
        # DoTime's near-zero exclusion is based on true_effect, not this baseline.
        p0 = np.zeros_like(effect)
        p1 = effect
    else:
        p0 = group["p0_metric"].to_numpy(float)
        p1 = group["p1_metric"].to_numpy(float)

    finite = np.isfinite(p0) & np.isfinite(p1) & np.isfinite(true)
    if not finite.any():
        return np.nan, 0

    result = metrics.direction_accuracy(p1[finite], p0[finite], true[finite])
    return float(result["accuracy"]), int(result.get("n_scored", result.get("n_valid", 0)))


def direction_summary(block, draws, alpha, reference=False):
    seeds = sorted(block["inference_seed"].dropna().unique().tolist()) or [None]
    groups = [
        (block if seed is None else block[block["inference_seed"] == seed])
        .sort_values("episode")
        for seed in seeds
    ]

    original = [direction_for(g, reference=reference) for g in groups]
    values = np.asarray([x[0] for x in original], float)
    if not np.isfinite(values).any():
        prefix = "reference_direction" if reference else "direction"
        return {
            f"{prefix}_accuracy": np.nan,
            f"{prefix}_accuracy_lo": np.nan,
            f"{prefix}_accuracy_hi": np.nan,
            f"{prefix}_accuracy_seed_std": np.nan,
            f"{prefix}_n_scored": 0,
        }

    reps = []
    for draw in draws:
        per_seed = []
        for g in groups:
            sampled = g.iloc[draw]
            acc, _ = direction_for(sampled, reference=reference)
            if np.isfinite(acc):
                per_seed.append(acc)
        if per_seed:
            reps.append(np.mean(per_seed))

    reps = np.asarray(reps, float)
    reps = reps[np.isfinite(reps)]
    lo, hi = (
        np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        if reps.size else (np.nan, np.nan)
    )

    prefix = "reference_direction" if reference else "direction"
    return {
        f"{prefix}_accuracy": float(np.nanmean(values)),
        f"{prefix}_accuracy_lo": float(lo),
        f"{prefix}_accuracy_hi": float(hi),
        f"{prefix}_accuracy_seed_std": (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ),
        f"{prefix}_n_scored": int(np.mean([x[1] for x in original])),
    }


def summarize_run(block, n_boot=N_BOOT, alpha=ALPHA, seed=BOOT_SEED):
    seeds = sorted(block["inference_seed"].dropna().unique().tolist()) or [None]
    groups, computed, ladders = [], [], []

    for value in seeds:
        g = (
            block if value is None
            else block[block["inference_seed"] == value]
        ).sort_values("episode")
        groups.append(g)
        m, qs = episode_metrics(g)
        computed.append(m)
        if qs is not None:
            ladders.append(metrics.coverage_ladder(qs, g["y_true"].to_numpy(float)))

    n = int(block["episode"].nunique())
    draws = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    result = {
        "n": n,
        "n_seeds": len(groups),
        "n_failed": int(block["n_failed_source"].max()),
    }

    for name in sorted(set().union(*(m.keys() for m in computed))):
        available = [m[name] for m in computed if name in m]
        by_seed = np.asarray([apply(v, agg) for v, agg in available], float)

        if not np.isfinite(by_seed).any():
            result.update({
                name: np.nan,
                f"{name}_lo": np.nan,
                f"{name}_hi": np.nan,
                f"{name}_seed_std": np.nan,
                f"{name}_n": 0,
            })
            continue

        reps = np.asarray([
            np.nanmean([apply(v, agg, draw) for v, agg in available])
            for draw in draws
        ])
        reps = reps[np.isfinite(reps)]
        lo, hi = (
            np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            if reps.size else (np.nan, np.nan)
        )

        result.update({
            name: float(np.nanmean(by_seed)),
            f"{name}_lo": float(lo),
            f"{name}_hi": float(hi),
            f"{name}_seed_std": (
                float(np.std(by_seed, ddof=1)) if len(by_seed) > 1 else 0.0
            ),
            f"{name}_n": int(max(np.isfinite(v).sum() for v, _ in available)),
        })

    role = block["evaluation_role"].iloc[0]
    if role == "standard":
        result.update(direction_summary(block, draws, alpha))
    elif np.isfinite(
        block["reference_implied_effect_metric"].to_numpy(float)
    ).any():
        result.update(direction_summary(block, draws, alpha, reference=True))

    if ladders:
        for nominal in sorted(ladders[0], reverse=True):
            result[f"coverage_{int(round(100 * nominal))}"] = float(
                np.mean([x[nominal] for x in ladders])
            )

    return result


def summarize(df, cols, **kwargs):
    rows = []
    for keys, block in df.groupby(cols, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(cols, keys))
        row.update(summarize_run(block, **kwargs))
        rows.append(row)
    return pd.DataFrame(rows)


def duration_accrual(df):
    # Main duration-accrual curves are TSFM-only. Anchors and causal/sanity
    # references are kept out of this file so they cannot be mistaken for
    # intervention-conditioning models.
    cont = df[
        (df["suite"] == CONT)
        & (df["evaluation_role"] == "standard")
        & (~df["model"].isin(ANCHORS))
    ]
    rows = []

    for (model, regime, length), block in cont.groupby(
        ["model", "regime", "intervention_length"], sort=True
    ):
        seed_rows = []
        seeds = sorted(block["inference_seed"].dropna().unique().tolist()) or [None]

        for seed in seeds:
            g = block if seed is None else block[block["inference_seed"] == seed]
            implied = g["implied_effect_metric"].to_numpy(float)
            true = g["true_effect"].to_numpy(float)
            error = implied - true
            seed_rows.append((
                apply(implied, mean),
                apply(true, mean),
                apply(error, rmse),
                apply(error, mae),
            ))

        a = np.asarray(seed_rows, float)
        rows.append({
            "model": model,
            "regime": regime,
            "intervention_length": int(length),
            "n": int(block["episode"].nunique()),
            "n_seeds": len(seeds),
            "mean_implied_effect": safe_nanmean(a[:, 0]),
            "mean_true_effect": safe_nanmean(a[:, 1]),
            "effect_rmse": safe_nanmean(a[:, 2]),
            "effect_mae": safe_nanmean(a[:, 3]),
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", choices=("intersection", "all"), default="intersection")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=BOOT_SEED)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    suites = load_suites()

    frames = []
    paths = [
        p
        for folder in FOLDERS
        for p in (RESULTS / folder).glob("*.json")
    ]

    for path in sorted(paths):
        parsed = parse_path(path)
        if parsed is None:
            continue
        model, seed, suite_name, regime = parsed
        print(f"Loading {model} | seed={seed} | {regime}")
        frame, failed = load_result(path, suites[suite_name])
        if not frame.empty:
            print(f"  OK={len(frame)} | FAILED={failed}")
            frames.append(frame)

    if not frames:
        raise RuntimeError("No successful prediction rows found.")

    # Keep the full schema from every source frame. Pandas 2.3 emits a
    # FutureWarning when concatenating all-NA columns from heterogeneous
    # sources (e.g. Oracle has no P0). Suppress only that warning rather than
    # dropping columns, which would make downstream schema assumptions fragile.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "The behavior of DataFrame concatenation with empty or all-NA "
                "entries is deprecated.*"
            ),
            category=FutureWarning,
        )
        predictions = pd.concat(frames, ignore_index=True, sort=False)

    # Final-run invariants: do not silently publish an aggregation that lost
    # either the Oracle alignment check or the Static TabPFN causal reference.
    oracle = predictions[predictions["model"] == "oracle"]
    if oracle.empty:
        raise RuntimeError("Oracle results were not loaded; alignment sanity check is missing.")
    oracle_max = float(np.nanmax(oracle["oracle_alignment_abs_error"].to_numpy(float)))
    if not np.isfinite(oracle_max) or oracle_max > ORACLE_TOLERANCE:
        raise RuntimeError(
            f"Oracle alignment check failed: max abs error = {oracle_max:.6g}"
        )
    print(f"Oracle alignment: PASS | max_abs_error={oracle_max:.3g}")

    paired_refs = predictions[
        predictions["evaluation_role"] == "reference_paired"
    ]
    if paired_refs.empty:
        raise RuntimeError(
            "Static TabPFN reference was not loaded from "
            "results/runs/identifiability/. Expected a JSON with "
            "model_type='structure_aware_causal_reference' (for example "
            "tabpfn-static.json)."
        )
    print(
        "Static TabPFN reference: LOADED | "
        f"models={sorted(paired_refs['model'].unique().tolist())}"
    )

    if args.population == "intersection":
        predictions = restrict_common(predictions)

    # Null-effect controls. In Identifiability the query sits at the
    # intervention onset (offset 0), so episodes without a same-step path
    # from treatment to outcome have a true effect of exactly zero (see the
    # DoTime errata). The headline is also reported on nonzero-effect
    # episodes only, where the zero-effect anchor is no longer optimal by
    # construction.
    ident_nonzero = predictions[
        (predictions["regime"] == "identifiability")
        & (predictions["true_effect"] != 0)
    ]
    
    counts = ident_nonzero.groupby("model")["episode"].nunique()

    print(f"Identifiability nonzero-effect episodes per model:\n{counts}")

    kw = {"n_boot": args.n_boot, "seed": args.seed}
    outputs = {
        "per_episode.csv": predictions,
        "summary_pooled.csv": summarize(predictions, ["model", "regime"], **kw),
        "summary_by_structure.csv": summarize(
            predictions, ["model", "regime", "structure"], **kw
        ),
        "summary_by_horizon.csv": summarize(
            predictions, ["model", "regime", "horizon"], **kw
        ),
        "duration_accrual.csv": duration_accrual(predictions),
        "summary_pooled_nonzero.csv": summarize(
            ident_nonzero, ["model", "regime"], **kw
        ),
        "summary_by_structure_nonzero.csv": summarize(
            ident_nonzero, ["model", "regime", "structure"], **kw
        ),
    }

    print("\nSaved:")
    for name, frame in outputs.items():
        path = OUT / name
        frame.to_csv(path, index=False)
        print(f"  {path}")


if __name__ == "__main__":
    main()
