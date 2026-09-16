#!/usr/bin/env python3
"""Run the corrected Static TabPFN reference on ALL 8 Identifiability structures.

Default protocol matches the DoTime Table-4 scale:
    60 episodes per structure x 8 structures = 480 episodes.

The suite is pinned to corrected dot-Identifiability-v1 v1.1.0:
    revision 03e330504eeac298a0063c7c694a902a9fd0180d

Outputs are written to:
    results/references/tabpfn-static-adjustment/

Important:
- This is NOT TabPFN-TS.
- This is a structure-aware causal reference.
- Back-door/front-door formulas are used only where applicable.
- RCT/IV/unobserved-confounder receive the released-reference mean fallback;
  IV2SLS remains a separate IV reference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import replace
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

from dotime.benchmarks import _SUITE_REGISTRY, _parse_suite_dir

from probe.references.tabpfn_adjustment import (
    ALL_STRUCTURES,
    BACKDOOR_STRUCTURES,
    FALLBACK_STRUCTURES,
    FRONTDOOR_STRUCTURES,
    ROLE_SPECS,
    run_adjustment,
)


RESULTS_DIR = Path("results")
DATA_DIR = Path("data")

IDENT_SUITE = "dot-Identifiability-v1"
IDENT_VERSION = "1.1.0"
IDENT_COMMIT = "03e330504eeac298a0063c7c694a902a9fd0180d"

MODEL_NAME = "tabpfn-static-adjustment"
DEFAULT_PER_STRUCTURE = 60
DEFAULT_RANDOM_STATE = 0

PACKAGES = (
    "numpy",
    "torch",
    "dotime",
    "tabpfn",
    "huggingface-hub",
)


def package_versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def git_commit() -> str | None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = ""

    return commit + ("-dirty" if dirty else "")


def atomic_json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def load_identifiability():
    """Load the exact corrected v1.1.0 artifact, not whatever 'latest' is."""
    suite_dir = DATA_DIR / f"{IDENT_SUITE}-{IDENT_COMMIT[:8]}"

    snapshot_download(
        repo_id="thummd/dot-Identifiability-v1",
        repo_type="dataset",
        revision=IDENT_COMMIT,
        local_dir=suite_dir,
    )

    meta = replace(
        _SUITE_REGISTRY[IDENT_SUITE],
        version=IDENT_VERSION,
    )
    suite = _parse_suite_dir(meta, suite_dir)

    if len(suite) != 10800:
        raise RuntimeError(
            f"Expected 10,800 Identifiability episodes, got {len(suite)}"
        )

    zeroed = sum(
        bool(
            np.all(np.asarray(ep.x_obs) == 0)
            or np.all(np.asarray(ep.x_int) == 0)
        )
        for ep in suite
    )
    if zeroed:
        raise RuntimeError(
            f"Corrected Identifiability {IDENT_VERSION} unexpectedly contains "
            f"{zeroed} all-zero episodes"
        )

    return suite


def normalize_structure_name(name: str) -> str:
    # The released suite uses rct_no_confounding; current sampler calls it
    # bi_variate. Treat them as one structure for validation/reporting.
    return "rct_no_confounding" if name == "bi_variate" else name


EXPECTED_RELEASED_STRUCTURES = {
    "rct_no_confounding",
    "observed_confounder",
    "back_door",
    "confounder_mediator",
    "mediator",
    "front_door",
    "instrumental_variable",
    "unobserved_confounder",
}


def validate_suite_roster(suite) -> Counter:
    counts = Counter(
        normalize_structure_name(str(ep.structure))
        for ep in suite
    )

    if set(counts) != EXPECTED_RELEASED_STRUCTURES:
        raise RuntimeError(
            "Unexpected Identifiability structure roster.\n"
            f"expected={sorted(EXPECTED_RELEASED_STRUCTURES)}\n"
            f"got={sorted(counts)}"
        )

    if any(counts[s] != 1350 for s in EXPECTED_RELEASED_STRUCTURES):
        raise RuntimeError(
            f"Expected 1,350 episodes per structure, got {dict(counts)}"
        )

    return counts


def select_first_per_structure(
    suite,
    *,
    per_structure: int,
) -> list[int]:
    """Same deterministic selection convention as released evaluator."""
    selected: list[int] = []
    counts: Counter = Counter()

    for idx, ep in enumerate(suite):
        s = normalize_structure_name(str(ep.structure))
        if counts[s] < per_structure:
            selected.append(idx)
            counts[s] += 1

    expected = {
        s: per_structure
        for s in EXPECTED_RELEASED_STRUCTURES
    }
    if dict(counts) != expected:
        raise RuntimeError(
            f"Could not select {per_structure}/structure: {dict(counts)}"
        )

    return selected


def result_path(*, full: bool, per_structure: int) -> Path:
    root = RESULTS_DIR / "runs" / "identifiability"

    if full:
        return root / "tabpfn-static.json"

    return root / f"tabpfn-static_{per_structure}ps.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
    )
    ap.add_argument(
        "--per-structure",
        type=int,
        default=DEFAULT_PER_STRUCTURE,
        help="Default 60 reproduces the released Table-4 sample scale.",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run all 10,800 episodes (very expensive).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output file.",
    )
    args = ap.parse_args()

    if args.per_structure < 1:
        raise ValueError("--per-structure must be >= 1")

    print("=== Static TabPFN causal-adjustment reference ===")
    print(f"suite:        {IDENT_SUITE} {IDENT_VERSION}")
    print(f"revision:     {IDENT_COMMIT}")
    print(f"device:       {args.device}")
    print(f"random_state: {args.random_state}")
    print("reference:    structure-aware; NOT TabPFN-TS")

    suite = load_identifiability()
    full_counts = validate_suite_roster(suite)
    print("full suite structures:", dict(full_counts))

    if args.full:
        indices = list(range(len(suite)))
    else:
        indices = select_first_per_structure(
            suite,
            per_structure=args.per_structure,
        )

    selected_counts = Counter(
        normalize_structure_name(str(suite[i].structure))
        for i in indices
    )

    print(f"selected episodes: {len(indices)}")
    print("selected structures:", dict(selected_counts))

    out = result_path(
        full=args.full,
        per_structure=args.per_structure,
    )

    if out.exists() and not args.force:
        raise SystemExit(
            f"{out} already exists. Use --force only if you intend to replace it."
        )

    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.time()

    rows = run_adjustment(
        suite,
        IDENT_SUITE,
        indices,
        device=args.device,
        random_state=args.random_state,
    )

    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.synchronize()

    elapsed = time.time() - t0

    n_ok = sum(r.get("status") == "OK" for r in rows)
    n_failed = len(rows) - n_ok

    mode_counts = Counter(
        r.get("adjustment_mode")
        for r in rows
        if r.get("status") == "OK"
    )
    applicable_counts = Counter(
        str(r.get("adjustment_applicable"))
        for r in rows
        if r.get("status") == "OK"
    )

    payload = {
        "model": MODEL_NAME,
        "model_type": "structure_aware_causal_reference",

        "suite": IDENT_SUITE,
        "suite_version": IDENT_VERSION,
        "suite_revision": IDENT_COMMIT,

        "protocol": "identifiability",
        "population": (
            "full_suite"
            if args.full
            else "stratified_reference_subsample"
        ),
        "selection_rule": (
            "all episodes"
            if args.full
            else f"first {args.per_structure} episodes per structure"
        ),
        "n_per_structure": (
            None if args.full else args.per_structure
        ),
        "structure_counts": dict(selected_counts),

        "inference_seed": None,
        "random_state": args.random_state,
        "n_episodes": len(indices),

        "model_config": {
            "tabpfn_constructor": "TabPFNRegressor",
            "device": args.device,
            "random_state": args.random_state,
            "temporal_training_rows": "strictly_pre_intervention",
            "canonical_roles": {
                name: {
                    "n_vars": spec.n_vars,
                    "treatment": spec.treatment,
                    "outcome": spec.outcome,
                    "confounder": spec.confounder,
                    "mediator": spec.mediator,
                    "instrument": spec.instrument,
                    "hidden": list(spec.hidden),
                }
                for name, spec in ROLE_SPECS.items()
            },
        },

        "methodology": {
            "role": "adapted structure-aware causal reference",
            "is_tabpfn_ts": False,
            "backdoor_structures": sorted(BACKDOOR_STRUCTURES),
            "frontdoor_structures": sorted(FRONTDOOR_STRUCTURES),
            "fallback_structures": sorted(FALLBACK_STRUCTURES),
            "iv_note": (
                "Static TabPFN has no released IV adjustment. "
                "instrumental_variable uses matched trajectory-mean fallback; "
                "IV2SLS is a separate reference."
            ),
            "unobserved_note": (
                "unobserved_confounder is non-identifiable by design and uses "
                "the matched trajectory-mean fallback."
            ),
            "rct_note": (
                "To remain aligned with the released Static TabPFN evaluator, "
                "rct_no_confounding/bi_variate uses the same off-family "
                "trajectory-mean fallback rather than a new direct estimator."
            ),
            "p0_p1_same_fitted_models": True,
            "effect_reconstruction": (
                "implied_effect = p1_query - p0_query; "
                "true_effect = y_true_query - y_obs_query"
            ),
            "ordinary_tsfm_usage_gap": False,
        },

        "adjustment_mode_counts": dict(mode_counts),
        "adjustment_applicable_counts": dict(applicable_counts),

        "package_versions": package_versions(),
        "git_commit": git_commit(),
        "elapsed_seconds": float(elapsed),

        "n_ok": n_ok,
        "n_failed": n_failed,
        "results": rows,
    }

    atomic_json_dump(payload, out)

    print(f"\nSAVED {out}")
    print(f"{n_ok} OK | {n_failed} failed | {elapsed/60:.1f} min")
    print("modes:", dict(mode_counts))

    if n_failed:
        print(
            "\nWARNING: failures were saved in the JSON. "
            "Inspect them before using this reference in the paper."
        )


if __name__ == "__main__":
    main()
