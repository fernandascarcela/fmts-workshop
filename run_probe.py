"""Run TSFMs and deterministic DoTime baselines on the final probe protocols.

Final protocols:
    - Identifiability: full corrected v1.1.0 suite, all 10,800 episodes.
    - Continuous in-window: 1,000 non-self episodes per structure.
    - Continuous post-window: 1,000 non-self episodes per structure.

Continuous episode selection uses seed 42 and is frozen once per protocol.
Inference seeds 42/43/44 are repeated only for models whose forward pass is
stochastic; a deterministic checkpoint would produce three identical files, so
it runs once.

Completed result files are validated and skipped before loading the model. A
run counts as complete when every selected episode was attempted, not when
every episode succeeded -- otherwise one permanently failing episode would
force a full rerun on every invocation. Writes are atomic so an interrupted run
cannot leave a partial JSON that is later mistaken for a completed experiment.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from importlib import metadata
from pathlib import Path
import json
import random
import subprocess

import numpy as np
import torch
from huggingface_hub import snapshot_download

from dotime.benchmarks import load_benchmark, _SUITE_REGISTRY, _parse_suite_dir

from probe.adapters.chronos import ChronosAdapter
from probe.adapters.moirai import MoiraiAdapter
from probe.adapters.moirai2 import Moirai2Adapter
from probe.adapters.tabpfnts import TabPFNTSAdapter
from probe.adapters.timesfm import TimesFMAdapter
from probe.adapters.tirex import TiRexAdapter
from probe.adapters.toto import TotoAdapter

from probe.interfaces import parse_episode, run_episode
from probe.filter import select_continuous

from probe.references.dotime_baselines import BASELINES, run_baseline, run_oracle


# --------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------- #

RESULTS_DIR = Path("results")
SELECTIONS_DIR = RESULTS_DIR / "selections"
RUNS_DIR = RESULTS_DIR / "runs"
DATA_DIR = Path("data")


# --------------------------------------------------------------------- #
# final protocol
# --------------------------------------------------------------------- #

SELECTION_SEED = 42
INFERENCE_SEEDS = (42, 43, 44)
N_PER_STRUCTURE = 1000

IDENT_SUITE = "dot-Identifiability-v1"
IDENT_VERSION = "1.1.0"
IDENT_COMMIT = "03e330504eeac298a0063c7c694a902a9fd0180d"

CONT_SUITE = "dot-Continuous-v1"
CONT_VERSION = "1.0.0"

CONT_STRUCTURES = ("back_door", "front_door", "instrumental_variable")

POST_WINDOW_MODELS = ("chronos-2", "tabpfn-ts", "tirex-2")
# Toto-1.0 is interface-limited for horizon-one Identifiability:
# its 64-step patch is generated before the supplied treatment is clamped.
IDENT_MODELS = (
    "chronos-2",
    "tabpfn-ts",
    "timesfm-2.5",
    "moirai-1.1",
    "moirai-2.0",
    "tirex-2",
)


# Models whose forward pass samples. Only these gain anything from repeated
# inference seeds; the rest are deterministic checkpoints and run once.
# Verify against each adapter before changing this set.
STOCHASTIC_MODELS = ("toto-1.0", "moirai-1.1")

ORACLE_TOLERANCE = 1e-6


# --------------------------------------------------------------------- #
# external TSFMs
# --------------------------------------------------------------------- #

MODELS = {
    "chronos-2": (lambda: ChronosAdapter(device="cuda"), {"device": "cuda"}),
    "tabpfn-ts": (lambda: TabPFNTSAdapter(device="cuda"), {"device": "cuda"}),
    "timesfm-2.5": (
        lambda: TimesFMAdapter(force_xreg_cpu=True, torch_compile=True),
        {"force_xreg_cpu": True, "torch_compile": True},
    ),
    "toto-1.0": (TotoAdapter, {}),
    "moirai-1.1": (MoiraiAdapter, {}),
    "moirai-2.0": (Moirai2Adapter, {}),
    "tirex-2": (lambda: TiRexAdapter(device="cpu"), {"device": "cpu"}),
}

PACKAGES = (
    "numpy",
    "torch",
    "dotime",
    "huggingface-hub",
    "chronos-forecasting",
    "tabpfn",
    "tabpfn-time-series",
    "timesfm",
    "uni2ts",
)


def seeds_for(model_name: str) -> tuple[int, ...]:
    """Inference seeds to run for one model.

    A deterministic checkpoint produces byte-identical output on every seed,
    so repeating it only multiplies GPU time.
    """
    if model_name in STOCHASTIC_MODELS:
        return INFERENCE_SEEDS
    return (INFERENCE_SEEDS[0],)


# --------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    """Set model-side RNG state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def package_versions() -> dict:
    """Return installed package versions for experiment provenance."""
    versions = {}

    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None

    return versions


def git_commit() -> str | None:
    """Return the current commit, suffixed when the tree is dirty.

    A clean hash over a modified working tree is worse than no hash, since it
    claims a provenance the results do not have.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = ""

    return commit + ("-dirty" if dirty else "")


# --------------------------------------------------------------------- #
# suite loading
# --------------------------------------------------------------------- #

def load_identifiability():
    """Load the exact corrected Identifiability v1.1.0 artifact."""
    suite_dir = DATA_DIR / f"{IDENT_SUITE}-{IDENT_COMMIT[:8]}"

    snapshot_download(
        repo_id="thummd/dot-Identifiability-v1",
        repo_type="dataset",
        revision=IDENT_COMMIT,
        local_dir=suite_dir,
    )

    meta = replace(_SUITE_REGISTRY[IDENT_SUITE], version=IDENT_VERSION)
    suite = _parse_suite_dir(meta, suite_dir)

    if len(suite) != 10800:
        raise RuntimeError(
            f"Expected 10,800 Identifiability episodes, got {len(suite)}."
        )

    # The protocol evaluates the full suite with no zero filter, which is only
    # sound if v1.1.0 really contains no degenerate episodes: a zeroed episode
    # has a true effect of exactly zero, handing a free correct answer to any
    # method that predicts no effect.
    zeroed = sum(
        bool(np.all(np.asarray(ep.x_obs) == 0) or np.all(np.asarray(ep.x_int) == 0))
        for ep in suite
    )
    if zeroed:
        raise RuntimeError(
            f"{zeroed} zeroed episodes in Identifiability {IDENT_VERSION}; "
            "the full-suite protocol assumes none. Reinstate the zero filter."
        )

    return suite


def load_continuous():
    """Load the released Continuous suite."""
    suite = load_benchmark(CONT_SUITE, version=CONT_VERSION)

    if len(suite) != 9999:
        raise RuntimeError(f"Expected 9,999 Continuous episodes, got {len(suite)}.")

    return suite


# --------------------------------------------------------------------- #
# continuous selections
# --------------------------------------------------------------------- #

def validate_continuous_selection(suite, indices, *, regime: str) -> None:
    """Validate final non-self Continuous selection semantics."""
    counts = Counter()

    if len(indices) != N_PER_STRUCTURE * len(CONT_STRUCTURES):
        raise RuntimeError(f"{regime}: expected 3,000 episodes, got {len(indices)}.")

    for index in indices:
        ep = suite[index]
        inp = parse_episode(ep, CONT_SUITE)

        if inp.treatment_index == inp.outcome_index:
            raise RuntimeError(f"{regime}: self-treatment episode selected: {index}")

        if regime == "in_window":
            valid_regime = inp.t_start <= inp.query_step < inp.t_end
        elif regime == "post_window":
            valid_regime = inp.query_step >= inp.t_end
        else:
            raise ValueError(f"Unknown regime: {regime}")

        if not valid_regime:
            raise RuntimeError(
                f"{regime}: episode {index} has incompatible query/window."
            )

        counts[str(ep.structure)] += 1

    expected = {structure: N_PER_STRUCTURE for structure in CONT_STRUCTURES}

    if dict(counts) != expected:
        raise RuntimeError(f"{regime}: unexpected structure counts: {dict(counts)}")


def atomic_json_dump(payload: dict, path: Path) -> None:
    """Write JSON atomically so an interrupted write leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    tmp.replace(path)


def freeze_selection(suite, indices, *, protocol: str) -> Path:
    """Write or validate one frozen Continuous selection manifest."""
    path = SELECTIONS_DIR / f"{protocol}_sel{SELECTION_SEED}.json"
    ids = [int(i) for i in indices]

    payload = {
        "suite": CONT_SUITE,
        "suite_version": CONT_VERSION,
        "protocol": protocol,
        "selection_seed": SELECTION_SEED,
        "n_per_structure": N_PER_STRUCTURE,
        "n_total": len(ids),
        "structure_counts": dict(Counter(str(suite[i].structure) for i in ids)),
        "episode_ids": ids,
    }

    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(
                f"Could not read existing selection file: {path}"
            ) from exc

        if existing != payload:
            raise RuntimeError(
                f"Existing selection does not match current protocol: {path}"
            )

        print(f"SELECTION OK {path}")
        return path

    atomic_json_dump(payload, path)
    print(f"SELECTION SAVED {path}")

    return path


# --------------------------------------------------------------------- #
# one TSFM episode
# --------------------------------------------------------------------- #

def _at_query(pred, horizon: int) -> tuple[float, dict]:
    """Read point and quantile predictions at the query step."""
    point = np.asarray(pred.point)

    if len(point) != horizon:
        raise ValueError(
            f"Expected horizon {horizon}, got {len(point)} prediction steps."
        )

    quantiles = {
        str(q): float(np.asarray(values)[-1])
        for q, values in pred.quantiles.items()
    }

    return float(point[-1]), quantiles


def run_one(model, suite, suite_name: str, index: int) -> dict:
    """Run P0 and the available intervention-conditioned path.

    A missing P0 or P1 is recorded as null rather than raising: an
    interface-limited model has no do-value channel, which is a finding.
    """
    ep = suite[index]
    inp = parse_episode(ep, suite_name)
    result = run_episode(model, ep, suite_name)

    p0, p0_quantiles = (
        _at_query(result.pred_p0, inp.horizon)
        if result.pred_p0 is not None
        else (None, None)
    )

    p1, p1_quantiles = (
        _at_query(result.pred_p1, inp.horizon)
        if result.pred_p1 is not None
        else (None, None)
    )

    return {
        "episode": int(index),
        "structure": str(ep.structure),
        "interface": str(result.interface),
        "horizon": int(inp.horizon),
        "query_step": int(inp.query_step),
        "p0_query": p0,
        "p1_query": p1,
        "p0_quantiles": p0_quantiles,
        "p1_quantiles": p1_quantiles,
        # Effects are differences of the fields above and are recomputed by the
        # aggregation layer, so they are not stored per episode: a stored copy
        # can only ever disagree with its own operands.
        "y_obs_query": float(inp.y_obs),
        "y_true_query": float(inp.y_true),
        "status": "OK",
    }


def run_model(model, suite, suite_name: str, indices) -> list[dict]:
    """Run one external model on fixed episode IDs."""
    rows = []

    for position, index in enumerate(indices, start=1):
        if position % 100 == 0:
            print(f"  {position}/{len(indices)}")

        try:
            rows.append(run_one(model, suite, suite_name, index))
        except Exception as exc:
            rows.append({
                "episode": int(index),
                "structure": str(suite[index].structure),
                "status": f"FAILED: {type(exc).__name__}: {exc}",
            })

    return rows


# --------------------------------------------------------------------- #
# result paths and skip logic
# --------------------------------------------------------------------- #

def result_path(name: str, protocol: str, inference_seed: int | None) -> Path:
    """Return the canonical result path for one run."""
    folder = RUNS_DIR / protocol

    if inference_seed is None:
        return folder / f"{name}.json"

    return folder / f"{name}_inf{inference_seed}.json"


def is_complete(path: Path, *, name: str, job: dict, inference_seed: int | None) -> bool:
    """Return True when this file is a finished run of exactly this job.

    Completeness means every selected episode was attempted, in order. Failed
    episodes are kept: a permanently failing adapter case would otherwise make
    the run un-completable and force a full rerun on every invocation. The
    aggregation layer already reports the effective denominator.
    """
    if not path.exists():
        return False

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    expected_ids = [int(i) for i in job["indices"]]
    rows = payload.get("results")

    if not isinstance(rows, list):
        return False

    checks = (
        payload.get("model") == name,
        payload.get("suite") == job["suite_name"],
        payload.get("suite_version") == job["suite_version"],
        payload.get("protocol") == job["protocol"],
        payload.get("inference_seed") == inference_seed,
        payload.get("n_episodes") == len(expected_ids),
        len(rows) == len(expected_ids),
    )

    if not all(checks):
        return False

    if [row.get("episode") for row in rows] != expected_ids:
        return False

    if job["protocol"] == "identifiability":
        if payload.get("population") != "full_suite":
            return False
        if payload.get("suite_revision") != IDENT_COMMIT:
            return False
    else:
        if payload.get("selection_seed") != SELECTION_SEED:
            return False
        if payload.get("selection_file") != job["selection_file"]:
            return False

    return True


# --------------------------------------------------------------------- #
# result writer
# --------------------------------------------------------------------- #

def run_and_save(name, job, get_rows, *, model_type, config, inference_seed):
    """Run one method unless its complete result already exists."""
    path = result_path(name, job["protocol"], inference_seed)

    if is_complete(path, name=name, job=job, inference_seed=inference_seed):
        print(f"SKIP {path} (complete)")
        return None

    if path.exists():
        print(f"REPLACE {path} (incomplete or incompatible)")

    print(
        f"\nRUN {name} | {job['protocol']}"
        f" | seed={inference_seed} | n={len(job['indices'])}"
    )

    rows = get_rows()
    n_ok = sum(row.get("status") == "OK" for row in rows)

    payload = {
        "model": name,
        "model_type": model_type,
        "model_config": config,
        "suite": job["suite_name"],
        "suite_version": job["suite_version"],
        "suite_revision": job["suite_revision"],
        "protocol": job["protocol"],
        "population": job["population"],
        "selection_seed": (
            None if job["protocol"] == "identifiability" else SELECTION_SEED
        ),
        "selection_file": job["selection_file"],
        "inference_seed": inference_seed,
        "n_episodes": len(job["indices"]),
        "package_versions": package_versions(),
        "git_commit": git_commit(),
        "n_ok": n_ok,
        "n_failed": len(rows) - n_ok,
        "results": rows,
    }

    atomic_json_dump(payload, path)
    print(f"SAVED {path} ({n_ok} OK, {len(rows) - n_ok} failed)")

    return rows


# --------------------------------------------------------------------- #
# deterministic references
# --------------------------------------------------------------------- #

def run_dotime_baseline(baseline_name, factory, job):
    """Run one deterministic released DoTime baseline."""

    def get_rows():
        return run_baseline(
            factory(), job["suite"], job["suite_name"], job["indices"]
        )

    run_and_save(
        baseline_name,
        job,
        get_rows,
        model_type="dotime_baseline",
        config={},
        inference_seed=None,
    )


def run_dotime_oracle(job):
    """Run the outcome-level Oracle and use it to validate query alignment.

    The Oracle returns the true interventional outcome, so its prediction must
    equal y_true. A deviation means the query step is being read at the wrong
    index -- fail here, before any GPU time is spent on the TSFMs.
    """

    def get_rows():
        return run_oracle(job["suite"], job["suite_name"], job["indices"])

    rows = run_and_save(
        "oracle",
        job,
        get_rows,
        model_type="dotime_oracle",
        config={},
        inference_seed=None,
    )

    if rows is None:      # skipped; already validated on the run that wrote it
        return

    ok = [row for row in rows if row.get("status") == "OK"]

    if not ok:
        raise RuntimeError(f"Oracle produced no successful rows on {job['protocol']}.")

    error = max(abs(row["p1_query"] - row["y_true_query"]) for row in ok)

    if error > ORACLE_TOLERANCE:
        raise RuntimeError(
            f"Oracle deviates from y_true by {error:.3e} on {job['protocol']}."
        )

    print(f"Oracle alignment OK (max |pred - y_true| = {error:.2e})")


# --------------------------------------------------------------------- #
# build final jobs
# --------------------------------------------------------------------- #

def build_jobs() -> list[dict]:
    """Load suites and freeze the final evaluation populations."""
    ident = load_identifiability()
    cont = load_continuous()

    ident_indices = list(range(len(ident)))

    in_ids, _ = select_continuous(
        cont,
        CONT_SUITE,
        regime="in_window",
        seed=SELECTION_SEED,
        n_per_structure=N_PER_STRUCTURE,
    )

    post_ids, _ = select_continuous(
        cont,
        CONT_SUITE,
        regime="post_window",
        seed=SELECTION_SEED,
        n_per_structure=N_PER_STRUCTURE,
    )

    in_ids = [int(i) for i in in_ids]
    post_ids = [int(i) for i in post_ids]

    validate_continuous_selection(cont, in_ids, regime="in_window")
    validate_continuous_selection(cont, post_ids, regime="post_window")

    in_path = freeze_selection(cont, in_ids, protocol="continuous_in_window")
    post_path = freeze_selection(cont, post_ids, protocol="continuous_post_window")

    jobs = [
        {
            "protocol": "identifiability",
            "suite": ident,
            "suite_name": IDENT_SUITE,
            "suite_version": IDENT_VERSION,
            "suite_revision": IDENT_COMMIT,
            "population": "full_suite",
            "indices": ident_indices,
            "selection_file": None,
            "models": IDENT_MODELS,
        },
        {
            "protocol": "continuous_in_window",
            "suite": cont,
            "suite_name": CONT_SUITE,
            "suite_version": CONT_VERSION,
            "suite_revision": None,
            "population": "selected",
            "indices": in_ids,
            "selection_file": str(in_path.relative_to(RESULTS_DIR)),
            "models": tuple(MODELS),
        },
        {
            "protocol": "continuous_post_window",
            "suite": cont,
            "suite_name": CONT_SUITE,
            "suite_version": CONT_VERSION,
            "suite_revision": None,
            "population": "selected",
            "indices": post_ids,
            "selection_file": str(post_path.relative_to(RESULTS_DIR)),
            "models": POST_WINDOW_MODELS,
        },
    ]

    for job in jobs:
        print(f"{job['protocol']}: {len(job['indices'])} episodes")

    return jobs


# --------------------------------------------------------------------- #
# TSFM execution
# --------------------------------------------------------------------- #

def run_tsfm_seed(model_name: str, inference_seed: int, jobs) -> None:
    """Load one TSFM once and run all pending protocols for one seed."""
    factory, config = MODELS[model_name]

    pending_jobs = []

    for job in jobs:
        if model_name not in job["models"]:
            continue

        path = result_path(model_name, job["protocol"], inference_seed)

        if is_complete(path, name=model_name, job=job, inference_seed=inference_seed):
            print(f"SKIP {path} (complete)")
        else:
            pending_jobs.append(job)

    if not pending_jobs:
        return

    model = None

    try:
        set_seed(inference_seed)
        print(f"\nLOADING {model_name} for inference seed {inference_seed}")
        model = factory()

        if model_name in ("moirai-1.1", "toto-1.0"):
            model.seed = inference_seed

        for job in pending_jobs:

            def get_rows(job=job):
                # Reseed per protocol so a job's draws do not depend on which
                # other protocols ran before it in this process.
                set_seed(inference_seed)
                return run_model(
                    model, job["suite"], job["suite_name"], job["indices"]
                )

            run_and_save(
                model_name,
                job,
                get_rows,
                model_type="tsfm",
                config=config,
                inference_seed=inference_seed,
            )

    finally:
        if model is not None:
            del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------- #

def main():
    jobs = build_jobs()

    planned = sum(
        len(job["indices"]) * len(seeds_for(name))
        for job in jobs
        for name in job["models"]
    )
    print(f"\nPlanned TSFM episode-runs: {planned:,}")

    print("\n=== deterministic baselines ===")

    for job in jobs:
        for baseline_name, factory in BASELINES.items():
            run_dotime_baseline(baseline_name, factory, job)

        run_dotime_oracle(job)

    print("\n=== external TSFMs ===")

    for model_name in MODELS:
        for inference_seed in seeds_for(model_name):
            run_tsfm_seed(model_name, inference_seed, jobs)

    print("\nALL REQUESTED RUNS FINISHED.")


if __name__ == "__main__":
    main()