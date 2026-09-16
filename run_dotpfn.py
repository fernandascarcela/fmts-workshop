#!/usr/bin/env python3
"""
Run the public Do-Over-Time-PFN interventional checkpoint as a single-arm
causal reference on the full FMTS evaluation populations.

Decisions
---------
- Public interventional checkpoint s9ho_all_causal/do_over_time_pfn_best.pt.
- Inference is deterministic in eval mode, so there is no seed axis; the
  preflight verifies this rather than assuming it.
- Single-arm reference: p0_query is unavailable, so usage gap, delta-error and
  the TSFM same-weights P1-P0 effect are not computed for this model.
- The common FMTS point prediction is q50 when available. The official DoTime
  predictive mean is retained separately as p1_mean_query for auditability.
- The training prior covers back_door, front_door and instrumental_variable;
  the remaining Identifiability structures are structure-OOD and are flagged
  per episode so the two populations can be reported apart.
- Native quantile head. Deciles the head does not emit (q40, q60 here) come
  from deterministic linear interpolation in tau-space, never extrapolation
  and never sampling.

Populations are reused from run.py: Identifiability v1.1.0 in full, and the
frozen 1,000/structure Continuous selections.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from dotime.reference.pfn import PFNRef, episode_to_batch_interp
from run_probe import (
    RESULTS_DIR,
    atomic_json_dump,
    build_jobs,
    git_commit,
    package_versions,
    parse_episode,
)


MODEL_NAME = "dotime-pfn"

DECILES = tuple(round(i / 10, 1) for i in range(1, 10))
TOL = 1e-6
QUANTILE_EXTRACTION_VERSION = "native_quantile_head_v2_interp_missing_deciles"

CHECKPOINT_REPO = "thummd/do-over-time-pfn"
CHECKPOINT_FILE = "s9ho_all_causal/do_over_time_pfn_best.pt"
CHECKPOINT_REVISION = "07fdf3f5890880e27598e4b2562d1f6b010a5f4e"
CHECKPOINT_SHA256 = "214db2fab16fb5fa17e68244abc0df3b7b0154b9fb2d2ce16b2be3f4d5d93618"
PUBLIC_RELEASE_DATE = "2026-06-25"

TRAINING_PRIOR_STRUCTURES = {"back_door", "front_door", "instrumental_variable"}


# --------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------- #

def checkpoint_provenance(path):
    """Resolve and verify the public checkpoint."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)

    digest = digest.hexdigest()

    if digest != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Checkpoint SHA256 mismatch.\nexpected: {CHECKPOINT_SHA256}\n"
            f"got:      {digest}"
        )

    return {
        "checkpoint_source": "huggingface_public",
        "checkpoint_repo": CHECKPOINT_REPO,
        "checkpoint_file": CHECKPOINT_FILE,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": digest,
        "public_release_date": PUBLIC_RELEASE_DATE,
        "observational_checkpoint": False,
        "training_seed": None,
        "training_prior_structures": sorted(TRAINING_PRIOR_STRUCTURES),
        "structure_ood_note": (
            "Structures outside back_door/front_door/instrumental_variable are "
            "structure-level OOD for s9ho_all_causal."
        ),
        "deterministic_inference": True,
        "inference_seed": None,
        "predictive_quantiles": True,
        "quantile_levels": list(DECILES),
        "quantile_extraction_version": QUANTILE_EXTRACTION_VERSION,
        "quantile_sampling": False,
        "point_output": "q50_common_fmts_protocol",
        "mean_output": "official_predictive_mean_retained_for_audit",
    }


# --------------------------------------------------------------------- #
# prediction
# --------------------------------------------------------------------- #

def quantile_head(model):
    """Fail early if the loaded checkpoint is not the expected quantile head."""
    core = model.model
    head = getattr(core, "quantile_head", None)

    if str(getattr(core, "head_type", "")).lower() != "quantile" or head is None:
        raise RuntimeError("Expected the public DoT-PFN quantile head.")

    if getattr(head, "tau_levels", None) is None:
        raise RuntimeError("Quantile head does not expose tau_levels.")

    return head


def common_deciles(row, tau):
    """Read the requested deciles off one native quantile row.

    Levels the head emits are taken as-is; the rest are linearly interpolated
    between the two nearest native levels. Extrapolation past the native range
    would invent tail behaviour the checkpoint never expressed, so it raises.
    """
    values, interpolated = [], []

    for level in DECILES:
        distance = torch.abs(tau - level)
        nearest = int(torch.argmin(distance))

        if float(distance[nearest]) <= TOL:
            values.append(row[nearest])
            continue

        lower = torch.nonzero(tau < level).flatten()
        upper = torch.nonzero(tau > level).flatten()

        if lower.numel() == 0 or upper.numel() == 0:
            raise RuntimeError(
                f"Cannot interpolate q={level:.2f}: outside native tau range."
            )

        lo, hi = int(lower[-1]), int(upper[0])
        weight = (level - float(tau[lo])) / (float(tau[hi]) - float(tau[lo]))
        values.append(row[lo] + weight * (row[hi] - row[lo]))
        interpolated.append(level)

    return torch.stack(values), interpolated


@torch.no_grad()
def predict_distribution(model, head, episode):
    """Return (predictive mean, deciles, interpolated levels) in raw units."""
    batch = episode_to_batch_interp(
        episode, model.n_max, model.device, model.observational
    )
    output = model.model(batch)

    tau = torch.as_tensor(head.tau_levels).detach().flatten().cpu().double()
    predicted = torch.as_tensor(output).reshape(-1, output.shape[-1])

    if predicted.shape[0] != 1 or predicted.shape[-1] != tau.numel():
        raise RuntimeError(
            f"Unexpected quantile output shape {tuple(predicted.shape)} "
            f"for {tau.numel()} tau levels."
        )

    mean_norm = torch.as_tensor(head.predict_mean(output)).reshape(-1)

    if mean_norm.numel() != 1:
        raise RuntimeError(f"Expected one predictive mean, got {mean_norm.numel()}.")

    quantiles_norm, interpolated = common_deciles(predicted[0], tau)

    # The head predicts in normalized space; undo the batch's per-variate
    # standardization before anything is stored or compared.
    outcome = int(episode.query_target[0])
    scale = batch["_norm_stds"][0, outcome]
    offset = batch["_norm_means"][0, outcome]

    mean = float((mean_norm[0] * scale + offset).cpu())
    quantiles = [
        float(v) for v in (quantiles_norm.to(scale.device, scale.dtype) * scale + offset).cpu()
    ]

    if not np.isfinite(mean) or not np.all(np.isfinite(quantiles)):
        raise RuntimeError("Non-finite DoT-PFN prediction.")

    if np.any(np.diff(quantiles) < -TOL):
        raise RuntimeError("Predictive quantiles cross.")

    return mean, dict(zip(map(str, DECILES), quantiles)), interpolated


def predict_one(model, head, job, index):
    ep = job["suite"][index]
    inp = parse_episode(ep, job["suite_name"])
    mean, quantiles, _ = predict_distribution(model, head, ep)

    return {
        "episode": int(index),
        "structure": str(ep.structure),
        "interface": "native_intervention_causal_reference",
        "horizon": int(inp.horizon),
        "query_step": int(inp.query_step),
        "p0_query": None,
        "p1_query": float(quantiles["0.5"]),
        "p1_mean_query": mean,
        "p0_quantiles": None,
        "p1_quantiles": quantiles,
        "y_obs_query": float(inp.y_obs),
        "y_true_query": float(inp.y_true),
        # Lets the causal reference be reported separately on the structures
        # its training prior covers and on those it does not.
        "structure_in_training_prior": str(ep.structure) in TRAINING_PRIOR_STRUCTURES,
        "status": "OK",
    }


def run_population(model, head, job, indices):
    rows = []
    start = time.time()

    for position, index in enumerate(indices, start=1):
        try:
            rows.append(predict_one(model, head, job, int(index)))
        except Exception as exc:
            rows.append({
                "episode": int(index),
                "structure": str(job["suite"][index].structure),
                "status": f"FAILED: {type(exc).__name__}: {exc}",
            })

        if position % 100 == 0 or position == len(indices):
            elapsed = time.time() - start
            rate = position / elapsed if elapsed else float("nan")
            eta = (len(indices) - position) / rate / 60 if rate else float("nan")
            print(f"  {position:>6}/{len(indices)} | {rate:6.2f} ep/s | ETA {eta:5.1f} min")

    return rows


# --------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------- #

def preflight(model, head, job):
    """Verify determinism and query alignment before spending any GPU time.

    Both checks guard claims the rest of the script depends on: that repeated
    calls agree (so one run suffices), and that y_true really is the
    interventional outcome at the queried step (so every metric downstream is
    reading the same cell).
    """
    ep = job["suite"][int(job["indices"][0])]
    inp = parse_episode(ep, job["suite_name"])

    from_pair = float(np.asarray(ep.x_int)[inp.query_step, inp.outcome_index])
    from_episode = float(np.asarray(ep.y_true, dtype=float).reshape(-1)[0])
    error = max(abs(from_pair - inp.y_true), abs(from_episode - inp.y_true))

    if error > TOL:
        raise RuntimeError(
            f"y_true is not the interventional outcome at the query step "
            f"(max deviation {error:.3e})."
        )

    first_mean, first_q, _ = predict_distribution(model, head, ep)
    second_mean, second_q, _ = predict_distribution(model, head, ep)

    first = np.asarray(
        [first_mean] + [first_q[str(q)] for q in DECILES], dtype=float
    )
    second = np.asarray(
        [second_mean] + [second_q[str(q)] for q in DECILES], dtype=float
    )
    drift = float(np.max(np.abs(first - second)))

    if drift > TOL:
        raise RuntimeError(
            f"Inference is not deterministic across mean+deciles "
            f"(max drift {drift:.3e})."
        )

    print(f"preflight OK | alignment {error:.1e} | determinism {drift:.1e}")

    return {
        "max_alignment_error": error,
        "determinism_max_drift_mean_and_deciles": drift,
    }


# --------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------- #

def result_path(protocol):
    return RESULTS_DIR / "runs" / protocol / f"{MODEL_NAME}.json"


def is_complete(path, *, job, indices, sha256):
    """A finished run of exactly this job, checkpoint and quantile protocol."""
    if not path.exists():
        return False

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    rows = payload.get("results")
    config = payload.get("model_config") or {}

    if not isinstance(rows, list):
        return False

    def has_deciles(row):
        quantiles = row.get("p1_quantiles")
        return isinstance(quantiles, dict) and all(
            str(q) in quantiles and np.isfinite(float(quantiles[str(q)]))
            for q in DECILES
        )

    return all((
        payload.get("model") == MODEL_NAME,
        payload.get("suite") == job["suite_name"],
        payload.get("suite_version") == job["suite_version"],
        payload.get("suite_revision") == job["suite_revision"],
        payload.get("protocol") == job["protocol"],
        payload.get("inference_seed") is None,
        payload.get("n_episodes") == len(indices),
        len(rows) == len(indices),
        [row.get("episode") for row in rows] == indices,
        config.get("checkpoint_sha256") == sha256,
        config.get("point_output") == "q50_common_fmts_protocol",
        config.get("quantile_extraction_version") == QUANTILE_EXTRACTION_VERSION,
        all(row.get("status") != "OK" or has_deciles(row) for row in rows),
    ))


def make_payload(job, indices, rows, provenance, checks, native_tau, interpolated, elapsed):
    n_ok = sum(row.get("status") == "OK" for row in rows)

    return {
        "model": MODEL_NAME,
        "model_type": "causal_reference_single_arm",
        "model_config": provenance,
        "suite": job["suite_name"],
        "suite_version": job["suite_version"],
        "suite_revision": job["suite_revision"],
        "protocol": job["protocol"],
        "population": job["population"],
        "selection_seed": None if job["protocol"] == "identifiability" else 42,
        "selection_file": job["selection_file"],
        "inference_seed": None,
        "n_episodes": len(indices),
        # Identical for every episode, so recorded once rather than per row.
        "native_tau_levels": native_tau,
        "interpolated_quantile_levels": interpolated,
        "preflight": checks,
        "methodology": {
            "role": "causal outcome-accuracy and calibration reference",
            "p0_available": False,
            "p1_point_semantics": "q50 on the common FMTS point-prediction protocol",
            "p1_mean_semantics": "official DoTime PFN predictive mean retained for audit",
            "exclude_from_usage_gap": True,
            "exclude_from_delta_err": True,
            "exclude_from_same_weights_p1_minus_p0": True,
            "effect_metric_semantics": (
                "If reported, predicted_effect = p1_query - y_obs_query against "
                "true_effect = y_true_query - y_obs_query. This is not the TSFM "
                "P1-P0 effect and must not be pooled with it."
            ),
            "calibration_target": "y_true_query",
        },
        "package_versions": package_versions(),
        "git_commit": git_commit(),
        "elapsed_seconds": float(elapsed),
        "n_ok": n_ok,
        "n_failed": len(rows) - n_ok,
        "results": rows,
    }


# --------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--protocol",
        action="append",
        choices=("identifiability", "continuous_in_window", "continuous_post_window"),
        help="Repeat to select protocols; omit to run all three.",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    jobs = build_jobs()

    if args.protocol:
        wanted = set(args.protocol)
        jobs = [job for job in jobs if job["protocol"] in wanted]

    checkpoint = hf_hub_download(
        repo_id=CHECKPOINT_REPO,
        filename=CHECKPOINT_FILE,
        revision=CHECKPOINT_REVISION,
    )
    provenance = checkpoint_provenance(checkpoint)

    print("=== DoT-PFN causal reference ===")
    print(f"checkpoint: {CHECKPOINT_FILE}")
    print(f"device:     {args.device}")

    model = PFNRef(checkpoint, device=args.device, observational=False)
    head = quantile_head(model)

    try:
        checks = preflight(model, head, jobs[0])

        native_tau = [
            float(t) for t in torch.as_tensor(head.tau_levels).flatten().tolist()
        ]
        _, _, interpolated = predict_distribution(
            model, head, jobs[0]["suite"][int(jobs[0]["indices"][0])]
        )

        for job in jobs:
            indices = [int(i) for i in job["indices"]]
            path = result_path(job["protocol"])

            if not args.force and is_complete(
                path, job=job, indices=indices,
                sha256=provenance["checkpoint_sha256"],
            ):
                print(f"\nSKIP {path} (complete)")
                continue

            print(f"\nRUN {job['protocol']} | n={len(indices)}")
            start = time.time()
            rows = run_population(model, head, job, indices)

            if args.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.time() - start
            payload = make_payload(
                job, indices, rows, provenance, checks,
                native_tau, interpolated, elapsed,
            )
            atomic_json_dump(payload, path)

            print(
                f"SAVED {path} | {payload['n_ok']} OK, "
                f"{payload['n_failed']} failed | {elapsed / 60:.1f} min"
            )

    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()