"""Episode selection for the FMTS probe.

Protocol (agreed with Dennis):
    filter first, then draw a balanced sample per causal structure.

Populations:
    identifiability  -- all non-zero episodes (every episode is non-self).
    continuous / in_window   -- non-self, t_start <= query_step <  t_end   (primary, R4)
    continuous / post_window -- non-self, query_step >= t_end              (secondary, S1)

Self-intervention episodes (query_target == intervention target) are excluded
everywhere: a query on the variable being clamped is not an interpretable
causal query under these identification structures.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .interfaces import parse_episode, reconstruct_window

# Target episodes per structure. Continuous in-window has ~1.09k per
# structure, so this sits close to the ceiling -- hence the strict guard
# in _balanced_sample.
DEFAULT_N_PER_STRUCTURE = 1000

# Controls episode selection only. Not the model inference seed.
DEFAULT_SELECTION_SEED = 42


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #

def _to_numpy(x) -> np.ndarray:
    """DoTime fields may be torch tensors, arrays or lists."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _scalar_int(x) -> int:
    """Extract one int from [2] / array([2]) / tensor([2])."""
    return int(_to_numpy(x).reshape(-1)[0])


def is_zero_episode(ep) -> bool:
    """True if either arm is all-zero (diverged during generation).

    Covers asymmetric cases where only one arm was zeroed.
    """
    return bool(
        np.all(_to_numpy(ep.x_obs) == 0) or np.all(_to_numpy(ep.x_int) == 0)
    )


def classify(ep, suite_name: str) -> str:
    """Label one Continuous episode.

    Returns: "zero" | "self" | "pre_window" | "in_window" | "post_window".
    The intervention is active on [t_start, t_end), so query_step == t_end
    is already post-window.
    """
    if is_zero_episode(ep):
        return "zero"

    if _scalar_int(ep.intervention.targets) == _scalar_int(ep.query_target):
        return "self"

    t_start, t_end = reconstruct_window(ep)
    query_step = parse_episode(ep, suite_name).query_step

    if query_step < t_start:
        return "pre_window"       # not expected; kept so it can't hide
    if query_step < t_end:
        return "in_window"
    return "post_window"


# --------------------------------------------------------------------- #
# balanced sampling
# --------------------------------------------------------------------- #

def _balanced_sample(
    valid_by_structure: dict[str, list[int]],
    *,
    seed: int,
    n_per_structure: int,
    strict: bool = True,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Draw n_per_structure episodes from every structure.

    candidates[:n] truncates silently when a structure is short, which
    would produce an unbalanced sample with no trace. strict=True turns
    that into an error; strict=False records it in the manifest instead.
    """
    counts = {s: len(idx) for s, idx in sorted(valid_by_structure.items())}
    short = {s: n for s, n in counts.items() if n < n_per_structure}

    if short and strict:
        raise ValueError(
            f"Cannot draw {n_per_structure} per structure. Short: {short}. "
            f"Use n_per_structure={min(counts.values())} or strict=False."
        )

    rng = random.Random(seed)
    selected: dict[str, list[int]] = {}

    for structure in sorted(valid_by_structure):          # order-independent
        candidates = list(valid_by_structure[structure])
        rng.shuffle(candidates)
        selected[structure] = sorted(candidates[:n_per_structure])

    balance = {
        "balanced": not short,
        "n_requested_per_structure": n_per_structure,
        "n_drawn_per_structure": {s: len(idx) for s, idx in selected.items()},
        "short_structures": short,
    }
    return selected, balance


def _flatten(selected: dict[str, list[int]]) -> list[int]:
    return sorted(i for idx in selected.values() for i in idx)


# --------------------------------------------------------------------- #
# selectors
# --------------------------------------------------------------------- #

def select_identifiability(
    suite,
    *,
    seed: int = DEFAULT_SELECTION_SEED,
    n_per_structure: int = DEFAULT_N_PER_STRUCTURE,
    strict: bool = True,
) -> tuple[list[int], dict[str, Any]]:
    """Identifiability subset: drop zeroed episodes, then sample."""
    valid: dict[str, list[int]] = defaultdict(list)
    n_zero = 0

    for i, ep in enumerate(suite):
        if is_zero_episode(ep):
            n_zero += 1
        else:
            valid[str(ep.structure)].append(i)

    selected, balance = _balanced_sample(
        valid, seed=seed, n_per_structure=n_per_structure, strict=strict
    )

    manifest = {
        "protocol": "identifiability",
        "rule": "drop zeroed episodes, then stratified sample by structure",
        "selection_seed": seed,
        "n_valid_by_structure": {s: len(v) for s, v in sorted(valid.items())},
        **balance,
        "selected_indices_by_structure": selected,
    }
    return _flatten(selected), manifest


def select_continuous(
    suite,
    suite_name: str,
    *,
    regime: str = "in_window",
    seed: int = DEFAULT_SELECTION_SEED,
    n_per_structure: int = DEFAULT_N_PER_STRUCTURE,
    strict: bool = True,
) -> tuple[list[int], dict[str, Any]]:
    """Continuous subset for one regime.

    regime="in_window"   -> primary analysis + R4 duration accrual
    regime="post_window" -> S1, NaN-tail protocol, eligible models only

    One classification pass keeps both regimes on the same rules.
    """
    if regime not in ("in_window", "post_window"):
        raise ValueError(f"regime must be in_window or post_window, got {regime!r}")

    valid: dict[str, list[int]] = defaultdict(list)
    tally: dict[str, int] = defaultdict(int)

    for i, ep in enumerate(suite):
        label = classify(ep, suite_name)
        tally[label] += 1
        if label == regime:
            valid[str(ep.structure)].append(i)

    selected, balance = _balanced_sample(
        valid, seed=seed, n_per_structure=n_per_structure, strict=strict
    )

    manifest = {
        "protocol": f"continuous_{regime}",
        "rule": (
            "drop zeroed and self-intervention episodes, keep "
            f"{regime} queries, then stratified sample by structure"
        ),
        "window_convention": "active on [t_start, t_end); query_step == t_end is post-window",
        "selection_seed": seed,
        "n_valid_by_structure": {s: len(v) for s, v in sorted(valid.items())},
        **balance,
        "selected_indices_by_structure": selected,
    }
    return _flatten(selected), manifest


# --------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------- #

def save_manifest(
    manifest: dict[str, Any],
    output_path: str | Path,
    *,
    suite_name: str,
    suite_version: str,
) -> Path:
    """Persist the selection so the exact episode set is recoverable."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"suite": suite_name, "suite_version": suite_version, **manifest}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return output_path