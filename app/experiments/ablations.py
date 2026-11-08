"""Ablation matrix helpers (spec 13.2)."""
from __future__ import annotations

from itertools import product

from .baselines import ABLATION_AXES, get_baseline


def build_matrix(axes: list[str], base: str = "pm_rag_full") -> list[dict]:
    """Return a list of config overrides covering the cartesian product of axes."""
    chosen = {a: ABLATION_AXES[a] for a in axes if a in ABLATION_AXES}
    keys = list(chosen.keys())
    combos = []
    for values in product(*[list(chosen[k].keys()) for k in keys]):
        override: dict = {}
        label_parts = []
        for k, v in zip(keys, values):
            override.update(chosen[k][v])
            label_parts.append(f"{k}={v}")
        combos.append({"label": ", ".join(label_parts), "overrides": override, "base": base})
    return combos


def default_retrieval_ablation() -> list[dict]:
    """Retrieval-method ablation for RQ2 / H2."""
    return build_matrix(["retrieval_method"])


def default_scope_ablation() -> list[dict]:
    """Retrieval-scope ablation for RQ3 / H3."""
    return build_matrix(["retrieval_scope"])
