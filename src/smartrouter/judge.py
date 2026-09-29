"""
Summary of the stored LLM-as-a-Judge results (historical run: apps/eval/test_inference_azure.py).

Scenarios: A = light only (gpt-5-nano), B = heavy only (gpt-5-mini), C = router.
Cost columns in the stored CSV are estimates (characters / 2.5 tokens x list price); they exclude
reasoning tokens and Judge cost. This module reproduces the historical arithmetic and adds paired
bootstrap confidence intervals; it does not call any API.
"""

from typing import Any

import numpy as np
import pandas as pd

SCENARIOS = ("A", "B", "C")
PAIRS = (("C", "A"), ("B", "A"), ("B", "C"))


def paired_bootstrap_ci(
    diff: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def summarize_judge(df: pd.DataFrame, n_boot: int = 5000, seed: int = 0) -> dict[str, Any]:
    required = [f"{k}_{s}" for k in ("score", "cost") for s in SCENARIOS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    mean_score = {s: float(df[f"score_{s}"].mean()) for s in SCENARIOS}
    total_cost = {s: float(df[f"cost_{s}"].sum()) for s in SCENARIOS}

    pairs = []
    for a, b in PAIRS:
        diff = (df[f"score_{a}"] - df[f"score_{b}"]).to_numpy(dtype=float)
        low, high = paired_bootstrap_ci(diff, n_boot=n_boot, seed=seed)
        pairs.append(
            {
                "pair": f"{a}-{b}",
                "mean_diff": float(diff.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "ci_includes_zero": low <= 0.0 <= high,
                "tie_rate": float((diff == 0).mean()),
            }
        )

    summary: dict[str, Any] = {
        "n": int(len(df)),
        "mean_score": mean_score,
        "estimated_total_cost_usd": total_cost,
        # Historical report arithmetic, kept for traceability.
        "historical_cost_reduction_C_vs_B_pct": 100 * (total_cost["B"] - total_cost["C"]) / total_cost["B"],
        "historical_score_ratio_C_over_B_pct": 100 * mean_score["C"] / mean_score["B"],
        "score_ratio_A_over_B_pct": 100 * mean_score["A"] / mean_score["B"],
        "paired_differences": pairs,
        "cost_method": "estimate: len(text)/2.5 tokens x list price; excludes reasoning tokens and judge cost",
    }
    if "router_decision" in df.columns:
        summary["heavy_share_C"] = float((df["router_decision"].astype(str).str.lower() == "hard").mean())
    if {"router_decision", "intensity"} <= set(df.columns):
        rule = df["intensity"].astype(str).str.upper() == "STRONG"
        routed = df["router_decision"].astype(str).str.lower() == "hard"
        summary["agreement_router_vs_intensity_rule"] = float((rule == routed).mean())
    return summary
