"""
Routing-classifier evaluation on stored probabilities.

Terminology is deliberate:
- light_share / heavy_share are routing shares, NOT cost savings.
- fn_rate_of_hard = FN / actual Hard (missed hard requests); fn_rate_of_total = FN / all rows.
"""

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

INTENSITY_PATTERN = r"^\[\s*(\w+)\s*\]"


@dataclass(frozen=True)
class RoutingMetrics:
    threshold: Optional[float]
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    heavy_share: float
    light_share: float
    fn_rate_of_hard: float
    fn_rate_of_total: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def routing_metrics(y_true: Iterable[int], y_pred: Iterable[int], threshold: Optional[float] = None) -> RoutingMetrics:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(y_pred), dtype=int)
    if y.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same length")

    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    n = len(y)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return RoutingMetrics(
        threshold=threshold,
        n=n,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        accuracy=_safe_div(tp + tn, n),
        precision=precision,
        recall=recall,
        f1=_safe_div(2 * precision * recall, precision + recall),
        heavy_share=_safe_div(tp + fp, n),
        light_share=_safe_div(tn + fn, n),
        fn_rate_of_hard=_safe_div(fn, tp + fn),
        fn_rate_of_total=_safe_div(fn, n),
    )


def rank_auc(y_true: Iterable[int], scores: Iterable[float]) -> Optional[float]:
    """ROC AUC via the Mann-Whitney rank statistic (ties get average ranks)."""
    y = np.asarray(list(y_true), dtype=int)
    s = pd.Series(np.asarray(list(scores), dtype=float))
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = s.rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def intensity_rule_predictions(intensities: Iterable[str]) -> np.ndarray:
    """Baseline: heavy iff intensity == STRONG."""
    return np.array([str(v).upper() == "STRONG" for v in intensities], dtype=int)


def resolve_intensity(df: pd.DataFrame, intensity_col: str, text_col: str) -> pd.Series:
    if intensity_col in df.columns:
        return df[intensity_col].astype(str).str.upper()
    if text_col in df.columns:
        return df[text_col].astype(str).str.extract(INTENSITY_PATTERN)[0].fillna("UNKNOWN").str.upper()
    raise KeyError(f"Neither '{intensity_col}' nor '{text_col}' found for intensity")


def evaluate_routing(
    df: pd.DataFrame,
    thresholds: Iterable[float] = (0.20, 0.25, 0.50),
    label_col: str = "true_label",
    prob_col: str = "hard_prob",
    intensity_col: str = "tag_intensity",
    text_col: str = "full_input_text",
    reference_threshold: float = 0.25,
) -> dict[str, Any]:
    """
    Full routing report: model metrics per threshold, intensity-rule baseline,
    model/baseline agreement and within-intensity discrimination.
    """
    missing = [c for c in (label_col, prob_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    if df[[label_col, prob_col]].isna().any().any():
        raise ValueError("Label/probability columns contain missing values")

    y = df[label_col].astype(int).to_numpy()
    prob = df[prob_col].astype(float).to_numpy()
    intensity = resolve_intensity(df, intensity_col, text_col)

    model_rows = [routing_metrics(y, (prob >= t).astype(int), threshold=t).as_dict() for t in thresholds]
    rule_pred = intensity_rule_predictions(intensity)
    ref_pred = (prob >= reference_threshold).astype(int)

    by_intensity = []
    for level, idx in intensity.groupby(intensity).groups.items():
        mask = intensity.index.isin(idx)
        by_intensity.append(
            {
                "intensity": level,
                "n": int(mask.sum()),
                "hard_rate": float(y[mask].mean()),
                "prob_min": float(prob[mask].min()),
                "prob_mean": float(prob[mask].mean()),
                "prob_max": float(prob[mask].max()),
                "auc_within": rank_auc(y[mask], prob[mask]),
            }
        )

    return {
        "n": int(len(df)),
        "actual_hard": int(y.sum()),
        "model": model_rows,
        "baseline_intensity_rule": routing_metrics(y, rule_pred).as_dict(),
        "agreement_model_vs_rule": float((ref_pred == rule_pred).mean()),
        "reference_threshold": reference_threshold,
        "auc_overall": rank_auc(y, prob),
        "by_intensity": by_intensity,
    }
