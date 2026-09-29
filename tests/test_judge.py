import numpy as np
import pandas as pd
import pytest

from smartrouter.judge import paired_bootstrap_ci, summarize_judge


def synthetic_judge() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score_A": [4, 5, 4, 3],
            "score_B": [5, 5, 4, 4],
            "score_C": [5, 5, 4, 3],
            "cost_A": [0.1, 0.1, 0.1, 0.1],
            "cost_B": [0.5, 0.5, 0.5, 0.5],
            "cost_C": [0.5, 0.5, 0.1, 0.1],
            "router_decision": ["Hard", "Hard", "Easy", "Easy"],
            "intensity": ["STRONG", "STRONG", "MODERATE", "WEAK"],
        }
    )


def test_summary_reproduces_historical_arithmetic():
    s = summarize_judge(synthetic_judge(), n_boot=200)
    assert s["mean_score"] == {"A": 4.0, "B": 4.5, "C": 4.25}
    assert s["historical_cost_reduction_C_vs_B_pct"] == pytest.approx(40.0)
    assert s["heavy_share_C"] == pytest.approx(0.5)
    assert s["agreement_router_vs_intensity_rule"] == pytest.approx(1.0)


def test_bootstrap_is_seeded_and_brackets_mean():
    diff = np.array([0.0, 1.0, 0.0, -1.0, 1.0])
    first = paired_bootstrap_ci(diff, n_boot=500, seed=1)
    assert first == paired_bootstrap_ci(diff, n_boot=500, seed=1)
    assert first[0] <= diff.mean() <= first[1]


def test_missing_columns_raise():
    with pytest.raises(KeyError):
        summarize_judge(synthetic_judge().drop(columns=["cost_C"]))
