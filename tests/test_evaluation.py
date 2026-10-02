import pandas as pd
import pytest

from smartrouter.evaluation import evaluate_routing, label_distribution, rank_auc, routing_metrics


def test_routing_metrics_counts_and_rates():
    m = routing_metrics([1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0])
    assert (m.tp, m.fp, m.tn, m.fn) == (2, 1, 2, 1)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.heavy_share == pytest.approx(0.5)
    assert m.light_share == pytest.approx(0.5)
    assert m.fn_rate_of_hard == pytest.approx(1 / 3)
    assert m.fn_rate_of_total == pytest.approx(1 / 6)


def test_rank_auc_known_values():
    assert rank_auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)
    assert rank_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert rank_auc([1, 1], [0.1, 0.2]) is None


def synthetic_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "full_input_text": [
                "[STRONG] [NONE] a",
                "[STRONG] [NONE] b",
                "[STRONG] [NONE] c",
                "[MODERATE] [NONE] d",
                "[MODERATE] [NONE] e",
                "[WEAK] [NONE] f",
            ],
            "true_label": [1, 1, 0, 1, 0, 0],
            "hard_prob": [0.80, 0.70, 0.60, 0.20, 0.10, 0.05],
        }
    )


def test_evaluate_routing_threshold_boundary_and_baseline():
    report = evaluate_routing(synthetic_predictions(), thresholds=[0.20, 0.25])
    at_020, at_025 = report["model"]
    assert at_020["tp"] == 3  # 0.20 >= 0.20 counts as heavy
    assert at_025["fn"] == 1

    rule = report["baseline_intensity_rule"]
    assert (rule["tp"], rule["fp"], rule["fn"]) == (2, 1, 1)
    assert report["agreement_model_vs_rule"] == pytest.approx(1.0)

    levels = {row["intensity"]: row for row in report["by_intensity"]}
    assert set(levels) == {"STRONG", "MODERATE", "WEAK"}  # parsed from full_input_text
    assert levels["STRONG"]["auc_within"] == pytest.approx(1.0)
    assert levels["WEAK"]["auc_within"] is None


def test_evaluate_routing_rejects_missing_values():
    df = synthetic_predictions()
    df.loc[0, "hard_prob"] = None
    with pytest.raises(ValueError):
        evaluate_routing(df)


def test_label_distribution_by_tag_and_source():
    df = pd.DataFrame(
        {
            "meta_intensity": ["strong", "STRONG", "STRONG", "WEAK", "MODERATE", "STRONG"],
            "meta_field": ["NONE", "NONE", "NONE", "NONE", "THESIS", "THESIS"],
            "label_difficulty": [1, 1, 0, 0, 0, 1],
            "source_origin": ["v2", "v2", "v2", "v2", "v1", "v1"],
        }
    )
    report = label_distribution(df)
    assert report["n"] == 6
    intensity = {row["value"]: row for row in report["overall"]["meta_intensity"]}
    assert intensity["STRONG"]["n"] == 4  # case-normalized
    assert intensity["STRONG"]["hard_rate"] == pytest.approx(0.75)
    assert intensity["WEAK"]["hard"] == 0
    v2 = {row["value"]: row for row in report["by_source"]["v2"]["meta_field"]}
    assert v2 == {"NONE": {"value": "NONE", "n": 4, "hard": 2, "hard_rate": 0.5}}


def test_label_distribution_requires_columns():
    with pytest.raises(KeyError):
        label_distribution(pd.DataFrame({"label_difficulty": [1]}))