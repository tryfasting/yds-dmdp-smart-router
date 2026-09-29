from unittest.mock import MagicMock

import pytest

from smartrouter.core import config
from smartrouter.router.dynamic import IntensityRuleRouter, RoBERTaDynamicRouter, format_input


def make_router(prob: float, threshold: float = 0.25) -> RoBERTaDynamicRouter:
    classifier = MagicMock()
    classifier.predict_probability.return_value = prob
    return RoBERTaDynamicRouter(classifier=classifier, threshold=threshold)


def test_format_input_matches_training_format():
    assert format_input("Hello world", "weak", "email") == "[WEAK] [EMAIL] Hello world"
    assert format_input("x", None, None) == "[WEAK] [NONE] x"


def test_light_below_threshold():
    result = make_router(0.15).predict("easy", "WEAK", "NONE")
    assert result["tier"] == config.TIER_LIGHT
    assert result["prob_hard"] == 0.15


def test_heavy_at_threshold_boundary():
    # Decision rule is prob >= threshold, identical to the stored pred_label.
    assert make_router(0.25).predict("t", "STRONG", "NONE")["tier"] == config.TIER_HEAVY


def test_classifier_error_propagates():
    classifier = MagicMock()
    classifier.predict_probability.side_effect = RuntimeError("inference failed")
    with pytest.raises(RuntimeError):
        RoBERTaDynamicRouter(classifier=classifier).predict("t", "WEAK", "NONE")


@pytest.mark.parametrize("intensity,tier", [("STRONG", "heavy"), ("strong", "heavy"), ("MODERATE", "light"), ("WEAK", "light")])
def test_intensity_rule_router(intensity, tier):
    assert IntensityRuleRouter().predict("t", intensity, "NONE")["tier"] == tier
