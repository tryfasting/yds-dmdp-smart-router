from typing import Any, Dict, Optional

from smartrouter.core import config
from smartrouter.router.base import BaseRouter


def format_input(text: str, intensity: Optional[str], field: Optional[str]) -> str:
    """
    Training input format of the submitted V8 model: "[INTENSITY] [FIELD] sentence".
    """
    safe_intensity = (intensity or config.DEFAULT_INTENSITY).upper()
    safe_field = (field or config.DEFAULT_FIELD).upper()
    return f"[{safe_intensity}] [{safe_field}] {text}"


class RoBERTaDynamicRouter(BaseRouter):
    """
    Routes a request to the heavy tier when P(Hard) >= threshold (default 0.25).
    Inference errors propagate; callers decide how to degrade.
    """

    name = "roberta"

    def __init__(self, classifier: Any = None, threshold: Optional[float] = None):
        if classifier is None:
            # Lazy import keeps torch out of the metrics-only CLI commands.
            from smartrouter.models.classifier import RoBERTaClassifier

            classifier = RoBERTaClassifier()
        self.classifier = classifier
        self.threshold = config.ROUTER_THRESHOLD if threshold is None else threshold

    def predict(self, text: str, intensity: str, field: str = config.DEFAULT_FIELD) -> Dict[str, Any]:
        formatted = format_input(text, intensity, field)
        prob_hard = float(self.classifier.predict_probability(formatted))
        is_heavy = prob_hard >= self.threshold
        return {
            "router": self.name,
            "tier": config.TIER_HEAVY if is_heavy else config.TIER_LIGHT,
            "prob_hard": round(prob_hard, 4),
            "threshold": self.threshold,
            "intensity": intensity,
            "field": field,
        }


class IntensityRuleRouter(BaseRouter):
    """
    Baseline: heavy tier iff the requested intensity is STRONG.
    On the saved V8 test predictions this rule agrees with the RoBERTa router on 99.65% of rows.
    """

    name = "intensity_rule"

    def predict(self, text: str, intensity: str, field: str = config.DEFAULT_FIELD) -> Dict[str, Any]:
        is_heavy = (intensity or "").upper() == "STRONG"
        return {
            "router": self.name,
            "tier": config.TIER_HEAVY if is_heavy else config.TIER_LIGHT,
            "intensity": intensity,
            "field": field,
        }
