from typing import Dict, Any
from smartrouter.router.base import BaseRouter
from smartrouter.models.classifier import RoBERTaClassifier
from smartrouter.core.logger import logger
from smartrouter.core import config

class RoBERTaDynamicRouter(BaseRouter):
    """
    Dynamic Router that evaluates input text difficulty using a fine-tuned RoBERTa model
    and routes requests accordingly based on a tunable threshold (default: 0.25, Max F1 balance point).
    """
    
    def __init__(self, classifier: RoBERTaClassifier = None, threshold: float = None):
        self.classifier = classifier or RoBERTaClassifier()
        self.threshold = threshold if threshold is not None else config.ROUTER_THRESHOLD
        logger.info(f"🚦 [RoBERTaDynamicRouter] Loaded with threshold value: {self.threshold}")

    def _format_input(self, text: str, intensity: str, field: str) -> str:
        """
        Formats input text identically to features.py and training data logic:
        Format: [INTENSITY] [FIELD] Input_sentence
        """
        safe_intensity = intensity if intensity else "WEAK"
        safe_field = field if field else config.DEFAULT_FIELD
        
        intensity_tag = f"[{safe_intensity.upper()}]"
        field_tag = f"[{safe_field.upper()}]"
        
        return f"{intensity_tag} {field_tag} {text}"

    def predict(self, text: str, intensity: str, field: str = "NONE") -> Dict[str, Any]:
        """
        Predict difficulty probability and select target model based on threshold.
        """
        try:
            # 1. Format input sentence
            formatted_input = self._format_input(text, intensity, field)
            
            # 2. Model Inference
            hard_prob = self.classifier.predict_probability(formatted_input)
            
            # 3. Decision Logic (configurable threshold, default 0.25)
            is_hard = hard_prob >= self.threshold
            decision = "HARD" if is_hard else "EASY"
            
            # 4. Target Model Mapping (Mini for Hard, Nano for Easy)
            target_model = config.MODEL_MINI if is_hard else config.MODEL_NANO
            
            # 5. Risk classification
            risk_level = self._calculate_risk_level(hard_prob)
            
            result = {
                "input_text": text,
                "decision": decision,
                "prob_hard": round(hard_prob, 4),
                "target_model": target_model,
                "risk_level": risk_level,
                "meta_intensity": intensity,
                "meta_field": field,
                "formatted_input_debug": formatted_input
            }
            return result
            
        except Exception as e:
            logger.error(f"❌ [RoBERTaDynamicRouter] Prediction failed: {e}", exc_info=True)
            # Safe Fallback to Cheap Model in case of failure (Graceful Degradation)
            return {
                "input_text": text,
                "target_model": config.MODEL_NANO,
                "decision": "FALLBACK_EASY",
                "risk_level": "unknown",
                "prob_hard": 0.0
            }

    def _calculate_risk_level(self, prob: float) -> str:
        if prob >= 0.85:
            return "critical"
        elif prob >= self.threshold:
            return "high"
        elif prob >= 0.30:
            return "medium"
        else:
            return "low"
