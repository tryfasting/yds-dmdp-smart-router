import pytest
from unittest.mock import MagicMock
from smartrouter.router.dynamic import RoBERTaDynamicRouter
from smartrouter.models.classifier import RoBERTaClassifier
from smartrouter.core import config

def test_format_input():
    # Instantiate router with a mock classifier to avoid heavy weight loading
    mock_classifier = MagicMock(spec=RoBERTaClassifier)
    router = RoBERTaDynamicRouter(classifier=mock_classifier)
    
    formatted = router._format_input("Hello world", "WEAK", "EMAIL")
    assert formatted == "[WEAK] [EMAIL] Hello world"

def test_routing_easy_sentence():
    mock_classifier = MagicMock(spec=RoBERTaClassifier)
    # Set probability to 0.15 (Easy sentence, below 0.25 threshold)
    mock_classifier.predict_probability.return_value = 0.15
    
    router = RoBERTaDynamicRouter(classifier=mock_classifier)
    result = router.predict("This is an easy sentence.", "WEAK", "NONE")
    
    assert result["decision"] == "EASY"
    assert result["target_model"] == config.MODEL_NANO
    assert result["risk_level"] == "low"
    assert result["prob_hard"] == 0.15

def test_routing_hard_sentence():
    mock_classifier = MagicMock(spec=RoBERTaClassifier)
    # Set probability to 0.85 (Hard sentence, above 0.25 threshold)
    mock_classifier.predict_probability.return_value = 0.85
    
    router = RoBERTaDynamicRouter(classifier=mock_classifier)
    result = router.predict("Topological Manifolds and backpropagation analysis.", "STRONG", "THESIS")
    
    assert result["decision"] == "HARD"
    assert result["target_model"] == config.MODEL_MINI
    assert result["risk_level"] == "critical"
    assert result["prob_hard"] == 0.85

def test_routing_fallback_on_exception():
    mock_classifier = MagicMock(spec=RoBERTaClassifier)
    # Simulate an inference failure
    mock_classifier.predict_probability.side_effect = RuntimeError("GPU out of memory")
    
    router = RoBERTaDynamicRouter(classifier=mock_classifier)
    result = router.predict("Error prone sentence", "MODERATE", "NONE")
    
    # Should degrade gracefully to Nano model
    assert result["target_model"] == config.MODEL_NANO
    assert result["decision"] == "FALLBACK_EASY"
    assert result["risk_level"] == "unknown"
