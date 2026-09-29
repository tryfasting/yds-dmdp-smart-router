"""Loads the real local checkpoint when present. Checks wiring only, not model quality."""

import pytest

from smartrouter.core import config

pytestmark = pytest.mark.skipif(
    not (config.MODEL_PATH / "model.safetensors").exists(), reason="local checkpoint not available"
)


def test_checkpoint_loads_and_returns_probabilities():
    from smartrouter.models.classifier import RoBERTaClassifier
    from smartrouter.router.dynamic import format_input

    classifier = RoBERTaClassifier(device="cpu")
    probs = classifier.predict_proba(
        [format_input("합성 예시 문장입니다.", "WEAK", "NONE"), format_input("합성 예시 문장입니다.", "STRONG", "NONE")]
    )
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_missing_checkpoint_raises(tmp_path):
    from smartrouter.models.classifier import ModelNotFoundError, RoBERTaClassifier

    with pytest.raises(ModelNotFoundError):
        RoBERTaClassifier(model_path=tmp_path)
