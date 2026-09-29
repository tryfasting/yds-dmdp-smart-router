from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from smartrouter.main import app  # noqa: E402
from smartrouter.router.dynamic import RoBERTaDynamicRouter  # noqa: E402


@pytest.fixture
def client():
    classifier = MagicMock()
    classifier.predict_probability.return_value = 0.9
    with TestClient(app) as test_client:
        app.state.router = RoBERTaDynamicRouter(classifier=classifier)
        yield test_client
    app.state.router = None


def test_health(client):
    assert client.get("/health").json() == {"status": "healthy"}


def test_route_returns_model_and_baseline(client):
    response = client.post("/route", json={"text": "sample", "intensity": "MODERATE", "field": "NONE"})
    assert response.status_code == 200
    body = response.json()
    assert body["tier"] == "heavy"
    assert body["baseline_tier"] == "light"
    assert "text" not in body  # input is not echoed back


def test_route_validation_error(client):
    assert client.post("/route", json={"text": ""}).status_code == 422


def test_unloaded_model_returns_503(client):
    app.state.router = None
    assert client.get("/health").status_code == 503
