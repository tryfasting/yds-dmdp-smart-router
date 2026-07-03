import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from smartrouter.main import app
from smartrouter.router.dynamic import RoBERTaDynamicRouter
from smartrouter.services.bedrock_generator import BedrockGenerator

# Helper fixture to inject mocked router and generator into app state
@pytest.fixture
def client():
    mock_router = MagicMock(spec=RoBERTaDynamicRouter)
    mock_generator = MagicMock(spec=BedrockGenerator)
    
    # Mock return values
    mock_router.predict.return_value = {
        "target_model": "anthropic.claude-3-haiku-20240307-v1:0",
        "decision": "EASY",
        "risk_level": "low",
        "prob_hard": 0.25
    }
    
    mock_generator.get_correction.return_value = "Corrected output sentence."
    
    with TestClient(app) as test_client:
        # Set application state with mock instances after lifespan runs
        app.state.router = mock_router
        app.state.generator = mock_generator
        yield test_client
        
    app.state.router = None
    app.state.generator = None

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_correct_sentence_endpoint(client):
    payload = {
        "text": "This is a raw input sentence.",
        "intensity": "WEAK",
        "field": "NONE"
    }
    response = client.post("/correct", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["input_text"] == payload["text"]
    assert data["final_output"] == "Corrected output sentence."
    assert data["decision"] == "EASY"
    assert data["target_model"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert data["risk_level"] == "low"
    assert "latency_ms" in data
    assert data["generator_used"] == "BedrockGenerator"

def test_correct_endpoint_validation_error(client):
    # Empty text payload should trigger pydantic validation error (422)
    payload = {
        "text": "",
        "intensity": "WEAK",
        "field": "NONE"
    }
    response = client.post("/correct", json=payload)
    assert response.status_code == 422
