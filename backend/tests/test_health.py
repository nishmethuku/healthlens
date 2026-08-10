from unittest.mock import patch

import main


def test_liveness_is_always_ok(client):
    res = client.get("/health/live")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_readiness_reflects_missing_llm_key(client):
    with patch.object(main.settings, "groq_api_key", None):
        res = client.get("/health/ready")
    assert res.status_code == 503
    body = res.json()
    assert body["llm_configured"] is False
    assert body["startup_complete"] is True
    assert body["corpus_index_loaded"] is False


def test_readiness_ok_when_llm_configured(client):
    with patch.object(main.settings, "groq_api_key", "test-key"):
        res = client.get("/health/ready")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
