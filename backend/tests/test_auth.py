from tests.conftest import AUTH_HEADERS


def test_query_without_api_key_rejected(client):
    res = client.post("/api/v1/query", json={"question": "what is diabetes?"})
    assert res.status_code == 401


def test_query_with_wrong_api_key_rejected(client):
    res = client.post(
        "/api/v1/query",
        json={"question": "what is diabetes?"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert res.status_code == 401


def test_query_with_correct_api_key_passes_auth(client):
    # Flagged question short-circuits before the LLM, so this exercises real
    # auth without needing a live Groq key.
    res = client.post(
        "/api/v1/query", json={"question": "how do I kill myself"}, headers=AUTH_HEADERS
    )
    assert res.status_code == 200


def test_health_endpoints_do_not_require_api_key(client):
    assert client.get("/health/live").status_code == 200
