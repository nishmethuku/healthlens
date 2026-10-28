import pytest
from starlette.requests import Request
from starlette.responses import Response

import feedback
import security
from config import Settings
from tests.conftest import AUTH_HEADERS


def _request(headers: dict[str, str], peer: str = "10.0.0.1") -> Request:
    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": (peer, 1234),
    }
    return Request(scope)


# --- secret comparison -------------------------------------------------------------


def test_constant_time_equals_matches_only_exact_secret():
    assert security.constant_time_equals("s3cret", "s3cret")
    assert not security.constant_time_equals("s3cre", "s3cret")
    assert not security.constant_time_equals("s3cret-extra", "s3cret")


def test_constant_time_equals_rejects_missing_values_and_handles_unicode():
    assert not security.constant_time_equals(None, "s3cret")
    assert not security.constant_time_equals("s3cret", None)
    assert not security.constant_time_equals("", "")
    assert not security.constant_time_equals("sécret", "secret")


# --- request id --------------------------------------------------------------------


def test_request_id_keeps_plain_values_and_replaces_unsafe_ones():
    assert security.sanitize_request_id("abc-123_X.9") == "abc-123_X.9"
    for bad in ["has space", "new\nline", "x" * 65, "<script>", "", None]:
        replaced = security.sanitize_request_id(bad)
        assert replaced != bad
        assert len(replaced) == 36  # a freshly minted uuid


# --- client ip for rate limiting ---------------------------------------------------


def test_client_ip_ignores_forwarded_header_with_no_trusted_hops():
    req = _request({"X-Forwarded-For": "1.2.3.4"})
    assert security.client_ip(req, 0) == "10.0.0.1"


def test_client_ip_reads_from_the_right_so_forged_entries_do_not_count():
    # The client claims to be 9.9.9.9; the trusted proxy appended the real address.
    req = _request({"X-Forwarded-For": "9.9.9.9, 203.0.113.7"})
    assert security.client_ip(req, 1) == "203.0.113.7"
    # Two trusted proxies: the real client is one further left.
    req = _request({"X-Forwarded-For": "9.9.9.9, 203.0.113.7, 10.1.1.1"})
    assert security.client_ip(req, 2) == "203.0.113.7"


def test_client_ip_falls_back_to_peer_when_header_is_shorter_than_hops():
    assert security.client_ip(_request({"X-Forwarded-For": "1.2.3.4"}), 2) == "10.0.0.1"
    assert security.client_ip(_request({}), 1) == "10.0.0.1"


# --- headers -----------------------------------------------------------------------


def test_security_headers_on_api_response(client):
    res = client.get("/health/live")
    assert res.headers["x-content-type-options"] == "nosniff"
    assert res.headers["x-frame-options"] == "DENY"
    assert res.headers["referrer-policy"] == "no-referrer"
    assert "default-src 'none'" in res.headers["content-security-policy"]
    assert res.headers["cache-control"] == "no-store"
    assert "strict-transport-security" not in res.headers  # only in production


def test_hsts_is_added_in_production_and_docs_keep_a_usable_policy():
    prod = Response()
    security.apply_security_headers(prod, "/api/v1/query", production=True)
    assert "max-age=31536000" in prod.headers["strict-transport-security"]
    docs = Response()
    security.apply_security_headers(docs, "/docs", production=False)
    assert "content-security-policy" not in docs.headers


def test_request_id_header_is_not_reflected_when_unsafe(client):
    res = client.get("/health/live", headers={"X-Request-ID": "bad id with spaces"})
    assert res.headers["x-request-id"] != "bad id with spaces"


# --- metrics -----------------------------------------------------------------------


def test_metrics_open_outside_production(client):
    assert client.get("/metrics").status_code == 200


def test_metrics_needs_token_in_production():
    prod = Settings(ENVIRONMENT="production", METRICS_TOKEN="tok")
    assert security.metrics_authorized("Bearer tok", prod)
    assert not security.metrics_authorized("Bearer wrong", prod)
    assert not security.metrics_authorized("tok", prod)
    assert not security.metrics_authorized(None, prod)
    closed = Settings(ENVIRONMENT="production")
    assert not security.metrics_authorized("Bearer anything", closed)


# --- production configuration ------------------------------------------------------


def test_production_config_problems():
    good = Settings(ENVIRONMENT="production", HEALTHLENS_API_KEY="k", CORS_ORIGINS="https://app.example.com")
    assert security.production_config_problems(good) == []
    bad = Settings(ENVIRONMENT="production", CORS_ORIGINS="*, http://app.example.com")
    problems = " ".join(security.production_config_problems(bad))
    assert "HEALTHLENS_API_KEY" in problems
    assert "wildcard" in problems
    assert "not https" in problems


# --- feedback endpoint -------------------------------------------------------------


@pytest.mark.parametrize("query_id", ["", "x" * 65, "has space", "new\nline", '{"a":1}'])
def test_feedback_rejects_unsafe_query_ids(client, query_id):
    res = client.post(
        "/api/v1/feedback",
        json={"query_id": query_id, "rating": "up", "answer_preview": "x"},
        headers=AUTH_HEADERS,
    )
    assert res.status_code == 422


def test_feedback_log_stops_growing_at_the_size_cap(tmp_path, monkeypatch):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)
    monkeypatch.setattr(feedback, "MAX_LOG_BYTES", 200)
    results = [feedback.append(f"id-{i}", "up", "preview text") for i in range(10)]
    assert results[0] is True
    assert results[-1] is False
    assert path.stat().st_size < 400


def test_wrong_and_missing_keys_are_rejected(client):
    body = {"question": "What does a statin do?"}
    assert client.post("/api/v1/query", json=body).status_code == 401
    assert client.post("/api/v1/query", json=body, headers={"X-API-Key": "wrong"}).status_code == 401


# --- rate limiting -----------------------------------------------------------------


def _feedback(client, headers):
    return client.post(
        "/api/v1/feedback",
        json={"query_id": "abc", "rating": "up", "answer_preview": "x"},
        headers={**AUTH_HEADERS, **headers},
    )


@pytest.fixture
def fresh_limiter(tmp_path, monkeypatch):
    import main

    monkeypatch.setattr(feedback, "FEEDBACK_PATH", tmp_path / "feedback.jsonl")
    main.limiter.reset()
    yield main
    main.limiter.reset()


def test_rate_limit_blocks_after_the_per_minute_cap(client, fresh_limiter):
    cap = fresh_limiter.settings.rate_limit_per_minute
    statuses = [_feedback(client, {}).status_code for _ in range(cap + 1)]
    assert statuses[:cap] == [200] * cap
    assert statuses[cap] == 429


def test_rate_limit_is_per_client_and_forged_header_entries_do_not_evade_it(
    client, fresh_limiter, monkeypatch
):
    monkeypatch.setattr(fresh_limiter.settings, "trusted_proxy_hops", 1)
    cap = fresh_limiter.settings.rate_limit_per_minute
    # Client A fills its bucket. The proxy appended its real address (1.1.1.1) on the right.
    for _ in range(cap):
        assert _feedback(client, {"X-Forwarded-For": "5.5.5.5, 1.1.1.1"}).status_code == 200
    # A changes the forged left-hand entry, but the right-hand entry is the same: still blocked.
    assert _feedback(client, {"X-Forwarded-For": "6.6.6.6, 1.1.1.1"}).status_code == 429
    # A different real client has its own bucket.
    assert _feedback(client, {"X-Forwarded-For": "5.5.5.5, 2.2.2.2"}).status_code == 200


def test_app_refuses_to_start_in_production_with_unsafe_config(monkeypatch):
    from fastapi.testclient import TestClient

    import main

    monkeypatch.setattr(main.settings, "environment", "production")
    monkeypatch.setattr(main.settings, "healthlens_api_key", None)
    with (
        pytest.raises(RuntimeError, match="unsafe production configuration"),
        TestClient(main.app),
    ):
        pass
