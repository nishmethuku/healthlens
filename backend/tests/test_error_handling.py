from unittest.mock import MagicMock, patch

import groq
import httpx

import main
import query_decomposition
from ingest import PubMedUnavailableError
from tests.conftest import AUTH_HEADERS

_REQUEST = httpx.Request("POST", "http://upstream.test")


def _rate_limit_error():
    return groq.RateLimitError(
        "rate limited", response=httpx.Response(429, request=_REQUEST), body=None
    )


async def _retrieve(question, decompose):
    return [{"title": "T", "abstract": "A long enough abstract.", "pmid": "1", "score": 1.0}]


def _raising(exc):
    def _fn(*args, **kwargs):
        raise exc

    return _fn


def _stream_events(text):
    return [block for block in text.strip().split("\n\n") if block]


def test_groq_rate_limit_is_a_429_with_a_clear_message(client):
    with (
        patch.object(main, "_retrieve_top", _retrieve),
        patch.object(main, "generate_answer", _raising(_rate_limit_error())),
    ):
        res = client.post("/api/v1/query", json={"question": "Do statins work?"}, headers=AUTH_HEADERS)
    assert res.status_code == 429
    assert "try again" in res.json()["detail"].lower()


def test_groq_connection_failure_is_a_503(client):
    error = groq.APIConnectionError(request=_REQUEST)
    with (
        patch.object(main, "_retrieve_top", _retrieve),
        patch.object(main, "generate_answer", _raising(error)),
    ):
        res = client.post("/api/v1/query", json={"question": "Do statins work?"}, headers=AUTH_HEADERS)
    assert res.status_code == 503


def test_pubmed_outage_is_a_503_not_a_generic_500(client):
    async def _down(question, decompose):
        raise PubMedUnavailableError("down")

    with patch.object(main, "_retrieve_top", _down):
        res = client.post("/api/v1/query", json={"question": "Do statins work?"}, headers=AUTH_HEADERS)
    assert res.status_code == 503
    assert "PubMed" in res.json()["detail"]


def test_unexpected_errors_stay_generic_and_leak_nothing(client):
    with (
        patch.object(main, "_retrieve_top", _retrieve),
        patch.object(main, "generate_answer", _raising(RuntimeError("secret path /etc/x"))),
    ):
        res = client.post("/api/v1/query", json={"question": "Do statins work?"}, headers=AUTH_HEADERS)
    assert res.status_code == 500
    assert "secret" not in res.text


def test_stream_reports_a_groq_rate_limit_as_an_error_event(client):
    with (
        patch.object(main, "_retrieve_top", _retrieve),
        patch.object(main, "stream_answer", _raising(_rate_limit_error())),
    ):
        res = client.post(
            "/api/v1/query/stream", json={"question": "Do statins work?"}, headers=AUTH_HEADERS
        )
    events = _stream_events(res.text)
    assert events[-1].startswith("event: error")
    assert "try again" in events[-1].lower()


def test_decomposition_failure_falls_back_to_the_original_question():
    index = MagicMock()
    index.retrieve.return_value = [
        {"title": "T", "abstract": "A", "pmid": "1", "score": 1.0}
    ]
    with (
        patch.object(
            query_decomposition, "decompose_query", _raising(groq.APIConnectionError(request=_REQUEST))
        ),
        patch.object(query_decomposition, "rerank_candidates", lambda q, c, top_k: c[:top_k]),
    ):
        result = query_decomposition.decomposed_retrieve("What treats gout?", index, top_k=5)

    assert [r["pmid"] for r in result] == ["1"]
    assert index.retrieve.call_args.args[0] == "What treats gout?"
