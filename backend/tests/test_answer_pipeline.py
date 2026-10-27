import json
from unittest.mock import patch

import main
from tests.conftest import AUTH_HEADERS

ABSTRACT = (
    "Statin therapy reduced LDL cholesterol in a randomized trial of adults at high "
    "cardiovascular risk. The effect was larger at higher doses."
)


async def _fake_retrieve_top(question, decompose):
    return [{"title": "Statins", "abstract": ABSTRACT, "pmid": "111", "score": 1.0}]


def _answer(question, sources):
    return {
        "answer": "Statins lower LDL cholesterol in high risk adults according to the trial [1].",
        "citations": [],
        "prompt_tokens": 1,
        "completion_tokens": 1,
    }


def _bad_answer(question, sources):
    return {
        "answer": "Statins cure heart disease completely in every patient who takes them [9].",
        "citations": [],
        "prompt_tokens": 1,
        "completion_tokens": 1,
    }


def _stream(question, sources, usage_sink=None):
    yield "Statins lower LDL cholesterol "
    yield "in high risk adults according to the trial [1]."


def _parse_events(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.split("\n")
        events.append((lines[0][len("event: "):], json.loads(lines[1][len("data: "):])))
    return events


def test_query_returns_snippet_and_grounding(client):
    with (
        patch.object(main, "_retrieve_top", _fake_retrieve_top),
        patch.object(main, "generate_answer", _answer),
    ):
        body = client.post(
            "/api/v1/query", json={"question": "Do statins work?"}, headers=AUTH_HEADERS
        ).json()

    assert body["sources"][0]["snippet"].startswith("Statin therapy reduced LDL")
    assert body["grounding"]["grounded"] is True
    assert body["grounding"]["cited"] == [1]


def test_query_flags_citation_to_nonexistent_source(client):
    with (
        patch.object(main, "_retrieve_top", _fake_retrieve_top),
        patch.object(main, "generate_answer", _bad_answer),
    ):
        body = client.post(
            "/api/v1/query", json={"question": "Do statins cure disease?"}, headers=AUTH_HEADERS
        ).json()

    assert body["grounding"]["grounded"] is False
    assert body["grounding"]["invalid"] == [9]


def test_stream_emits_grounding_event_before_done(client):
    with (
        patch.object(main, "_retrieve_top", _fake_retrieve_top),
        patch.object(main, "stream_answer", _stream),
    ):
        res = client.post(
            "/api/v1/query/stream", json={"question": "Do statins work?"}, headers=AUTH_HEADERS
        )

    events = _parse_events(res.text)
    names = [name for name, _ in events]
    assert names[0] == "sources"
    assert names[-2:] == ["grounding", "done"]
    assert events[0][1]["sources"][0]["snippet"].startswith("Statin therapy")
    grounding = dict(events)["grounding"]
    assert grounding["grounded"] is True


def test_cors_preflight_allows_the_api_key_header(client):
    res = client.options(
        "/api/v1/query/stream",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-api-key",
        },
    )
    assert res.status_code == 200
    assert "x-api-key" in res.headers["access-control-allow-headers"].lower()


def test_stream_serves_repeated_question_from_cache(client):
    calls = {"n": 0}

    def counting_stream(question, sources, usage_sink=None):
        calls["n"] += 1
        yield "Statins lower LDL cholesterol in high risk adults according to the trial [1]."

    with (
        patch.object(main, "_retrieve_top", _fake_retrieve_top),
        patch.object(main, "stream_answer", counting_stream),
    ):
        first = client.post(
            "/api/v1/query/stream", json={"question": "Do statins work?"}, headers=AUTH_HEADERS
        )
        second = client.post(
            "/api/v1/query/stream", json={"question": "  do STATINS work? "}, headers=AUTH_HEADERS
        )

    assert calls["n"] == 1, "second request should be replayed from the cache"
    names = [name for name, _ in _parse_events(second.text)]
    assert names == ["sources", "token", "grounding", "done"]
    # Each request still gets its own query_id so feedback stays per request.
    id_first = _parse_events(first.text)[0][1]["query_id"]
    id_second = _parse_events(second.text)[0][1]["query_id"]
    assert id_first != id_second


def test_stream_answer_is_shared_with_non_streaming_endpoint(client):
    calls = {"n": 0}

    def counting_generate(question, sources):
        calls["n"] += 1
        return _answer(question, sources)

    with (
        patch.object(main, "_retrieve_top", _fake_retrieve_top),
        patch.object(main, "stream_answer", _stream),
        patch.object(main, "generate_answer", counting_generate),
    ):
        client.post("/api/v1/query/stream", json={"question": "Do statins work?"}, headers=AUTH_HEADERS)
        body = client.post(
            "/api/v1/query", json={"question": "Do statins work?"}, headers=AUTH_HEADERS
        ).json()

    assert calls["n"] == 0, "non-streaming call should hit the entry the stream stored"
    assert body["sources"][0]["pmid"] == "111"
    assert body["grounding"]["grounded"] is True
