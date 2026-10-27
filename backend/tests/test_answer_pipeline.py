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
