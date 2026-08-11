from unittest.mock import patch

import cache
import main
from tests.conftest import AUTH_HEADERS


async def _fake_retrieve_top(question, decompose):
    return [{"title": "T", "abstract": "A", "pmid": "123", "score": 1.0}]


def _fake_generate_answer(call_count):
    def _generate(question, sources):
        call_count["n"] += 1
        return {"answer": "the answer", "citations": [], "prompt_tokens": 10, "completion_tokens": 5}

    return _generate


def test_repeated_normalized_question_hits_cache(client):
    call_count = {"n": 0}
    with (
        patch.object(main, "_retrieve_top", _fake_retrieve_top),
        patch.object(main, "generate_answer", _fake_generate_answer(call_count)),
    ):
        r1 = client.post(
            "/api/v1/query", json={"question": "What is diabetes?"}, headers=AUTH_HEADERS
        )
        r2 = client.post(
            "/api/v1/query", json={"question": "  what IS diabetes?  "}, headers=AUTH_HEADERS
        )

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["answer"] == r2.json()["answer"] == "the answer"
    assert call_count["n"] == 1, "second call should have been served from cache"


def test_decompose_flag_is_part_of_cache_key(client):
    call_count = {"n": 0}
    with (
        patch.object(main, "_retrieve_top", _fake_retrieve_top),
        patch.object(main, "generate_answer", _fake_generate_answer(call_count)),
    ):
        client.post(
            "/api/v1/query",
            json={"question": "What is diabetes?", "decompose": False},
            headers=AUTH_HEADERS,
        )
        client.post(
            "/api/v1/query",
            json={"question": "What is diabetes?", "decompose": True},
            headers=AUTH_HEADERS,
        )

    assert call_count["n"] == 2, "decompose=True/False must be distinct cache entries"


def test_make_key_normalizes_whitespace_and_case():
    k1 = cache.make_key("What is Diabetes?", False)
    k2 = cache.make_key("  what   is diabetes?  ", False)
    assert k1 == k2


def test_make_key_distinguishes_decompose_flag():
    assert cache.make_key("question", True) != cache.make_key("question", False)
