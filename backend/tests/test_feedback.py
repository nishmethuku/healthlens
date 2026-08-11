import json

import feedback
from tests.conftest import AUTH_HEADERS


def test_feedback_appends_jsonl_record(client, tmp_path, monkeypatch):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", path)

    res = client.post(
        "/api/v1/feedback",
        json={"query_id": "abc-123", "rating": "up", "answer_preview": "some answer text"},
        headers=AUTH_HEADERS,
    )
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["query_id"] == "abc-123"
    assert record["rating"] == "up"
    assert record["answer_preview"] == "some answer text"
    assert "timestamp" in record


def test_feedback_rejects_invalid_rating(client):
    res = client.post(
        "/api/v1/feedback",
        json={"query_id": "abc-123", "rating": "sideways", "answer_preview": "x"},
        headers=AUTH_HEADERS,
    )
    assert res.status_code == 422


def test_query_response_includes_query_id(client):
    res = client.post(
        "/api/v1/query", json={"question": "how do I kill myself"}, headers=AUTH_HEADERS
    )
    assert res.status_code == 200
    assert "query_id" in res.json()
    assert len(res.json()["query_id"]) > 0
