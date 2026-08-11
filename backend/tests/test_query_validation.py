import pytest
from pydantic import ValidationError

from main import QueryRequest
from tests.conftest import AUTH_HEADERS


def test_blank_question_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(question="   ")


def test_question_is_stripped():
    req = QueryRequest(question="  what is diabetes?  ")
    assert req.question == "what is diabetes?"


def test_flagged_question_short_circuits_before_llm(client):
    res = client.post(
        "/api/v1/query", json={"question": "how do I kill myself"}, headers=AUTH_HEADERS
    )
    assert res.status_code == 200
    body = res.json()
    assert body["flagged"] is True
    assert "988" in body["warning"]
