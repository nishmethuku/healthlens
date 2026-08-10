from guardrails import check_query


def test_benign_question_not_flagged():
    flagged, warning = check_query("What are the latest treatments for type 2 diabetes?")
    assert flagged is False
    assert warning is None


def test_self_harm_question_flagged():
    flagged, warning = check_query("What's the best way to kill myself?")
    assert flagged is True
    assert warning is not None
    assert "988" in warning


def test_empty_query_not_flagged():
    flagged, warning = check_query("   ")
    assert flagged is False
    assert warning is None


def test_clinical_use_of_sensitive_term_still_flagged():
    # Guardrail is keyword-based, not intent-aware — documenting current
    # (over-cautious) behavior rather than asserting it's ideal.
    flagged, _ = check_query("What is the epidemiology of suicidal ideation in teens?")
    assert flagged is True
