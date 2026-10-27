from grounding import analyze_citations, extract_citations, make_snippet


def test_extracts_single_grouped_and_ranged_citations():
    assert extract_citations("A [1]. B [2, 3]. C [4-5].") == [1, 2, 3, 4, 5]


def test_well_cited_answer_is_grounded():
    answer = (
        "Statins lower LDL cholesterol in adults with high cardiovascular risk [1]. "
        "Higher doses produced larger reductions in two trials [2][3]. "
        "This is educational information, not medical advice."
    )
    result = analyze_citations(answer, num_sources=3)
    assert result["grounded"] is True
    assert result["cited"] == [1, 2, 3]
    assert result["invalid"] == []
    assert result["coverage"] == 1.0


def test_citation_to_missing_source_is_flagged():
    answer = "Metformin is the first line therapy for type 2 diabetes in most adults [7]."
    result = analyze_citations(answer, num_sources=5)
    assert result["invalid"] == [7]
    assert result["cited"] == []
    assert result["grounded"] is False


def test_uncited_factual_answer_is_not_grounded():
    answer = (
        "Aspirin reduces the risk of a second heart attack in many patients. "
        "It also increases the chance of gastrointestinal bleeding in older adults."
    )
    result = analyze_citations(answer, num_sources=5)
    assert result["coverage"] == 0.0
    assert result["grounded"] is False


def test_answer_that_declines_is_grounded_without_citations():
    answer = "The provided abstracts do not support a firm answer to this question."
    assert analyze_citations(answer, num_sources=5)["grounded"] is True


def test_disclaimer_sentence_does_not_count_as_uncited_claim():
    answer = "Exercise lowers blood pressure in adults with hypertension [1]. Consult a clinician before changing treatment."
    result = analyze_citations(answer, num_sources=2)
    assert result["claim_sentences"] == 1
    assert result["coverage"] == 1.0


def test_snippet_keeps_short_text_and_cuts_long_text_at_a_boundary():
    assert make_snippet("Short abstract.") == "Short abstract."
    long_text = "First sentence is here. " + "word " * 100
    snippet = make_snippet(long_text, limit=40)
    assert snippet == "First sentence is here."
    assert len(make_snippet("word " * 200, limit=60)) <= 61
