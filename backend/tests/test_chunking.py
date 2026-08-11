from retrieval import CorpusIndex, chunk_abstract


def test_chunk_abstract_splits_on_sentences():
    chunks = chunk_abstract(
        "Diabetes Study",
        "Diabetes is a chronic disease. It affects blood sugar levels. Treatment includes insulin.",
    )
    assert len(chunks) == 3
    assert all(c.startswith("Diabetes Study ") for c in chunks)
    assert chunks[0].endswith("Diabetes is a chronic disease.")
    assert chunks[2].endswith("Treatment includes insulin.")


def test_chunk_abstract_empty_falls_back_to_title():
    assert chunk_abstract("Some Title", "") == ["Some Title"]
    assert chunk_abstract("", "") == []


def test_corpus_index_chunk_pmid_mapping_length_matches():
    abstracts = [
        {"pmid": "1", "title": "Diabetes Study", "abstract": "Insulin helps. Diet also matters."},
        {"pmid": "2", "title": "Cardiac Study", "abstract": "Statins reduce risk."},
    ]
    idx = CorpusIndex(abstracts)  # no cache_dir: tiny corpus, builds fresh
    assert len(idx.chunk_texts) == len(idx.chunk_pmids)
    assert len(idx.chunk_texts) == 3  # 2 sentences + 1 sentence
    assert set(idx.chunk_pmids) == {"1", "2"}


def test_retrieve_dedupes_by_pmid_even_with_multiple_matching_chunks():
    # A PMID with many sentences shouldn't be able to occupy more than one
    # of the top_k result slots.
    abstracts = [
        {
            "pmid": "1",
            "title": "Diabetes Study",
            "abstract": (
                "Diabetes affects blood sugar. Diabetes treatment includes insulin. "
                "Diabetes management requires diet control. Diabetes is chronic."
            ),
        },
        {"pmid": "2", "title": "Unrelated Cardiac Study", "abstract": "Statins reduce risk."},
    ]
    idx = CorpusIndex(abstracts)
    results = idx.retrieve("diabetes treatment", top_k=5, mode="bm25")
    pmids = [r["pmid"] for r in results]
    assert len(pmids) == len(set(pmids)), "no PMID should appear twice in results"
    assert pmids[0] == "1"
