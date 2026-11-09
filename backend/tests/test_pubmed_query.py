import asyncio

import ingest


def test_terms_drop_filler_words():
    terms = ingest.pubmed_terms("What are the benefits of exercise for depression?")
    assert terms == ["exercise", "depression"]


def test_terms_keep_medical_words_and_cap():
    q = "Does metformin reduce cardiovascular mortality in type 2 diabetes patients with kidney disease"
    terms = ingest.pubmed_terms(q)
    assert "metformin" in terms and "diabetes" in terms
    assert len(terms) <= 6 and "does" not in terms


def test_fetch_retries_with_fewer_terms(monkeypatch):
    calls: list[str] = []

    async def fake_esearch(client, term, retmax):
        calls.append(term)
        return ["1", "2", "3", "4", "5", "6", "7", "8", "9"] if term.count(" AND ") <= 2 else []

    async def fake_efetch(client, pmids):
        return [{"title": "t", "abstract": "a", "pmid": p} for p in pmids]

    monkeypatch.setattr(ingest, "_esearch", fake_esearch)
    monkeypatch.setattr(ingest, "_efetch_abstracts", fake_efetch)
    out = asyncio.run(ingest.fetch_pubmed_abstracts("effect of sleep deprivation on memory consolidation in adults"))
    assert len(out) == 9
    assert len(calls) >= 2 and calls[0].endswith("hasabstract")
