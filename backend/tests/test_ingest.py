import httpx
import pytest

import ingest
from ingest import (
    PubMedUnavailableError,
    _parse_pubmed_xml,
    _search_and_fetch,
    search_keywords,
)

XML = """<PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID>111</PMID><Article>
<ArticleTitle>Has abstract</ArticleTitle><Abstract><AbstractText>Real findings.</AbstractText></Abstract>
</Article></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID>222</PMID><Article>
<ArticleTitle>No abstract here</ArticleTitle>
</Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _esearch_response(ids):
    return httpx.Response(200, json={"esearchresult": {"idlist": ids}})


def test_search_keywords_drops_question_words_and_keeps_content_terms():
    assert (
        search_keywords("What are the latest treatments for type 2 diabetes?")
        == "treatments type 2 diabetes"
    )


def test_papers_without_an_abstract_are_skipped():
    results = _parse_pubmed_xml(XML)
    assert [r["pmid"] for r in results] == ["111"]


@pytest.mark.asyncio
async def test_original_question_is_searched_first_and_keywords_only_on_empty_result():
    terms = []

    def handler(request):
        if request.url.path.endswith("esearch.fcgi"):
            terms.append(request.url.params["term"])
            ids = [] if len(terms) == 1 else ["111"]
            return _esearch_response(ids)
        return httpx.Response(200, text=XML)

    async with _client(handler) as client:
        results = await _search_and_fetch(
            client, "What are the best treatments for gout?", 30, None
        )

    assert terms == ["What are the best treatments for gout?", "treatments gout"]
    assert [r["pmid"] for r in results] == ["111"]


@pytest.mark.asyncio
async def test_no_second_search_when_the_first_one_finds_papers():
    searches = []

    def handler(request):
        if request.url.path.endswith("esearch.fcgi"):
            searches.append(request.url.params["term"])
            return _esearch_response(["111"])
        return httpx.Response(200, text=XML)

    async with _client(handler) as client:
        await _search_and_fetch(client, "What is gout?", 30, None)

    assert len(searches) == 1


@pytest.mark.asyncio
async def test_api_key_is_sent_only_when_configured():
    seen = []

    def handler(request):
        seen.append(request.url.params.get("api_key"))
        if request.url.path.endswith("esearch.fcgi"):
            return _esearch_response(["111"])
        return httpx.Response(200, text=XML)

    async with _client(handler) as client:
        await _search_and_fetch(client, "gout", 30, "KEY123")
        await _search_and_fetch(client, "gout", 30, None)

    assert seen == ["KEY123", "KEY123", None, None]


@pytest.mark.asyncio
async def test_network_failure_becomes_pubmed_unavailable():
    def handler(request):
        raise httpx.ConnectError("boom")

    async with _client(handler) as client:
        with pytest.raises(PubMedUnavailableError):
            await _search_and_fetch(client, "gout", 30, None)


def test_module_exports_default_retmax():
    assert ingest.DEFAULT_RETMAX == 30
