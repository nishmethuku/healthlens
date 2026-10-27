import re
import xml.etree.ElementTree as ET
from typing import TypedDict

import httpx

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_RETMAX = 30

# Question words and filler that carry no search meaning. PubMed combines search
# terms with AND, so a natural-language question can find nothing when a filler
# word is not in any matching paper.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "for", "in", "on", "at", "to", "with", "by", "from", "as",
        "and", "or", "but", "not", "is", "are", "was", "were", "be", "been", "being", "do",
        "does", "did", "can", "could", "should", "would", "will", "may", "might", "what",
        "which", "who", "whom", "whose", "how", "why", "when", "where", "i", "me", "my",
        "we", "you", "your", "it", "its", "this", "that", "these", "those", "there",
        "their", "latest", "newest", "recent", "current", "best",
    }
)


class PubMedUnavailableError(Exception):
    """PubMed could not be reached or returned an error."""


def search_keywords(question: str) -> str:
    """The question reduced to its content words, for a fallback PubMed search."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", question)
    return " ".join(w for w in words if w.lower() not in _STOPWORDS)


class Abstract(TypedDict):
    title: str
    abstract: str
    pmid: str


async def fetch_pubmed_abstracts(
    query: str, retmax: int = DEFAULT_RETMAX, api_key: str | None = None
) -> list[Abstract]:
    """Fetch PubMed abstracts for a question via NCBI E-utilities.

    api_key is optional; with one NCBI allows a higher request rate.
    Raises PubMedUnavailableError if PubMed cannot be reached.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        return await _search_and_fetch(client, query, retmax, api_key)


async def _search_and_fetch(
    client: httpx.AsyncClient, query: str, retmax: int, api_key: str | None
) -> list[Abstract]:
    try:
        pmids = await _esearch(client, query, retmax, api_key)
        if not pmids:
            # The question as written found nothing, so retry with only its
            # content words. The original is tried first so questions that
            # already work behave exactly as before.
            keywords = search_keywords(query)
            if keywords and keywords.lower() != query.strip().lower():
                pmids = await _esearch(client, keywords, retmax, api_key)
        if not pmids:
            return []
        return await _efetch_abstracts(client, pmids, api_key)
    except httpx.HTTPError as exc:
        raise PubMedUnavailableError(str(exc)) from exc


def _with_key(params: dict, api_key: str | None) -> dict:
    return {**params, "api_key": api_key} if api_key else params


async def _esearch(
    client: httpx.AsyncClient, query: str, retmax: int, api_key: str | None = None
) -> list[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
    }
    resp = await client.get(f"{EUTILS_BASE}/esearch.fcgi", params=_with_key(params, api_key))
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


async def _efetch_abstracts(
    client: httpx.AsyncClient, pmids: list[str], api_key: str | None = None
) -> list[Abstract]:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    resp = await client.get(f"{EUTILS_BASE}/efetch.fcgi", params=_with_key(params, api_key))
    resp.raise_for_status()
    return _parse_pubmed_xml(resp.text)


def _parse_pubmed_xml(xml_text: str) -> list[Abstract]:
    root = ET.fromstring(xml_text)
    results: list[Abstract] = []

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_els = article.findall(".//AbstractText")

        pmid = (pmid_el.text or "").strip() if pmid_el is not None else ""
        title = _element_text(title_el)
        abstract = " ".join(_element_text(el) for el in abstract_els).strip()

        # A paper with no abstract gives the answer nothing to cite, so skip it.
        if pmid and title and abstract:
            results.append({"title": title, "abstract": abstract, "pmid": pmid})

    return results


def _element_text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    parts: list[str] = []
    if el.text:
        parts.append(el.text.strip())
    for child in el:
        if child.text:
            parts.append(child.text.strip())
        if child.tail:
            parts.append(child.tail.strip())
    return " ".join(p for p in parts if p).strip()
