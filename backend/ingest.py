import re
import xml.etree.ElementTree as ET
from typing import TypedDict

import httpx

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_RETMAX = 30


class Abstract(TypedDict):
    title: str
    abstract: str
    pmid: str


_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can", "could", "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "may", "might", "more", "most", "my", "of", "on", "or", "our", "should", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "us", "was", "we", "were", "what", "when", "where", "which", "who", "whom", "why", "will", "with", "would", "you", "your", "about", "after", "also", "any", "been", "between", "both", "but", "during", "each", "few", "get", "gets", "getting", "give", "just", "like", "make", "many", "much", "no", "not", "only", "other", "over", "own", "same", "such", "too", "very", "benefit", "benefits", "effect", "effects", "effective", "effectiveness", "good", "bad", "help", "helps", "use", "used", "using", "work", "works", "tell", "know", "say", "said", "causes", "cause", "risk", "risks"]
)
_MIN_RESULTS = 8


def pubmed_terms(question: str, max_terms: int = 6) -> list[str]:
    """Pick the content words of a question to use as a PubMed search.

    PubMed ANDs every word it is given, so sending the whole sentence ("what are the
    benefits of exercise for depression?") requires filler words to appear in each paper
    and often finds nothing useful. Dropping filler keeps the medical terms.
    """
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", question.lower())
    seen: list[str] = []
    for w in words:
        w = w.removesuffix("'s")
        if len(w) > 1 and w not in _STOPWORDS and w not in seen:
            seen.append(w)
    return seen[:max_terms]


async def fetch_pubmed_abstracts(query: str, retmax: int = DEFAULT_RETMAX) -> list[Abstract]:
    """Fetch PubMed abstracts for a question via NCBI E-utilities (no API key).

    Tries the question's content words first, then fewer words if too few papers match,
    then the raw question as a last resort.
    """
    terms = pubmed_terms(query)
    attempts: list[str] = []
    for n in dict.fromkeys([len(terms), min(len(terms), 4), min(len(terms), 2)]):
        if n:
            attempts.append(" AND ".join(terms[:n]) + " AND hasabstract")
    attempts.append(query)

    async with httpx.AsyncClient(timeout=60.0) as client:
        pmids: list[str] = []
        for term in attempts:
            found = await _esearch(client, term, retmax)
            if len(found) > len(pmids):
                pmids = found
            if len(found) >= _MIN_RESULTS:
                break
        if not pmids:
            return []
        return await _efetch_abstracts(client, pmids)


async def _esearch(client: httpx.AsyncClient, query: str, retmax: int) -> list[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
    }
    resp = await client.get(f"{EUTILS_BASE}/esearch.fcgi", params=params)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


async def _efetch_abstracts(client: httpx.AsyncClient, pmids: list[str]) -> list[Abstract]:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    resp = await client.get(f"{EUTILS_BASE}/efetch.fcgi", params=params)
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

        if not abstract:
            abstract = "(No abstract available.)"

        if pmid and title:
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
