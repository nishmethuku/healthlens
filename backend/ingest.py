import xml.etree.ElementTree as ET
from typing import TypedDict

import httpx

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_RETMAX = 30


class Abstract(TypedDict):
    title: str
    abstract: str
    pmid: str


async def fetch_pubmed_abstracts(query: str, retmax: int = DEFAULT_RETMAX) -> list[Abstract]:
    """Fetch PubMed abstracts for a search query via NCBI E-utilities (no API key)."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        pmids = await _esearch(client, query, retmax)
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
