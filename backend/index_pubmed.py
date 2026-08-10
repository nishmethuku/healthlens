#!/usr/bin/env python3
"""
One-time bulk PubMed indexer.

Fetches abstracts across major medical topics via NCBI E-utilities and writes
them to data/pubmed_abstracts.jsonl. Resumable: PMIDs already in the output
file are skipped.

Usage:
    python index_pubmed.py
    python index_pubmed.py --target 200000 --max-rps 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_PATH = DATA_DIR / "pubmed_abstracts.jsonl"

TARGET_TOTAL = 200_000
ESEARCH_PAGE_SIZE = 10_000
MAX_ESEARCH_IDS = 10_000  # NCBI hard cap on esearch idlist per query
EFETCH_BATCH_SIZE = 200
MAX_RPS_CAP = 10
DEFAULT_RPS_NO_KEY = 3
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0

# Broad MeSH queries with abstract filter; each topic has 100K+ PubMed hits.
TOPIC_QUERIES: dict[str, str] = {
    "cardiology": '"Cardiovascular Diseases"[MeSH] AND hasabstract[text]',
    "oncology": '"Neoplasms"[MeSH] AND hasabstract[text]',
    "neurology": '"Nervous System Diseases"[MeSH] AND hasabstract[text]',
    "diabetes": '"Diabetes Mellitus"[MeSH] AND hasabstract[text]',
    "hypertension": '"Hypertension"[MeSH] AND hasabstract[text]',
    "mental_health": '"Mental Disorders"[MeSH] AND hasabstract[text]',
    "infectious_disease": '"Communicable Diseases"[MeSH] AND hasabstract[text]',
    "pediatrics": '"Pediatrics"[MeSH] AND hasabstract[text]',
    "pharmacology": '"Pharmacology"[MeSH] AND hasabstract[text]',
    "surgery": '"General Surgery"[MeSH] AND hasabstract[text]',
}


def topic_subqueries(base_query: str, year_end: int = 2026, year_start: int = 1995) -> list[str]:
    """Split a topic into sub-queries so each stays within NCBI's 10K esearch cap."""
    queries = [base_query]
    for year in range(year_end, year_start - 1, -1):
        queries.append(f"({base_query}) AND {year}[PDAT]")
    return queries


class RateLimiter:
    """Enforce a maximum requests-per-second ceiling."""

    def __init__(self, max_rps: float) -> None:
        self.min_interval = 1.0 / max_rps
        self._last_request = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.monotonic()


def load_existing_pmids(path: Path) -> set[str]:
    """Load PMIDs already written to the JSONL output."""
    if not path.exists():
        return set()

    pmids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                pmid = str(record.get("pmid", "")).strip()
                if pmid:
                    pmids.add(pmid)
            except json.JSONDecodeError:
                continue
    return pmids


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


def parse_pubmed_xml(xml_text: str) -> list[dict[str, str]]:
    """Extract pmid, title, abstract from PubMed XML."""
    root = ET.fromstring(xml_text)
    records: list[dict[str, str]] = []

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_els = article.findall(".//AbstractText")

        pmid = (pmid_el.text or "").strip() if pmid_el is not None else ""
        title = _element_text(title_el)
        abstract = " ".join(_element_text(el) for el in abstract_els).strip()

        if pmid and title and abstract:
            records.append({"pmid": pmid, "title": title, "abstract": abstract})

    return records


def _base_params(tool: str, email: str | None, api_key: str | None) -> dict[str, str]:
    params: dict[str, str] = {"tool": tool}
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    return params


def parse_ncbi_json(response: httpx.Response) -> dict:
    """Parse NCBI JSON; tolerate occasional control characters in responses."""
    try:
        return response.json()
    except json.JSONDecodeError:
        return json.loads(response.text, strict=False)


def request_with_retry(
    limiter: RateLimiter,
    request_fn: Callable[[], httpx.Response],
) -> httpx.Response:
    """Execute an HTTP request with rate limiting and exponential backoff."""
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            limiter.wait()
            response = request_fn()
            if response.status_code == 429:
                delay = RETRY_BASE_DELAY * (2**attempt)
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response
        except (httpx.HTTPError, httpx.TimeoutException, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == MAX_RETRIES - 1:
                break
            time.sleep(RETRY_BASE_DELAY * (2**attempt))

    raise RuntimeError(f"Request failed after {MAX_RETRIES} retries") from last_error


def esearch_begin(
    client: httpx.Client,
    query: str,
    limiter: RateLimiter,
    base_params: dict[str, str],
) -> tuple[str, str, int]:
    """Start an esearch with usehistory=y; return (webenv, query_key, total_count)."""
    params = {
        **base_params,
        "db": "pubmed",
        "term": query,
        "retmax": 0,
        "usehistory": "y",
        "retmode": "json",
    }

    def do_request() -> httpx.Response:
        return client.get(f"{EUTILS_BASE}/esearch.fcgi", params=params)

    response = request_with_retry(limiter, do_request)
    result = parse_ncbi_json(response).get("esearchresult", {})
    count = int(result.get("count", "0"))
    webenv = result.get("webenv", "")
    query_key = result.get("querykey", "")
    if not webenv or not query_key:
        raise RuntimeError("esearch did not return WebEnv/query_key (usehistory required)")
    return webenv, query_key, count


def esearch_page(
    client: httpx.Client,
    webenv: str,
    query_key: str,
    retstart: int,
    retmax: int,
    limiter: RateLimiter,
    base_params: dict[str, str],
) -> list[str]:
    """Fetch a page of PMIDs from a stored esearch result (supports >10K via usehistory)."""
    params = {
        **base_params,
        "db": "pubmed",
        "query_key": query_key,
        "WebEnv": webenv,
        "retstart": retstart,
        "retmax": retmax,
        "retmode": "json",
    }

    def do_request() -> httpx.Response:
        return client.get(f"{EUTILS_BASE}/esearch.fcgi", params=params)

    response = request_with_retry(limiter, do_request)
    data = parse_ncbi_json(response)
    return data.get("esearchresult", {}).get("idlist", [])


def efetch_batch(
    client: httpx.Client,
    pmids: list[str],
    limiter: RateLimiter,
    base_params: dict[str, str],
) -> list[dict[str, str]]:
    params = {
        **base_params,
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }

    def do_request() -> httpx.Response:
        return client.get(f"{EUTILS_BASE}/efetch.fcgi", params=params)

    response = request_with_retry(limiter, do_request)
    return parse_pubmed_xml(response.text)


def chunked(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def collect_pmids(
    client: httpx.Client,
    limiter: RateLimiter,
    base_params: dict[str, str],
    existing_pmids: set[str],
    target_new: int,
) -> list[tuple[str, str]]:
    """
    Collect up to target_new unique PMIDs not already in existing_pmids.
    Returns list of (pmid, topic) tuples.
    """
    per_topic_target = (target_new // len(TOPIC_QUERIES)) + 5_000
    global_seen = set(existing_pmids)
    queue: list[tuple[str, str]] = []

    topic_bar = tqdm(TOPIC_QUERIES.items(), desc="Topics (esearch)", unit="topic")

    for topic, query in topic_bar:
        if len(queue) >= target_new:
            break

        topic_bar.set_postfix(topic=topic, queued=len(queue))

        topic_new = 0

        for subquery in topic_subqueries(query):
            if topic_new >= per_topic_target or len(queue) >= target_new:
                break

            try:
                webenv, query_key, total_available = esearch_begin(
                    client, subquery, limiter, base_params
                )
            except RuntimeError as exc:
                tqdm.write(f"Warning: esearch init failed for {topic}: {exc}")
                continue

            if total_available == 0:
                continue

            retstart = 0
            fetch_limit = min(total_available, MAX_ESEARCH_IDS)

            while retstart < fetch_limit and topic_new < per_topic_target and len(queue) < target_new:
                page_size = min(ESEARCH_PAGE_SIZE, fetch_limit - retstart)
                if page_size <= 0:
                    break

                try:
                    idlist = esearch_page(
                        client, webenv, query_key, retstart, page_size, limiter, base_params
                    )
                except RuntimeError as exc:
                    tqdm.write(
                        f"Warning: esearch failed for {topic} @ {retstart}: {exc}"
                    )
                    break

                if not idlist:
                    break

                for pmid in idlist:
                    if pmid in global_seen:
                        continue
                    global_seen.add(pmid)
                    queue.append((pmid, topic))
                    topic_new += 1
                    if topic_new >= per_topic_target or len(queue) >= target_new:
                        break

                retstart += len(idlist)
                topic_bar.set_postfix(topic=topic, queued=len(queue), scanned=retstart)

    return queue


def fetch_and_write(
    client: httpx.Client,
    limiter: RateLimiter,
    base_params: dict[str, str],
    queue: list[tuple[str, str]],
    output_path: Path,
) -> tuple[int, int]:
    """
    Fetch abstracts for queued PMIDs and append to JSONL.
    Returns (written_count, skipped_no_abstract_count).
    """
    pmid_to_topic = dict(queue)
    pmids = [pmid for pmid, _ in queue]

    written = 0
    skipped = 0

    fetch_bar = tqdm(total=len(pmids), desc="Fetching abstracts", unit="pmid")

    with output_path.open("a", encoding="utf-8") as out_f:
        for batch in chunked(pmids, EFETCH_BATCH_SIZE):
            try:
                records = efetch_batch(client, batch, limiter, base_params)
            except RuntimeError as exc:
                tqdm.write(f"Warning: efetch failed for batch starting {batch[0]}: {exc}")
                fetch_bar.update(len(batch))
                skipped += len(batch)
                continue

            records_by_pmid = {r["pmid"]: r for r in records}

            for pmid in batch:
                record = records_by_pmid.get(pmid)
                if not record or not record.get("abstract", "").strip():
                    skipped += 1
                    fetch_bar.update(1)
                    continue

                line = {
                    "pmid": record["pmid"],
                    "title": record["title"],
                    "abstract": record["abstract"],
                    "topic": pmid_to_topic[pmid],
                }
                out_f.write(json.dumps(line, ensure_ascii=False) + "\n")
                written += 1
                fetch_bar.update(1)

            out_f.flush()

    fetch_bar.close()
    return written, skipped


def resolve_max_rps(explicit: float | None, api_key: str | None) -> float:
    if explicit is not None:
        return min(explicit, MAX_RPS_CAP)
    if api_key:
        return MAX_RPS_CAP
    return DEFAULT_RPS_NO_KEY


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk index PubMed abstracts")
    parser.add_argument(
        "--target",
        type=int,
        default=TARGET_TOTAL,
        help=f"Target total abstracts in output file (default: {TARGET_TOTAL})",
    )
    parser.add_argument(
        "--max-rps",
        type=float,
        default=None,
        help=f"Max NCBI requests per second, capped at {MAX_RPS_CAP} (default: 3 without API key, 10 with)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output JSONL path",
    )
    args = parser.parse_args()

    api_key = os.getenv("NCBI_API_KEY")
    email = os.getenv("NCBI_EMAIL")
    tool = os.getenv("NCBI_TOOL", "healthlens_indexer")
    max_rps = resolve_max_rps(args.max_rps, api_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    existing_pmids = load_existing_pmids(args.output)
    already_collected = len(existing_pmids)
    target_new = max(0, args.target - already_collected)

    print(f"Output file:       {args.output}")
    print(f"Already collected: {already_collected:,} abstracts")
    print(f"Target total:      {args.target:,} abstracts")
    print(f"New to fetch:      {target_new:,} abstracts")
    print(f"Rate limit:        {max_rps} req/s")
    print(f"Topics:            {', '.join(TOPIC_QUERIES.keys())}")
    print()

    if target_new == 0:
        print(f"Target already met. Total abstracts collected: {already_collected:,}")
        return

    limiter = RateLimiter(max_rps)
    base_params = _base_params(tool, email, api_key)

    with httpx.Client(timeout=120.0) as client:
        print("Phase 1: Collecting PMIDs via esearch …")
        queue = collect_pmids(
            client, limiter, base_params, existing_pmids, target_new
        )

        if not queue:
            print("No new PMIDs found.")
            sys.exit(1)

        print(f"\nPhase 2: Fetching {len(queue):,} abstracts via efetch …")
        written, skipped = fetch_and_write(
            client, limiter, base_params, queue, args.output
        )

    final_total = already_collected + written
    print()
    print(f"New abstracts written this run: {written:,}")
    print(f"Skipped (no abstract text):       {skipped:,}")
    print(f"Total abstracts collected:        {final_total:,}")


if __name__ == "__main__":
    main()
