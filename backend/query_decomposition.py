import json
import os
import re
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from groq import Groq

from retrieval import RankedAbstract

if TYPE_CHECKING:
    from retrieval import CorpusIndex

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
SUB_QUERY_TOP_K = 10
FINAL_TOP_K = 5


def _get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError("GROQ_API_KEY is not set. Add your key to backend/.env")
    return Groq(api_key=api_key)


def decompose_query(question: str) -> list[str]:
    """Break a complex medical question into 2-3 focused sub-queries via Groq."""
    client = _get_groq_client()

    system = (
        "You decompose complex medical questions into 2-3 focused sub-queries "
        "suitable for searching PubMed literature. Each sub-query should target "
        "a distinct aspect of the original question. Return ONLY valid JSON."
    )
    user_message = (
        f"Question: {question}\n\n"
        'Return JSON: {"sub_queries": ["sub-query 1", "sub-query 2", ...]}\n'
        "Use exactly 2-3 sub-queries."
    )

    completion = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
    )

    raw = completion.choices[0].message.content or ""
    sub_queries = _parse_sub_queries(raw, question)
    return sub_queries[:3]


def _parse_sub_queries(raw: str, fallback: str) -> list[str]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "sub_queries" in data:
            queries = [str(q).strip() for q in data["sub_queries"] if str(q).strip()]
            if len(queries) >= 2:
                return queries
    except json.JSONDecodeError:
        pass

    return [fallback]


def _merge_and_rerank(
    result_lists: list[list[RankedAbstract]], top_k: int = FINAL_TOP_K
) -> list[RankedAbstract]:
    """Merge sub-query results, deduplicate by PMID, re-rank by max combined score."""
    by_pmid: dict[str, RankedAbstract] = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            pmid = item["pmid"]
            # Blend retrieval score with rank decay so multiple hits boost relevance
            rank_bonus = 1.0 / (rank + 1)
            combined = item["score"] + 0.1 * rank_bonus

            if pmid not in by_pmid or combined > by_pmid[pmid]["score"]:
                by_pmid[pmid] = {**item, "score": combined}
            else:
                by_pmid[pmid]["score"] = max(by_pmid[pmid]["score"], combined)

    ranked = sorted(by_pmid.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def decomposed_retrieve(
    question: str,
    corpus_index: "CorpusIndex",
    top_k: int = FINAL_TOP_K,
) -> list[RankedAbstract]:
    """Decompose question, retrieve per sub-query, merge and return top abstracts."""
    sub_queries = decompose_query(question)
    result_lists = [
        corpus_index.retrieve(sq, top_k=SUB_QUERY_TOP_K, mode="hybrid")
        for sq in sub_queries
    ]
    return _merge_and_rerank(result_lists, top_k=top_k)
