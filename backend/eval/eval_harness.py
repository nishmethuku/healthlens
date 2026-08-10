#!/usr/bin/env python3
"""
Evaluate HealthLens retrieval (BM25, dense, hybrid) against benchmark.json.

Usage (from backend/, using project venv):
    ./venv/bin/python eval/eval_harness.py
    ./venv/bin/python eval/eval_harness.py --benchmark eval/benchmark.json --corpus data/pubmed_abstracts.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Use locally cached Hugging Face weights; never hit the Hub during eval.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BACKEND_DIR = Path(__file__).resolve().parent.parent
VENV_DIR = BACKEND_DIR / "venv"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _ensure_backend_venv() -> None:
    """Require the backend venv so eval uses the same deps as the API."""
    expected = (VENV_DIR / "bin" / "python").resolve()
    if not expected.exists():
        return
    current = Path(sys.executable).resolve()
    if current != expected:
        print(
            f"Warning: run with backend venv for consistent deps:\n"
            f"  {expected} {Path(__file__).name}",
            file=sys.stderr,
        )

from ingest import Abstract
from retrieval import INDEX_CACHE_DIR, CorpusIndex

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = EVAL_DIR / "benchmark.json"
DEFAULT_CORPUS = BACKEND_DIR / "data" / "pubmed_abstracts.jsonl"
DEFAULT_RESULTS = EVAL_DIR / "results.json"

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")
TOP_K = 5


def load_corpus(path: Path) -> list[Abstract]:
    abstracts: list[Abstract] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            abstracts.append(
                {
                    "title": rec["title"],
                    "abstract": rec["abstract"],
                    "pmid": str(rec["pmid"]),
                }
            )
    return abstracts


def load_benchmark(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def recall_at_k(relevant: set[str], retrieved_pmids: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(relevant & set(retrieved_pmids[:k]))
    return hits / len(relevant)


def precision_at_k(relevant: set[str], retrieved_pmids: list[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = len(relevant & set(retrieved_pmids[:k]))
    return hits / k


def reciprocal_rank(relevant: set[str], retrieved_pmids: list[str]) -> float:
    for rank, pmid in enumerate(retrieved_pmids, start=1):
        if pmid in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_mode(
    benchmark: list[dict],
    corpus_index: CorpusIndex,
    mode: str,
) -> dict[str, float]:
    recalls: list[float] = []
    precisions: list[float] = []
    rrs: list[float] = []

    for item in benchmark:
        question = item["question"]
        relevant = {str(p) for p in item["ground_truth_pmids"]}
        if mode == "hybrid_rerank":
            results = corpus_index.retrieve(question, top_k=TOP_K, mode="hybrid", rerank=True)
        else:
            results = corpus_index.retrieve(question, top_k=TOP_K, mode=mode)
        retrieved = [r["pmid"] for r in results]

        recalls.append(recall_at_k(relevant, retrieved, TOP_K))
        precisions.append(precision_at_k(relevant, retrieved, TOP_K))
        rrs.append(reciprocal_rank(relevant, retrieved))

    n = len(benchmark)
    return {
        "recall_at_5": sum(recalls) / n,
        "precision_at_5": sum(precisions) / n,
        "mrr": sum(rrs) / n,
    }


def print_results_table(results: dict[str, dict[str, float]]) -> None:
    metrics = ["recall_at_5", "mrr", "precision_at_5"]
    col_w = 14
    header = f"{'Mode':<10}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print(header)
    print("-" * len(header))
    for mode in MODES:
        row = results[mode]
        print(
            f"{mode:<10}"
            + "".join(f"{row[m]:>{col_w}.4f}" for m in metrics)
        )


def main() -> None:
    _ensure_backend_venv()

    parser = argparse.ArgumentParser(description="HealthLens retrieval evaluation harness")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    if not args.benchmark.exists():
        sys.exit(f"Benchmark not found: {args.benchmark}")
    if not args.corpus.exists():
        sys.exit(f"Corpus not found: {args.corpus}")

    benchmark = load_benchmark(args.benchmark)
    print(f"Loaded {len(benchmark)} benchmark questions")
    print(f"Loading corpus from {args.corpus} ...")
    abstracts = load_corpus(args.corpus)
    print(f"Indexed {len(abstracts)} abstracts\n")

    print("Building retrieval index (BM25 + dense embeddings) ...")
    corpus_index = CorpusIndex(abstracts, cache_dir=INDEX_CACHE_DIR)

    results: dict[str, dict[str, float]] = {}
    for mode in MODES:
        print(f"Evaluating {mode} ...")
        results[mode] = evaluate_mode(benchmark, corpus_index, mode)

    print("\nRetrieval evaluation (top-5)\n")
    print_results_table(results)

    payload = {
        "top_k": TOP_K,
        "num_questions": len(benchmark),
        "corpus_size": len(abstracts),
        "modes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
