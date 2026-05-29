import json
import os
import re
from pathlib import Path
from typing import TypedDict

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ingest import Abstract

CORPUS_PATH = Path(__file__).parent / "data" / "pubmed_abstracts.jsonl"

_corpus_index: "CorpusIndex | None" = None
_corpus_checked: bool = False

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_SHORT = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


class RankedAbstract(TypedDict):
    title: str
    abstract: str
    pmid: str
    score: float


def _hf_hub_cache_dir() -> Path:
    if cache := os.environ.get("HF_HUB_CACHE"):
        return Path(cache)
    if hf_home := os.environ.get("HF_HOME"):
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _local_embedding_model_path() -> Path | None:
    """Return the newest local snapshot for the embedding model, if present."""
    repo_dir = (
        _hf_hub_cache_dir()
        / "models--sentence-transformers--all-MiniLM-L6-v2"
        / "snapshots"
    )
    if not repo_dir.is_dir():
        return None
    snapshots = sorted(p for p in repo_dir.iterdir() if p.is_dir())
    return snapshots[-1] if snapshots else None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        local_path = _local_embedding_model_path()
        if local_path is not None:
            _model = SentenceTransformer(str(local_path), local_files_only=True)
        else:
            _model = SentenceTransformer(
                EMBEDDING_MODEL_SHORT, local_files_only=True
            )
    return _model


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


RetrievalMode = str  # "hybrid" | "bm25" | "dense"


class CorpusIndex:
    """Pre-built BM25 + dense indices for repeated queries over a fixed corpus."""

    def __init__(self, abstracts: list[Abstract]) -> None:
        self.abstracts = abstracts
        self.corpus = [f"{a['title']} {a['abstract']}" for a in abstracts]
        self.tokenized = [_tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized) if abstracts else None
        self.doc_embs: np.ndarray | None = None
        self.faiss_index: faiss.IndexFlatIP | None = None
        if abstracts:
            model = _get_model()
            self.doc_embs = model.encode(
                self.corpus, normalize_embeddings=True
            ).astype(np.float32)
            dim = self.doc_embs.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.doc_embs)

    def retrieve(
        self, query: str, top_k: int = 5, mode: RetrievalMode = "hybrid"
    ) -> list[RankedAbstract]:
        return _retrieve_from_index(self, query, top_k, mode)


def _rank_abstracts(
    abstracts: list[Abstract], scores: np.ndarray, top_k: int
) -> list[RankedAbstract]:
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "title": abstracts[i]["title"],
            "abstract": abstracts[i]["abstract"],
            "pmid": abstracts[i]["pmid"],
            "score": float(scores[i]),
        }
        for i in ranked_indices
    ]


def _retrieve_from_index(
    corpus_index: CorpusIndex,
    query: str,
    top_k: int,
    mode: RetrievalMode,
) -> list[RankedAbstract]:
    abstracts = corpus_index.abstracts
    if not abstracts:
        return []

    if len(abstracts) <= top_k:
        return [{**a, "score": 1.0} for a in abstracts[:top_k]]  # type: ignore[misc]

    if mode == "bm25":
        assert corpus_index.bm25 is not None
        scores = np.array(
            corpus_index.bm25.get_scores(_tokenize(query)), dtype=np.float32
        )
        return _rank_abstracts(abstracts, scores, top_k)

    model = _get_model()
    query_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    assert corpus_index.faiss_index is not None
    dense_scores, _ = corpus_index.faiss_index.search(query_emb, len(abstracts))
    dense_scores = dense_scores[0].astype(np.float32)

    if mode == "dense":
        return _rank_abstracts(abstracts, dense_scores, top_k)

    assert corpus_index.bm25 is not None
    bm25_scores = np.array(
        corpus_index.bm25.get_scores(_tokenize(query)), dtype=np.float32
    )
    combined = 0.5 * _normalize(bm25_scores) + 0.5 * _normalize(dense_scores)
    return _rank_abstracts(abstracts, combined, top_k)


def retrieve(
    query: str,
    abstracts: list[Abstract],
    top_k: int = 5,
    mode: RetrievalMode = "hybrid",
) -> list[RankedAbstract]:
    """Retrieve top_k abstracts using BM25-only, dense-only, or hybrid fusion."""
    return CorpusIndex(abstracts).retrieve(query, top_k=top_k, mode=mode)


def hybrid_retrieve(query: str, abstracts: list[Abstract], top_k: int = 5) -> list[RankedAbstract]:
    """Hybrid BM25 + dense (FAISS) retrieval with score fusion; return top_k abstracts."""
    return retrieve(query, abstracts, top_k=top_k, mode="hybrid")


def load_corpus_from_file(path: Path = CORPUS_PATH) -> list[Abstract]:
    """Load indexed abstracts from JSONL."""
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


def get_corpus_index() -> CorpusIndex | None:
    """Return a cached CorpusIndex built from pubmed_abstracts.jsonl, or None if missing."""
    global _corpus_index, _corpus_checked
    if _corpus_checked:
        return _corpus_index

    _corpus_checked = True
    if CORPUS_PATH.exists():
        abstracts = load_corpus_from_file(CORPUS_PATH)
        if abstracts:
            _corpus_index = CorpusIndex(abstracts)
    return _corpus_index


def hybrid_retrieve_indexed(query: str, top_k: int = 5) -> list[RankedAbstract]:
    """Hybrid retrieve against the indexed local corpus."""
    index = get_corpus_index()
    if index is None:
        raise FileNotFoundError(
            f"Indexed corpus not found at {CORPUS_PATH}. "
            "Run index_pubmed.py or rely on live PubMed fetch."
        )
    return index.retrieve(query, top_k=top_k, mode="hybrid")


def _normalize(scores: np.ndarray) -> np.ndarray:
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-9:
        return np.ones_like(scores)
    return (scores - min_s) / (max_s - min_s)
