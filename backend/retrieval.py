import json
import logging
import os
import re
from pathlib import Path
from typing import TypedDict

import faiss
import nltk
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import get_settings
from ingest import Abstract

_nltk_ready = False


def _ensure_nltk_data() -> None:
    """Download the sentence tokenizer's data file on first use, not at
    import time — most callers (tests, tooling) never chunk anything."""
    global _nltk_ready
    if _nltk_ready:
        return
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        import ssl

        try:
            import certifi

            ssl._create_default_https_context = lambda: ssl.create_default_context(
                cafile=certifi.where()
            )
        except ImportError:
            pass
        nltk.download("punkt_tab", quiet=True)
    _nltk_ready = True


def chunk_abstract(title: str, abstract: str) -> list[str]:
    """Split an abstract into sentence-level chunks, each prefixed with the
    title for context — a lone sentence like "Results were significant" is
    meaningless without it. Falls back to the title alone if there's no
    abstract body (e.g. the ingest-side "(No abstract available.)" case)."""
    _ensure_nltk_data()
    sentences = [s.strip() for s in nltk.sent_tokenize(abstract) if s.strip()] if abstract else []
    if not sentences:
        return [title] if title else []
    return [f"{title} {s}" for s in sentences]

logger = logging.getLogger(__name__)

CORPUS_PATH = Path(__file__).parent / "data" / "pubmed_abstracts.jsonl"
INDEX_CACHE_DIR = Path(__file__).parent / "data" / "index_cache"

_corpus_index: "CorpusIndex | None" = None
_corpus_checked: bool = False

# Swapping to a domain-adapted model (e.g. a PubMedBERT-based sentence encoder)
# is a config change via EMBEDDING_MODEL_ID in .env, not a code change — the
# corpus index cache key includes this value so it invalidates automatically.
EMBEDDING_MODEL_ID = get_settings().embedding_model_id

# Candidates pulled from hybrid fusion before the cross-encoder reranks them down
# to top_k. Wider than top_k because the dense leg is prone to "hubness" — a
# handful of generic-sounding abstracts scoring deceptively high cosine
# similarity against almost any query (see rerank() docstring).
RERANK_CANDIDATE_POOL = 25

_model: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None


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


def _local_snapshot_path(repo_id: str) -> Path | None:
    """Return the newest local HF Hub snapshot for a repo id, if present."""
    cache_dirname = "models--" + repo_id.replace("/", "--")
    repo_dir = _hf_hub_cache_dir() / cache_dirname / "snapshots"
    if not repo_dir.is_dir():
        return None
    snapshots = sorted(p for p in repo_dir.iterdir() if p.is_dir())
    return snapshots[-1] if snapshots else None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        local_path = _local_snapshot_path(EMBEDDING_MODEL_ID)
        if local_path is not None:
            _model = SentenceTransformer(str(local_path), local_files_only=True)
        else:
            _model = SentenceTransformer(EMBEDDING_MODEL_ID, local_files_only=True)
    return _model


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


RetrievalMode = str  # "hybrid" | "bm25" | "dense"


def _embedding_cache_paths(
    abstracts: list[Abstract], num_chunks: int, cache_dir: Path
) -> tuple[Path, Path]:
    """Cache key = corpus size + chunk count + first/last PMID + embedding model,
    so a changed corpus, a chunking-logic change, or a model swap all invalidate
    automatically instead of silently serving stale/mismatched vectors."""
    import hashlib

    fingerprint = (
        f"chunked:{len(abstracts)}:{num_chunks}:"
        f"{abstracts[0]['pmid']}:{abstracts[-1]['pmid']}:{EMBEDDING_MODEL_ID}"
    )
    digest = hashlib.sha1(fingerprint.encode()).hexdigest()[:16]
    return cache_dir / f"embs_{digest}.npy", cache_dir / f"faiss_{digest}.index"


class CorpusIndex:
    """Pre-built BM25 + dense indices over sentence-level chunks of the corpus.

    A whole-abstract embedding blurs every sentence in it into one vector, so
    a single relevant claim buried in an otherwise off-topic abstract can't
    surface on its own. Indexing per-sentence chunks instead lets that claim
    be found directly, at the cost of ~N_sentences more vectors to build and
    search. Chunk granularity is purely an internal indexing detail —
    retrieve() always dedupes back to one (highest-scoring) result per PMID,
    so callers still get the same PMID-level RankedAbstract shape as before.

    Dense embeddings are the expensive part to build (one encode() pass over
    every chunk) and cheap to persist, so they're cached to disk keyed on a
    corpus+chunking+model fingerprint. BM25 rebuilds from tokenized chunk text
    every time since it's fast and rank_bm25 has no built-in serialization.
    """

    def __init__(self, abstracts: list[Abstract], cache_dir: Path | None = None) -> None:
        self.abstracts = abstracts
        self.abstract_by_pmid = {a["pmid"]: a for a in abstracts}

        self.chunk_texts: list[str] = []
        self.chunk_pmids: list[str] = []
        for a in abstracts:
            for chunk in chunk_abstract(a["title"], a["abstract"]):
                self.chunk_texts.append(chunk)
                self.chunk_pmids.append(a["pmid"])

        self.tokenized = [_tokenize(c) for c in self.chunk_texts]
        self.bm25 = BM25Okapi(self.tokenized) if self.chunk_texts else None
        self.doc_embs: np.ndarray | None = None
        self.faiss_index: faiss.IndexFlatIP | None = None
        if self.chunk_texts:
            self.doc_embs, self.faiss_index = self._load_or_build_dense(cache_dir)

    def _load_or_build_dense(
        self, cache_dir: Path | None
    ) -> tuple[np.ndarray, faiss.IndexFlatIP]:
        emb_path = index_path = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            emb_path, index_path = _embedding_cache_paths(
                self.abstracts, len(self.chunk_texts), cache_dir
            )
            if emb_path.exists() and index_path.exists():
                try:
                    doc_embs = np.load(emb_path)
                    faiss_index = faiss.read_index(str(index_path))
                    logger.info(
                        "loaded cached dense index", extra={"num_chunks": len(self.chunk_texts)}
                    )
                    return doc_embs, faiss_index
                except Exception:
                    logger.warning("cached index unreadable, rebuilding", exc_info=True)

        model = _get_model()
        doc_embs = model.encode(self.chunk_texts, normalize_embeddings=True).astype(np.float32)
        dim = doc_embs.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(doc_embs)

        if emb_path is not None and index_path is not None:
            try:
                np.save(emb_path, doc_embs)
                faiss.write_index(faiss_index, str(index_path))
            except OSError:
                logger.warning("failed to persist dense index cache", exc_info=True)

        return doc_embs, faiss_index

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: RetrievalMode = "hybrid",
        rerank: bool = False,
    ) -> list[RankedAbstract]:
        if rerank and mode == "hybrid":
            candidates = _retrieve_from_index(
                self, query, max(top_k * 5, RERANK_CANDIDATE_POOL), mode
            )
            return rerank_candidates(query, candidates, top_k)
        return _retrieve_from_index(self, query, top_k, mode)


def _rank_chunks_dedup_by_pmid(
    corpus_index: CorpusIndex, scores: np.ndarray, top_k: int
) -> list[RankedAbstract]:
    """Walk chunks best-score-first, keeping only the first (highest-scoring)
    chunk seen per PMID, until top_k distinct documents are collected."""
    order = np.argsort(scores)[::-1]
    seen_pmids: set[str] = set()
    results: list[RankedAbstract] = []
    for idx in order:
        pmid = corpus_index.chunk_pmids[idx]
        if pmid in seen_pmids:
            continue
        seen_pmids.add(pmid)
        abstract = corpus_index.abstract_by_pmid[pmid]
        results.append(
            {
                "title": abstract["title"],
                "abstract": abstract["abstract"],
                "pmid": pmid,
                "score": float(scores[idx]),
            }
        )
        if len(results) >= top_k:
            break
    return results


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
        return _rank_chunks_dedup_by_pmid(corpus_index, scores, top_k)

    model = _get_model()
    query_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    assert corpus_index.faiss_index is not None
    dense_scores, _ = corpus_index.faiss_index.search(query_emb, len(corpus_index.chunk_texts))
    dense_scores = dense_scores[0].astype(np.float32)

    if mode == "dense":
        return _rank_chunks_dedup_by_pmid(corpus_index, dense_scores, top_k)

    assert corpus_index.bm25 is not None
    bm25_scores = np.array(
        corpus_index.bm25.get_scores(_tokenize(query)), dtype=np.float32
    )
    combined = 0.5 * _normalize(bm25_scores) + 0.5 * _normalize(dense_scores)
    return _rank_chunks_dedup_by_pmid(corpus_index, combined, top_k)


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        settings = get_settings()
        _reranker = CrossEncoder(settings.reranker_model_id)
    return _reranker


def rerank_candidates(
    query: str, candidates: list[RankedAbstract], top_k: int
) -> list[RankedAbstract]:
    """Cross-encoder reranking pass over a hybrid-retrieved candidate pool.

    Bi-encoder dense retrieval (a shared embedding space for queries and docs)
    is prone to "hubness": a small set of generically-worded documents end up
    with high cosine similarity to nearly every query, regardless of topic,
    because their embedding sits close to the centroid of the space. On this
    corpus that manifested as dense-only retrieval returning the *same* 5
    PMIDs for unrelated cardiology, anticoagulation, and TAVR questions.
    A cross-encoder scores (query, doc) pairs jointly instead of via a shared
    vector space, so it isn't susceptible to that failure mode — it becomes
    the final relevance judge, with hybrid fusion only responsible for
    building a reasonable candidate pool.
    """
    if not candidates:
        return []
    if len(candidates) <= top_k:
        pass  # still worth reranking order even if the pool is small

    reranker = _get_reranker()
    pairs = [(query, f"{c['title']} {c['abstract'][:1000]}") for c in candidates]
    scores = reranker.predict(pairs)
    order = np.argsort(scores)[::-1][:top_k]
    return [{**candidates[i], "score": float(scores[i])} for i in order]


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
            _corpus_index = CorpusIndex(abstracts, cache_dir=INDEX_CACHE_DIR)
    return _corpus_index


def warm_reranker() -> None:
    """Force-load the cross-encoder at startup instead of on the first request,
    so p99 latency doesn't spike for whichever user triggers the cold load."""
    _get_reranker()


def hybrid_retrieve_indexed(query: str, top_k: int = 5) -> list[RankedAbstract]:
    """Hybrid retrieve + cross-encoder rerank against the indexed local corpus."""
    index = get_corpus_index()
    if index is None:
        raise FileNotFoundError(
            f"Indexed corpus not found at {CORPUS_PATH}. "
            "Run index_pubmed.py or rely on live PubMed fetch."
        )
    return index.retrieve(query, top_k=top_k, mode="hybrid", rerank=True)


def _normalize(scores: np.ndarray) -> np.ndarray:
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-9:
        return np.ones_like(scores)
    return (scores - min_s) / (max_s - min_s)
