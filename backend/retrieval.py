import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import TypedDict

import faiss
import numpy as np
from scipy import sparse
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import get_settings
from ingest import Abstract

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


class _FastBM25:
    """Drop-in replacement for rank_bm25.BM25Okapi, scored via a sparse
    term-document matrix instead of one Python dict per document.

    rank_bm25's get_scores() does `doc.get(term)` against every document's
    frequency dict for every query term — O(corpus_size) per query term. At
    ~200k docs that made a single BM25 query take on the order of tens of
    seconds under an earlier sentence-chunked indexing experiment (2.2M
    chunks), and eval_harness.py calls get_scores() 150 times (50 questions
    x 3 BM25-touching modes), dominating the eval runtime. A term with zero
    frequency in a document contributes exactly 0 to that document's score,
    so restricting the computation to a term's nonzero postings (its sparse
    matrix column) reproduces identical scores while only touching documents
    that actually contain the term — verified to match rank_bm25's output
    bit-for-bit (within float64 rounding) on randomized corpora.
    """

    def __init__(
        self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(tokenized_corpus)

        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        data: list[int] = []
        doc_len = np.empty(self.corpus_size, dtype=np.float64)
        doc_freq: dict[str, int] = {}

        for doc_idx, tokens in enumerate(tokenized_corpus):
            doc_len[doc_idx] = len(tokens)
            freqs: dict[str, int] = {}
            for tok in tokens:
                freqs[tok] = freqs.get(tok, 0) + 1
            for tok, freq in freqs.items():
                col = vocab.setdefault(tok, len(vocab))
                rows.append(doc_idx)
                cols.append(col)
                data.append(freq)
                doc_freq[tok] = doc_freq.get(tok, 0) + 1

        self.avgdl = float(doc_len.mean()) if self.corpus_size else 0.0
        self.doc_len = doc_len
        self.vocab = vocab

        # Matches rank_bm25.BM25Okapi._calc_idf: negative-idf terms (those
        # appearing in over half the corpus) get floored to epsilon * the
        # mean idf computed *before* flooring, not after.
        idf = np.empty(len(vocab), dtype=np.float64)
        for tok, col in vocab.items():
            freq = doc_freq[tok]
            idf[col] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
        negative = idf < 0
        if negative.any():
            idf[negative] = epsilon * idf.mean()
        self.idf = idf

        matrix = sparse.coo_matrix(
            (data, (rows, cols)), shape=(self.corpus_size, len(vocab)), dtype=np.float64
        )
        self.term_doc_freq = matrix.tocsc()

    def get_scores(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float64)
        indptr = self.term_doc_freq.indptr
        indices = self.term_doc_freq.indices
        data = self.term_doc_freq.data
        for term, qf in Counter(query).items():
            col = self.vocab.get(term)
            if col is None:
                continue
            idf_val = self.idf[col]
            start, end = indptr[col], indptr[col + 1]
            doc_idx = indices[start:end]
            freqs = data[start:end]
            denom = freqs + self.k1 * (1 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl)
            scores[doc_idx] += qf * idf_val * (freqs * (self.k1 + 1)) / denom
        return scores


RetrievalMode = str  # "hybrid" | "bm25" | "dense"


def _embedding_cache_paths(abstracts: list[Abstract], cache_dir: Path) -> tuple[Path, Path]:
    """Cache key = corpus size + first/last PMID + embedding model, so a changed
    corpus or model swap invalidates automatically instead of silently serving
    stale vectors."""
    import hashlib

    fingerprint = f"{len(abstracts)}:{abstracts[0]['pmid']}:{abstracts[-1]['pmid']}:{EMBEDDING_MODEL_ID}"
    digest = hashlib.sha1(fingerprint.encode()).hexdigest()[:16]
    return cache_dir / f"embs_{digest}.npy", cache_dir / f"faiss_{digest}.index"


class CorpusIndex:
    """Pre-built BM25 + dense indices for repeated queries over a fixed corpus.

    Dense embeddings are the expensive part to build (one encode() pass over the
    whole corpus) and cheap to persist, so they're cached to disk keyed on corpus
    fingerprint + embedding model. BM25 rebuilds from tokenized text every time
    since building the sparse matrix is fast even at ~200k docs and there's no
    built-in serialization for it.
    """

    def __init__(self, abstracts: list[Abstract], cache_dir: Path | None = None) -> None:
        self.abstracts = abstracts
        self.corpus = [f"{a['title']} {a['abstract']}" for a in abstracts]
        self.tokenized = [_tokenize(doc) for doc in self.corpus]
        self.bm25 = _FastBM25(self.tokenized) if abstracts else None
        self.doc_embs: np.ndarray | None = None
        self.faiss_index: faiss.IndexFlatIP | None = None
        if abstracts:
            self.doc_embs, self.faiss_index = self._load_or_build_dense(abstracts, cache_dir)

    def _load_or_build_dense(
        self, abstracts: list[Abstract], cache_dir: Path | None
    ) -> tuple[np.ndarray, faiss.IndexFlatIP]:
        emb_path = index_path = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            emb_path, index_path = _embedding_cache_paths(abstracts, cache_dir)
            if emb_path.exists() and index_path.exists():
                try:
                    doc_embs = np.load(emb_path)
                    faiss_index = faiss.read_index(str(index_path))
                    logger.info(
                        "loaded cached dense index", extra={"num_docs": len(abstracts)}
                    )
                    return doc_embs, faiss_index
                except Exception:
                    logger.warning("cached index unreadable, rebuilding", exc_info=True)

        model = _get_model()
        doc_embs = model.encode(self.corpus, normalize_embeddings=True).astype(np.float32)
        dim = doc_embs.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(doc_embs)

        if emb_path is not None and index_path is not None:
            try:
                np.save(emb_path, doc_embs)
                faiss.write_index(faiss_index, str(index_path))
            except (OSError, RuntimeError):
                # faiss raises RuntimeError (not OSError) for I/O failures like
                # a full disk — confirmed in production: "No space left on
                # device" surfaced as RuntimeError and wasn't caught here,
                # which let it propagate out of CorpusIndex entirely instead
                # of degrading to in-memory-only as intended. A partially
                # written index file from the failed attempt may be left on
                # disk; delete it so a later run doesn't try to load garbage.
                logger.warning("failed to persist dense index cache", exc_info=True)
                index_path.unlink(missing_ok=True)

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
    """Return a cached CorpusIndex built from pubmed_abstracts.jsonl, or None if missing.

    _corpus_checked is only set to True after construction succeeds. Setting
    it beforehand (the original bug) meant a mid-construction exception left
    _corpus_index permanently None for the rest of the process — every future
    call short-circuited on the "already checked" branch and silently fell
    back to live-fetch-per-query forever, instead of retrying on next call.
    """
    global _corpus_index, _corpus_checked
    if _corpus_checked:
        return _corpus_index

    if CORPUS_PATH.exists():
        abstracts = load_corpus_from_file(CORPUS_PATH)
        if abstracts:
            _corpus_index = CorpusIndex(abstracts, cache_dir=INDEX_CACHE_DIR)
    _corpus_checked = True
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
