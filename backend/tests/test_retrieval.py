import hashlib
from unittest.mock import patch

import numpy as np
import pytest

import retrieval
from retrieval import CorpusIndex, _fuse, _reciprocal_rank_fusion

DIM = 64


class _BagOfWordsModel:
    """Deterministic stand-in for SentenceTransformer: hashes each word into a
    fixed-size vector, so texts sharing words have high cosine similarity."""

    def encode(self, texts, normalize_embeddings=True):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
                out[row, bucket] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.maximum(norms, 1e-9)
        return out


def _abstracts():
    topics = {
        "1001": ("Statins", "statin therapy lowers cholesterol and cardiovascular risk"),
        "1002": ("Insulin", "insulin dosing improves glycemic control in diabetes"),
        "1003": ("Asthma", "inhaled corticosteroids reduce asthma exacerbations"),
        "1004": ("Warfarin", "warfarin anticoagulation prevents stroke in atrial fibrillation"),
        "1005": ("Migraine", "triptans relieve acute migraine headache"),
        "1006": ("Hypertension", "ace inhibitors lower blood pressure in hypertension"),
        "1007": ("Sepsis", "early antibiotics improve sepsis survival"),
        "1008": ("Osteoporosis", "bisphosphonates reduce fracture risk in osteoporosis"),
    }
    return [{"title": t, "abstract": a, "pmid": p} for p, (t, a) in topics.items()]


@pytest.fixture
def index():
    with patch.object(retrieval, "_get_model", return_value=_BagOfWordsModel()):
        yield CorpusIndex(_abstracts())


def test_dense_retrieval_returns_the_matching_document_not_corpus_order(index):
    # The target is deliberately NOT among the first five corpus documents.
    results = index.retrieve("antibiotics sepsis survival", top_k=3, mode="dense")
    assert results[0]["pmid"] == "1007"


def test_dense_retrieval_differs_across_queries(index):
    a = [r["pmid"] for r in index.retrieve("asthma corticosteroids", top_k=3, mode="dense")]
    b = [r["pmid"] for r in index.retrieve("fracture osteoporosis bisphosphonates", top_k=3, mode="dense")]
    assert a[0] == "1003"
    assert b[0] == "1008"
    assert a != b


def test_hybrid_retrieval_finds_the_matching_document(index):
    results = index.retrieve("warfarin stroke atrial fibrillation", top_k=3, mode="hybrid")
    assert results[0]["pmid"] == "1004"


def test_rrf_fusion_retrieval_finds_the_matching_document(index):
    results = index.retrieve("migraine triptans", top_k=3, mode="hybrid", fusion="rrf")
    assert results[0]["pmid"] == "1005"


def test_reciprocal_rank_fusion_rewards_agreement_between_legs():
    # Doc 2 is second in both legs; doc 0 and doc 1 each win only one leg.
    leg_a = np.array([0.9, 0.1, 0.8, 0.0], dtype=np.float32)
    leg_b = np.array([0.1, 0.9, 0.8, 0.0], dtype=np.float32)
    fused = _reciprocal_rank_fusion([leg_a, leg_b])
    assert int(np.argmax(fused)) in (0, 1)  # a leg winner
    assert fused[2] > fused[3]


def test_weighted_fusion_respects_dense_weight():
    bm25 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    dense = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert int(np.argmax(_fuse(bm25, dense, "weighted", dense_weight=0.2))) == 0
    assert int(np.argmax(_fuse(bm25, dense, "weighted", dense_weight=0.8))) == 1


def test_lite_mode_uses_keywords_only_and_loads_no_models(monkeypatch):
    from config import get_settings

    monkeypatch.setenv("LITE_MODE", "true")
    get_settings.cache_clear()

    def boom():
        raise AssertionError("lite mode must not load an embedding model")

    monkeypatch.setattr(retrieval, "_get_model", boom)
    monkeypatch.setattr(retrieval, "_get_reranker", boom)
    docs = [
        {"title": "Metformin for diabetes", "abstract": "Metformin lowers glucose in type 2 diabetes.", "pmid": "1"},
        {"title": "Knee surgery", "abstract": "Recovery after knee arthroplasty.", "pmid": "2"},
        {"title": "Asthma inhalers", "abstract": "Inhaled steroids reduce asthma attacks.", "pmid": "3"},
        {"title": "Sleep and memory", "abstract": "Sleep supports memory consolidation.", "pmid": "4"},
    ]
    try:
        out = retrieval.hybrid_retrieve("metformin diabetes", docs, top_k=1)
        assert out[0]["pmid"] == "1"
        retrieval.warm_reranker()
    finally:
        monkeypatch.delenv("LITE_MODE")
        get_settings.cache_clear()
