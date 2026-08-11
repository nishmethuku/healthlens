"""Prometheus metric definitions, scraped via GET /metrics."""
from prometheus_client import Counter, Histogram

request_latency_seconds = Histogram(
    "request_latency_seconds", "Request latency in seconds", ["endpoint"]
)
requests_total = Counter(
    "requests_total", "Total requests handled", ["endpoint", "status"]
)
cache_hits_total = Counter("cache_hits_total", "Query cache hits on /api/v1/query")
groq_tokens_used_total = Counter(
    "groq_tokens_used_total", "Total Groq tokens consumed", ["kind"]  # kind: prompt|completion
)
# Not a real recall metric (that needs ground truth, which isn't available at
# request time) — a cheap proxy for "did retrieval return a full top-k or did
# it come up short," e.g. from a sparse corpus or an over-narrow query.
retrieval_recall_proxy = Histogram(
    "retrieval_recall_proxy",
    "docs_returned / k for each retrieval call (proxy signal, not true recall)",
    buckets=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
)


def record_token_usage(prompt_tokens: int, completion_tokens: int) -> None:
    groq_tokens_used_total.labels(kind="prompt").inc(prompt_tokens)
    groq_tokens_used_total.labels(kind="completion").inc(completion_tokens)
