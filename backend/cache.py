"""In-memory TTL/LRU cache for /api/v1/query responses."""
import hashlib
import threading
from typing import Any

from cachetools import TTLCache

MAX_ENTRIES = 500
TTL_SECONDS = 3600

_lock = threading.Lock()
_cache: TTLCache = TTLCache(maxsize=MAX_ENTRIES, ttl=TTL_SECONDS)


def normalize_question(question: str) -> str:
    return " ".join(question.lower().split())


def make_key(question: str, decompose: bool) -> str:
    raw = f"{normalize_question(question)}::{decompose}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get(key: str) -> Any | None:
    with _lock:
        return _cache.get(key)


def set(key: str, value: Any) -> None:
    with _lock:
        _cache[key] = value


def clear() -> None:
    with _lock:
        _cache.clear()


def stats() -> dict:
    with _lock:
        return {"size": len(_cache), "maxsize": MAX_ENTRIES, "ttl_seconds": TTL_SECONDS}
