"""Thumbs up/down feedback log, one JSON record per line."""
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict

FEEDBACK_PATH = Path(__file__).parent / "data" / "feedback.jsonl"

_lock = threading.Lock()


class FeedbackRecord(TypedDict):
    query_id: str
    rating: Literal["up", "down"]
    answer_preview: str
    timestamp: str


def append(query_id: str, rating: Literal["up", "down"], answer_preview: str) -> None:
    record: FeedbackRecord = {
        "query_id": query_id,
        "rating": rating,
        "answer_preview": answer_preview[:500],
        "timestamp": datetime.now(UTC).isoformat(),
    }
    line = json.dumps(record) + "\n"
    with _lock:
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
            f.write(line)
