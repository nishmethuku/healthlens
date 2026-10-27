"""Checks that a generated answer actually cites the sources it was given.

These are cheap, deterministic checks that run on every answer. They cannot
prove a claim is true, but they catch the failures that are visible without a
second model call: citations to sources that do not exist, and answers whose
factual sentences carry no citation at all.
"""
import re
from typing import TypedDict

# Matches "[1]", "[2, 3]", "[1-3]" and "[1,2,3]".
_CITATION_RE = re.compile(r"\[(\d+(?:\s*(?:,|-|–)\s*\d+)*)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_MIN_WORDS_FOR_CLAIM = 6
_DISCLAIMER_MARKERS = ("not medical advice", "consult a", "educational")
_NO_EVIDENCE_MARKERS = (
    "do not support",
    "does not support",
    "do not contain",
    "does not contain",
    "not enough information",
    "no information",
    "cannot answer",
)
MIN_COVERAGE = 0.6


class Grounding(TypedDict):
    cited: list[int]
    invalid: list[int]
    claim_sentences: int
    cited_sentences: int
    coverage: float
    grounded: bool


def _expand(group: str) -> list[int]:
    numbers: list[int] = []
    for part in re.split(r"\s*,\s*", group):
        if re.search(r"[-–]", part):
            low, high = (int(x) for x in re.split(r"\s*[-–]\s*", part))
            if low <= high and high - low < 20:
                numbers.extend(range(low, high + 1))
        else:
            numbers.append(int(part))
    return numbers


def extract_citations(text: str) -> list[int]:
    found: list[int] = []
    for match in _CITATION_RE.finditer(text):
        found.extend(_expand(match.group(1)))
    return found


def _is_claim(sentence: str) -> bool:
    lowered = sentence.lower()
    if any(marker in lowered for marker in _DISCLAIMER_MARKERS):
        return False
    return len(re.findall(r"\w+", sentence)) >= _MIN_WORDS_FOR_CLAIM


def analyze_citations(answer: str, num_sources: int) -> Grounding:
    """Report which sources were cited, which citations point at nothing, and
    what fraction of the answer's factual sentences carry a citation."""
    all_cited = extract_citations(answer)
    invalid = sorted({n for n in all_cited if n < 1 or n > num_sources})
    cited = sorted({n for n in all_cited if 1 <= n <= num_sources})

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer) if s.strip()]
    claims = [s for s in sentences if _is_claim(s)]
    cited_claims = [s for s in claims if _CITATION_RE.search(s)]
    coverage = len(cited_claims) / len(claims) if claims else 1.0

    lowered = answer.lower()
    declines_to_answer = any(marker in lowered for marker in _NO_EVIDENCE_MARKERS)
    grounded = not invalid and (coverage >= MIN_COVERAGE or declines_to_answer)

    return {
        "cited": cited,
        "invalid": invalid,
        "claim_sentences": len(claims),
        "cited_sentences": len(cited_claims),
        "coverage": round(coverage, 3),
        "grounded": grounded,
    }


def make_snippet(text: str, limit: int = 280) -> str:
    """First ~limit characters of an abstract, cut at a sentence or word boundary."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    window = text[:limit]
    end = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
    if end >= limit // 2:
        return window[: end + 1]
    return window.rsplit(" ", 1)[0] + "…"
