import re

UNSAFE_PATTERNS = [
    r"\bkill\s+myself\b",
    r"\bkill\s+my\s*self\b",
    r"\bend\s+my\s+life\b",
    r"\bwant\s+to\s+die\b",
    r"\bwish\s+(i\s+)?(was|were)\s+dead\b",
    r"\bcommit\s+suicide\b",
    r"\bhow\s+to\s+(commit\s+)?suicide\b",
    r"\bways?\s+to\s+(die|kill\s+myself)\b",
    r"\bself[\s-]?harm\b",
    r"\bcut\s+myself\b",
    r"\bhurt\s+myself\b",
    r"\boverdose\s+on\b",
    r"\bsuicidal\b",
    r"\bsuicide\b",
]

_compiled = [re.compile(p, re.IGNORECASE) for p in UNSAFE_PATTERNS]


def check_query(query: str) -> tuple[bool, str | None]:
    """Return (flagged, reason). flagged=True if query matches unsafe patterns."""
    text = query.strip()
    if not text:
        return False, None

    for pattern in _compiled:
        if pattern.search(text):
            return True, (
                "This question appears to involve self-harm or crisis content. "
                "HealthLens cannot provide medical guidance for this topic. "
                "If you are in crisis, please contact emergency services (911 in the US) "
                "or the 988 Suicide & Crisis Lifeline (call or text 988)."
            )

    return False, None
