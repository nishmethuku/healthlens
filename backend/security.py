"""Small, dependency-free security helpers used by main.py.

Kept separate from the FastAPI wiring so each rule can be unit tested on its own.
"""

from __future__ import annotations

import hmac
import re
import uuid

from starlette.requests import Request
from starlette.responses import Response

from config import Settings

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_DOC_PATHS = ("/docs", "/redoc", "/openapi.json")


def constant_time_equals(supplied: str | None, expected: str | None) -> bool:
    """Compare secrets without leaking, through timing, how many leading characters match.

    A plain `==` stops at the first differing character, so response time can reveal a
    correct prefix. Both sides are encoded so non-ASCII input cannot raise.
    """
    if not supplied or not expected:
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def sanitize_request_id(value: str | None) -> str:
    """Accept a caller's correlation ID only if it is short and plain; else mint one.

    The ID is echoed in a response header and written to logs, so unbounded or
    control-character input would allow log forging and header abuse.
    """
    if value and _REQUEST_ID.match(value):
        return value
    return str(uuid.uuid4())


def client_ip(request: Request, trusted_hops: int) -> str:
    """Best estimate of the real client address, for rate limiting.

    Behind a reverse proxy, the socket address is the proxy, so every user would share
    one rate-limit bucket. The proxy appends the address it saw to X-Forwarded-For, so
    the real client is `trusted_hops` entries from the RIGHT. Entries further left are
    supplied by the client and can be forged, so they are never used. With 0 trusted
    hops the header is ignored entirely.
    """
    peer = request.client.host if request.client else "unknown"
    if trusted_hops <= 0:
        return peer
    forwarded = request.headers.get("x-forwarded-for", "")
    hops = [h.strip() for h in forwarded.split(",") if h.strip()]
    if len(hops) < trusted_hops:
        return peer
    return hops[-trusted_hops]


def apply_security_headers(response: Response, path: str, production: bool) -> None:
    """Defensive headers for every response. The API only returns JSON or an event
    stream, so it can forbid framing, sniffing, caching, and all content loading."""
    h = response.headers
    h.setdefault("X-Content-Type-Options", "nosniff")
    h.setdefault("X-Frame-Options", "DENY")
    h.setdefault("Referrer-Policy", "no-referrer")
    h.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    h.setdefault("Cross-Origin-Resource-Policy", "same-site")
    if not path.startswith(_DOC_PATHS):
        # Interactive docs load scripts from a CDN, so they are exempt from the strict policy.
        h.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
        h.setdefault("Cache-Control", "no-store")
    if production:
        h.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")


def metrics_authorized(authorization: str | None, settings: Settings) -> bool:
    """/metrics reveals traffic and error patterns. In production it needs a bearer token;
    with no token configured it is closed. Outside production it stays open for local use."""
    if not settings.is_production:
        return True
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        return False
    return constant_time_equals(authorization[len(prefix) :], settings.metrics_token)


def production_config_problems(settings: Settings) -> list[str]:
    """Misconfigurations that are unsafe in production. Empty list means fine."""
    problems: list[str] = []
    if not settings.healthlens_api_key:
        problems.append("HEALTHLENS_API_KEY is not set, so every request would be rejected")
    origins = settings.cors_origin_list
    if not origins:
        problems.append("CORS_ORIGINS is empty")
    for origin in origins:
        if origin == "*":
            problems.append("CORS_ORIGINS must not be a wildcard")
        elif not origin.startswith("https://"):
            problems.append(f"CORS origin {origin!r} is not https")
    return problems
