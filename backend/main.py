import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import get_settings
from guardrails import check_query
from ingest import fetch_pubmed_abstracts
from llm import generate_answer
from logging_config import configure_logging, request_id_var
from query_decomposition import decomposed_retrieve
from retrieval import (
    get_corpus_index,
    hybrid_retrieve,
    hybrid_retrieve_indexed,
    warm_reranker,
)

load_dotenv()
settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger("healthlens")

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the (slow, CPU-bound) corpus index off the event loop so the
    first real request doesn't pay embedding-build latency, and so /health/ready
    has something meaningful to report."""
    app.state.startup_complete = False
    try:
        await asyncio.to_thread(get_corpus_index)
        await asyncio.to_thread(warm_reranker)
    except Exception:
        logger.exception("failed to warm corpus/reranker index")
    app.state.startup_complete = True
    yield


app = FastAPI(
    title="HealthLens API",
    version="1.0.0",
    description=(
        "Grounded medical Q&A over a hybrid-retrieval PubMed index. "
        "Not medical advice — see the /query endpoint disclaimer."
    ),
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Stamp every request with a correlation ID and log latency/outcome."""
    req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    token = request_id_var.set(req_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "unhandled request error",
            extra={"path": request.url.path, "method": request.method},
        )
        raise
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        request_id_var.reset(token)
    logger.info(
        "request handled",
        extra={
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    response.headers["x-request-id"] = req_id
    return response


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Fail closed: with no configured key, or a missing/wrong header, reject.

    Known limitation: the frontend is a public browser SPA, so any key it
    sends is baked into the built JS bundle and readable by anyone who opens
    devtools — it is not a real secret in that context. This gates casual/
    automated abuse of a cost-bearing LLM endpoint, not a determined attacker
    with access to the frontend. A proper fix is per-user auth or a
    backend-for-frontend that holds the real secret server-side.
    """
    expected = settings.healthlens_api_key
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="A medical question to answer using indexed PubMed literature.",
        examples=["What are the latest treatments for type 2 diabetes?"],
    )
    decompose: bool = Field(
        default=False,
        description="Split complex multi-part questions into sub-queries before retrieval.",
    )

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("question must not be blank or whitespace-only")
        return stripped


class Source(BaseModel):
    title: str
    pmid: str


class QueryResponse(BaseModel):
    answer: str | None = None
    sources: list[Source] = []
    flagged: bool = False
    warning: str | None = None


live_router = APIRouter(tags=["health"])


@live_router.get("/health/live", summary="Liveness probe")
async def health_live():
    """Process is up and serving requests. Does not imply the corpus/model are loaded."""
    return {"status": "ok"}


@live_router.get("/health/ready", summary="Readiness probe")
async def health_ready():
    """Ready to serve /query: startup warm-up finished and an LLM key configured.

    A missing local corpus is NOT a readiness failure — /query transparently
    falls back to live PubMed fetch in that case. Only report the corpus as a
    hard dependency if that fallback is intentionally disabled for this deploy.
    """
    startup_complete = getattr(app.state, "startup_complete", False)
    llm_configured = bool(settings.groq_api_key)
    ok = startup_complete and llm_configured
    body = {
        "status": "ok" if ok else "not_ready",
        "startup_complete": startup_complete,
        "corpus_index_loaded": get_corpus_index() is not None,
        "llm_configured": llm_configured,
    }
    return JSONResponse(status_code=200 if ok else 503, content=body)


v1_router = APIRouter(prefix="/api/v1", tags=["query"])


@v1_router.post(
    "/query",
    response_model=QueryResponse,
    summary="Answer a medical question grounded in PubMed abstracts",
    dependencies=[Depends(verify_api_key)],
    responses={
        401: {"description": "Missing or invalid API key"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "LLM or retrieval backend unavailable"},
    },
)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def query(request: Request, req: QueryRequest):
    question = req.question

    flagged, warning = check_query(question)
    if flagged:
        logger.info("query flagged by guardrails")
        return QueryResponse(flagged=True, warning=warning)

    try:
        corpus_index = await asyncio.to_thread(get_corpus_index)

        if corpus_index is not None:
            if req.decompose:
                top = await asyncio.to_thread(
                    decomposed_retrieve, question, corpus_index, 5
                )
            else:
                top = await asyncio.to_thread(hybrid_retrieve_indexed, question, 5)
        else:
            abstracts = await fetch_pubmed_abstracts(question)
            if not abstracts:
                return QueryResponse(
                    answer="No PubMed abstracts were found for this query. Try rephrasing your question.",
                    sources=[],
                )
            top = await asyncio.to_thread(hybrid_retrieve, question, abstracts, 5)

        if not top:
            return QueryResponse(
                answer="No relevant abstracts were found for this query. Try rephrasing your question.",
                sources=[],
            )

        result = await asyncio.to_thread(generate_answer, question, top)
        sources = [Source(title=s["title"], pmid=s["pmid"]) for s in top]
        return QueryResponse(answer=result["answer"], sources=sources)

    except ValueError:
        # Missing/invalid API key etc. — a config problem, not a client error.
        # Never echo exception text to the caller: it can carry internal
        # paths, key-shape hints, or stack detail.
        logger.exception("service misconfiguration handling query")
        raise HTTPException(
            status_code=503, detail="HealthLens is temporarily unavailable. Please try again shortly."
        ) from None
    except Exception:
        logger.exception("unhandled error handling query")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your question.",
        ) from None


app.include_router(live_router)
app.include_router(v1_router)


@app.get("/health", include_in_schema=False)
async def health_legacy():
    """Deprecated: use /health/live or /health/ready. Kept for existing monitors."""
    return {"status": "ok"}
