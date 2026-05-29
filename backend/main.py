from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from guardrails import check_query
from ingest import fetch_pubmed_abstracts
from llm import generate_answer
from query_decomposition import decomposed_retrieve
from retrieval import get_corpus_index, hybrid_retrieve, hybrid_retrieve_indexed

load_dotenv()

app = FastAPI(title="HealthLens", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    decompose: bool = False


class Source(BaseModel):
    title: str
    pmid: str


class QueryResponse(BaseModel):
    answer: str | None = None
    sources: list[Source] = []
    flagged: bool = False
    warning: str | None = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    question = req.question.strip()

    flagged, warning = check_query(question)
    if flagged:
        return QueryResponse(flagged=True, warning=warning)

    try:
        corpus_index = get_corpus_index()

        if corpus_index is not None:
            if req.decompose:
                top = decomposed_retrieve(question, corpus_index, top_k=5)
            else:
                top = hybrid_retrieve_indexed(question, top_k=5)
        else:
            abstracts = await fetch_pubmed_abstracts(question)
            if not abstracts:
                return QueryResponse(
                    answer="No PubMed abstracts were found for this query. Try rephrasing your question.",
                    sources=[],
                )
            top = hybrid_retrieve(question, abstracts, top_k=5)

        if not top:
            return QueryResponse(
                answer="No relevant abstracts were found for this query. Try rephrasing your question.",
                sources=[],
            )

        result = generate_answer(question, top)

        sources = [
            Source(title=s["title"], pmid=s["pmid"])
            for s in top
        ]

        return QueryResponse(answer=result["answer"], sources=sources)

    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your question: {e}",
        ) from e
