import os
from typing import TypedDict

from dotenv import load_dotenv
from groq import Groq

from retrieval import RankedAbstract

load_dotenv()

MODEL = "llama-3.3-70b-versatile"


class LLMResponse(TypedDict):
    answer: str
    citations: list[dict[str, str]]


def generate_answer(question: str, sources: list[RankedAbstract]) -> LLMResponse:
    """Call Groq with retrieved abstracts; return grounded answer and citation metadata."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "GROQ_API_KEY is not set. Add your key to backend/.env"
        )

    context = _build_context(sources)
    client = Groq(api_key=api_key)

    system = (
        "You are HealthLens, a medical literature assistant. Answer ONLY using the "
        "provided PubMed abstracts. Be clear, accurate, and concise. "
        "Cite sources inline as [1], [2], etc., matching the numbered abstracts. "
        "If the abstracts do not support an answer, say so. "
        "Include a brief disclaimer that this is not medical advice."
    )

    user_message = (
        f"Question: {question}\n\n"
        f"PubMed abstracts:\n{context}\n\n"
        "Provide a grounded answer with inline citations [1], [2], ..."
    )

    completion = client.chat.completions.create(
        model=MODEL,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
    )

    answer = completion.choices[0].message.content or ""
    citations = [
        {"index": str(i + 1), "title": s["title"], "pmid": s["pmid"]}
        for i, s in enumerate(sources)
    ]

    return {"answer": answer.strip(), "citations": citations}


def _build_context(sources: list[RankedAbstract]) -> str:
    blocks = []
    for i, s in enumerate(sources, start=1):
        blocks.append(
            f"[{i}] PMID: {s['pmid']}\n"
            f"Title: {s['title']}\n"
            f"Abstract: {s['abstract'][:2000]}"
        )
    return "\n\n".join(blocks)
