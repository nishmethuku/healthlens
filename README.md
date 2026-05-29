# HealthLens

HealthLens is a full-stack medical Q&A app that answers questions using PubMed literature. It runs hybrid retrieval (BM25 + dense embeddings + FAISS) over a local indexed corpus, optionally decomposes complex questions into sub-queries, and generates grounded answers with citations via Groq (Llama 3.3 70B).

## Architecture

```
  User
   │
   ▼
┌──────────────────┐
│  React frontend  │  (Vite + Tailwind, localhost:5173)
└────────┬─────────┘
         │ POST /query
         ▼
┌──────────────────┐
│     FastAPI      │  (localhost:8000)
└────────┬─────────┘
         │
    ┌────┴────┬──────────────┬─────────────────┐
    ▼         ▼              ▼                 ▼
┌─────────┐ ┌────────────┐ ┌─────────────────────────────┐
│Guardrails│ │Query decomp│ │ Hybrid retrieval            │
│(keywords)│ │(Groq LLM)  │ │ BM25 + dense + FAISS        │
└─────────┘ └─────┬──────┘ └──────────────┬──────────────┘
                  │                       │
                  └───────────┬───────────┘
                              ▼
                   ┌─────────────────────┐
                   │  PubMed corpus      │
                   │  pubmed_abstracts   │
                   │  .jsonl (~200K)     │
                   └──────────┬──────────┘
                              │ top-5 abstracts
                              ▼
                   ┌─────────────────────┐
                   │  Groq LLM           │
                   │  llama-3.3-70b      │
                   │  (grounded answer)  │
                   └─────────────────────┘
```

## Benchmark results

Evaluated on **50 medical questions** against a corpus of **164,639 abstracts** (Recall@5, top-5 retrieval):

| Mode       | Recall@5 | MRR   | Precision@5 |
|------------|----------|-------|-------------|
| BM25-only  | 0.072    | 0.139 | 0.072       |
| Dense-only | 0.000    | 0.000 | 0.000       |
| **Hybrid** | **0.104**| **0.199** | **0.104** |

Hybrid retrieval **outperforms BM25-only by 44% on Recall@5** (0.104 vs 0.072).

Run the eval harness yourself:

```bash
cd backend && python eval/eval_harness.py
```

## Project structure

```
healthlens/
├── backend/
│   ├── main.py
│   ├── retrieval.py
│   ├── query_decomposition.py
│   ├── ingest.py
│   ├── llm.py
│   ├── guardrails.py
│   ├── index_pubmed.py
│   ├── eval/
│   │   ├── eval_harness.py
│   │   └── benchmark.json
│   ├── data/
│   │   └── pubmed_abstracts.jsonl
│   ├── requirements.txt
│   └── .env
├── frontend/
└── README.md
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- A [Groq API key](https://console.groq.com/)
- Optional: [NCBI API key](https://www.ncbi.nlm.nih.gov/account/) for bulk indexing

## Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Edit `backend/.env`:

```
GROQ_API_KEY=your_actual_key_here
NCBI_API_KEY=your_ncbi_key_here   # optional, for index_pubmed.py
```

Start the API:

```bash
uvicorn main:app --reload --port 8000
```

The first query loads the local corpus index and may take a minute while embeddings are built. If `data/pubmed_abstracts.jsonl` is missing, the API falls back to live PubMed fetch per query.

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Usage

1. Start the backend (`uvicorn` on port 8000).
2. Start the frontend (`npm run dev` on port 5173).
3. Enter a medical question and submit.
4. HealthLens retrieves relevant abstracts from the indexed corpus and returns a cited answer.

For complex multi-part questions, enable query decomposition:

```json
{
  "question": "Compare SGLT2 inhibitors vs GLP-1 agonists for heart failure and kidney disease",
  "decompose": true
}
```

Unsafe queries (e.g. self-harm) are blocked and show a crisis-resources warning instead of calling the LLM.

## API

**POST** `/query`

```json
{
  "question": "What are the latest treatments for type 2 diabetes?",
  "decompose": false
}
```

Response:

```json
{
  "answer": "...",
  "sources": [{ "title": "...", "pmid": "12345678" }],
  "flagged": false,
  "warning": null
}
```

## Disclaimer

HealthLens is for educational and research purposes only. It does not provide medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.
