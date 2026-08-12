# --- builder: compile deps into a venv, keep build tools out of the final image ---
FROM python:3.12-slim AS builder

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- runtime: slim image, non-root user, only the venv + source copied in ---
FROM python:3.12-slim AS runtime

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache

COPY . .

# Bake the embedding + reranker weights into the image so the container never
# needs Hub access at runtime (faster cold start, works in network-locked prod).
# Read model ids from config.py instead of hardcoding them here, so a model
# swap in one place can't silently drift between dev and the built image.
RUN python -c "\
from config import get_settings; \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
s = get_settings(); \
SentenceTransformer(s.embedding_model_id); \
CrossEncoder(s.reranker_model_id)" \
    && chown -R appuser:appuser /opt/hf-cache /app

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health/live').raise_for_status()"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
