"""Gemini embedding wrapper with batching and rate-limit handling."""
from __future__ import annotations

import time

from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings

_client = genai.Client(api_key=settings.GOOGLE_API_KEY)

# free-tier: 100 text-items/min – back off up to 65s so the window fully resets
_retry = retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=5, max=65),
    reraise=True,
)


@_retry
def _embed_batch(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    result = _client.models.embed_content(
        model=settings.EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [list(e.values) for e in result.embeddings]


class GeminiEmbeddingModel:
    """Batch-embeds texts via Gemini, respecting the free-tier rate limit."""

    def __init__(self, batch_size: int = settings.EMBEDDING_BATCH_SIZE):
        self.batch_size = batch_size
        self.dimension = settings.EMBEDDING_DIMENSION

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed corpus documents. Sleeps 35s between batches (50 items + 35s < 100/min)."""
        all_vecs: list[list[float]] = []
        total = max(1, (len(texts) + self.batch_size - 1) // self.batch_size)

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.info(f"Embedding batch {i // self.batch_size + 1}/{total} ({len(batch)} texts) ...")
            all_vecs.extend(_embed_batch(batch, task_type="RETRIEVAL_DOCUMENT"))
            if i + self.batch_size < len(texts):
                logger.debug("Rate-limit cooldown: 35s ...")
                time.sleep(35)

        logger.success(f"Embedded {len(all_vecs)} documents.")
        return all_vecs

    def embed_query(self, query: str) -> list[float]:
        return _embed_batch([query], task_type="RETRIEVAL_QUERY")[0]
