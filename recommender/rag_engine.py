"""RAG pipeline: enrich query -> FAISS search -> rerank -> structured output."""
from __future__ import annotations

import re

from loguru import logger

from config import settings
from embeddings.vector_store import FAISSVectorStore
from recommender.query_processor import QueryProcessor
from recommender.reranker import Reranker


def _format_result(assessment: dict, score: float) -> dict:
    """Map raw assessment metadata to the API response schema."""
    duration: int | None = assessment.get("duration_minutes") or None
    if duration is None:
        m = re.search(r"(\d+)", assessment.get("assessment_length", "") or "")
        duration = int(m.group(1)) if m else None
    return {
        "url": assessment.get("url", ""),
        "name": assessment.get("name", ""),
        "adaptive_support": "Yes" if assessment.get("adaptive_irt") else "No",
        "description": assessment.get("description", ""),
        "duration": duration,
        "remote_support": "Yes" if assessment.get("remote_testing") else "No",
        "test_type": assessment.get("test_type_labels", assessment.get("test_type", [])),
    }


class RAGEngine:
    """End-to-end recommendation engine: enrich query, retrieve, rerank."""

    def __init__(self, use_llm_rerank: bool = True, enrich_query: bool = True):
        self.vector_store = FAISSVectorStore()
        self.query_processor = QueryProcessor(enrich=enrich_query)
        self.reranker = Reranker(use_llm=use_llm_rerank)
        self._loaded = False

    def load(self) -> None:
        """Load FAISS index from disk. Call once at startup."""
        if not self._loaded:
            self.vector_store.load()
            self._loaded = True
            logger.success("RAGEngine ready.")

    def recommend(self, query: str, n: int = settings.MAX_RECOMMENDATIONS) -> list[dict]:
        """Return up to n ranked assessment recommendations for query."""
        if not self._loaded:
            self.load()

        n = max(settings.MIN_RECOMMENDATIONS, min(n, settings.MAX_RECOMMENDATIONS))

        logger.info("Processing query ...")
        enriched = self.query_processor.process(query)

        logger.info(f"Retrieving top-{settings.RETRIEVAL_K} candidates from FAISS ...")
        candidates = self.vector_store.search(enriched, k=settings.RETRIEVAL_K)
        logger.info(f"Retrieved {len(candidates)} candidates.")

        if not candidates:
            logger.warning("No candidates retrieved.")
            return []

        final = self.reranker.rerank(enriched, candidates, n=n)
        results = [_format_result(a, s) for a, s in final]
        logger.success(f"Returning {len(results)} recommendations.")
        return results
