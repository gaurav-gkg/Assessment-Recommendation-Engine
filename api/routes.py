from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from api.models import AssessmentResult, HealthResponse, RecommendRequest, RecommendResponse
from recommender.rag_engine import RAGEngine

_engine: RAGEngine | None = None


def _get_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
        _engine.load()
    return _engine


router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@router.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
async def recommend(payload: RecommendRequest) -> RecommendResponse:
    """Return up to 10 recommended SHL assessments for the given query."""
    engine = _get_engine()
    try:
        raw = engine.recommend(query=payload.query, n=payload.num_results)
    except FileNotFoundError as exc:
        logger.error(f"Index not found: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector index not found. Run `python scripts/build_index.py`.",
        )
    except Exception as exc:
        logger.exception(f"Recommendation error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return RecommendResponse(recommended_assessments=[AssessmentResult(**r) for r in raw])
