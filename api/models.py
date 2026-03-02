from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class RecommendRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=10_000,
        examples=[
            "We need a mid-level Java developer with good communication skills.",
            "https://example.com/job/data-scientist",
        ],
    )
    num_results: int = Field(default=10, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()


class AssessmentResult(BaseModel):
    url: str
    name: str
    adaptive_support: str = "No"  # "Yes" | "No"
    description: str = ""
    duration: Optional[int] = None  # minutes
    remote_support: str = "No"  # "Yes" | "No"
    test_type: list[str] = Field(default_factory=list)


class RecommendResponse(BaseModel):
    recommended_assessments: list[AssessmentResult] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "healthy"
