"""Integration tests for the FastAPI recommendation endpoints."""

from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock, patch

# Helpers

MOCK_RECOMMENDATIONS = [
    {
        "url": "https://www.shl.com/solutions/products/product-catalog/view/general-ability/",
        "name": "General Ability",
        "adaptive_support": "Yes",
        "description": "Measures general cognitive ability.",
        "duration": 36,
        "remote_support": "Yes",
        "test_type": ["Ability & Aptitude"],
    }
    for _ in range(5)
]

def _make_mock_engine():
    engine = MagicMock()
    engine.recommend.return_value = MOCK_RECOMMENDATIONS
    return engine

# Tests

@pytest.mark.asyncio
async def test_health():
    from api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_recommend_returns_200():
    from api.main import app
    import api.routes as routes_module

    mock_engine = _make_mock_engine()

    with patch.object(routes_module, "_get_engine", return_value=mock_engine):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/recommend",
                json={"query": "Java developer with communication skills"},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert "recommended_assessments" in body
    assert len(body["recommended_assessments"]) >= 1

@pytest.mark.asyncio
async def test_recommend_validates_empty_query():
    from api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/recommend", json={"query": "   "})

    assert resp.status_code == 422  # Validation error

@pytest.mark.asyncio
async def test_recommend_result_schema():
    from api.main import app
    import api.routes as routes_module

    mock_engine = _make_mock_engine()

    with patch.object(routes_module, "_get_engine", return_value=mock_engine):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/recommend",
                json={"query": "Senior data analyst", "num_results": 5},
            )

    assert resp.status_code == 200
    recs = resp.json()["recommended_assessments"]
    for rec in recs:
        assert "name" in rec
        assert "url" in rec
        assert "test_type" in rec
        assert "remote_support" in rec
        assert rec["remote_support"] in ("Yes", "No")
        assert "adaptive_support" in rec
        assert rec["adaptive_support"] in ("Yes", "No")
        assert "duration" in rec

@pytest.mark.asyncio
async def test_root_redirect():
    from api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/")

    assert resp.status_code == 200
    body = resp.json()
    assert "docs" in body
