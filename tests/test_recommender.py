"""Unit tests for QueryProcessor and Reranker."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

# QueryProcessor tests

class TestQueryProcessor:
    def test_plain_text_passthrough(self):
        """Without enrichment, plain text should pass through unchanged."""
        from recommender.query_processor import QueryProcessor

        processor = QueryProcessor(enrich=False)
        result = processor.process("I need a strong leader for a senior role.")
        assert "leader" in result.lower() or "senior" in result.lower()

    def test_empty_query_raises(self):
        from recommender.query_processor import QueryProcessor

        processor = QueryProcessor(enrich=False)
        with pytest.raises(ValueError):
            processor.process("   ")

    def test_url_detection(self):
        from recommender.query_processor import _is_url

        assert _is_url("https://example.com/job") is True
        assert _is_url("http://example.com") is True
        assert _is_url("This is a job description text.") is False
        assert _is_url("") is False

# Reranker tests

SAMPLE_CANDIDATES = [
    (
        {
            "name": f"Assessment {i}",
            "url": f"https://shl.com/{i}",
            "description": "Test",
            "test_type": ["A"] if i % 2 == 0 else ["P"],
            "test_type_labels": ["Ability & Aptitude"] if i % 2 == 0 else ["Personality & Behavior"],
        },
        0.9 - i * 0.05,
    )
    for i in range(15)
]

class TestReranker:
    def test_rerank_without_llm_returns_balanced(self):
        from recommender.reranker import Reranker

        reranker = Reranker(use_llm=False)
        result = reranker.rerank("test query", SAMPLE_CANDIDATES, n=10)

        assert len(result) <= 10
        assert len(result) >= 5

    def test_rerank_never_exceeds_max(self):
        from recommender.reranker import Reranker
        from config import settings

        reranker = Reranker(use_llm=False)
        result = reranker.rerank("test", SAMPLE_CANDIDATES, n=100)
        assert len(result) <= settings.MAX_RECOMMENDATIONS

    def test_rerank_never_below_min(self):
        from recommender.reranker import Reranker
        from config import settings

        reranker = Reranker(use_llm=False)
        # Provide more candidates than min
        big_candidates = SAMPLE_CANDIDATES * 2
        result = reranker.rerank("test", big_candidates, n=1)
        # Should be clamped up to MIN_RECOMMENDATIONS
        assert len(result) >= settings.MIN_RECOMMENDATIONS

    def test_empty_candidates(self):
        from recommender.reranker import Reranker

        reranker = Reranker(use_llm=False)
        result = reranker.rerank("test", [], n=10)
        assert result == []

    def test_balance_reduces_single_category_dominance(self):
        from recommender.reranker import _balance_results

        # All 10 candidates are category "A"
        mono_candidates = [
            ({"name": f"A{i}", "url": f"u{i}", "test_type": ["A"]}, 0.9)
            for i in range(8)
        ] + [
            ({"name": f"P{i}", "url": f"p{i}", "test_type": ["P"]}, 0.7)
            for i in range(7)
        ]

        result = _balance_results(mono_candidates, target_count=10)
        types = [t for doc, _ in result for t in doc["test_type"]]
        n_A = types.count("A")
        # "A" should not dominate > 70 % of 10 results
        assert n_A / 10 <= 0.75
