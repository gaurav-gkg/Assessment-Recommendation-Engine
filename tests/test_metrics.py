"""Unit tests for the evaluation metrics module."""

import pytest
from evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    average_precision,
    ndcg_at_k,
    mean_recall_at_k,
    evaluate_system,
)

class TestRecallAtK:
    def test_perfect_recall(self):
        relevant = ["url1", "url2", "url3"]
        retrieved = ["url1", "url2", "url3", "url4"]
        assert recall_at_k(relevant, retrieved, k=4) == 1.0

    def test_zero_recall(self):
        assert recall_at_k(["url1"], ["url2", "url3"], k=2) == 0.0

    def test_partial_recall(self):
        relevant = ["url1", "url2", "url3", "url4"]
        retrieved = ["url1", "url5", "url2", "url6"]
        assert recall_at_k(relevant, retrieved, k=4) == 0.5

    def test_empty_relevant_returns_1(self):
        assert recall_at_k([], ["url1"], k=1) == 1.0

    def test_k_less_than_retrieved(self):
        # Only top-k items should be considered
        relevant = ["url3"]
        retrieved = ["url1", "url2", "url3"]
        assert recall_at_k(relevant, retrieved, k=2) == 0.0
        assert recall_at_k(relevant, retrieved, k=3) == 1.0

    def test_case_insensitive(self):
        assert recall_at_k(["URL1"], ["url1"], k=1) == 1.0

class TestPrecisionAtK:
    def test_full_precision(self):
        relevant = ["a", "b"]
        retrieved = ["a", "b"]
        assert precision_at_k(relevant, retrieved, k=2) == 1.0

    def test_empty_retrieved(self):
        assert precision_at_k(["a"], [], k=5) == 0.0

    def test_k_zero(self):
        assert precision_at_k(["a"], ["a"], k=0) == 0.0

class TestAveragePrecision:
    def test_perfect_ap(self):
        relevant = ["a", "b"]
        retrieved = ["a", "b", "c"]
        ap = average_precision(relevant, retrieved)
        assert ap == pytest.approx(1.0)

    def test_ap_with_misses(self):
        relevant = ["a", "b"]
        retrieved = ["c", "a", "d", "b"]
        # Hit at position 2 (P=1/2), hit at position 4 (P=2/4)
        expected = (0.5 + 0.5) / 2
        assert average_precision(relevant, retrieved) == pytest.approx(expected)

class TestNDCGAtK:
    def test_perfect_ndcg(self):
        relevant = ["a", "b"]
        retrieved = ["a", "b", "c"]
        assert ndcg_at_k(relevant, retrieved, k=2) == pytest.approx(1.0)

    def test_zero_ndcg(self):
        assert ndcg_at_k(["a"], ["b", "c"], k=2) == 0.0

class TestMeanRecallAtK:
    def test_mean_of_two(self):
        all_rel = [["a", "b"], ["c"]]
        all_ret = [["a", "x"], ["c", "y"]]
        # Query 1: recall@2 = 1/2 = 0.5
        # Query 2: recall@2 = 1/1 = 1.0
        assert mean_recall_at_k(all_rel, all_ret, k=2) == pytest.approx(0.75)

    def test_empty(self):
        assert mean_recall_at_k([], [], k=5) == 0.0

class TestEvaluateSystem:
    def test_full_report_keys(self):
        queries = ["q1", "q2"]
        all_rel = [["a"], ["b"]]
        all_ret = [["a", "c"], ["d", "b"]]
        report = evaluate_system(queries, all_rel, all_ret, ks=[5, 10])
        assert "mean_recall@5" in report
        assert "mean_recall@10" in report
        assert "map" in report
        assert "per_query" in report
        assert len(report["per_query"]) == 2
