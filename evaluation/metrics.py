"""Evaluation metrics: Recall@K, Precision@K, AP, NDCG@K, MAP."""
from __future__ import annotations

import math
from typing import Any


def recall_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 1.0
    rel = {u.strip().lower() for u in relevant}
    return len(rel & {u.strip().lower() for u in retrieved[:k]}) / len(rel)


def precision_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    if not retrieved or k == 0:
        return 0.0
    rel = {u.strip().lower() for u in relevant}
    top = [u.strip().lower() for u in retrieved[:k]]
    return len(rel & set(top)) / min(k, len(top))


def average_precision(relevant: list[str], retrieved: list[str]) -> float:
    if not relevant:
        return 1.0
    rel = {u.strip().lower() for u in relevant}
    ap, hits = 0.0, 0
    for i, url in enumerate([u.strip().lower() for u in retrieved], 1):
        if url in rel:
            hits += 1
            ap += hits / i
    return ap / len(rel)


def ndcg_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    rel = {u.strip().lower() for u in relevant}
    top_k = [u.strip().lower() for u in retrieved[:k]]
    dcg = sum(1.0 / math.log2(i + 2) for i, u in enumerate(top_k) if u in rel)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg > 0 else 0.0


def mean_recall_at_k(all_relevant: list[list[str]], all_retrieved: list[list[str]], k: int) -> float:
    if not all_relevant:
        return 0.0
    return sum(recall_at_k(r, p, k) for r, p in zip(all_relevant, all_retrieved)) / len(all_relevant)


def mean_average_precision(all_relevant: list[list[str]], all_retrieved: list[list[str]]) -> float:
    if not all_relevant:
        return 0.0
    return sum(average_precision(r, p) for r, p in zip(all_relevant, all_retrieved)) / len(all_relevant)


def evaluate_system(
    queries: list[str],
    all_relevant: list[list[str]],
    all_retrieved: list[list[str]],
    ks: list[int] | None = None,
) -> dict[str, Any]:
    """Compute full evaluation report with per-query breakdown."""
    ks = ks or [5, 10]
    report: dict[str, Any] = {}
    for k in ks:
        recalls = [recall_at_k(r, p, k) for r, p in zip(all_relevant, all_retrieved)]
        precs = [precision_at_k(r, p, k) for r, p in zip(all_relevant, all_retrieved)]
        report[f"mean_recall@{k}"] = round(sum(recalls) / len(recalls), 4)
        report[f"mean_precision@{k}"] = round(sum(precs) / len(precs), 4)
    aps = [average_precision(r, p) for r, p in zip(all_relevant, all_retrieved)]
    report["map"] = round(sum(aps) / len(aps), 4)
    report["per_query"] = [
        {
            "query": q,
            **{f"recall@{k}": round(recall_at_k(r, p, k), 4) for k in ks},
            "ap": round(average_precision(r, p), 4),
            "num_relevant": len(r),
            "num_retrieved": len(p),
        }
        for q, r, p in zip(queries, all_relevant, all_retrieved)
    ]
    return report
