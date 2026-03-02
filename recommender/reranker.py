"""Reranker: LLM-based pointwise scoring + category-balance enforcement."""
from __future__ import annotations

import json
import re
from collections import Counter

from google import genai
from google.genai import types
from loguru import logger

from config import settings

_client = genai.Client(api_key=settings.GOOGLE_API_KEY)

# max fraction a single test-type category may occupy in the final list
MAX_SINGLE_CATEGORY_FRACTION = 0.7

_RERANK_PROMPT = (
    "You are an expert talent-assessment consultant at SHL.\n\n"
    "A recruiter submitted this hiring query:\n\"{query}\"\n\n"
    "Below are {n} candidate SHL assessments (JSON list).\n"
    "For each, assign a relevance score 0-10 (10 = perfectly relevant).\n\n"
    "Return ONLY a JSON array in the same order:\n"
    "[{{\"index\": 0, \"score\": <float>}}, ...]\n\n"
    "Assessments:\n{assessments_json}"
)


def _llm_rerank(query: str, candidates: list) -> list:
    """Ask Gemini to score candidates; falls back to FAISS order on failure."""
    prompt_items = [
        {
            "index": i,
            "name": c["name"],
            "description": c.get("description", "")[:300],
            "test_type": c.get("test_type_labels", c.get("test_type", [])),
        }
        for i, (c, _) in enumerate(candidates)
    ]
    prompt = _RERANK_PROMPT.format(
        query=query,
        n=len(candidates),
        assessments_json=json.dumps(prompt_items, indent=2),
    )
    try:
        response = _client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=1024),
        )
        raw = response.text.strip()
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON array in LLM response")
        score_map = {item["index"]: float(item["score"]) for item in json.loads(json_match.group())}
        return [c for _, c in sorted(enumerate(candidates), key=lambda x: score_map.get(x[0], 0.0), reverse=True)]
    except Exception as exc:
        logger.warning(f"LLM reranking failed, using FAISS order: {exc}")
        return candidates


def _balance_results(candidates: list, target_count: int) -> list:
    """Swap dominant-category items for alternatives until no category exceeds 70%."""
    if not candidates:
        return candidates

    selected = list(candidates[:target_count])
    reserve = list(candidates[target_count:])

    def _dominant(items):
        counts: Counter = Counter(t for a, _ in items for t in a.get("test_type", []))
        return counts.most_common(1)[0] if counts else ("", 0)

    for _ in range(10):
        cat, count = _dominant(selected)
        if not cat or count / len(selected) <= MAX_SINGLE_CATEGORY_FRACTION:
            break
        worst_idx = max(
            (i for i, (a, _) in enumerate(selected) if cat in a.get("test_type", [])),
            key=lambda i: -selected[i][1],
            default=None,
        )
        if worst_idx is None:
            break
        reserve.insert(0, selected.pop(worst_idx))
        alt_idx = next(
            (i for i, (a, _) in enumerate(reserve) if cat not in a.get("test_type", [])),
            None,
        )
        if alt_idx is None:
            selected.append(reserve.pop(0))
            break
        selected.append(reserve.pop(alt_idx))

    return selected


class Reranker:
    """Post-processes FAISS candidates into a balanced recommendation list."""

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm

    def rerank(self, query: str, candidates: list, n: int = settings.MAX_RECOMMENDATIONS) -> list:
        n = max(settings.MIN_RECOMMENDATIONS, min(n, settings.MAX_RECOMMENDATIONS))
        if not candidates:
            return []
        if self.use_llm:
            logger.info("Running LLM reranking ...")
            candidates = _llm_rerank(query, candidates)
        logger.info("Applying category balance ...")
        return _balance_results(candidates, n)[:n]
