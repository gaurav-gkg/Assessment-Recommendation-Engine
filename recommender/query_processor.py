"""Converts any input (text, JD, URL) into an enriched query for the RAG engine."""
from __future__ import annotations

import re
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from loguru import logger

from config import settings

_client = genai.Client(api_key=settings.GOOGLE_API_KEY)


def _is_url(text: str) -> bool:
    try:
        p = urlparse(text.strip())
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


def _fetch_text_from_url(url: str, timeout: int = 20) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"}
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
    except Exception as exc:
        raise ValueError(f"Could not fetch URL '{url}': {exc}") from exc
    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True)).strip()[:8000]


_ENRICHMENT_PROMPT = (
    "You are an expert I/O psychologist and talent-assessment specialist.\n\n"
    "Given the input below (hiring query or job description), extract and summarise "
    "the key requirements for selecting the right SHL assessments.\n\n"
    "Return a concise summary (3-5 sentences) covering:\n"
    "- Required cognitive abilities or aptitude areas\n"
    "- Relevant personality or behavioural traits\n"
    "- Technical skills or knowledge domains\n"
    "- Job level (entry / mid / senior / executive)\n"
    "- Any other context useful for assessment selection\n\n"
    "Input:\n{input_text}\n\nSummary:"
)


def _enrich_query(raw_text: str) -> str:
    """Use Gemini to extract an assessment-selection summary from raw text."""
    try:
        response = _client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=_ENRICHMENT_PROMPT.format(input_text=raw_text[:4000]),
            config=types.GenerateContentConfig(
                temperature=settings.LLM_TEMPERATURE,
                max_output_tokens=512,
            ),
        )
        return response.text.strip()
    except Exception as exc:
        logger.warning(f"Query enrichment failed, using raw text: {exc}")
        return raw_text


class QueryProcessor:
    """Normalises and optionally enriches any input before RAG retrieval."""

    def __init__(self, enrich: bool = True):
        self.enrich = enrich

    def process(self, raw_input: str) -> str:
        """Return enriched query text ready for embedding."""
        raw_input = raw_input.strip()
        if not raw_input:
            raise ValueError("Input query is empty.")
        if _is_url(raw_input):
            logger.info(f"Input is a URL - fetching content from {raw_input}")
            raw_input = _fetch_text_from_url(raw_input)
        if self.enrich:
            logger.info("Enriching query with Gemini ...")
            enriched = _enrich_query(raw_input)
            logger.debug(f"Enriched: {enriched[:200]} ...")
            return enriched
        return raw_input