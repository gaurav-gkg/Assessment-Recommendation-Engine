"""Cleans and normalises raw scraped data into a canonical schema."""

import json
import re
from pathlib import Path

from loguru import logger

from config import settings

# Canonical test-type mapping (letter codes  →  full names)

TYPE_CODE_MAP: dict[str, str] = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgment",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

# Reverse: full name → code
TYPE_NAME_MAP: dict[str, str] = {v.lower(): k for k, v in TYPE_CODE_MAP.items()}

# Helpers

def _clean_text(text: str) -> str:
    """Remove excess whitespace and non-printable characters."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _normalise_test_types(raw: list[str]) -> list[str]:
    """Return canonical type codes from raw strings (codes or full names)."""
    codes: list[str] = []
    for item in raw:
        item = item.strip()
        if item.upper() in TYPE_CODE_MAP:
            codes.append(item.upper())
        elif item.lower() in TYPE_NAME_MAP:
            codes.append(TYPE_NAME_MAP[item.lower()])
        elif item:
            # Keep as-is (single char) or take first letter
            codes.append(item[0].upper())
    return list(dict.fromkeys(codes))  # deduplicate, preserving order

def _parse_duration_minutes(text: str) -> "int | None":
    """Try to extract an integer minute count from a duration string."""
    m = re.search(r"(\d+)\s*(?:min|minute)", text, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", text)
    if m:
        return int(m.group(1))
    return None

def _build_search_text(assessment: dict) -> str:
    """
    Combine all textual fields into a single string used for embedding.
    This gives the vector store richer context for semantic matching.
    """
    parts = [
        assessment.get("name", ""),
        assessment.get("description", ""),
        " ".join(
            TYPE_CODE_MAP.get(t, t)
            for t in assessment.get("test_type", [])
        ),
        assessment.get("assessment_length", ""),
        " ".join(assessment.get("job_levels", [])),
    ]
    return _clean_text(" | ".join(p for p in parts if p))

# Processor

class DataProcessor:
    """Loads raw scraped data, cleans it, and saves a processed copy."""

    def __init__(
        self,
        raw_path: Path = settings.RAW_DATA_PATH,
        output_path: Path = settings.PROCESSED_DATA_PATH,
    ):
        self.raw_path = Path(raw_path)
        self.output_path = Path(output_path)

    def process(self) -> list[dict]:
        """Full processing pipeline. Returns the list of clean assessments."""
        logger.info(f"Loading raw data from {self.raw_path} …")
        with open(self.raw_path, "r", encoding="utf-8") as fh:
            raw: list[dict] = json.load(fh)

        logger.info(f"Processing {len(raw)} raw assessments …")
        processed = [self._process_one(r) for r in raw if r.get("name") and r.get("url")]

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as fh:
            json.dump(processed, fh, indent=2, ensure_ascii=False)

        logger.success(f"Processed {len(processed)} assessments → {self.output_path}")
        return processed

    def _process_one(self, raw: dict) -> dict:
        """Normalise a single raw assessment record."""
        test_types = _normalise_test_types(raw.get("test_type", []))

        duration_raw = raw.get("assessment_length", "")
        duration_mins = _parse_duration_minutes(duration_raw)

        return {
            "name": _clean_text(raw.get("name", "")),
            "url": raw.get("url", "").strip(),
            "description": _clean_text(raw.get("description", "")),
            "test_type": test_types,
            "test_type_labels": [TYPE_CODE_MAP.get(t, t) for t in test_types],
            "assessment_length": _clean_text(duration_raw),
            "duration_minutes": duration_mins,
            "remote_testing": bool(raw.get("remote_testing", False)),
            "adaptive_irt": bool(raw.get("adaptive_irt", False)),
            "job_levels": raw.get("job_levels", []),
            "languages": raw.get("languages", []),
            "search_text": _build_search_text(raw),
        }

    def load_processed(self) -> list[dict]:
        """Load the already-processed assessments from disk."""
        if not self.output_path.exists():
            raise FileNotFoundError(
                f"No processed data at {self.output_path}. "
                "Run process() first."
            )
        with open(self.output_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
