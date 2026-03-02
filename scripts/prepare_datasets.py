"""Scrape + process SHL catalogue and split into train/test datasets.

Usage: python scripts/prepare_datasets.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from loguru import logger

from config import settings

XLSX_DEFAULT = Path(__file__).resolve().parents[1] / "Gen_AI Dataset.xlsx"

def _parse_labelled_sheet(df: pd.DataFrame) -> list[dict]:
    """
    Parse a labelled sheet.

    Expected columns (case-insensitive):
        query / question / Query
        url / URL / Assessment_url / assessment_url / relevant_url
    """
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Detect query column
    query_col = next(
        (c for c in df.columns if "query" in c or "question" in c), None
    )
    # Detect URL column
    url_col = next(
        (c for c in df.columns if "url" in c), None
    )

    if not query_col or not url_col:
        logger.warning(
            f"Could not auto-detect columns. Available: {list(df.columns)}"
        )
        # Fallback: assume col 0 = query, col 1+ = URLs
        query_col = df.columns[0]
        url_col = df.columns[1] if len(df.columns) > 1 else None

    if not url_col:
        raise ValueError("Cannot detect URL column in labelled sheet.")

    # Group multiple URLs per query
    grouped: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        q = str(row[query_col]).strip()
        u = str(row[url_col]).strip()
        if not q or q.lower() in ("nan", "none", "") :
            continue
        if not u or u.lower() in ("nan", "none", ""):
            continue
        grouped.setdefault(q, []).append(u)

    return [{"query": q, "relevant_urls": urls} for q, urls in grouped.items()]

def _parse_unlabelled_sheet(df: pd.DataFrame) -> list[dict]:
    """Parse an unlabelled test sheet (query column only)."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    query_col = next(
        (c for c in df.columns if "query" in c or "question" in c),
        df.columns[0],
    )
    queries = []
    for _, row in df.iterrows():
        q = str(row[query_col]).strip()
        if q and q.lower() not in ("nan", "none"):
            queries.append({"query": q})
    return queries

def prepare_datasets(xlsx_path: Path) -> None:
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel file not found: {xlsx_path}")

    xl = pd.ExcelFile(xlsx_path)
    sheet_names = xl.sheet_names
    logger.info(f"Sheets found: {sheet_names}")

    out_dir = settings.TRAIN_DATASET_PATH.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Training / labelled sheet ---
    # Heuristic: look for a sheet named 'train', 'labelled', 'training'
    train_sheet = next(
        (s for s in sheet_names if any(k in s.lower() for k in ("train", "label"))),
        sheet_names[0],
    )
    logger.info(f"Using sheet '{train_sheet}' as training data.")
    train_df = pd.read_excel(xlsx_path, sheet_name=train_sheet)
    train_data = _parse_labelled_sheet(train_df)
    with open(settings.TRAIN_DATASET_PATH, "w", encoding="utf-8") as fh:
        json.dump(train_data, fh, indent=2, ensure_ascii=False)
    logger.success(
        f"Written {len(train_data)} training queries → {settings.TRAIN_DATASET_PATH}"
    )

    # --- Test / unlabelled sheet ---
    test_sheet = next(
        (s for s in sheet_names if any(k in s.lower() for k in ("test", "unlabel"))),
        sheet_names[1] if len(sheet_names) > 1 else None,
    )
    if test_sheet and test_sheet != train_sheet:
        logger.info(f"Using sheet '{test_sheet}' as test data.")
        test_df = pd.read_excel(xlsx_path, sheet_name=test_sheet)
        # Test sheet might be labelled too – just extract queries
        test_df.columns = [c.strip().lower().replace(" ", "_") for c in test_df.columns]
        query_col = next(
            (c for c in test_df.columns if "query" in c or "question" in c),
            test_df.columns[0],
        )
        test_data = [
            {"query": str(row[query_col]).strip()}
            for _, row in test_df.iterrows()
            if str(row[query_col]).strip().lower() not in ("nan", "none", "")
        ]
        with open(settings.TEST_DATASET_PATH, "w", encoding="utf-8") as fh:
            json.dump(test_data, fh, indent=2, ensure_ascii=False)
        logger.success(
            f"Written {len(test_data)} test queries → {settings.TEST_DATASET_PATH}"
        )
    else:
        logger.warning("No separate test sheet found. Using placeholder test.json.")

def main():
    parser = argparse.ArgumentParser(
        description="Convert Gen_AI Dataset.xlsx to JSON format."
    )
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=XLSX_DEFAULT,
        help=f"Path to the Excel file (default: {XLSX_DEFAULT}).",
    )
    args = parser.parse_args()
    prepare_datasets(args.xlsx)

if __name__ == "__main__":
    main()
