"""Load processed assessments, compute Gemini embeddings, build FAISS index.

Usage: python scripts/build_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from scraper.data_processor import DataProcessor
from embeddings.vector_store import FAISSVectorStore

def main():
    # Load processed catalogue data
    processor = DataProcessor()
    try:
        assessments = processor.load_processed()
    except FileNotFoundError:
        logger.warning(
            "Processed data not found. Attempting to run scrape_catalogue.py first …"
        )
        try:
            assessments = processor.process()
        except FileNotFoundError:
            logger.error(
                "Raw scraped data also not found. "
                "Please run: python scripts/scrape_catalogue.py"
            )
            sys.exit(1)

    logger.info(f"Loaded {len(assessments)} processed assessments.")

    # Build FAISS index
    vector_store = FAISSVectorStore()
    vector_store.build(assessments)

    logger.success("FAISS index built successfully.")
    logger.info(
        f"Index saved at:    {vector_store.index_path}\n"
        f"Metadata saved at: {vector_store.meta_path}"
    )

if __name__ == "__main__":
    main()
