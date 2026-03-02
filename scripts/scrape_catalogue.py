"""Scrape the SHL product catalogue and save raw JSON.

Usage: python scripts/scrape_catalogue.py
"""

import argparse
import sys
from pathlib import Path

# --- Path setup ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from scraper.shl_scraper import SHLScraper
from scraper.data_processor import DataProcessor

def main():
    parser = argparse.ArgumentParser(
        description="Scrape the SHL product catalogue."
    )
    parser.add_argument(
        "--no-detail",
        action="store_true",
        help="Skip enrichment from individual detail pages (faster, less data).",
    )
    parser.add_argument(
        "--no-process",
        action="store_true",
        help="Skip the data-processing step after scraping.",
    )
    args = parser.parse_args()

    scraper = SHLScraper()

    if scraper.output_path.exists():
        logger.info(f"Cached raw data exists at {scraper.output_path}.")
        ans = input("Re-scrape anyway? [y/N]: ").strip().lower()
        if ans != "y":
            logger.info("Using cached data.")
            if not args.no_process:
                _run_processing()
            return

    logger.info("Starting scrape …")
    data = scraper.scrape(enrich_details=not args.no_detail)
    logger.success(f"Scraped {len(data)} assessments.")

    if not args.no_process:
        _run_processing()

def _run_processing():
    logger.info("Processing scraped data …")
    processor = DataProcessor()
    processed = processor.process()
    logger.success(f"Processed {len(processed)} assessments.")

if __name__ == "__main__":
    main()
