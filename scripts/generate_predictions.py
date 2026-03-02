"""Generate prediction CSV for unlabelled test queries.

Usage: python scripts/generate_predictions.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from evaluation.evaluator import generate_predictions

def main():
    parser = argparse.ArgumentParser(
        description="Generate prediction CSV for the unlabelled test set."
    )
    parser.add_argument("--test", type=Path, default=None, help="Test set JSON path.")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path.")
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of recommendations per query (default: 10).",
    )
    args = parser.parse_args()

    from config import settings
    test_path = args.test or settings.TEST_DATASET_PATH
    output_path = args.output or settings.PREDICTIONS_PATH

    out = generate_predictions(
        test_set_path=test_path,
        output_path=output_path,
        k=args.k,
    )
    logger.success(f"Predictions CSV written to: {out}")

if __name__ == "__main__":
    main()
