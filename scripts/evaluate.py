"""Run evaluation metrics over the labelled training dataset.

Usage: python scripts/evaluate.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from evaluation.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate the recommendation engine.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to the labelled training JSON (default: config.TRAIN_DATASET_PATH).",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[5, 10],
        help="Cutoff values for Recall@K (default: 5 10).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the JSON report to disk.",
    )
    args = parser.parse_args()

    from config import settings
    dataset_path = args.dataset or settings.TRAIN_DATASET_PATH

    evaluator = Evaluator(dataset_path=dataset_path)
    report = evaluator.run(k_values=args.k, save_report=not args.no_save)

    logger.success(
        "Evaluation complete. "
        "Summary: " + ", ".join(
            f"{k}={v}" for k, v in report.items() if k != "per_query"
        )
    )

if __name__ == "__main__":
    main()
