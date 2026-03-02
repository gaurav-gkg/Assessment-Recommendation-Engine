"""Evaluator: runs the engine over labelled data, computes metrics, saves reports."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.table import Table

from config import settings
from evaluation.metrics import evaluate_system
from recommender.rag_engine import RAGEngine

console = Console()


class Evaluator:
    """Evaluate recommendation quality against a labelled JSON dataset.

    Dataset format: [{"query": "...", "relevant_urls": ["url", ...]}, ...]
    """

    def __init__(self, dataset_path: Path = settings.TRAIN_DATASET_PATH, engine: RAGEngine | None = None):
        self.dataset_path = Path(dataset_path)
        self.engine = engine or RAGEngine()

    def load_dataset(self) -> list[dict]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        with open(self.dataset_path, encoding="utf-8") as fh:
            return json.load(fh)

    def run(self, k_values: list[int] | None = None, save_report: bool = True) -> dict:
        """Run evaluation and return report dict."""
        k_values = k_values or [5, 10]
        dataset = self.load_dataset()
        self.engine.load()

        queries, all_relevant, all_retrieved = [], [], []
        for item in dataset:
            q, relevant = item["query"], item.get("relevant_urls", [])
            logger.info(f"Evaluating: {q[:80]} ...")
            try:
                preds = [r["url"] for r in self.engine.recommend(q, n=max(k_values))]
            except Exception as exc:
                logger.warning(f"Engine error for '{q[:40]}': {exc}")
                preds = []
            queries.append(q)
            all_relevant.append(relevant)
            all_retrieved.append(preds)

        report = evaluate_system(queries, all_relevant, all_retrieved, ks=k_values)
        self._print_report(report, k_values)

        if save_report:
            out = self.dataset_path.parent / "eval_report.json"
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)
            logger.success(f"Report saved -> {out}")

        return report

    def _print_report(self, report: dict, k_values: list[int]) -> None:
        console.rule("[bold cyan]Evaluation Report[/bold cyan]")
        tbl = Table(title="Summary", show_header=True)
        tbl.add_column("Metric", style="cyan")
        tbl.add_column("Value", style="green")
        for k in k_values:
            tbl.add_row(f"Mean Recall@{k}", f"{report.get(f'mean_recall@{k}', 0):.4f}")
            tbl.add_row(f"Mean Precision@{k}", f"{report.get(f'mean_precision@{k}', 0):.4f}")
        tbl.add_row("MAP", f"{report.get('map', 0):.4f}")
        console.print(tbl)


def generate_predictions(
    test_set_path: Path = settings.TEST_DATASET_PATH,
    output_path: Path = settings.PREDICTIONS_PATH,
    engine: RAGEngine | None = None,
    k: int = settings.MAX_RECOMMENDATIONS,
) -> Path:
    """Run engine on unlabelled test queries and write submission CSV (Query, Assessment_url)."""
    if not Path(test_set_path).exists():
        raise FileNotFoundError(f"Test dataset not found: {test_set_path}")

    with open(test_set_path, encoding="utf-8") as fh:
        test_data = json.load(fh)

    _engine = engine or RAGEngine()
    _engine.load()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str]] = []
    for item in test_data:
        q = item["query"]
        logger.info(f"Predicting: {q[:80]} ...")
        try:
            for r in _engine.recommend(q, n=k):
                rows.append((q, r["url"]))
        except Exception as exc:
            logger.warning(f"Prediction error for '{q[:40]}': {exc}")

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Query", "Assessment_url"])
        writer.writerows(rows)

    logger.success(f"Predictions saved -> {output_path}  ({len(rows)} rows)")
    return output_path
