"""FAISS vector store: build, persist, and search assessment embeddings."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from loguru import logger

from config import settings
from embeddings.embedding_model import GeminiEmbeddingModel


class FAISSVectorStore:
    """Flat-IP (cosine) FAISS index with a JSON metadata sidecar."""

    def __init__(
        self,
        index_path: Path = settings.FAISS_INDEX_PATH,
        meta_path: Path = settings.FAISS_META_PATH,
    ):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.index: faiss.Index | None = None
        self.metadata: list[dict] = []
        self._embedder = GeminiEmbeddingModel()

    def build(self, assessments: list[dict]) -> None:
        """Embed all assessments and build the FAISS index."""
        logger.info(f"Building FAISS index for {len(assessments)} assessments …")
        texts = [a["search_text"] for a in assessments]
        vecs = self._embedder.embed_documents(texts)
        matrix = self._to_matrix(vecs)
        faiss.normalize_L2(matrix)  # cosine similarity via inner product
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)
        self.metadata = assessments
        self._save()
        logger.success(f"FAISS index built: {self.index.ntotal} vectors (dim={matrix.shape[1]})")

    def _save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as fh:
            json.dump(self.metadata, fh, indent=2, ensure_ascii=False)
        logger.info(f"Saved FAISS index → {self.index_path}")
        logger.info(f"Saved metadata    → {self.meta_path}")

    def load(self) -> None:
        """Load a previously saved index from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"No FAISS index at {self.index_path}. Run build() first.")
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "r", encoding="utf-8") as fh:
            self.metadata = json.load(fh)
        logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors")

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.metadata) > 0

    def search(self, query: str, k: int = settings.RETRIEVAL_K) -> list[tuple[dict, float]]:
        """Embed query and return top-k (assessment_dict, similarity_score) pairs."""
        if not self.is_ready:
            raise RuntimeError("Index not loaded. Call load() or build() first.")
        q_vec = np.array([self._embedder.embed_query(query)], dtype="float32")
        faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, min(k, self.index.ntotal))
        return [
            (self.metadata[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        ]

    @staticmethod
    def _to_matrix(embeddings: list[list[float]]) -> np.ndarray:
        return np.array(embeddings, dtype=np.float32)
