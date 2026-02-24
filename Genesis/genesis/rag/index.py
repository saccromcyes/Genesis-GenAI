from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from genesis.utils.text import clean_text, chunk_text

@dataclass
class DocChunk:
    doc_id: str
    chunk_id: int
    text: str
    source_path: str

class RAGIndex:
    def __init__(self, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.embedder = SentenceTransformer(embed_model)
        self.index = None  # faiss.Index
        self.chunks: List[DocChunk] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        embs = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embs, dtype="float32")

    def build_from_folder(self, folder: Path, exts: Tuple[str, ...] = (".md", ".txt")) -> None:
        docs: List[Tuple[str, str, str]] = []
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                docs.append((p.stem, p.read_text(encoding="utf-8", errors="ignore"), str(p)))
        self.chunks = []
        for doc_id, raw, src in docs:
            text = clean_text(raw)
            parts = chunk_text(text)
            for i, part in enumerate(parts):
                self.chunks.append(DocChunk(doc_id=doc_id, chunk_id=i, text=part, source_path=src))

        if not self.chunks:
            raise RuntimeError(f"No documents found in {folder}")

        vecs = self._embed([c.text for c in self.chunks])
        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vecs)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Index not built. Call build_from_folder() first.")
        q = self._embed([query])
        scores, ids = self.index.search(q, k)
        out = []
        for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
            if idx < 0 or idx >= len(self.chunks):
                continue
            c = self.chunks[idx]
            out.append({
                "score": float(score),
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source_path": c.source_path,
            })
        return out

    def save(self, path: Path) -> None:
        if self.index is None:
            raise RuntimeError("No index to save.")
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path.with_suffix(".faiss")))
        meta = {
            "chunks": [c.__dict__ for c in self.chunks],
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        self.index = faiss.read_index(str(path.with_suffix(".faiss")))
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        self.chunks = [DocChunk(**d) for d in meta["chunks"]]
