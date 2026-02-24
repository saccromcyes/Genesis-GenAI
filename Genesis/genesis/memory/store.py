from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

@dataclass
class MemoryItem:
    id: int
    question: str
    final_answer: str
    critic_feedback: str
    score: float

class MemoryStore:
    def __init__(self, db_path: Path, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}")
        self.embedder = SentenceTransformer(embed_model)
        self._init_db()

    def _init_db(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                final_answer TEXT NOT NULL,
                critic_feedback TEXT NOT NULL,
                score REAL NOT NULL,
                embedding BLOB NOT NULL
            );
            """))

    def _embed(self, text_in: str) -> bytes:
        v = self.embedder.encode([text_in], normalize_embeddings=True, show_progress_bar=False)[0].astype("float32")
        return v.tobytes()

    def add(self, question: str, final_answer: str, critic_feedback: str, score: float) -> int:
        emb = self._embed(question + "\n" + critic_feedback)
        with self.engine.begin() as conn:
            res = conn.execute(text("""
                INSERT INTO memory(question, final_answer, critic_feedback, score, embedding)
                VALUES (:q, :a, :c, :s, :e)
            """), {"q": question, "a": final_answer, "c": critic_feedback, "s": float(score), "e": emb})
            return int(res.lastrowid)

    def _bytes_to_vec(self, b: bytes) -> np.ndarray:
        return np.frombuffer(b, dtype="float32")

    def search(self, query: str, k: int = 5) -> List[MemoryItem]:
        qv = self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].astype("float32")
        items: List[MemoryItem] = []
        with self.engine.begin() as conn:
            rows = conn.execute(text("SELECT id, question, final_answer, critic_feedback, score, embedding FROM memory")).fetchall()
        if not rows:
            return []

        # brute force cosine sim (small memory)
        scored = []
        for r in rows:
            mv = self._bytes_to_vec(r.embedding)
            sim = float(np.dot(qv, mv) / (np.linalg.norm(qv) * np.linalg.norm(mv) + 1e-9))
            scored.append((sim, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        for sim, r in scored[:k]:
            items.append(MemoryItem(id=r.id, question=r.question, final_answer=r.final_answer, critic_feedback=r.critic_feedback, score=float(r.score)))
        return items
