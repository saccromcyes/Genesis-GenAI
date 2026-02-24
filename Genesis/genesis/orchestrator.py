from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import time
import json

from genesis.llm.llama_cpp_backend import LLMConfig, OfflineLLM
from genesis.rag.index import RAGIndex
from genesis.memory.store import MemoryStore
from genesis.agents.loop import run_genesis_loop, LoopConfig
from genesis.eval.metrics import compute_metrics
from genesis.utils.trace import Tracer

@dataclass
class RunConfig:
    model_path: str
    n_ctx: int = 4096
    n_threads: int = 8
    n_gpu_layers: int = 0
    top_k: int = 5
    max_iters: int = 2
    min_score: float = 0.78

def run_once(
    question: str,
    ingest_folder: Path,
    run_dir: Path,
    cfg: RunConfig,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    tracer = Tracer()

    # Build RAG
    idx = RAGIndex()
    idx.build_from_folder(ingest_folder)
    retrieved = idx.search(question, k=cfg.top_k)
    context_text = "\n\n".join([r["text"] for r in retrieved])
    tracer.add("rag_retrieve", {"question": question, "top_k": cfg.top_k}, {"hits": [{"score": r["score"], "source": r["source_path"]} for r in retrieved]}, {})

    # Memory
    mem = MemoryStore(run_dir.parent / "memory.sqlite")
    mem_hits = mem.search(question, k=3)
    mem_summaries: List[str] = []
    for m in mem_hits:
        mem_summaries.append(f"Past issue: {m.critic_feedback[:140]}... (score={m.score:.2f})")
    tracer.add("memory_recall", {"question": question}, {"hits": [m.__dict__ for m in mem_hits]}, {})

    # LLM
    llm = OfflineLLM(LLMConfig(
        model_path=cfg.model_path,
        n_ctx=cfg.n_ctx,
        n_threads=cfg.n_threads,
        n_gpu_layers=cfg.n_gpu_layers,
    ))

    out = run_genesis_loop(
        llm=llm,
        question=question,
        context_blocks=retrieved,
        memory_summaries=mem_summaries,
        cfg=LoopConfig(max_iters=cfg.max_iters, min_score=cfg.min_score),
        tracer=tracer,
    )

    # Save artifacts
    (run_dir / "final_answer.md").write_text(out["final_answer"], encoding="utf-8")
    (run_dir / "plan.txt").write_text(out["plan"], encoding="utf-8")
    (run_dir / "retrieved.json").write_text(json.dumps(retrieved, indent=2), encoding="utf-8")
    (run_dir / "trace.json").write_text(json.dumps(out["trace"], indent=2), encoding="utf-8")

    metrics = compute_metrics(out["final_answer"], out["score"], context_text, iters=cfg.max_iters)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Add to memory (learning)
    mem.add(question=question, final_answer=out["final_answer"], critic_feedback=out["critic_feedback"], score=out["score"])

    return {"run_dir": str(run_dir), **out, "metrics": metrics}
