from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json

from genesis.llm.llama_cpp_backend import OfflineLLM
from genesis.agents.prompts import PLANNER_SYS, EXECUTOR_SYS, CRITIC_SYS, REFINER_SYS, JUDGE_SYS
from genesis.utils.trace import Tracer

@dataclass
class LoopConfig:
    max_iters: int = 2
    min_score: float = 0.78

def _safe_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        # attempt to extract JSON object
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                pass
    return {"score": 0.0, "rationale": "Failed to parse judge output."}

def run_genesis_loop(
    llm: OfflineLLM,
    question: str,
    context_blocks: Optional[List[Dict[str, Any]]] = None,
    memory_summaries: Optional[List[str]] = None,
    cfg: Optional[LoopConfig] = None,
    tracer: Optional[Tracer] = None,
) -> Dict[str, Any]:
    cfg = cfg or LoopConfig()
    tracer = tracer or Tracer()
    context_blocks = context_blocks or []
    memory_summaries = memory_summaries or []

    ctx_text = ""
    if context_blocks:
        lines = []
        for i, c in enumerate(context_blocks, 1):
            src = c.get("source_path", "")
            fname = src.split("/")[-1] if src else "unknown"
            lines.append(f"[{i}] {fname}\n{c['text']}")
        ctx_text = "\n\n".join(lines)

    mem_text = ""
    if memory_summaries:
        mem_text = "\n".join([f"- {m}" for m in memory_summaries])

    planner_user = f"""QUESTION:
{question}

CONTEXT (optional):
{ctx_text if ctx_text else "(none)"}

PAST MEMORY (optional):
{mem_text if mem_text else "(none)"}
"""
    plan = llm.generate(PLANNER_SYS, planner_user)
    tracer.add("planner", {"question": question, "context_n": len(context_blocks)}, {"plan": plan}, {})

    draft_user = f"""PLAN:
{plan}

QUESTION:
{question}

CONTEXT (use if relevant):
{ctx_text if ctx_text else "(none)"}

PAST MEMORY (use to avoid repeating mistakes):
{mem_text if mem_text else "(none)"}
"""
    draft = llm.generate(EXECUTOR_SYS, draft_user)
    tracer.add("executor", {"plan": plan}, {"draft": draft}, {})

    best = draft
    best_score = 0.0
    last_crit = ""

    for it in range(cfg.max_iters):
        critic_user = f"""DRAFT ANSWER:
{best}

CONTEXT:
{ctx_text if ctx_text else "(none)"}

Now critique it."""
        critique = llm.generate(CRITIC_SYS, critic_user)
        tracer.add("critic", {"iter": it}, {"critique": critique}, {})

        refine_user = f"""CRITIQUE:
{critique}

Rewrite the answer.

QUESTION:
{question}

CONTEXT:
{ctx_text if ctx_text else "(none)"}
"""
        refined = llm.generate(REFINER_SYS, refine_user)
        tracer.add("refiner", {"iter": it}, {"refined": refined}, {})

        judge_user = f"""QUESTION:
{question}

ANSWER:
{refined}

Return JSON only."""
        judge_raw = llm.generate(JUDGE_SYS, judge_user, extra={"temperature": 0.2, "max_tokens": 200})
        judge = _safe_json(judge_raw)
        score = float(judge.get("score", 0.0))
        tracer.add("judge", {"iter": it}, {"judge_raw": judge_raw, "score": score, "rationale": judge.get("rationale", "")}, {})

        if score > best_score:
            best_score = score
            best = refined
            last_crit = critique

        if best_score >= cfg.min_score:
            break

    return {
        "plan": plan,
        "final_answer": best,
        "score": best_score,
        "critic_feedback": last_crit,
        "context_used": context_blocks,
        "trace": tracer.to_dict(),
    }
