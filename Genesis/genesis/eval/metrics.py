from __future__ import annotations
from typing import Dict, Any
import re

def simple_groundedness(answer: str, context: str) -> float:
    # heuristic: percentage of answer sentences containing at least one rare token present in context
    ctx = context.lower()
    sents = [s.strip() for s in re.split(r"[\.!?]\s+", answer) if s.strip()]
    if not sents:
        return 0.0
    hit = 0
    for s in sents:
        toks = [t for t in re.findall(r"[a-zA-Z]{5,}", s.lower())]
        if any(t in ctx for t in toks[:8]):  # only check a few
            hit += 1
    return hit / max(1, len(sents))

def compute_metrics(final_answer: str, score: float, context_text: str, iters: int) -> Dict[str, Any]:
    return {
        "judge_score": float(score),
        "groundedness_heuristic": float(simple_groundedness(final_answer, context_text)),
        "iters": int(iters),
        "answer_chars": int(len(final_answer)),
        "context_chars": int(len(context_text)),
    }
