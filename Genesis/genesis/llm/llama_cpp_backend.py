from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
from llama_cpp import Llama

@dataclass
class LLMConfig:
    model_path: str
    n_ctx: int = 4096
    n_threads: int = 8
    n_gpu_layers: int = 0
    temperature: float = 0.4
    top_p: float = 0.9
    max_tokens: int = 600

class OfflineLLM:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.llm = Llama(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=cfg.n_gpu_layers,
            logits_all=False,
            verbose=False,
        )

    def generate(self, system: str, user: str, extra: Optional[Dict[str, Any]] = None) -> str:
        extra = extra or {}
        # Chat format works for most instruct models; if your model needs a different template,
        # adjust here.
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        resp = self.llm.create_chat_completion(
            messages=messages,
            temperature=float(extra.get("temperature", self.cfg.temperature)),
            top_p=float(extra.get("top_p", self.cfg.top_p)),
            max_tokens=int(extra.get("max_tokens", self.cfg.max_tokens)),
        )
        return resp["choices"][0]["message"]["content"].strip()
