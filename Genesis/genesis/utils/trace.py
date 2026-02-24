from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import json
import time
from pathlib import Path

@dataclass
class TraceStep:
    name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    meta: Dict[str, Any]

class Tracer:
    def __init__(self) -> None:
        self.steps: List[TraceStep] = []
        self.t0 = time.time()

    def add(self, name: str, input: Dict[str, Any], output: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> None:
        meta = meta or {}
        meta["t_rel_s"] = round(time.time() - self.t0, 4)
        self.steps.append(TraceStep(name=name, input=input, output=output, meta=meta))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "steps": [asdict(s) for s in self.steps],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
