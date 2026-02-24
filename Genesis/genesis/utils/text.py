from __future__ import annotations
import re
from typing import List

def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    # Simple semantic-ish chunking by paragraphs, then merge to target size.
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)

    # Add overlap by carrying tail words
    if overlap > 0 and len(chunks) > 1:
        out: List[str] = []
        for i, c in enumerate(chunks):
            if i == 0:
                out.append(c)
            else:
                prev = out[-1]
                tail = " ".join(prev.split()[-overlap:]) if len(prev.split()) > overlap else prev
                out.append((tail + "\n\n" + c).strip())
        return out
    return chunks
