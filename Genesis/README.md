# GENESIS (Offline GenAI) — Self-Improving Multi‑Agent RAG System (No API Keys)

GENESIS is **not a chatbot wrapper**. It's an **offline** GenAI system that:
- **Plans → Executes → Critiques → Refines → Judges** in a loop
- Uses **RAG** (FAISS + sentence-transformers) for grounded answers
- Stores **long‑term memory** (SQLite + embeddings) to avoid repeating mistakes
- Runs **fully offline** once you download an open-weight model (GGUF)

✅ No proprietary APIs  
✅ No API keys  
✅ Engineering-first: modular, testable, reproducible

---

## 1) Quick Start (CLI)

### A) Create env
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/mac:
source .venv/bin/activate
pip install -r requirements.txt
```

### B) Download an offline LLM (GGUF)
GENESIS uses `llama-cpp-python` to run GGUF models locally.

Recommended (fast + strong):
- **Mistral 7B Instruct (GGUF)** or **Llama 3 Instruct (GGUF)**

Place your model here:
```
models/
  llm.gguf
```

Then run:
```bash
python -m genesis.cli --model models/llm.gguf --ingest sample_kb --question "Explain RAG and how GENESIS reduces hallucinations."
```

---

## 2) Streamlit UI (Optional)
```bash
streamlit run app/streamlit_app.py
```

---

## 3) What to Submit to Recruiters
- This repo + your demo output in `runs/`
- Include an example run:
  ```bash
  python -m genesis.cli --model models/llm.gguf --ingest sample_kb --question "Summarize the system and propose improvements."
  ```
- The system will generate:
  - `runs/<timestamp>/final_answer.md`
  - `runs/<timestamp>/trace.json`
  - `runs/<timestamp>/metrics.json`

---

## 4) Project Structure
```
GENESIS_offline_genai/
  app/                 # Streamlit UI
  genesis/             # Core package
    agents/            # Planner/Executor/Critic/Refiner/Judge
    rag/               # Ingestion + retrieval
    memory/            # SQLite long-term memory
    eval/              # Lightweight evaluation
    llm/               # llama.cpp backend wrapper
    utils/
  sample_kb/           # Sample knowledge base
  scripts/             # helper scripts
  tests/
  runs/                # generated outputs
  models/              # put your GGUF here (ignored by git)
```

---

## 5) Notes
- If you have no GPU: use a **smaller quant** (e.g., Q4_K_M) and lower context.
- On RTX 1650 Ti: 7B Q4 models are usually reasonable; tune `--n_gpu_layers`.

---

## License
MIT
