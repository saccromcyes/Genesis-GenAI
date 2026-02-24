# GENESIS  
### Offline Self-Improving Multi-Agent GenAI System (No API Keys Required)

GENESIS is a production-style, fully offline Generative AI system that plans, reasons, critiques itself, refines its responses, and learns from long-term memory — without relying on proprietary APIs.

Unlike typical chatbot demos, GENESIS implements a modular multi-agent architecture combined with Retrieval-Augmented Generation (RAG), evaluation tracking, and persistent memory to produce grounded, traceable AI responses.

---

## Why GENESIS?

Most GenAI projects:
- Just wrap an external API  
- Have no evaluation  
- Have no memory  
- Cannot run offline  

GENESIS:
- Runs entirely offline using open-weight LLMs (GGUF via llama.cpp)  
- Implements a Planner → Executor → Critic → Refiner → Judge loop  
- Uses FAISS-based RAG for grounded responses  
- Stores long-term memory using SQLite + embeddings  
- Produces traceable artifacts (plans, critiques, metrics, traces)  
- Designed with production-grade modular architecture  

---

## System Architecture

User Query  
↓  
RAG Retrieval (FAISS + Sentence Transformers)  
↓  
Planner Agent  
↓  
Executor Agent  
↓  
Critic Agent  
↓  
Refiner Agent  
↓  
Judge Agent (Score 0–1)  
↓  
Memory Storage + Metrics Logging  

---

## Tech Stack

| Layer | Technology |
|--------|------------|
| LLM Runtime | llama-cpp-python |
| Model Format | GGUF (Mistral / LLaMA / Phi) |
| Retrieval | FAISS |
| Embeddings | Sentence Transformers |
| Memory | SQLite |
| Evaluation | Custom groundedness + scoring |
| CLI | Typer |
| UI | Streamlit |
| Packaging | Docker-ready |

---

## Project Structure

```
GENESIS/
│
├── genesis/
│   ├── agents/
│   ├── rag/
│   ├── memory/
│   ├── eval/
│   ├── llm/
│   └── utils/
│
├── app/
├── sample_kb/
├── runs/
├── models/ (not committed)
├── requirements.txt
└── README.md
```

---

## Installation

### Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Download an Open-Weight GGUF Model

Place it inside:

```
models/llm.gguf
```

(Note: Model files are excluded from Git to keep repository lightweight.)

---

## Run via CLI

```bash
python -m genesis.cli \
  --model models/llm.gguf \
  --ingest sample_kb \
  --question "Explain how GENESIS reduces hallucinations."
```

Outputs will be saved in:

```
runs/<timestamp>/
```

Including:
- final_answer.md  
- plan.txt  
- retrieved.json  
- trace.json  
- metrics.json  

---

## Run with UI (Optional)

```bash
streamlit run app/streamlit_app.py
```

---

## valuation & Traceability

GENESIS automatically generates:
- Agent execution traces  
- Retrieval logs  
- Self-critique feedback  
- Judge confidence scores  
- Groundedness heuristics  

This ensures responses are transparent, measurable, and reproducible.

---

## Key Engineering Highlights

- Modular multi-agent orchestration  
- Autonomous self-improvement loop  
- Long-term semantic memory  
- Retrieval-based hallucination reduction  
- Configurable runtime parameters  
- CPU and GPU compatibility  
- Production-ready folder structure  

---

## Future Improvements

- Hybrid BM25 + vector retrieval  
- Fine-tuning pipeline integration  
- Quantization benchmarking  
- Multi-modal (vision + text) support  
- Distributed deployment  

---
