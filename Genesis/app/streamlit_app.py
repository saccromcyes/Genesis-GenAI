import time
from pathlib import Path
import streamlit as st

from genesis.orchestrator import RunConfig, run_once

st.set_page_config(page_title="GENESIS Offline GenAI", layout="wide")
st.title("GENESIS — Offline Self‑Improving Multi‑Agent RAG (No API Keys)")

with st.sidebar:
    st.subheader("Model & Runtime")
    model_path = st.text_input("GGUF model path", value="models/llm.gguf")
    n_ctx = st.slider("Context length (n_ctx)", 1024, 8192, 4096, 256)
    n_threads = st.slider("CPU threads", 1, 32, 8, 1)
    n_gpu_layers = st.slider("GPU layers (0=CPU)", 0, 60, 0, 1)

    st.subheader("RAG")
    ingest_folder = st.text_input("Knowledge base folder", value="sample_kb")
    top_k = st.slider("Top-k chunks", 1, 10, 5, 1)

    st.subheader("Self-Improve")
    max_iters = st.slider("Max iterations", 0, 4, 2, 1)
    min_score = st.slider("Min judge score", 0.0, 1.0, 0.78, 0.01)

question = st.text_area("Ask a question", value="Explain how GENESIS reduces hallucinations and stores long-term memory.")
run_btn = st.button("Run GENESIS")

if run_btn:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / ts
    cfg = RunConfig(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        top_k=top_k,
        max_iters=max_iters,
        min_score=min_score,
    )
    with st.spinner("Running agents..."):
        result = run_once(question=question, ingest_folder=Path(ingest_folder), run_dir=run_dir, cfg=cfg)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.success(f"Saved to {result['run_dir']}  |  score={result['score']:.2f}")
        st.markdown("### Final Answer")
        st.markdown(result["final_answer"])
    with col2:
        st.markdown("### Plan")
        st.code(result["plan"])
        st.markdown("### Metrics")
        st.json(result["metrics"])
        st.markdown("### Retrieved Context")
        st.json(result["context_used"])
        st.markdown("### Trace (debug)")
        st.json(result["trace"])
