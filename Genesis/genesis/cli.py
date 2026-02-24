from __future__ import annotations
from pathlib import Path
import time
import typer
from rich import print
from rich.panel import Panel
from rich.console import Console

from genesis.orchestrator import RunConfig, run_once

app = typer.Typer(add_completion=False)
console = Console()

@app.command()
def main(
    model: str = typer.Option(..., help="Path to GGUF model file (offline)."),
    ingest: str = typer.Option("sample_kb", help="Folder to ingest for RAG (md/txt)."),
    question: str = typer.Option(..., help="User question."),
    n_ctx: int = typer.Option(4096, help="Context length."),
    n_threads: int = typer.Option(8, help="CPU threads."),
    n_gpu_layers: int = typer.Option(0, help="GPU layers for llama.cpp (0 = CPU)."),
    top_k: int = typer.Option(5, help="RAG top-k retrieval."),
    max_iters: int = typer.Option(2, help="Max self-improvement iterations."),
    min_score: float = typer.Option(0.78, help="Stop when judge score >= min_score."),
):
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / ts

    cfg = RunConfig(
        model_path=model,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        top_k=top_k,
        max_iters=max_iters,
        min_score=min_score,
    )
    result = run_once(question=question, ingest_folder=Path(ingest), run_dir=run_dir, cfg=cfg)

    console.print(Panel.fit(f"[bold]Run saved to:[/bold] {result['run_dir']}\n[bold]Final score:[/bold] {result['score']:.2f}"))
    print("\n[bold]FINAL ANSWER[/bold]\n")
    print(result["final_answer"])

if __name__ == "__main__":
    app()
