"""End-to-end Standard RAG pipeline for TriviaQA evaluation.

Entry point: ``uv run python -m rag.pipeline`` (or ``make eval-mistral``).

Standard RAG (as described in Speculative RAG, ICLR 2025):
  1. Retrieve top-k documents from a FAISS vector store.
  2. Concatenate all retrieved documents into a single prompt.
  3. Pass the full prompt to an LLM and generate a final answer.
  4. Evaluate accuracy with the containment metric (Self-RAG convention).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from rag.data.loader import iter_samples
from rag.evaluation.metrics import EvalResult
from rag.generation.prompts import build_prompt
from rag.generation.vllm_server import GenerationConfig, VLLMGenerator
from rag.retrieval.index import FAISSIndex
from rag.retrieval.retriever import ContrieverRetriever

load_dotenv()
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(add_completion=False)


def _load_retriever(index_path: str, meta_path: str, device: str | None) -> ContrieverRetriever:
    index = FAISSIndex.load(index_path, meta_path)
    return ContrieverRetriever(index, device=device)


def _print_results_table(model_name: str, accuracy: float, paper_accuracy: float) -> None:
    table = Table(title="TriviaQA — Standard RAG Results")
    table.add_column("Model", style="cyan")
    table.add_column("TriviaQA (ours)", justify="right")
    table.add_column("TriviaQA (paper)", justify="right")
    table.add_row(model_name, f"{accuracy:.2f}", f"{paper_accuracy:.2f}")
    console.print(table)


_PAPER_ACCURACY = {
    "mistralai/Mistral-7B-Instruct-v0.1": 67.11,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 73.91,
}


@app.command()
def main(
    model: str = typer.Option(
        os.getenv("DRAFTER_MODEL", "mistralai/Mistral-7B-Instruct-v0.1"),
        help="HuggingFace model ID",
    ),
    index_path: str = typer.Option(
        os.getenv("INDEX_PATH", "/data/faiss_contriever.index"),
        help="Path to FAISS index file",
    ),
    meta_path: str = typer.Option(
        os.getenv("PASSAGES_META_PATH", "/data/passages_meta.pkl"),
        help="Path to passages metadata pickle",
    ),
    top_k: int = typer.Option(int(os.getenv("TOP_K", "10")), help="Number of passages to retrieve"),
    max_new_tokens: int = typer.Option(
        int(os.getenv("MAX_NEW_TOKENS", "100")), help="Max tokens to generate"
    ),
    split: str = typer.Option("validation", help="TriviaQA split to evaluate on"),
    sample: int | None = typer.Option(
        None, "--sample", "-n", help="Evaluate on first N examples only"
    ),
    batch_size: int = typer.Option(32, help="Generation batch size"),
    tensor_parallel_size: int = typer.Option(1, help="Tensor parallel size for vLLM"),
    results_path: str = typer.Option(
        os.getenv("RESULTS_PATH", "output/results.json"), help="Output JSON path"
    ),
    device: str | None = typer.Option(None, help="Retriever device (cuda/cpu)"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    console.print(f"[bold]Loading FAISS index[/bold] from {index_path} …")
    retriever = _load_retriever(index_path, meta_path, device)

    console.print(f"[bold]Loading vLLM engine[/bold] for {model} …")
    gen_config = GenerationConfig(
        model=model,
        max_new_tokens=max_new_tokens,
        tensor_parallel_size=tensor_parallel_size,
    )
    generator = VLLMGenerator(gen_config)

    console.print(f"[bold]Streaming TriviaQA[/bold] ({split} split) …")
    samples = list(iter_samples(split))
    if sample is not None:
        samples = samples[:sample]
    console.print(f"Evaluating on {len(samples):,} examples (top_k={top_k})")

    eval_result = EvalResult()
    retrieval_latencies: list[float] = []
    generation_latencies: list[float] = []

    total_start = time.perf_counter()

    for batch_start in tqdm(range(0, len(samples), batch_size), desc="Batches"):
        batch = samples[batch_start : batch_start + batch_size]
        questions = [s.question for s in batch]

        t0 = time.perf_counter()
        passages_batch = retriever.retrieve_batch(questions, top_k=top_k)
        retrieval_latencies.append((time.perf_counter() - t0) * 1000 / len(batch))

        prompts = [build_prompt(q, p) for q, p in zip(questions, passages_batch, strict=True)]

        t0 = time.perf_counter()
        responses = generator.generate(prompts)
        generation_latencies.append((time.perf_counter() - t0) * 1000 / len(batch))

        for sample_obj, response in zip(batch, responses, strict=True):
            eval_result.update(
                question_id=sample_obj.question_id,
                question=sample_obj.question,
                gold_answers=sample_obj.answers,
                model_response=response,
            )

    total_elapsed = time.perf_counter() - total_start
    accuracy = eval_result.accuracy * 100

    avg_retrieval_ms = sum(retrieval_latencies) / len(retrieval_latencies)
    avg_generation_ms = sum(generation_latencies) / len(generation_latencies)

    paper_acc = _PAPER_ACCURACY.get(model, float("nan"))
    _print_results_table(model.split("/")[-1], accuracy, paper_acc)
    console.print(
        f"\nAvg retrieval latency : {avg_retrieval_ms:.1f} ms/example"
        f"\nAvg generation latency: {avg_generation_ms:.1f} ms/example"
        f"\nTotal runtime         : {total_elapsed / 60:.1f} min"
    )

    output = {
        "model": model,
        "split": split,
        "n_examples": eval_result.total,
        "top_k": top_k,
        **eval_result.summary(),
        "paper_accuracy": paper_acc,
        "avg_retrieval_latency_ms": round(avg_retrieval_ms, 2),
        "avg_generation_latency_ms": round(avg_generation_ms, 2),
        "total_runtime_s": round(total_elapsed, 1),
        "per_example": eval_result.details,
    }
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    Path(results_path).write_text(json.dumps(output, indent=2))
    console.print(f"Results saved → [cyan]{results_path}[/cyan]")


if __name__ == "__main__":
    app()
