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

import random

import time

from datetime import datetime

from pathlib import Path


import numpy as np

import torch

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


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

default_results = f"output/results_{timestamp}.json"


def _load_retriever(index_path: str, meta_path: str, device: str | None) -> ContrieverRetriever:

    index = FAISSIndex.load(index_path, meta_path)

    return ContrieverRetriever(index, device=device)


def _print_results_table(
    model_name: str, accuracy: float, paper_accuracy: float, p95: float
) -> None:

    table = Table(title="TriviaQA — Standard RAG Results")

    table.add_column("Model", style="cyan")

    table.add_column("TriviaQA (ours)", justify="right")

    table.add_column("TriviaQA (paper)", justify="right")

    table.add_column("P95 Latency (ms)", justify="right")

    table.add_row(model_name, f"{accuracy:.2f}", f"{paper_accuracy:.2f}", f"{p95:.1f}")

    console.print(table)


_PAPER_ACCURACY = {
    "mistralai/Mistral-7B-Instruct-v0.1": 67.11,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 73.91,
}


def _select_samples(samples: list, sample: int | None, sample_seed: int | None) -> tuple[list, str]:

    if sample is None:
        return samples, "all"

    if sample_seed is None:
        return samples[:sample], "first_n"

    rng = random.Random(sample_seed)

    selected = list(samples)

    rng.shuffle(selected)

    return selected[:sample], "random_seed"


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
    sample: int | None = typer.Option(1000, "--sample", "-n", help="Evaluate on N examples"),
    sample_seed: int | None = typer.Option(
        None,
        "--sample-seed",
        help="Shuffle examples with this seed before taking --sample examples.",
    ),
    batch_size: int = typer.Option(32, help="Generation batch size"),
    tensor_parallel_size: int = typer.Option(1, help="Tensor parallel size for vLLM"),
    results_path: str = typer.Option(default_results, help="Output JSON path"),
    profile_dir: str = typer.Option(
        os.getenv("PROFILE_DIR", f"output/profiles_{timestamp}"),
        help="Directory for torch profiler traces",
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

    total_available_examples = len(samples)

    samples, sample_strategy = _select_samples(samples, sample, sample_seed)

    console.print(
        f"Evaluating on {len(samples):,} examples "
        f"(top_k={top_k}, sample_strategy={sample_strategy}, sample_seed={sample_seed})"
    )

    eval_result = EvalResult()

    retrieval_latencies: list[float] = []

    generation_latencies: list[float] = []

    total_generated_tokens = 0

    total_prompt_tokens = 0

    generation_elapsed_s = 0.0

    profiler_stats: dict = {}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    total_start = time.perf_counter()

    for batch_start in tqdm(range(0, len(samples), batch_size), desc="Batches"):
        batch = samples[batch_start : batch_start + batch_size]

        questions = [s.question for s in batch]

        t0 = time.perf_counter()

        passages_batch = retriever.retrieve_batch(questions, top_k=top_k)

        retrieval_latencies.append((time.perf_counter() - t0) * 1000 / len(batch))

        prompts = [build_prompt(q, p) for q, p in zip(questions, passages_batch, strict=True)]

        t0 = time.perf_counter()

        if batch_start == 0:
            responses = generator.generate_with_profiler(prompts, profile_dir)

            profiler_stats = generator.last_profile_stats

        else:
            responses = generator.generate(prompts)

        batch_gen_time_s = time.perf_counter() - t0

        batch_gen_time_ms = batch_gen_time_s * 1000

        generation_elapsed_s += batch_gen_time_s

        generation_latencies.append(batch_gen_time_ms / len(batch))

        total_generated_tokens += generator.last_generation_token_count

        total_prompt_tokens += generator.last_prompt_token_count

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

    total_latencies = [
        r + g for r, g in zip(retrieval_latencies, generation_latencies, strict=True)
    ]

    p50_gen_ms = np.percentile(generation_latencies, 50)

    p95_gen_ms = np.percentile(generation_latencies, 95)

    p50_total_ms = np.percentile(total_latencies, 50)

    p95_total_ms = np.percentile(total_latencies, 95)

    generation_token_throughput = (
        total_generated_tokens / generation_elapsed_s if generation_elapsed_s > 0 else 0.0
    )

    end_to_end_token_throughput = (
        total_generated_tokens / total_elapsed if total_elapsed > 0 else 0.0
    )

    peak_allocated_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0

    paper_acc = _PAPER_ACCURACY.get(model, float("nan"))

    _print_results_table(model.split("/")[-1], accuracy, paper_acc, p95_gen_ms)

    console.print(
        f"\nAvg retrieval latency : {avg_retrieval_ms:.1f} ms/example"
        f"\nAvg generation latency: {avg_generation_ms:.1f} ms/example"
        f"\nP95 generation latency: {p95_gen_ms:.1f} ms/example"
        f"\nP50 total latency     : {p50_total_ms:.1f} ms/example"
        f"\nP95 total latency     : {p95_total_ms:.1f} ms/example"
        f"\nGeneration throughput : {generation_token_throughput:.2f} output tokens/s"
        f"\nPeak GPU allocated    : {peak_allocated_gb:.2f} GB"
        f"\nPeak GPU reserved     : {peak_reserved_gb:.2f} GB"
        f"\nTotal runtime         : {total_elapsed / 60:.1f} min"
    )

    output = {
        "timestamp": timestamp,
        "model": model,
        "split": split,
        "n_examples": eval_result.total,
        "total_available_examples": total_available_examples,
        "sample_requested": sample,
        "sample_strategy": sample_strategy,
        "sample_seed": sample_seed,
        "sample_question_ids": [sample_obj.question_id for sample_obj in samples],
        "top_k": top_k,
        **eval_result.summary(),
        "paper_accuracy": paper_acc,
        "avg_retrieval_latency_ms": round(avg_retrieval_ms, 2),
        "avg_generation_latency_ms": round(avg_generation_ms, 2),
        "p50_generation_latency_ms": round(p50_gen_ms, 2),
        "p95_generation_latency_ms": round(p95_gen_ms, 2),
        "p50_total_latency_ms": round(p50_total_ms, 2),
        "p95_total_latency_ms": round(p95_total_ms, 2),
        "generated_tokens": total_generated_tokens,
        "prompt_tokens": total_prompt_tokens,
        "generation_token_throughput_s": round(generation_token_throughput, 2),
        "end_to_end_token_throughput_s": round(end_to_end_token_throughput, 2),
        "peak_gpu_allocated_gb": round(peak_allocated_gb, 2),
        "peak_gpu_reserved_gb": round(peak_reserved_gb, 2),
        "profile_dir": profile_dir,
        "profiler": profiler_stats,
        "sm_utilization_avg_pct": profiler_stats.get("sm_utilization_avg_pct"),
        "sm_utilization_p50_pct": profiler_stats.get("sm_utilization_p50_pct"),
        "sm_utilization_p95_pct": profiler_stats.get("sm_utilization_p95_pct"),
        "sm_utilization_max_pct": profiler_stats.get("sm_utilization_max_pct"),
        "profiler_cuda_time_coverage_pct": profiler_stats.get("profiler_cuda_time_coverage_pct"),
        "total_runtime_s": round(total_elapsed, 1),
        "per_example": eval_result.details,
    }

    Path(results_path).parent.mkdir(parents=True, exist_ok=True)

    Path(results_path).write_text(json.dumps(output, indent=2))

    console.print(f"Results saved → [cyan]{results_path}[/cyan]")


if __name__ == "__main__":
    app()
