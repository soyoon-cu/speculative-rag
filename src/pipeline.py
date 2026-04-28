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

# Updated Imports (Removing 'rag.' prefix to match your new structure)
from data.loader import iter_samples
from data.preprocess import answer_in_response
from sampling.index import FAISSIndex
from sampling.retriever import ContrieverRetriever
from sampling.multi_perspective import MultiPerspectiveSampler

# Placeholder imports for your teammates' future work
# from drafter.draft_model import Drafter 
# from verifier.verify_model import Verifier

load_dotenv()
logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(add_completion=False)

def _load_retriever(index_path: str, meta_path: str, device: str | None) -> ContrieverRetriever:
    index = FAISSIndex.load(index_path, meta_path)
    return ContrieverRetriever(index, device=device)

@app.command()
def main(
    index_path: str = typer.Option(os.getenv("INDEX_PATH", "/data/faiss_contriever.index")),
    meta_path: str = typer.Option(os.getenv("PASSAGES_META_PATH", "/data/passages_meta.pkl")),
    top_n: int = typer.Option(20, help="Initial retrieval pool size (n)"),
    m_drafts: int = typer.Option(5, help="Number of drafts (m)"),
    k_subset: int = typer.Option(2, help="Docs per subset (k)"),
    split: str = typer.Option("validation", help="TriviaQA split"),
    sample_n: int | None = typer.Option(None, "--sample", "-n"),
    results_path: str = typer.Option("output/results.json"),
    device: str | None = typer.Option(None, help="cuda/cpu"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # 1. Initialize Components
    console.print(f"[bold cyan]Initializing Speculative RAG Components...[/bold cyan]")
    retriever = _load_retriever(index_path, meta_path, device)
    sampler = MultiPerspectiveSampler(device=device)

    # 2. Load Data
    samples = list(iter_samples(split))
    if sample_n is not None:
        samples = samples[:sample_n]
    console.print(f"Evaluating {len(samples)} samples from TriviaQA {split}")

    # 3. Metrics Tracking
    results = []
    total_correct = 0
    latencies = {"retrieval": [], "sampling": [], "total": []}

    # 4. Main Evaluation Loop
    for sample in tqdm(samples, desc="Processing Questions"):
        start_time = time.perf_counter()

        # STEP A: Retrieval (Fetch top-N candidates)
        t0 = time.perf_counter()
        passages = retriever.retrieve(sample.question, top_k=top_n)
        latencies["retrieval"].append(time.perf_counter() - t0)

        # STEP B: Multi-Perspective Sampling (Your Algorithm 1)
        t1 = time.perf_counter()
        subsets = sampler.generate_subsets(
            sample.question, 
            passages, 
            m=m_drafts, 
            k=k_subset
        )
        latencies["sampling"].append(time.perf_counter() - t1)

        # STEP C: Drafting & Verification (MOCK for now)
        # In the final version, this is where you call the Mistral and Mixtral models.
        # For your current test, we assume the Drafter picked a passage 
        # and the Verifier picked the best draft.
        mock_response = "This is a mock answer based on the sampled subsets."
        
        # STEP D: Score Result
        is_correct = answer_in_response(sample.answers, mock_response)
        if is_correct:
            total_correct += 1
            
        latencies["total"].append(time.perf_counter() - start_time)

        results.append({
            "question_id": sample.question_id,
            "question": sample.question,
            "correct": is_correct,
            "subsets_count": len(subsets)
        })

    # 5. Summary & Save
    final_accuracy = (total_correct / len(samples)) * 100
    avg_sampling_ms = (sum(latencies["sampling"]) / len(samples)) * 1000
    
    console.print(f"\n[bold green]Evaluation Complete![/bold green]")
    console.print(f"Accuracy: {final_accuracy:.2f}%")
    console.print(f"Avg Sampling Latency: {avg_sampling_ms:.2f} ms")

    output_data = {
        "accuracy": final_accuracy,
        "avg_latencies_ms": {k: (sum(v)/len(v))*1000 for k, v in latencies.items()},
        "details": results
    }
    
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    app()