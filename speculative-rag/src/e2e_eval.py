import multiprocessing
import subprocess
import sys
import time
import json
import os
from pathlib import Path
import logging
import numpy as np # NEW: Required for p50/p95 latency math

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# import drafter test functions
from drafter.run_tests import (
    run_test, 
    run_test_profiler, 
    run_no_opt, 
    run_nf4, 
    run_int8, 
    run_vllm, 
    run_m, 
    run_k
)

EXPERIMENT_MAP = {
    "test":         {"func": run_test,          "outputs": ["drafter_output/test.json"]},
    "test_p":       {"func": run_test_profiler, "outputs": ["drafter_output/test_profiler.json"]},
    "no_opt":       {"func": run_no_opt,        "outputs": ["drafter_output/no_opt.json"]},
    "nf4":          {"func": run_nf4,           "outputs": ["drafter_output/bnb.json"]},
    "int8":         {"func": run_int8,          "outputs": ["drafter_output/int8.json"]},
    "run_vllm":     {"func": run_vllm,          "outputs": ["drafter_output/vllm_m5_k2.json"]},
    "run_m":        {"func": run_m,             "outputs": [f"drafter_output/vllm_m{m}.json" for m in [5, 10, 15, 20]]},
    "run_k":        {"func": run_k,             "outputs": [f"drafter_output/vllm_k{k}.json" for k in [2, 4, 6, 10]]},
}

VERIFIER_OUTPUT_DIR = Path(os.getenv("RESULTS_PATH", "./verifier_output"))
VERIFIER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

Path("drafter_output").mkdir(parents=True, exist_ok=True)

# VERIFIER_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
VERIFIER_MODEL = os.getenv(
    "VERIFIER_MODEL_PATH",
    "/gcs/standard-rag-results-2026/models/mistral-7b-instruct-v0.1"
)

def _drafter_worker(target_func):
    """Isolated worker to run the drafter function and release VRAM."""
    target_func()

def evaluate_pipeline(experiment_name: str):
    if experiment_name not in EXPERIMENT_MAP:
        raise ValueError(f"Unknown experiment: {experiment_name}")
        
    target_func = EXPERIMENT_MAP[experiment_name]["func"]
    expected_outputs = EXPERIMENT_MAP[experiment_name]["outputs"]
    
    logger.info(f"=== Starting E2E Pipeline for: {experiment_name} ===")

    # ---------------------------------------------------------
    # PHASE 1: DRAFTER
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    p = multiprocessing.Process(target=_drafter_worker, args=(target_func,))
    p.start()
    p.join()
    
    if p.exitcode != 0:
        raise RuntimeError(f"Drafter phase crashed with exit code {p.exitcode}. Halting pipeline.")
        
    drafter_latency = time.perf_counter() - t0
    logger.info(f"Drafter Phase Complete. Total Draft Latency: {drafter_latency:.2f}s")

    # AUTO-BACKUP
    import shutil
    backup_dir = VERIFIER_OUTPUT_DIR.parent / "drafter_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if Path("drafter_output").exists():
        shutil.copytree("drafter_output", backup_dir, dirs_exist_ok=True)
        logger.info(f"IMMORTALIZED: Safely backed up raw drafts to {backup_dir}")

    # ---------------------------------------------------------
    # PHASE 2 & 3: VERIFIER & METRICS
    # ---------------------------------------------------------
    for drafter_file in expected_outputs:
        drafter_out_path = Path(drafter_file)
        
        if not drafter_out_path.exists():
            logger.warning(f"Expected output file {drafter_out_path} not found. Skipping.")
            continue
            
        verifier_out_path = VERIFIER_OUTPUT_DIR / drafter_out_path.name
        
        t1 = time.perf_counter()
        cmd = [
            sys.executable, "verifier/verifier.py",
            "--input-path", str(drafter_out_path),
            "--output-path", str(verifier_out_path),
            "--model-name", VERIFIER_MODEL 
        ]

        verifier_profile_dir = str(VERIFIER_OUTPUT_DIR / "profiler_traces" / f"{drafter_out_path.stem}_verifier")
        if experiment_name == "run_vllm":
            cmd.append("--use-vllm")
        elif experiment_name == "nf4":
            cmd.extend(["--use-bnb-nf4", "--profile-run", "--profile-dir", verifier_profile_dir])
        elif experiment_name == "int8":
            cmd.extend(["--use-int8", "--profile-run", "--profile-dir", verifier_profile_dir])
        elif experiment_name == "no_opt":
            cmd.extend(["--profile-run", "--profile-dir", verifier_profile_dir])
        
        # logger.info(f"Launching Verifier + Nsys Profiler on {drafter_out_path}...")
        subprocess.run(cmd, check=True)
        verifier_latency = time.perf_counter() - t1
        logger.info(f"Verifier logic complete in {verifier_latency:.2f}s")

        # --- ADVANCED METRICS PACKAGING ---
        with open(verifier_out_path, 'r') as f:
            data = json.load(f)

        records = data.get("results", data)
        verifier_summary = data.get("summary", {})

        correct_count = sum(1 for r in records if r.get("is_correct", False))
        total_examples = len(records)
        accuracy = (correct_count / total_examples) * 100 if total_examples > 0 else 0
        total_latency = drafter_latency + verifier_latency

        final_payload = {
            "experiment_name": drafter_out_path.stem,
            "n_examples": total_examples,
            "accuracy_em": round(accuracy, 2),

            # --- MACRO TIMING ---
            "total_drafter_latency_s": round(drafter_latency, 2),
            "total_verifier_latency_s": round(verifier_latency, 2),
            "total_pipeline_latency_s": round(total_latency, 2),

            # --- THROUGHPUT (totals + per-question averages) ---
            "total_drafter_tok_in":   verifier_summary.get("total_drafter_tok_in"),
            "total_drafter_tok_out":  verifier_summary.get("total_drafter_tok_out"),
            "total_verifier_tok_in":  verifier_summary.get("total_verifier_tok_in"),
            "avg_drafter_tok_in":     verifier_summary.get("avg_drafter_tok_in"),
            "avg_drafter_tok_out":    verifier_summary.get("avg_drafter_tok_out"),
            "avg_verifier_tok_in":    verifier_summary.get("avg_verifier_tok_in"),

            # --- MICRO TIMING (PERCENTILES) ---
            "latency_p50_ms": {
                "retrieve": verifier_summary.get("p50_retrieve_ms"),
                "sample":   verifier_summary.get("p50_sample_ms"),
                "draft":    verifier_summary.get("p50_draft_ms"),
                "verify":   verifier_summary.get("p50_verify_ms"),
                "e2e":      verifier_summary.get("p50_e2e_ms"),
            },
            "latency_p95_ms": {
                "retrieve": verifier_summary.get("p95_retrieve_ms"),
                "sample":   verifier_summary.get("p95_sample_ms"),
                "draft":    verifier_summary.get("p95_draft_ms"),
                "verify":   verifier_summary.get("p95_verify_ms"),
                "e2e":      verifier_summary.get("p95_e2e_ms"),
            },

            # Keep raw data for custom analysis
            "per_example": records
        }
        
        metrics_out_path = verifier_out_path.with_name(f"{verifier_out_path.stem}_metrics.json")
        with open(metrics_out_path, 'w') as f:
            json.dump(final_payload, f, indent=2)
            
        logger.info(f"SUCCESS: {drafter_out_path.stem} | Acc: {accuracy:.2f}% | p95 Draft: {verifier_summary.get('p95_draft_ms', 0):.1f}ms")

    logger.info("=== E2E Pipeline Fully Completed ===")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=list(EXPERIMENT_MAP.keys()) + ["verify_saved"], default="test")
    args = parser.parse_args()

    # ==========================================
    # THE VERIFIER-ONLY BYPASS
    # ==========================================
    if args.run == "verify_saved":
        logger.info("=== VERIFIER ONLY MODE: Reading from GCS Vault ===")
        
        # 1. Point directly to the cloud backup
        backup_file = Path("/gcs/speculative-rag-results-2026/drafter_backups/vllm_m5_k2.json")
        verifier_out_path = VERIFIER_OUTPUT_DIR / "vllm_m5_k2.json"
        
        if not backup_file.exists():
            logger.error(f"Cannot find backup at {backup_file}! Did the bucket name change?")
            sys.exit(1)

        # 2. Run the Verifier Subprocess
        t1 = time.perf_counter()
        cmd = [
            sys.executable, "verifier/verifier.py",
            "--input-path", str(backup_file),
            "--output-path", str(verifier_out_path),
            "--model-name", VERIFIER_MODEL, 
            "--n-samples", "100" # Explicitly match your test slice
        ]

        subprocess.run(cmd, check=True)
        verifier_latency = time.perf_counter() - t1
        
        # 3. Package the Advanced Metrics
        with open(verifier_out_path, 'r') as f:
            records = json.load(f)
            
        correct_count = sum(1 for r in records if r.get("is_correct", False))
        total_examples = len(records)
        accuracy = (correct_count / total_examples) * 100 if total_examples > 0 else 0
        
        # CRITICAL: We hardcode the 7973.75s from your crashed logs so the Tokens/Sec math remains accurate!
        drafter_latency = 7973.75 
        total_latency = drafter_latency + verifier_latency

        retrieve_lats = [r.get("retrieval_time_s", 0) * 1000 for r in records]
        sample_lats = [r.get("sampling_time_s", 0) * 1000 for r in records]
        draft_lats = [r.get("drafting_time_s", 0) * 1000 for r in records]
        
        prompt_tokens = sum(r.get("drafts_tokens_in", 0) + r.get("verifier_tokens_in", 0) for r in records)
        completion_tokens = sum(r.get("drafts_tokens_out", 0) + r.get("verifier_tokens_out", 0) for r in records)
        total_tokens = prompt_tokens + completion_tokens

        final_payload = {
            "experiment_name": "vllm_m5_k2_verify_only",
            "n_examples": total_examples,
            "accuracy_em": round(accuracy, 2),
            "total_drafter_latency_s": drafter_latency,
            "total_verifier_latency_s": round(verifier_latency, 2),
            "total_pipeline_latency_s": round(total_latency, 2),
            "throughput_examples_per_sec": round(total_examples / total_latency, 2) if total_latency > 0 else 0,
            "throughput_tokens_per_sec": round(total_tokens / total_latency, 2) if total_latency > 0 else 0,
            "total_prompt_tokens": prompt_tokens,
            "total_completion_tokens": completion_tokens,
            "retrieve_latency_ms": {"p50": round(np.percentile(retrieve_lats, 50), 2) if retrieve_lats else 0, "p95": round(np.percentile(retrieve_lats, 95), 2) if retrieve_lats else 0},
            "sample_latency_ms": {"p50": round(np.percentile(sample_lats, 50), 2) if sample_lats else 0, "p95": round(np.percentile(sample_lats, 95), 2) if sample_lats else 0},
            "draft_latency_ms": {"p50": round(np.percentile(draft_lats, 50), 2) if draft_lats else 0, "p95": round(np.percentile(draft_lats, 95), 2) if draft_lats else 0},
        }
        
        metrics_out_path = verifier_out_path.with_name("vllm_m5_k2_metrics.json")
        with open(metrics_out_path, 'w') as f:
            json.dump(final_payload, f, indent=2)
            
        logger.info(f"=== VERIFICATION COMPLETE | Acc: {accuracy:.2f}% ===")
        sys.exit(0) # Stop the script so it doesn't try to run evaluate_pipeline
    
    evaluate_pipeline(args.run)