import multiprocessing
import subprocess
import sys
import time
import json
import os
from pathlib import Path
import logging

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

# Map each test function to a list of outputs
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

# ADD THIS: Ensure the local drafter directory exists so it doesn't crash on save!
Path("drafter_output").mkdir(parents=True, exist_ok=True)

VERIFIER_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

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
    # PHASE 1: DRAFTER (Run in isolation to clear VRAM)
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    p = multiprocessing.Process(target=_drafter_worker, args=(target_func,))
    p.start()
    p.join()
    
    if p.exitcode != 0:
        raise RuntimeError(f"Drafter phase crashed with exit code {p.exitcode}. Halting pipeline.")
        
    drafter_latency = time.perf_counter() - t0
    logger.info(f"Drafter Phase Complete. Total Draft Latency: {drafter_latency:.2f}s")

    # ==========================================
    # NEW: AUTO-BACKUP THE DRAFTS TO YOUR BUCKET
    # ==========================================
    import shutil
    # This points to /gcs/speculative-rag-results-2026/drafter_backups
    backup_dir = VERIFIER_OUTPUT_DIR.parent / "drafter_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    if Path("drafter_output").exists():
        # Instantly copy the local drafts to the permanent bucket
        shutil.copytree("drafter_output", backup_dir, dirs_exist_ok=True)
        logger.info(f"IMMORTALIZED: Safely backed up raw drafts to {backup_dir}")
    # ==========================================

    # ---------------------------------------------------------
    # PHASE 2 & 3: VERIFIER & METRICS (Loop over all generated files)
    # ---------------------------------------------------------
    # For sweeps like run_m, the Drafter latency is the total time for the whole sweep.
    # log the individual Verifier times for each specific file.
    
    for drafter_file in expected_outputs:
        drafter_out_path = Path(drafter_file)
        
        if not drafter_out_path.exists():
            logger.warning(f"Expected output file {drafter_out_path} not found. Skipping verification for this file.")
            continue
            
        verifier_out_path = VERIFIER_OUTPUT_DIR / drafter_out_path.name
        
        # --- VERIFIER EXECUTION ---
        t1 = time.perf_counter()
        cmd = [
            sys.executable, "verifier/verifier.py",
            "--input-path", str(drafter_out_path),
            "--output-path", str(verifier_out_path),
            "--model-name", VERIFIER_MODEL 
        ]
        
        logger.info(f"Launching Verifier on {drafter_out_path}...")
        subprocess.run(cmd, check=True)
        verifier_latency = time.perf_counter() - t1
        logger.info(f"Verifier logic for {drafter_out_path.name} complete in {verifier_latency:.2f}s")

        # --- METRICS PACKAGING ---
        with open(verifier_out_path, 'r') as f:
            records = json.load(f)
            
        correct_count = sum(1 for r in records if r.get("is_correct", False))
        total_examples = len(records)
        accuracy = (correct_count / total_examples) * 100 if total_examples > 0 else 0
        total_latency = drafter_latency + verifier_latency
        
        final_payload = {
            "experiment_name": drafter_out_path.stem,  # e.g., 'vllm_m10'
            "n_examples": total_examples,
            "accuracy": round(accuracy, 2),
            "drafter_latency_s": round(drafter_latency, 2), # Note: For sweeps, this is cumulative drafting time
            "verifier_latency_s": round(verifier_latency, 2),
            "total_pipeline_latency_s": round(total_latency, 2),
            "throughput_examples_per_sec": round(total_examples / total_latency, 2) if total_latency > 0 else 0,
            "per_example": records
        }
        
        # Save metrics to a safely distinct file
        metrics_out_path = verifier_out_path.with_name(f"{verifier_out_path.stem}_metrics.json")
        with open(metrics_out_path, 'w') as f:
            json.dump(final_payload, f, indent=2)
            
        logger.info(f"SUCCESS: {drafter_out_path.stem} | Acc: {accuracy:.2f}%")

    logger.info("=== E2E Pipeline Fully Completed ===")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=list(EXPERIMENT_MAP.keys()), default="test")
    args = parser.parse_args()
    
    evaluate_pipeline(args.run)