import multiprocessing

import subprocess

import sys

import time

import json

import os

from pathlib import Path

import logging

import numpy as np


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

logger = logging.getLogger(__name__)


from drafter.run_tests import (
    run_test,
    run_test_profiler,
    run_no_opt,
    run_nf4,
    run_int8,
    run_vllm,
    run_m,
    run_k,
)


EXPERIMENT_MAP = {
    "test": {"func": run_test, "outputs": ["drafter_output/test.json"]},
    "test_p": {"func": run_test_profiler, "outputs": ["drafter_output/test_profiler.json"]},
    "no_opt": {"func": run_no_opt, "outputs": ["drafter_output/no_opt.json"]},
    "nf4": {"func": run_nf4, "outputs": ["drafter_output/bnb.json"]},
    "int8": {"func": run_int8, "outputs": ["drafter_output/int8.json"]},
    "run_vllm": {"func": run_vllm, "outputs": ["drafter_output/vllm_m5_k2.json"]},
    "run_m": {
        "func": run_m,
        "outputs": [f"drafter_output/vllm_m{m}.json" for m in [5, 10, 15, 20]],
    },
    "run_k": {"func": run_k, "outputs": [f"drafter_output/vllm_k{k}.json" for k in [2, 4, 6, 10]]},
}


VERIFIER_OUTPUT_DIR = Path(os.getenv("RESULTS_PATH", "./verifier_output"))

VERIFIER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


Path("drafter_output").mkdir(parents=True, exist_ok=True)


VERIFIER_MODEL = os.getenv(
    "VERIFIER_MODEL_PATH", "/gcs/standard-rag-results-2026/models/mistral-7b-instruct-v0.1"
)


VERIFY_ONLY_RUNS = {
    "verify_saved": {
        "backup_file": Path("/gcs/speculative-rag-results-2026/drafter_backups/vllm_m5_k2.json"),
        "stem": "vllm_m5_k2",
        "label": "vllm_m5_k2_verify_only",
        "drafter_latency_s": 7973.75,
    },
    "verify_no_opt": {
        "backup_file": VERIFIER_OUTPUT_DIR.parent / "drafter_backups" / "no_opt.json",
        "stem": "no_opt",
        "label": "no_opt_verify_only",
        "drafter_latency_s": float(os.getenv("NO_OPT_DRAFTER_LATENCY_S", "1430.13")),
    },
}


def _drafter_worker(target_func):
    """Isolated worker to run the drafter function and release VRAM."""

    target_func()


def _unpack_verifier_payload(data):
    """Verifier output may be a bare list or {"summary": ..., "results": ...}."""

    if isinstance(data, dict):
        return data.get("results", []), data.get("summary", {})

    return data, {}


def run_verifier_only(run_name: str):

    cfg = VERIFY_ONLY_RUNS[run_name]

    backup_file = cfg["backup_file"]

    verifier_out_path = VERIFIER_OUTPUT_DIR / f"{cfg['stem']}.json"

    logger.info("=== VERIFIER ONLY MODE: Reading %s ===", backup_file)

    if not backup_file.exists():
        raise FileNotFoundError(f"Cannot find saved drafter output at {backup_file}")

    t1 = time.perf_counter()

    cmd = [
        sys.executable,
        "verifier/verifier.py",
        "--input-path",
        str(backup_file),
        "--output-path",
        str(verifier_out_path),
        "--model-name",
        VERIFIER_MODEL,
        "--n-samples",
        "100",
    ]

    subprocess.run(cmd, check=True)

    verifier_latency = time.perf_counter() - t1

    with open(verifier_out_path, "r") as f:
        records, verifier_summary = _unpack_verifier_payload(json.load(f))

    correct_count = sum(1 for r in records if r.get("is_correct", False))

    total_examples = len(records)

    accuracy = (correct_count / total_examples) * 100 if total_examples else 0

    drafter_latency = cfg["drafter_latency_s"]

    total_latency = drafter_latency + verifier_latency

    final_payload = {
        "experiment_name": cfg["label"],
        "n_examples": total_examples,
        "accuracy_em": round(accuracy, 2),
        "total_drafter_latency_s": round(drafter_latency, 2),
        "total_verifier_latency_s": round(verifier_latency, 2),
        "total_pipeline_latency_s": round(total_latency, 2),
        "total_drafter_tok_in": verifier_summary.get("total_drafter_tok_in"),
        "total_drafter_tok_out": verifier_summary.get("total_drafter_tok_out"),
        "total_verifier_tok_in": verifier_summary.get("total_verifier_tok_in"),
        "avg_drafter_tok_in": verifier_summary.get("avg_drafter_tok_in"),
        "avg_drafter_tok_out": verifier_summary.get("avg_drafter_tok_out"),
        "avg_verifier_tok_in": verifier_summary.get("avg_verifier_tok_in"),
        "latency_p50_ms": {
            "retrieve": verifier_summary.get("p50_retrieve_ms"),
            "sample": verifier_summary.get("p50_sample_ms"),
            "draft": verifier_summary.get("p50_draft_ms"),
            "verify": verifier_summary.get("p50_verify_ms"),
            "e2e": verifier_summary.get("p50_e2e_ms"),
        },
        "latency_p95_ms": {
            "retrieve": verifier_summary.get("p95_retrieve_ms"),
            "sample": verifier_summary.get("p95_sample_ms"),
            "draft": verifier_summary.get("p95_draft_ms"),
            "verify": verifier_summary.get("p95_verify_ms"),
            "e2e": verifier_summary.get("p95_e2e_ms"),
        },
        "per_example": records,
    }

    metrics_out_path = verifier_out_path.with_name(f"{cfg['stem']}_metrics.json")

    with open(metrics_out_path, "w") as f:
        json.dump(final_payload, f, indent=2)

    logger.info("=== VERIFICATION COMPLETE | Acc: %.2f%% ===", accuracy)


def evaluate_pipeline(experiment_name: str):

    if experiment_name not in EXPERIMENT_MAP:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    target_func = EXPERIMENT_MAP[experiment_name]["func"]

    expected_outputs = EXPERIMENT_MAP[experiment_name]["outputs"]

    logger.info(f"=== Starting E2E Pipeline for: {experiment_name} ===")

    t0 = time.perf_counter()

    p = multiprocessing.Process(target=_drafter_worker, args=(target_func,))

    p.start()

    p.join()

    if p.exitcode != 0:
        raise RuntimeError(f"Drafter phase crashed with exit code {p.exitcode}. Halting pipeline.")

    drafter_latency = time.perf_counter() - t0

    logger.info(f"Drafter Phase Complete. Total Draft Latency: {drafter_latency:.2f}s")

    import shutil

    backup_dir = VERIFIER_OUTPUT_DIR.parent / "drafter_backups"

    backup_dir.mkdir(parents=True, exist_ok=True)

    if Path("drafter_output").exists():
        shutil.copytree("drafter_output", backup_dir, dirs_exist_ok=True)

        logger.info(f"IMMORTALIZED: Safely backed up raw drafts to {backup_dir}")

    for drafter_file in expected_outputs:
        drafter_out_path = Path(drafter_file)

        if not drafter_out_path.exists():
            logger.warning(f"Expected output file {drafter_out_path} not found. Skipping.")

            continue

        verifier_out_path = VERIFIER_OUTPUT_DIR / drafter_out_path.name

        t1 = time.perf_counter()

        cmd = [
            sys.executable,
            "verifier/verifier.py",
            "--input-path",
            str(drafter_out_path),
            "--output-path",
            str(verifier_out_path),
            "--model-name",
            VERIFIER_MODEL,
        ]

        verifier_profile_dir = str(
            VERIFIER_OUTPUT_DIR / "profiler_traces" / f"{drafter_out_path.stem}_verifier"
        )

        if experiment_name == "run_vllm":
            cmd.append("--use-vllm")

        elif experiment_name == "nf4":
            cmd.extend(["--use-bnb-nf4", "--profile-run", "--profile-dir", verifier_profile_dir])

        elif experiment_name == "int8":
            cmd.extend(["--use-int8", "--profile-run", "--profile-dir", verifier_profile_dir])

        elif experiment_name == "no_opt":
            cmd.extend(["--profile-run", "--profile-dir", verifier_profile_dir])

        subprocess.run(cmd, check=True)

        verifier_latency = time.perf_counter() - t1

        logger.info(f"Verifier logic complete in {verifier_latency:.2f}s")

        with open(verifier_out_path, "r") as f:
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
            "total_drafter_latency_s": round(drafter_latency, 2),
            "total_verifier_latency_s": round(verifier_latency, 2),
            "total_pipeline_latency_s": round(total_latency, 2),
            "total_drafter_tok_in": verifier_summary.get("total_drafter_tok_in"),
            "total_drafter_tok_out": verifier_summary.get("total_drafter_tok_out"),
            "total_verifier_tok_in": verifier_summary.get("total_verifier_tok_in"),
            "avg_drafter_tok_in": verifier_summary.get("avg_drafter_tok_in"),
            "avg_drafter_tok_out": verifier_summary.get("avg_drafter_tok_out"),
            "avg_verifier_tok_in": verifier_summary.get("avg_verifier_tok_in"),
            "latency_p50_ms": {
                "retrieve": verifier_summary.get("p50_retrieve_ms"),
                "sample": verifier_summary.get("p50_sample_ms"),
                "draft": verifier_summary.get("p50_draft_ms"),
                "verify": verifier_summary.get("p50_verify_ms"),
                "e2e": verifier_summary.get("p50_e2e_ms"),
            },
            "latency_p95_ms": {
                "retrieve": verifier_summary.get("p95_retrieve_ms"),
                "sample": verifier_summary.get("p95_sample_ms"),
                "draft": verifier_summary.get("p95_draft_ms"),
                "verify": verifier_summary.get("p95_verify_ms"),
                "e2e": verifier_summary.get("p95_e2e_ms"),
            },
            "per_example": records,
        }

        metrics_out_path = verifier_out_path.with_name(f"{verifier_out_path.stem}_metrics.json")

        with open(metrics_out_path, "w") as f:
            json.dump(final_payload, f, indent=2)

        logger.info(
            f"SUCCESS: {drafter_out_path.stem} | Acc: {accuracy:.2f}% | p95 Draft: {verifier_summary.get('p95_draft_ms', 0):.1f}ms"
        )

    logger.info("=== E2E Pipeline Fully Completed ===")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run", choices=list(EXPERIMENT_MAP.keys()) + list(VERIFY_ONLY_RUNS.keys()), default="test"
    )

    args = parser.parse_args()

    if args.run in VERIFY_ONLY_RUNS:
        run_verifier_only(args.run)

        sys.exit(0)

    evaluate_pipeline(args.run)
