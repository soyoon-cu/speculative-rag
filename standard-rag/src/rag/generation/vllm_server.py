"""vLLM-based LLM engine wrapper for greedy generation with profiling support."""

from __future__ import annotations


import logging

import os

import statistics

import subprocess

import threading

import time

from dataclasses import dataclass, field

from pathlib import Path


import torch

from torch.profiler import ProfilerActivity, profile, record_function


try:
    from vllm import LLM, SamplingParams

except ImportError as _vllm_err:
    LLM = None

    SamplingParams = None


logger = logging.getLogger(__name__)


class NvidiaSmiSampler:
    """Sample GPU utilization while profiler traces are being captured."""

    def __init__(self, interval_ms: int = 200) -> None:

        self.interval_ms = interval_ms

        self.samples: list[int] = []

        self._stop = threading.Event()

        self._thread: threading.Thread | None = None

    def start(self) -> None:

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

        self._thread.start()

    def stop(self) -> dict[str, float | int | None]:

        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=2)

        if not self.samples:
            return {
                "sm_utilization_avg_pct": None,
                "sm_utilization_p50_pct": None,
                "sm_utilization_p95_pct": None,
                "sm_utilization_max_pct": None,
                "sm_utilization_samples": 0,
            }

        sorted_samples = sorted(self.samples)

        p95_index = min(len(sorted_samples) - 1, int(0.95 * (len(sorted_samples) - 1)))

        return {
            "sm_utilization_avg_pct": round(statistics.fmean(self.samples), 2),
            "sm_utilization_p50_pct": round(statistics.median(self.samples), 2),
            "sm_utilization_p95_pct": float(sorted_samples[p95_index]),
            "sm_utilization_max_pct": max(self.samples),
            "sm_utilization_samples": len(self.samples),
        }

    def _sample_loop(self) -> None:

        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ]

        while not self._stop.is_set():
            try:
                output = subprocess.check_output(command, text=True, timeout=2)

                first_line = output.strip().splitlines()[0]

                self.samples.append(int(first_line.strip()))

            except Exception as exc:
                logger.warning("Unable to sample nvidia-smi GPU utilization: %s", exc)

                return

            self._stop.wait(self.interval_ms / 1000)


@dataclass
class GenerationConfig:
    model: str = "mistralai/Mistral-7B-Instruct-v0.1"

    temperature: float = 0.0

    max_new_tokens: int = 100

    tensor_parallel_size: int = 1

    gpu_memory_utilization: float = 0.90

    dtype: str = "float16"

    max_model_len: int = 4096

    hf_token: str | None = field(default_factory=lambda: os.getenv("HF_TOKEN"))


class VLLMGenerator:
    """Thin wrapper around vllm.LLM for batched greedy generation."""

    def __init__(self, config: GenerationConfig) -> None:

        if LLM is None:
            raise RuntimeError(
                "vllm is not installed. Install it on a Linux+CUDA machine: "
                "uv sync (requires sys_platform == 'linux')."
            )

        self.config = config

        if config.hf_token:
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", config.hf_token)

        self._llm = LLM(
            model=config.model,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
            enforce_eager=True,
            trust_remote_code=False,
        )

        self._sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
        )

        self.last_generation_token_count = 0

        self.last_prompt_token_count = 0

        self.last_profile_stats: dict[str, float | int | str | None] = {}

    def generate(self, prompts: list[str]) -> list[str]:
        """Generate responses for a batch of prompts."""

        outputs = self._llm.generate(prompts, self._sampling_params)

        self.last_generation_token_count = sum(
            len(request_output.outputs[0].token_ids) for request_output in outputs
        )

        self.last_prompt_token_count = sum(
            len(getattr(request_output, "prompt_token_ids", []) or []) for request_output in outputs
        )

        return [output.outputs[0].text for output in outputs]

    def generate_with_profiler(self, prompts: list[str], profile_dir: str) -> list[str]:
        """
        Generates responses while capturing hardware traces for SM utilization.
        Outputs TensorBoard events and Chromium traces for your report.
        """

        Path(profile_dir).mkdir(parents=True, exist_ok=True)

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

        logger.info("Profiling TensorBoard traces: %s", profile_dir)

        sampler = NvidiaSmiSampler()

        sampler.start()

        wall_start = time.perf_counter()

        with profile(
            activities=activities,
            record_shapes=True,
            with_flops=True,
            profile_memory=True,
            with_stack=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
        ) as prof:
            with record_function("vllm_generation_step"):
                results = self.generate(prompts)

            prof.step()

        profiler_wall_time_s = time.perf_counter() - wall_start

        sm_stats = sampler.stop()

        chrome_trace_path = str(Path(profile_dir) / "standard_rag_profile_trace.json")

        try:
            prof.export_chrome_trace(chrome_trace_path)

        except Exception as exc:
            logger.warning("Unable to export Chrome profiler trace: %s", exc)

            chrome_trace_path = None

        cuda_time_us = 0

        for event in prof.key_averages():
            cuda_time_us += getattr(event, "self_cuda_time_total", 0)

        cuda_time_s = cuda_time_us / 1_000_000

        cuda_time_coverage_pct = (
            min(100.0, 100.0 * cuda_time_s / profiler_wall_time_s)
            if profiler_wall_time_s > 0
            else 0.0
        )

        self.last_profile_stats = {
            **sm_stats,
            "profiler_wall_time_s": round(profiler_wall_time_s, 4),
            "profiler_self_cuda_time_s": round(cuda_time_s, 4),
            "profiler_cuda_time_coverage_pct": round(cuda_time_coverage_pct, 2),
            "profiler_trace_path": chrome_trace_path,
            "sm_utilization_source": "nvidia-smi sampled during torch profiler window",
        }

        print("\n" + "-" * 70)

        print(f"  GCP A100 Profiler Summary ({len(prompts)} prompts)")

        print("-" * 70)

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

        print("SM utilization samples:", self.last_profile_stats)

        return results
