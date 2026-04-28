"""vLLM-based LLM engine wrapper for greedy generation.

Uses vllm.LLM (offline / batch mode) so that the full evaluation loop runs
in a single process without a separate HTTP server.  For online serving the
vLLM OpenAI-compatible server can be used instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from vllm import LLM, SamplingParams
except ImportError as _vllm_err:
    LLM = None  # type: ignore[assignment,misc]
    SamplingParams = None  # type: ignore[assignment,misc]


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

    def generate(self, prompts: list[str]) -> list[str]:
        """Generate responses for a batch of prompts.

        Returns a list of response strings in the same order as ``prompts``.
        """
        outputs = self._llm.generate(prompts, self._sampling_params)
        return [output.outputs[0].text for output in outputs]
