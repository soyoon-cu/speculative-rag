"""Speculative RAG Verifier.

Reads serialized drafter outputs, computes three-component confidence scores
per draft, selects the best answer, and emits all required metrics.

Score (log space, Algorithm 1):
    ρ_final_j = ρ_Draft_j + ρ_SC_j + ρ_SR_j
    ρ_SC  = log P(α_j, β_j | Q)           — self-consistency  (summed, NOT normalised)
    ρ_SR  = log P("Yes" | Q, α_j, β_j, R) — self-reflection

Metrics emitted
───────────────
  EM            — uses answer_in_response() matching drafter's draft-coverage metric
  p50 / p95     — per stage and end-to-end (retrieve + sample + draft + verify)
  Token throughput — drafter (in/out) forwarded from JSON + verifier in (out = 0)
  PyTorch profiler — wraps verifier.select on FIRST question only (matches drafter)
  NVTX           — pipeline.verify pushed/popped per question (nsys timeline)
"""

import json

import time

from pathlib import Path


import numpy as np

import typer

import torch

import torch.nn.functional as F

import torch.cuda.nvtx as nvtx

from rich.console import Console

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from torch.profiler import profile, record_function, ProfilerActivity


console = Console()

app = typer.Typer(add_completion=False)


REFLECTION_STATEMENT = "Do you think the explanation supports the answers? (Yes or No)"


def sync_time() -> float:

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.perf_counter()


def load_data(json_path: str) -> list[dict]:
    """Load drafter JSON — supports both bare list and wrapped {"results": [...]}."""

    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        return data["results"]

    return data


def answer_in_response(gold_answers: list[str], response: str) -> bool:
    """
    FIX 3: mirrors preprocess.answer_in_response exactly — case-insensitive
    substring match — so EM numbers are comparable to draft-coverage metric.
    Replicated here so the file is also standalone-runnable outside the package.
    """

    resp_lower = response.lower().strip()

    return any(gold.lower().strip() in resp_lower for gold in gold_answers)


def _percentile(arr: list[float], p: float) -> float:

    return float(np.percentile(arr, p)) if arr else 0.0


def build_prompt_and_boundaries(
    tokenizer,
    question: str,
    ans_draft: str,
    rationale: str,
) -> tuple[list[int], int, int, int, int, int]:
    """
    Build the full scoring token sequence and return span boundaries.

    Sequence layout
    ───────────────
        [prefix] [alpha_tokens] [beta_tokens] [reflect_+_Yes_tokens]

    Returns
    ───────
    full_token_ids                  complete token id list
    alpha_start, alpha_end          slice [alpha_start:alpha_end) covers α tokens
    beta_start,  beta_end           slice [beta_start:beta_end)   covers β tokens
    yes_idx                         position of the final "Yes" token
    """

    prefix_str = (
        f"[INST] Answer the following question and provide a rationale.\n"
        f"Question: {question} [/INST]\n"
    )

    alpha_str = f"Draft: {ans_draft}\n"

    beta_str = f"Rationale: {rationale}\n"

    reflect_str = f"[INST] {REFLECTION_STATEMENT} [/INST] Yes"

    tok_prefix = tokenizer.encode(prefix_str, add_special_tokens=True)

    tok_alpha = tokenizer.encode(alpha_str, add_special_tokens=False)

    tok_beta = tokenizer.encode(beta_str, add_special_tokens=False)

    tok_reflect = tokenizer.encode(reflect_str, add_special_tokens=False)

    full_token_ids = tok_prefix + tok_alpha + tok_beta + tok_reflect

    alpha_start = len(tok_prefix)

    alpha_end = alpha_start + len(tok_alpha)

    beta_start = alpha_end

    beta_end = beta_start + len(tok_beta)

    yes_idx = len(full_token_ids) - 1

    return full_token_ids, alpha_start, alpha_end, beta_start, beta_end, yes_idx


def score_draft_hf(
    hf_model,
    input_tensor: torch.Tensor,
    alpha_start: int,
    alpha_end: int,
    beta_start: int,
    beta_end: int,
    yes_idx: int,
) -> tuple[float, float]:
    """
    Teacher-forced forward pass → ρ_SC and ρ_SR for one draft.

    Auto-regressive shift
    ─────────────────────
        logits[t] predicts token[t+1]
        → log P(token at position p) uses log_probs_shifted[p - 1]

    FIX 1: ρ_SC is a SUM of log-probs — never divided by token count.
           Normalising changes ranking (systematically favours short rationales).
    FIX 2: ρ_SR uses the stored yes_idx (not hardcoded -1) for consistency
           with the vLLM path and robustness to prompt-format changes.
    """

    with torch.no_grad():
        outputs = hf_model(input_tensor)

        logits = outputs.logits[0]

    log_probs_shifted = F.log_softmax(logits[:-1, :], dim=-1)

    target_ids = input_tensor[0, 1:]

    token_log_probs = (
        torch.gather(log_probs_shifted, 1, target_ids.unsqueeze(-1)).squeeze(-1).cpu().tolist()
    )

    log_p_sc = 0.0

    for p in range(alpha_start, alpha_end):
        if p >= 1:
            log_p_sc += token_log_probs[p - 1]

    for p in range(beta_start, beta_end):
        if p >= 1:
            log_p_sc += token_log_probs[p - 1]

    log_p_sr = token_log_probs[yes_idx - 1] if yes_idx >= 1 else 0.0

    return log_p_sc, log_p_sr


def score_drafts_vllm(
    llm,
    sampling_params,
    question_prompt_token_ids: list[list[int]],
    drafts: list[dict],
) -> list[tuple[float, float]]:
    """
    Submit all m draft prompts at once via vLLM prompt_logprobs.
    Returns list of (log_p_sc, log_p_sr) per draft — FIX 1 applied here too.
    """

    outputs = llm.generate(
        prompt_token_ids=question_prompt_token_ids,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    results = []

    for d_idx, output in enumerate(outputs):
        d = drafts[d_idx]

        prompt_logprobs = output.prompt_logprobs

        token_ids = question_prompt_token_ids[d_idx]

        log_p_sc = 0.0

        for p in range(d["_alpha_start"], d["_alpha_end"]):
            if prompt_logprobs[p] is not None:
                log_p_sc += prompt_logprobs[p][token_ids[p]].logprob

        for p in range(d["_beta_start"], d["_beta_end"]):
            if prompt_logprobs[p] is not None:
                log_p_sc += prompt_logprobs[p][token_ids[p]].logprob

        yes_idx = d["_yes_idx"]

        log_p_sr = (
            prompt_logprobs[yes_idx][token_ids[yes_idx]].logprob
            if prompt_logprobs[yes_idx] is not None
            else 0.0
        )

        results.append((log_p_sc, log_p_sr))

    return results


@app.command()
def main(
    input_path: str = typer.Option("drafter_output/vllm_m5_k2.json", help="Drafter output JSON"),
    output_path: str = typer.Option("verifier_output/final_results.json", help="Output path"),
    model_name: str = typer.Option(
        "mistralai/Mistral-7B-Instruct-v0.1", help="Generalist verifier LM"
    ),
    n_samples: int = typer.Option(1000, help="Questions to process (0 = all)"),
    tensor_parallel_size: int = typer.Option(1, help="GPUs for vLLM"),
    use_vllm: bool = typer.Option(False, help="Use vLLM backend"),
    use_bnb_nf4: bool = typer.Option(False, help="NF4 4-bit quantisation"),
    use_int8: bool = typer.Option(False, help="INT8 quantisation"),
    profile_run: bool = typer.Option(False, help="PyTorch Profiler (1st question only)"),
    profile_dir: str = typer.Option("verifier_output/profiles", help="Profiler trace output dir"),
):

    console.print(f"[bold cyan]Loading Verifier LM ({model_name})[/bold cyan]...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm = None

    hf_model = None

    sampling_params = None

    if use_vllm:
        from vllm import LLM, SamplingParams

        console.print("[bold green]Booting vLLM Backend...[/bold green]")

        llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=4096,
            enforce_eager=True,
            gpu_memory_utilization=0.85,
        )

        sampling_params = SamplingParams(prompt_logprobs=1, max_tokens=1, temperature=0.0)

    else:
        console.print("[bold green]Booting HuggingFace Backend...[/bold green]")

        quant_config = None

        if use_bnb_nf4:
            console.print("  NF4 4-bit quantisation enabled")

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        elif use_int8:
            console.print("  INT8 quantisation enabled")

            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        hf_model.eval()

    records = load_data(input_path)

    if n_samples and n_samples > 0:
        records = records[:n_samples]

    console.print(f"Processing {len(records)} questions...")

    correct_count = 0

    total_questions = len(records)

    total_drafts_evaluated = 0

    retrieve_ms_list = []

    sample_ms_list = []

    draft_ms_list = []

    verify_ms_list = []

    total_drafter_tok_in = 0

    total_drafter_tok_out = 0

    total_verifier_tok_in = 0

    for q_idx, record in enumerate(records):
        question = record["question"]

        drafts = record["drafts"]

        question_prompt_token_ids = []

        verifier_tokens_in = 0

        for d_idx, draft in enumerate(drafts):
            (
                full_token_ids,
                alpha_start,
                alpha_end,
                beta_start,
                beta_end,
                yes_idx,
            ) = build_prompt_and_boundaries(
                tokenizer,
                question,
                draft["answer_draft"],
                draft["rationale"],
            )

            question_prompt_token_ids.append(full_token_ids)

            verifier_tokens_in += len(full_token_ids)

            draft["_alpha_start"] = alpha_start

            draft["_alpha_end"] = alpha_end

            draft["_beta_start"] = beta_start

            draft["_beta_end"] = beta_end

            draft["_yes_idx"] = yes_idx

        nvtx.range_push(f"pipeline.verify qid={record.get('question_id', q_idx)}")

        t0 = sync_time()

        is_first_question = q_idx == 0

        should_profile_hf = profile_run and is_first_question and not use_vllm

        if should_profile_hf:
            Path(profile_dir).mkdir(parents=True, exist_ok=True)

            _prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
            )

            _prof.__enter__()

        if use_vllm:
            sc_sr_pairs = score_drafts_vllm(llm, sampling_params, question_prompt_token_ids, drafts)

            for d_idx, (log_p_sc, log_p_sr) in enumerate(sc_sr_pairs):
                d = drafts[d_idx]

                d["score_sc"] = log_p_sc

                d["score_sr"] = log_p_sr

                d["total_score"] = d["draft_logprob"] + log_p_sc + log_p_sr

                for k in ("_alpha_start", "_alpha_end", "_beta_start", "_beta_end", "_yes_idx"):
                    del d[k]

        else:
            for d_idx, token_ids in enumerate(question_prompt_token_ids):
                d = drafts[d_idx]

                input_tensor = torch.tensor([token_ids]).to(hf_model.device)

                with record_function("verifier.select"):
                    log_p_sc, log_p_sr = score_draft_hf(
                        hf_model,
                        input_tensor,
                        d["_alpha_start"],
                        d["_alpha_end"],
                        d["_beta_start"],
                        d["_beta_end"],
                        d["_yes_idx"],
                    )

                d["score_sc"] = log_p_sc

                d["score_sr"] = log_p_sr

                d["total_score"] = d["draft_logprob"] + log_p_sc + log_p_sr

                for k in ("_alpha_start", "_alpha_end", "_beta_start", "_beta_end", "_yes_idx"):
                    del d[k]

        if should_profile_hf:
            _prof.__exit__(None, None, None)

            console.print(f"[bold magenta]Verifier profiler trace → {profile_dir}[/bold magenta]")

            _prof_obj = getattr(_prof, "profiler", None) or _prof

            print("\n" + "─" * 70)

            print(f"  Verifier profiler summary  ({len(drafts)} drafts)")

            print("─" * 70)

            try:
                print(_prof_obj.key_averages().table(sort_by="cuda_time_total", row_limit=15))

            except Exception:
                pass

        verify_time = sync_time() - t0

        nvtx.range_pop()

        best_draft = max(drafts, key=lambda d: d["total_score"])

        record["selected_draft"] = best_draft["subset_index"]

        is_correct = answer_in_response(record["gold_answers"], best_draft["answer_draft"])

        record["is_correct"] = is_correct

        if is_correct:
            correct_count += 1

        record["verify_time_s"] = verify_time

        verify_ms_list.append(verify_time * 1000)

        retrieve_ms_list.append(record.get("retrieval_time_s", 0.0) * 1000)

        sample_ms_list.append(record.get("sampling_time_s", 0.0) * 1000)

        draft_ms_list.append(record.get("drafting_time_s", 0.0) * 1000)

        record["drafter_tokens_in"] = record.get("drafts_tokens_in", 0)

        record["drafter_tokens_out"] = record.get("drafts_tokens_out", 0)

        record["verifier_tokens_in"] = verifier_tokens_in

        record["verifier_tokens_out"] = 0

        total_drafter_tok_in += record["drafter_tokens_in"]

        total_drafter_tok_out += record["drafter_tokens_out"]

        total_verifier_tok_in += verifier_tokens_in

        total_drafts_evaluated += len(drafts)

        if (q_idx + 1) % 100 == 0:
            console.print(
                f"  [{q_idx + 1}/{total_questions}] EM so far: "
                f"{correct_count / (q_idx + 1) * 100:.2f}%"
            )

    e2e_ms_list = [
        retrieve_ms_list[i] + sample_ms_list[i] + draft_ms_list[i] + verify_ms_list[i]
        for i in range(total_questions)
    ]

    accuracy = correct_count / total_questions * 100 if total_questions else 0.0

    summary = {
        "n_questions": total_questions,
        "em": correct_count / total_questions if total_questions else 0.0,
        "em_hits": correct_count,
        "p50_retrieve_ms": _percentile(retrieve_ms_list, 50),
        "p95_retrieve_ms": _percentile(retrieve_ms_list, 95),
        "p50_sample_ms": _percentile(sample_ms_list, 50),
        "p95_sample_ms": _percentile(sample_ms_list, 95),
        "p50_draft_ms": _percentile(draft_ms_list, 50),
        "p95_draft_ms": _percentile(draft_ms_list, 95),
        "p50_verify_ms": _percentile(verify_ms_list, 50),
        "p95_verify_ms": _percentile(verify_ms_list, 95),
        "p50_e2e_ms": _percentile(e2e_ms_list, 50),
        "p95_e2e_ms": _percentile(e2e_ms_list, 95),
        "total_drafter_tok_in": total_drafter_tok_in,
        "total_drafter_tok_out": total_drafter_tok_out,
        "total_verifier_tok_in": total_verifier_tok_in,
        "total_verifier_tok_out": 0,
        "avg_drafter_tok_in": total_drafter_tok_in / total_questions if total_questions else 0.0,
        "avg_drafter_tok_out": total_drafter_tok_out / total_questions if total_questions else 0.0,
        "avg_verifier_tok_in": total_verifier_tok_in / total_questions if total_questions else 0.0,
        "total_drafts_evaluated": total_drafts_evaluated,
    }

    out_path = Path(output_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(json.dumps({"summary": summary, "results": records}, indent=2))

    console.print("=" * 62)

    console.print("[bold]  Speculative RAG — Verifier Stage Metrics[/bold]")

    console.print("=" * 62)

    console.print(f"  Questions evaluated      : {total_questions}")

    console.print(
        f"  Exact Match (EM)         : {accuracy:.2f}%  ({correct_count}/{total_questions})"
    )

    console.print("")

    console.print("  Latency (p50 / p95) [ms]")

    console.print(
        f"    retrieve               : {summary['p50_retrieve_ms']:8.1f}  /  {summary['p95_retrieve_ms']:.1f}"
    )

    console.print(
        f"    sample                 : {summary['p50_sample_ms']:8.1f}  /  {summary['p95_sample_ms']:.1f}"
    )

    console.print(
        f"    draft                  : {summary['p50_draft_ms']:8.1f}  /  {summary['p95_draft_ms']:.1f}"
    )

    console.print(
        f"    verify                 : {summary['p50_verify_ms']:8.1f}  /  {summary['p95_verify_ms']:.1f}"
    )

    console.print(
        f"    end-to-end             : {summary['p50_e2e_ms']:8.1f}  /  {summary['p95_e2e_ms']:.1f}"
    )

    console.print("")

    console.print("  Token throughput")

    console.print(f"    drafter  in  (total)   : {total_drafter_tok_in:>10,}")

    console.print(f"    drafter  out (total)   : {total_drafter_tok_out:>10,}")

    console.print(f"    verifier in  (total)   : {total_verifier_tok_in:>10,}")

    console.print(f"    drafter  in  (avg/q)   : {summary['avg_drafter_tok_in']:>10.1f}")

    console.print(f"    drafter  out (avg/q)   : {summary['avg_drafter_tok_out']:>10.1f}")

    console.print(f"    verifier in  (avg/q)   : {summary['avg_verifier_tok_in']:>10.1f}")

    console.print(f"  Results saved → {output_path}")

    console.print("=" * 62)


if __name__ == "__main__":
    app()
