"""Speculative RAG Verifier.

Reads serialized drafts, calculates self-consistency and self-reflection scores
using a Generalist LM via vLLM prompt logprobs, and evaluates the final accuracy.
"""

import json
import time
from pathlib import Path

import typer
import torch
import torch.cuda.nvtx as nvtx
from rich.console import Console
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

console = Console()
app = typer.Typer(add_completion=False)

REFLECTION_STATEMENT = "Do you think the explanation supports the answers? (Yes or No)"

def sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

def load_data(json_path: str) -> list[dict]:
    with open(json_path, 'r') as f:
        return json.load(f)

@app.command()
def main(
    input_path: str = typer.Option("drafter_output/vllm_m5_k2.json", help="Path to drafter output"),
    output_path: str = typer.Option("verifier_output/final_results.json", help="Path to save final scored results"),
    model_name: str = typer.Option("mistralai/Mistral-7B-Instruct-v0.1", help="Generalist LM model ID"),
    tensor_parallel_size: int = typer.Option(1, help="GPUs to use for the Verifier"),
):
    console.print(f"[bold cyan]Loading Verifier LM ({model_name})[/bold cyan]...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)
    llm = LLM(
        model=model_name, 
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=4096,           # Restrict context window to 4K tokens
        enforce_eager=True,           # Disable CUDA graphs to save 1-3 GiB VRAM
        gpu_memory_utilization=0.85   # Cap VRAM usage to leave breathing room
    )

    # We want prompt logprobs, but we don't want the model to generate any new text
    sampling_params = SamplingParams(prompt_logprobs=1, max_tokens=1, temperature=0.0)
    
    records = load_data(input_path)
    
    all_prompt_token_ids = []
    draft_mappings = [] # Tracks (Question_Index, Draft_Index, original_draft_logprob, bounds)
    
    console.print("[bold]Tokenizing prompts and mapping boundaries[/bold]...")
    
    correct_count = 0
    total_questions = len(records)

    total_verifier_time_s = 0.0
    total_drafts_evaluated = 0

    for q_idx, record in enumerate(records):
        question = record["question"]
        question_prompt_token_ids = []

        # Initialize Verifier token counters for this question
        verifier_tokens_in = 0
        verifier_tokens_out = 0
        
        for d_idx, draft in enumerate(record["drafts"]):
            ans_draft = draft["answer_draft"]
            rationale = draft["rationale"]
            
            # 1. Build the prompt chunks with Mistral-Instruct formatting
            prefix_str = f"[INST] Answer the following question and provide a rationale.\nQuestion: {question} [/INST]\n"
            alpha_str = f"Draft: {ans_draft}\n"
            beta_str = f"Rationale: {rationale}\n"
            reflect_str = f"[INST] {REFLECTION_STATEMENT} [/INST] Yes"
            
            # 2. Tokenize chunks independently (avoiding BOS token repetition where necessary)
            tok_prefix = tokenizer.encode(prefix_str, add_special_tokens=True)
            tok_alpha = tokenizer.encode(alpha_str, add_special_tokens=False)
            tok_beta = tokenizer.encode(beta_str, add_special_tokens=False)
            tok_reflect = tokenizer.encode(reflect_str, add_special_tokens=False)
            
            # 3. Concatenate to form the final prompt
            full_token_ids = tok_prefix + tok_alpha + tok_beta + tok_reflect
            question_prompt_token_ids.append(full_token_ids)

            # Add this draft's tokens to the question's total
            verifier_tokens_in += len(full_token_ids)
            verifier_tokens_out += 1 # The Verifier only generates 1 token ("Yes") per draft
            
            # Map boundaries for scoring later
            record["drafts"][d_idx]["_alpha_bounds"] = (len(tok_prefix), len(tok_prefix) + len(tok_alpha))
            record["drafts"][d_idx]["_beta_bounds"] = (len(tok_prefix) + len(tok_alpha), len(tok_prefix) + len(tok_alpha) + len(tok_beta))
            record["drafts"][d_idx]["_yes_idx"] = len(full_token_ids) - 1

            # 4. Map the exact index boundaries for scoring
        #     alpha_start = len(tok_prefix)
        #     alpha_end = alpha_start + len(tok_alpha)
            
        #     beta_start = alpha_end
        #     beta_end = beta_start + len(tok_beta)
            
        #     reflect_yes_idx = len(full_token_ids) - 1 # The very last token is "Yes"
            
        #     all_prompt_token_ids.append(full_token_ids)
        #     draft_mappings.append({
        #         "q_idx": q_idx,
        #         "d_idx": d_idx,
        #         "draft_logprob": draft["draft_logprob"],
        #         "alpha_bounds": (alpha_start, alpha_end),
        #         "beta_bounds": (beta_start, beta_end),
        #         "yes_idx": reflect_yes_idx
        #     })
        # # Save the Verifier token counts into the record before it gets written to JSON
        record["verifier_tokens_in"] = verifier_tokens_in
        record["verifier_tokens_out"] = verifier_tokens_out

    # console.print(f"[bold yellow]Executing Batched Verification Pass ({len(all_prompt_token_ids)} drafts)[/bold yellow]...")

        # Measure Verifier Latency for this question
        t0 = sync_time()
        nvtx.range_push(f"pipeline.verify qid={record.get('question_id', q_idx)}")
        
        # Generate just the m drafts for this question
        outputs = llm.generate(prompt_token_ids=question_prompt_token_ids, sampling_params=sampling_params, use_tqdm=False)
        
        nvtx.range_pop()
        verifier_time = sync_time() - t0

        # Save latency to the record
        record["verifier_time_s"] = verifier_time
        total_verifier_time_s += verifier_time           
        total_drafts_evaluated += len(record["drafts"])
    
    # console.print("[bold]Extracting LogProbs and Computing Final Scores[/bold]...")
        for d_idx, output in enumerate(outputs):
            draft = record["drafts"][d_idx]
            prompt_logprobs = output.prompt_logprobs
            token_ids = question_prompt_token_ids[d_idx]
            
            log_p_sc = 0.0
            
            # Sum alpha and beta logprobs
            for idx in range(draft["_alpha_bounds"][0], draft["_alpha_bounds"][1]):
                if prompt_logprobs[idx] is not None:
                    log_p_sc += prompt_logprobs[idx][token_ids[idx]].logprob

            for idx in range(draft["_beta_bounds"][0], draft["_beta_bounds"][1]):
                if prompt_logprobs[idx] is not None:
                    log_p_sc += prompt_logprobs[idx][token_ids[idx]].logprob
                    
            # Extract Yes logprob
            yes_idx = draft["_yes_idx"]
            log_p_sr = prompt_logprobs[yes_idx][token_ids[yes_idx]].logprob if prompt_logprobs[yes_idx] else 0.0
            
            total_score = draft["draft_logprob"] + log_p_sc + log_p_sr
            
            draft["score_sc"] = log_p_sc
            draft["score_sr"] = log_p_sr
            draft["total_score"] = total_score
            
            # Cleanup temporary mapping keys
            del draft["_alpha_bounds"], draft["_beta_bounds"], draft["_yes_idx"]

    # for i, output in enumerate(outputs):
    #     mapping = draft_mappings[i]
    #     q_idx, d_idx = mapping["q_idx"], mapping["d_idx"]
    #     prompt_logprobs = output.prompt_logprobs
    #     token_ids = all_prompt_token_ids[i]
        
    #     log_p_sc = 0.0
        
    #     # Calculate log(p_SC) by summing probabilities of alpha (draft) and beta (rationale) tokens
    #     # Note: vLLM's prompt_logprobs[0] is always None (no preceding context for the first token)
    #     for idx in range(mapping["alpha_bounds"][0], mapping["alpha_bounds"][1]):
    #         if prompt_logprobs[idx] is not None:
    #             tok_id = token_ids[idx]
    #             log_p_sc += prompt_logprobs[idx][tok_id].logprob

    #     for idx in range(mapping["beta_bounds"][0], mapping["beta_bounds"][1]):
    #         if prompt_logprobs[idx] is not None:
    #             tok_id = token_ids[idx]
    #             log_p_sc += prompt_logprobs[idx][tok_id].logprob
                
    #     # Calculate log(p_SR) from the final "Yes" token
    #     yes_idx = mapping["yes_idx"]
    #     yes_tok_id = token_ids[yes_idx]
    #     log_p_sr = prompt_logprobs[yes_idx][yes_tok_id].logprob if prompt_logprobs[yes_idx] else 0.0
        
    #     # Calculate Total Score: \rho_j = \rho_{Draft,j} * \rho_{SC} * \rho_{SR} (in log space)
    #     # log_p_sc = log_p_sc / num_sc_tokens
    #     total_score = mapping["draft_logprob"] + log_p_sc + log_p_sr
        
    #     records[q_idx]["drafts"][d_idx]["score_sc"] = log_p_sc
    #     records[q_idx]["drafts"][d_idx]["score_sr"] = log_p_sr
    #     records[q_idx]["drafts"][d_idx]["total_score"] = total_score

    # Evaluation phase
    
    # for record in records:
        # Sort drafts by total_score descending and pick the highest
        best_draft = max(record["drafts"], key=lambda x: x["total_score"])
        record["selected_draft"] = best_draft["subset_index"]
        
        # Exact Match validation
        is_correct = any(gold.lower() in best_draft["answer_draft"].lower() for gold in record["gold_answers"])
        record["is_correct"] = is_correct
        if is_correct:
            correct_count += 1

    accuracy = (correct_count / total_questions) * 100
    
    # Save the finalized outputs
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2))
    
    console.print("="*50)
    console.print(f"[bold green]Speculative RAG Accuracy:[/bold green] {accuracy:.2f}% ({correct_count}/{total_questions})")
    console.print(f"Verifier Overhead Latency: {total_verifier_time_s:.2f} seconds")
    console.print(f"Total drafts evaluated   : {len(all_prompt_token_ids)}")
    console.print(f"Results saved to         : {output_path}")
    console.print("="*50)

if __name__ == "__main__":
    app()