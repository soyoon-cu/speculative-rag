"""Speculative RAG Verifier.

Reads serialized drafts, calculates self-consistency and self-reflection scores
using a Generalist LM via vLLM prompt logprobs, and evaluates the final accuracy.
"""

import json
import time
from pathlib import Path

import typer
import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from rich.console import Console
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from torch.profiler import profile, record_function, ProfilerActivity
import contextlib
from data.preprocess import answer_in_response

console = Console()
app = typer.Typer(add_completion=False)

REFLECTION_STATEMENT = "Do you think the explanation supports the answers? (Yes or No)"

@contextlib.contextmanager
def optional_profiler(enable: bool, out_dir: str):
    """Seamlessly wraps code in PyTorch profiler only if enabled."""
    if enable:
        console.print("[bold magenta]PyTorch Profiler Enabled for Verifier[/bold magenta]")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            yield prof
        
        # Export trace when the block finishes
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            trace_path = f"{out_dir}/verifier_pytorch_trace.json"
            prof.export_chrome_trace(trace_path)
            console.print(f"[bold magenta]Saved Verifier PyTorch Trace -> {trace_path}[/bold magenta]")
    else:
        yield None

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
    use_vllm: bool = typer.Option(False, help="Use vLLM backend"),
    use_bnb_nf4: bool = typer.Option(False, help="Use 4-bit quantization"),
    use_int8: bool = typer.Option(False, help="Use 8-bit quantization"),
    profile_run: bool = typer.Option(False, help="Enable PyTorch Profiler"),
    profile_dir: str = typer.Option("verifier_output/profiles", help="Path to save trace"),
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
            gpu_memory_utilization=0.85
        )
        sampling_params = SamplingParams(prompt_logprobs=1, max_tokens=1, temperature=0.0)
    else:
        console.print("[bold green]Booting Native PyTorch/HF Backend...[/bold green]")
        quant_config = None
        if use_bnb_nf4:
            console.print("Applying NF4 Quantization...")
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        elif use_int8:
            console.print("Applying INT8 Quantization...")
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=quant_config, 
            device_map="auto",
            torch_dtype=torch.float16
        )
        hf_model.eval()

    records = load_data(input_path)
      
    console.print("[bold]Tokenizing prompts and mapping boundaries[/bold]...")
    
    correct_count = 0
    total_questions = len(records)

    total_verifier_time_s = 0.0
    total_drafts_evaluated = 0

    should_profile = profile_run and not use_vllm
    
    with optional_profiler(should_profile, profile_dir):
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

            # Save the Verifier token counts into the record before it gets written to JSON
            record["verifier_tokens_in"] = verifier_tokens_in
            record["verifier_tokens_out"] = verifier_tokens_out

            # Measure Verifier Latency for this question
            t0 = sync_time()
            nvtx.range_push(f"pipeline.verify qid={record.get('question_id', q_idx)}")
            
            if use_vllm:
                # Generate just the m drafts for this question
                outputs = llm.generate(prompt_token_ids=question_prompt_token_ids, sampling_params=sampling_params, use_tqdm=False)
                
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
            else:
                # Wrap the exact forward pass in a PyTorch profiler record marker
                with record_function(f"verifier_hf_forward_qid_{q_idx}"):
                    for d_idx, token_ids in enumerate(question_prompt_token_ids):
                        draft = record["drafts"][d_idx]
                        input_tensor = torch.tensor([token_ids]).to(hf_model.device)
                        
                        with torch.no_grad():
                            outputs = hf_model(input_tensor)
                            logits = outputs.logits[0]
                        
                        log_probs = F.log_softmax(logits[:-1, :], dim=-1)
                        target_ids = input_tensor[0, 1:]
                        token_log_probs = torch.gather(log_probs, 1, target_ids.unsqueeze(-1)).squeeze(-1).cpu().tolist()
                        
                        log_p_sc = 0.0
                        num_sc_tokens = 0
                        
                        for idx in range(draft["_alpha_bounds"][0], draft["_alpha_bounds"][1]):
                            log_p_sc += token_log_probs[idx - 1]
                            num_sc_tokens += 1
                            
                        for idx in range(draft["_beta_bounds"][0], draft["_beta_bounds"][1]):
                            log_p_sc += token_log_probs[idx - 1]
                            num_sc_tokens += 1
                            
                        if num_sc_tokens > 0:
                            log_p_sc = log_p_sc / num_sc_tokens
                            
                        log_p_sr = token_log_probs[-1]
                        
                        draft["score_sc"] = log_p_sc
                        draft["score_sr"] = log_p_sr
                        draft["total_score"] = draft["draft_logprob"] + log_p_sc + log_p_sr
                        del draft["_alpha_bounds"], draft["_beta_bounds"], draft["_yes_idx"]
            nvtx.range_pop()
            verifier_time = sync_time() - t0

            # Save latency to the record
            record["verifier_time_s"] = verifier_time
            total_verifier_time_s += verifier_time           
            total_drafts_evaluated += len(record["drafts"])
        
            # Sort drafts by total_score descending and pick the highest
            best_draft = max(record["drafts"], key=lambda x: x["total_score"])
            record["selected_draft"] = best_draft["subset_index"]
            
            # Exact Match validation
            is_correct = answer_in_response(record["gold_answers"], best_draft["answer_draft"])
            record["is_correct"] = is_correct
            if is_correct:
                correct_count += 1

    accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0
    
    # Save the finalized outputs
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2))
    
    console.print("="*50)
    console.print(f"[bold green]Speculative RAG Accuracy:[/bold green] {accuracy:.2f}% ({correct_count}/{total_questions})")
    console.print(f"Verifier Overhead Latency: {total_verifier_time_s:.2f} seconds")
    console.print(f"Total drafts evaluated   : {total_drafts_evaluated}")
    console.print(f"Results saved to         : {output_path}")
    console.print("="*50)

if __name__ == "__main__":
    app()