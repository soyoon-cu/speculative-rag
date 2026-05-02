
from __future__ import annotations

'''
Specialist RAG drafter : generates m draft (answer, rationale) pairs from m document subsets
in a SINGLE batched forward pass
'''


import torch.cuda.nvtx as nvtx
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

from draft_output import DraftOutput
from vllm_ import VLLM



MODEL_MISTRAL_7B   = "mistralai/Mistral-7B-v0.1"


class BatchedDrafter:
    def __init__(
        self, device,
        max_new_tokens,
        max_input_len ,
        use_vllm,
        use_bnb_nf4,
        use_int8,
        DO_SAMPLE,
        TEMPERATURE,
        model_name = None
    ):
        
        self.device = device
        self.DO_SAMPLE = DO_SAMPLE
        self.TEMPERATURE = TEMPERATURE
        self.use_int8 = use_int8
        
        self.is_cuda = device.type == 'cuda'
        self.is_mps = device.type == 'mps'
        self.is_cpu = device.type == 'cpu'

        if model_name is None:
            model_name = MODEL_MISTRAL_7B 

        self.max_input_len = max_input_len
        self.max_new_tokens = max_new_tokens

        self.use_vllm = use_vllm and self.is_cuda
        self.use_bnb_nf4 = use_bnb_nf4 and self.is_cuda

        if use_vllm and not self.use_vllm:
            logger.warning("use_vllm=True ignored: vLLM unavailable or device is not CUDA.")
        if use_bnb_nf4 and not self.use_bnb_nf4:
            logger.warning("use_bnb_nf4=True ignored: bitsandbytes unavailable or device is not CUDA.")

        logger.info(
            'BatchedDrafter : device = %s | model = %s | vllm = %s | bnb_nf4 = %s',
            self.device, model_name, self.use_vllm, self.use_bnb_nf4
        )
        if self.use_vllm:
            logger.info('Using vLLM...')
            self._vllm = VLLM()
            self.vllm_llm, self.vllm_sampling = self._vllm.load_vllm(model_name,max_new_tokens, 
                                                                     max_input_len, TEMPERATURE)
        else:
            self.load_transformers(model_name)

    # Timing helper
    @staticmethod
    def sync_time():
        '''Synchronizes GPU'''
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()


    def load_transformers(self, model_name):
        'Load tokenizer + model via HuggingFace'
        if self.is_mps : dtype = torch.float16
        elif self.is_cuda : dtype = torch.bfloat16
        else: dtype = torch.float32

        # Tokenizer
        # left pad- so that all sequences end at same position - batched generation
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast = True, padding_side = 'left'
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
    
        model_kwargs = {
            'torch_dtype':dtype,
            'device_map' : 'auto' if self.is_cuda else None,
        }
    
        # NF4 quantization (L4)
        if self.use_bnb_nf4:
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_quant_type = 'nf4',
                bnb_4bit_compute_dtype = torch.bfloat16,
                bnb_4bit_use_double_quant = True, # QLoRA
            )
            model_kwargs['quantization_config'] = bnb_cfg
            model_kwargs.pop('torch_dtype') # BnB own dtype
    
            logger.info('NF4 4-bit quantization enabled via bitsandbytes')
    
        if self.is_cuda and self.use_int8 and not self.use_bnb_nf4:
            model_kwargs['load_in_8bit'] = True 
            
            logger.info('INT8 quantization enabled via bitsandbytes')
    
        logger.info("Loading %s  dtype=%s  device=%s …", model_name, dtype, self.device)
        t0=self.sync_time()
    
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        if not self.is_cuda:
            self.model = self.model.to(self.device)
        self.model.eval()
        t1 = self.sync_time()
        logger.info("Model ready in %.2f s", t1 - t0)


    @torch.inference_mode()
    def generate_transformers(self, prompts):
        '''
        Tokenize all m prompts , pad to uniform length, run 1 batched
        generate() call, decode model response, and compute per-draft log-probs

        Every prompt in the batch goes through model's KV attention layers in the same
        forward pass-> maximum GPU utilization (single - GPU parallelism)

        returns list of (response_text, draft_logprob) - one per prompt
        '''

        enc = self.tokenizer(
            prompts,
            padding = True,
            truncation = True,
            max_length = self.max_input_len,
            return_tensors = 'pt',
        )
        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc['attention_mask'].to(self.device)

        input_len = input_ids.size(1)

        # Batched generation

        gen_out = self.model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_new_tokens = self.max_new_tokens,
            do_sample = self.DO_SAMPLE,
            temperature = self.TEMPERATURE,
            pad_token_id = self.tokenizer.pad_token_id,
            eos_token_id = self.tokenizer.eos_token_id,
            output_scores = True, # captures per-step logit tensors for log probs
            return_dict_in_generate = True,
        )

        gen_ids = gen_out.sequences[:, input_len:]
        # per-draft log_probs
        # gen_out.scores -> tuple of len = gen_len, each tensor (m, vocab_size)
        
        logprobs_batch = DraftOutput.compute_seq_logprob(gen_out.scores, gen_ids) # (m, )
        
        # Decode
        completions = self.tokenizer.batch_decode(
            gen_ids,
            skip_special_tokens = True,
            clean_up_tokenization_spaces = True,
        )
        return list(zip(completions, logprobs_batch.cpu().tolist()))


    def generate_drafts(self,question,subsets,profile_run=False, profile_dir = '/profiler_traces'):
        '''
        For each doc subset:
        -Build drafter prompt from DraftOutput
        -Batch all m prompts into one forward pass
        -Decode completions and parse (rationale, answer_draft)
        -Record log prob of answer draft per subset

        Args:
        -question - query Q
        -subsets : from  MultiPerspectiveSampler.generate_subsets(),
                    shape: list of m lists, each with k document strings
        -profile_run : True-> PyTorch profiler and writes a Chrome trace + TensorBoard events
        -profile_dir : output directory for profiler traces 

        Returns:
        list[DraftOutput]  — m DraftOutputs, one per (valid) subset.
        '''

        if not subsets:
            logger.warning('call generate_subsets()')
            return []

        valid_pairs = [(i,s) for i, s in enumerate(subsets) if s]
        if not valid_pairs:
            logger.warning("All subsets were empty — returning no drafts.")
            return []
        indices = [i for i, _ in valid_pairs]
        docs_list = [s for _,s in valid_pairs] # list containing list of docs
        m = len(docs_list)

        # Build all m prompts
        prompts = [DraftOutput.build_drafter_prompt(question, docs) for docs in docs_list]

        logger.info(
            "Generating %d drafts  k=%d docs/subset  device=%s",
            m, len(docs_list[0]), self.device,
        )


        if self.use_vllm:
            if profile_run : nvtx.range_push(f"drafter.vllm_generate  m={m}")
            raw_outputs = self._vllm.generate_vllm(prompts, self.vllm_llm, self.vllm_sampling, profile_run)
            if profile_run : nvtx.range_pop()
        elif profile_run:
            raw_outputs = self.generate_with_profiler(prompts, profile_dir)
        else:
            raw_outputs = self.generate_transformers(prompts)

        # Parse 
        drafts = []
        for subset_idx, (completion, logprob) in zip(indices, raw_outputs):
            rationale, answer_draft = DraftOutput.parse_draft_output(completion)
            drafts.append(DraftOutput(
                subset_index = subset_idx,
                answer_draft = answer_draft,
                rationale = rationale,
                draft_logprob = logprob,
                raw_model_output = completion,
            ))

        logger.info("Drafting complete — %d/%d drafts produced.", len(drafts), m)
        return drafts

    def generate_with_profiler(self, prompts, profile_dir):
        '''
        Outputs
        -TensorBoard trace events
        '''
        Path(profile_dir).mkdir(parents = True, exist_ok = True)
        
        activities = [ProfilerActivity.CPU]
        if self.is_cuda:
            activities.append(ProfilerActivity.CUDA)
    
        if self.is_mps and hasattr(ProfilerActivity, 'MPS'):
            activities.append(ProfilerActivity.MPS)
    
        logger.info("Profiling → TensorBoard events: %s", profile_dir)
    
        with profile(
            activities = activities,
            record_shapes = True,
            with_flops = True,
            profile_memory = True, # tracks peak VRAM
            with_stack = False,
            on_trace_ready = torch.profiler.tensorboard_trace_handler(profile_dir),
        ) as prof:
            with record_function('batched_drafter.generate'):
                result = self.generate_transformers(prompts)
            prof.step()
    
        
        # Summary table sorted by CPU time
        print("\n" + "─" * 70)
        print(f"  Profiler summary  ({len(prompts)} prompts, device={self.device})")
        print("─" * 70)
        print(
            prof.key_averages().table(
                sort_by  = "cuda_time_total" if self.is_cuda else "cpu_time_total",
                row_limit= 15,
            )
        )
    
        return result

