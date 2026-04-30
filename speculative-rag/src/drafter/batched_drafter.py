#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


# MAX_NEW_TOKENS = 300   
# MAX_INPUT_LEN  = 1024  
# DO_SAMPLE      = False
# TEMPERATURE    = 1.0


# In[ ]:


import drafter_pipeline as dp
from draft_output import DraftOutput
import vllm_ 


# In[ ]:


from __future__ import annotations

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


# In[ ]:


'''
Specialist RAG drafter : generates m draft (answer, rationale) pairs from m document subsets
in a SINGLE batched forward pass
'''
class BatchedDrafter:
    def __init__(
        self, model_name = None, device= dp.DEVICE,
        max_new_tokens = dp.MAX_NEW_TOKENS,
        max_input_len = dp.MAX_INPUT_LEN,
        use_vllm = dp.VLLM_AVAILABLE,
        use_bnb_nf4 = dp.BNB_AVAILABLE
    ):

        self.device = DEVICE



        self.is_cuda = DEVICE.type == 'cuda'
        self.is_mps = DEVICE.type == 'mps'
        self.is_cpu = DEVICE.type == 'cpu'

        if model_name is None:
            model_name = dp.MODEL_MISTRAL_7B 

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
            vllm_.load_vllm(model_name)
        else:
            self.load_transformers(model_name)

    # Timing helper
    def sync_time():
        '''Synchronizes GPU'''
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()


    def load_transformers(self, model_name):
        'Load tokenizer + model via HuggingFace'
        if self.is_mps : dtype = torch.float16
        elif self.is_cuda : dtype = torch.bfloat32
        else: dtype = torch.float32

        # Tokenizer
        # left pad- so that all sequences end at same position - batched generation
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast = True, padding_side = 'left'
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


        model_kwargs = dict{
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

        if self.is_cuda and not self.use_bnb_nf4:
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
            do_sample = dp.DO_SAMPLE,
            temperature = dp.TEMPERATURE,
            pad_token_id = self.tokenizer.pad_token_id,
            eos_token_id = self.tokenizer.eos_token_id,
            output_scores = True, # captures per-step logit tensors for log probs
            return_dict_in_generate = True,
        )

        gen_ids = gen_out.sequences[:, input_len:]
        # per-draft log_probs
        # gen_out.scores -> tuple of len = gen_len, each tensor (m, vocab_size)
        DraftOutput_ = DraftOutput()
        logprobs_batch = DraftOutput_.compute_seq_logprob(gen_out.scores, gen_ids) # (m, )

        # Decode
        completions = self.tokenizer.batch_decode(
            gen_ids,
            skip_special_tokens = True,
            clean_up_tokenization_spaces = True,
        )
        return list(zip(completions, logprobs_batch.cpu().tolist()))














# In[ ]:




