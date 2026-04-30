#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('jupyter nbconvert --to script draft_output.ipynb')


# In[2]:


'''
Batched Parallel Drafting for Speculative RAG

On a single GPU we implement parallel drafting as a single BATCHED
forward pass : all m prompts are tokenized together, left-padded to uniform length,
and sent through the model in one call. This maximises GPU SM utilization
through continous batching (vLLM)

Profiling : PyTorch Profiler
'''

from __future__ import annotations


import re
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F



# In[4]:


# (L4/A100)
# !pip install vllm bitsandbytes
# from transformers import BitsAndBytesConfig
# VLLM_AVAILABLE = True
# BNB_AVAILABLE = True

# mps
# VLLM_AVAILABLE = False
# BNB_AVAILABLE = False

# Model names
# MODEL_MISTRAL_7B        = "mistralai/Mistral-7B-v0.1"
# MODEL_MISTRAL_INSTRUCT  = "mistralai/Mistral-7B-Instruct-v0.1"

# mps
# MODEL_PHI2   = "microsoft/phi-2"           
# MODEL_TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  



# In[ ]:


@dataclass
class DraftOutput:
    '''
    Specialist RAG drafter output
    Handles drafter prompt generation
    Given prompt and docs subset -> generates (rationale, asnwer_draft)
    Computes logprob
    '''
    subset_index : int # jth subset out of m
    answer_draft : str
    rationale : str
    draft_logprob : float
    model_output : str = field(repr = False, default = '')

    # Prompt construction (paper Appendix H verbatim)
    SYSTEM_PROMPT = (
        'Response to the instruction. '
        'Also provide rationale for your response.\n\n'
    )

    def build_drafter_prompt(question, docs):
        '''
        Format (Q, doc subset) into drafter input
        Returns full prompt string
        '''
        evidence_ = '\n'.join(
            f"[{i}] {doc.strip()}" for i, doc in enumerate(docs, 1)
        )
        return (
            f"{SYSTEM_PROMPT} "
            f"## Instruction : {question.strip()}\n"
            f"## Evidence :\n{evidence_}\n\n"
            f"## Rationale :"
        )

    def parse_draft_output(response):
        '''
        Parse the model response into (rationale, answer_draft)
        Fallback strategy : if no ## Response marker - pick last non-empty line as answer
        '''

        RATIONALE = re.compile(
            r"^(.*?)(?=\n\s*#+\s*Response:|\Z)", re.DOTALL | re.IGNORECASE
        )
        RESPONSE = re.compile(
            r"\s*#+\s*Response:\s*(.*?)(?=\n#+|\Z)", re.DOTALL | re.IGNORECASE
        )
        rat_match = RATIONALE.search(response)
        ans_match = RESPONSE.search(response)

        rationale = rat_match.group(1).strip() if rat_match else response.strip()
        answer_draft = ans_match.group(1).strip() if ans_match else ''

        # Fallback strategy
        if not answer_draft and rationale :
            lines = [ln.strip() for ln in rationale.splitlines() if ln.strip()]
            if lines : 
                answer_draft = lines[-1]
                rationale = '\n'.join(lines[:-1]).strip() or rationale
        return rationale, answer_draft


    def compute_seq_logprob(scores, generated_ids):
        '''

        Computes sum of per-token log probs over a generated sequence
        Paper : log P(β_j|Q,docs) + log P(α_j|Q,docs,β_j)
        Equivalent to Σ_t log P(token_t | context, token<t)
        scores : (m drafts, vocab_size) logit tensors, per generated token
        generated_ids : (m drafts, )
        Returns tensor of shape (m drafts, ) - one scalar log-prob per draft
        '''
        total_drafts = generated_ids.size(0)
        per_token = []

        for token_idx, logits in enumerate(scores):
            # logits : (m, vocab_size)
            log_probs = F.log_softmax(logits.float(), dim = -1) # (m, vocab)
            chosen_ids = generated_ids[:, token_idx] # (m, )
            token_lp = log_probs[torch.arange(total_drafts, device = logits.device), chosen_ids]
            per_token.append(token_lp)

        # (m, total_tokens) -> (m, )
        return torch.stack(per_token, dim = 1).sum(dim = 1)






