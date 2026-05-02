
from __future__ import annotations

'''
Specialist RAG drafter output

On a single GPU we implement parallel drafting as a single BATCHED
forward pass : all m prompts are tokenized together, left-padded to uniform length,
and sent through the model in one call. This maximises GPU SM utilization
through continous batching (vLLM)

Profiling : PyTorch Profiler
'''




import re
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

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
    raw_model_output : str = field(repr = False, default = '')

   
    # Prompt construction 
    @staticmethod
    def build_drafter_prompt(question, docs):
        '''
        Format (Q, doc subset) into drafter input
        Returns full prompt string
        '''
        evidence_ = '\n'.join(
            f"[{i}] {doc.strip()}" for i, doc in enumerate(docs, 1)
        )
        user_content = (
        "You are a precise question-answering assistant. "
        "Answer the instruction using ONLY the provided evidence. "
        "Reply in exactly this format:\n"
        "## Rationale: <one or two sentences explaining which evidence supports your answer>\n"
        "## Response: <concise answer>\n\n"
        f"## Instruction: {question.strip()}\n\n"
        f"## Evidence:\n{evidence_}"
    )
        return f"[INST] {user_content} [/INST]"

    @staticmethod
    def parse_draft_output(response):
        '''
        Parse the model response into (rationale, answer_draft)
        Expected format:
        ## Rationale: <text>
        ## Response: <text>
        Fallback strategy : if no ## Response marker - pick last non-empty line as answer
        '''
    
        RAT_RE = re.compile(
            r"\s*#+\s*Rationale\s*[:\-]\s*(.*?)(?=\n\s*#+\s*Response\s*[:\-]|\Z)",
            re.DOTALL | re.IGNORECASE,
    )
        ANS_RE = re.compile(
            r"\s*#+\s*Response\s*[:\-]\s*(.*?)(?=\n\s*#+|\Z)",
            re.DOTALL | re.IGNORECASE,
    )
        rat_match = RAT_RE.search(response)
        ans_match = ANS_RE.search(response)

        rationale = rat_match.group(1).strip() if rat_match else response.strip()
        answer_draft = ans_match.group(1).strip() if ans_match else ''

        # Fallback strategy
        if not answer_draft and rationale :
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", rationale) if s.strip()]
            if len(sentences) > 1:
                answer_draft = sentences[-1]
                rationale    = " ".join(sentences[:-1])
            else:
                answer_draft = rationale 
        elif not rationale and not answer_draft:
            rationale    = response.strip()
            answer_draft = response.strip()
        return rationale, answer_draft


    @staticmethod
    def compute_seq_logprob(scores, generated_ids):
        '''
        
        Computes sum of per-token log probs over a generated sequence
        Paper : log P(β_j|Q,docs) + log P(α_j|Q,docs,β_j)
        Equivalent to Σ_t log P(token_t | context, token<t)
        scores : (m drafts, vocab_size) logit tensors, per generated token
        generated_ids : (m drafts, )
        Returns tensor of shape (m drafts, ) - one scalar log-prob per draft
        '''
        # guard against empty generation
        if len(scores) == 0:
            total_drafts = generated_ids.size(0)
            return torch.full((total_drafts,), float('-inf'))
        
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
            
            
 
