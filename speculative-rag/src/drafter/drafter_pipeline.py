#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# In[ ]:


'''
Full Speculative RAG drafting pipeline.
Stages:
-Load TriviaQA question + gold answers      
-Retrieve top-k passages from FAISS index    
-Multi-perspective subset sampling           
-Batched parallel draft generation           
-Pack VerifierInput for downstream verifier 

run_test() : real TriviaQA data, small subset (n=10 by default)
           -to verify end-to-end correctness before full run
run_full() : complete TriviaQA validation split 

Output:
list[VerifierInput], one per question, contains
- question Q
- list of DraftOutput (α_j, β_j, ρ_Draft,j) for j = 1..m
-gold_answers for downstream EM scoring

'''


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


from index import FAISSIndex
from retriever   import ContrieverRetriever
from multi_perspective import MultiPerspectiveSampler
from batched_drafter  import BatchedDrafter, DraftOutput
from loader import iter_samples, TriviaQASample
from preprocess import answer_in_response


# In[ ]:


# (L4/A100)
# VLLM_AVAILABLE = True

# A100 quantization
# INT8_Q = True

# L4 quantization
# BNB_AVAILABLE = True
# from transformers import BitsAndBytesConfig

# mps
VLLM_AVAILABLE = False
BNB_AVAILABLE = False
INT8_Q = False


# Model names
MODEL_MISTRAL_7B        = "mistralai/Mistral-7B-v0.1"
MODEL_MISTRAL_INSTRUCT  = "mistralai/Mistral-7B-Instruct-v0.1"

# mps
MODEL_PHI2   = "microsoft/phi-2"           
MODEL_TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  

# hyperparamters


PROFILE_RUN = False
PROFILE_BASE_DIR = './profiler_traces'


if torch.cuda.is_available():
    DEVICE= torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    logger.warning('No GPU/MPS - falling back to CPU')
    DEVICE = torch.device('cpu')


# In[ ]:


INDEX_PATH = Path("data/index.faiss")
META_PATH  = Path("data/index_meta.pkl")


# In[ ]:


# Speculative RAG hyperparameters
MAX_NEW_TOKENS = 300   
MAX_INPUT_LEN  = 1024  
DO_SAMPLE      = False
TEMPERATURE    = 1.0  # greedy decoding
TOP_K = 10


# In[ ]:


@dataclass
class VerifierInput:
    '''
    Fields:
    question_id  : TriviaQA question ID
    question: the raw question string Q
    drafts : m DraftOutput objects, each carrying:
               .answer_draft   α_j
               .rationale  β_j
               .draft_logprob  log ρ_Draft,j  (from drafter)
               .subset_index  j
    gold_answers : list of acceptable gold answers from loader.py
                   used for EM scoring after verifier selects best draft
    retrieval_time_s  : wall-clock seconds spent in retrieve_with_embeddings()
    sampling_time_s  : wall-clock seconds spent in generate_subsets()
    drafting_time_s : wall-clock seconds spent in generate_drafts()
    '''
    question_id : str
    question : str
    drafts : list[DraftOutput]
    gold_answers : list[str]
    retrieval_time_s : float = 0.0
    sampling_time_s : float = 0.0
    drafting_time_s : float = 0.0



# In[ ]:


@dataclass
class PipelineResult:
    '''
    Aggregate result over a set of questions.
    Reported after drafting stage (before verifier) to measure
    draft-only accuracy 
    '''
    total_questions  : int   = 0
    drafts_hit : int   = 0    # ≥1 draft contains gold answer
    total_drafts : int   = 0
    avg_retrieval_ms   : float = 0.0
    avg_sampling_ms : float = 0.0
    avg_drafting_ms : float = 0.0
    avg_total_ms  : float = 0.0

    @property
    def draft_coverage(self):
        'Fraction of questions where at least one draft is correct.'
        if self.total_questions == 0: return 0.0
        return self.drafts_hit / self.total_questions



# In[ ]:


def sync_time():
    'Synchronizes GPU'
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


# In[ ]:


def process_one(
    sample, retriever,sampler,drafter,
    top_k, m, k_docs, profile_run, profile_dir
):
    ' Run one TriviaQASample through the full drafting pipeline.'
    question = sample.question
    question_id = sample.question_id
    gold_answers = sample.answers

    # Stage 1 - Retrieve top-k passages + precomputed embeddings
    t0 = sync_time()
    texts, embs = retriever.retrieve_with_embeddings(question, top_k)
    retrieval_time = sync_time() - t0
    if not texts:
        logger.warning('No passages retrieved for qid = %s- skipping', question_id)
        return VerifierInput(
            question_id = question_id,
            question = question,
            drafts = [],
            gold_answers = gold_answers,
            retrieval_time_s = retrieval_time,
        )

    # Stage 2 -Multi-perspective subset sampling
    # precomputed_emb bypasses Contriever re-encoding (use FAISS cached vectors)
    # kmeans runs on top-k already computed vectors
    t0 = sync_time()
    subsets = sampler.generate_subsets(
        question = question,
        passages = texts,
        m = m,
        k = k_docs,
        precomputed_emb = embs,
    )
    sampling_time = sync_time() - t0
    if not subsets:
        logger.warning("No subsets generated for qid=%s — skipping.", question_id)
        return VerifierInput(
            question_id = question_id,
            question = question,
            drafts = [],
            gold_answers = gold_answers,
            retrieval_time_s = retrieval_time,
            sampling_time_s = sampling_time,
        )

    # Stage 3 - Batched parallel draft generation
    # All m prompts go through model in ONE batched generate() call
    t0 = sync_time()
    drafts = drafter.generate_drafts(
        question = question,
        subsets = subsets,
        profile_run=profile_run,
        profile_dir = profile_dir)

    drafting_time = sync_time()-t0

    return VerifierInput(
        question_id  = question_id,
        question = question,
        drafts = drafts,
        gold_answers  = gold_answers,
        retrieval_time_s = retrieval_time,
        sampling_time_s  = sampling_time,
        drafting_time_s  = drafting_time,
    )






# In[ ]:


def load_pipeline(model_name, use_vllm ,use_bnb_nf4, index_path = INDEX_PATH, meta_path = META_PATH):
    'Load all three pipelinecomponents once'

    logger.info("Loading FAISS index from %s …", index_path)
    faiss_index = FAISSIndex.load(index_path, meta_path)

    logger.info("Loading ContrieverRetriever …")
    retriever = ContrieverRetriever(index=faiss_index, device=str(DEVICE))

    logger.info("Loading MultiPerspectiveSampler …")
    sampler = MultiPerspectiveSampler(device=str(DEVICE))

    logger.info("Loading BatchedDrafter (%s) …", MODEL_NAME)
    drafter = BatchedDrafter(
        device = DEVICE,
        max_new_tokens = MAX_NEW_TOKENS,
        max_input_len  = MAX_INPUT_LEN,
        use_vllm ,
        use_bnb_nf4,
        DO_SAMPLE  = DO_SAMPLE,
        TEMPERATURE  = TEMPERATURE,
        model_name,
    )
    return retriever, sampler, drafter


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




