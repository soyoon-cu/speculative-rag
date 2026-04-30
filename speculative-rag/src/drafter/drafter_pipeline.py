#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('jupyter nbconvert --to script drafter_pipeline.ipynb')


# In[ ]:


import draft_output
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
# MODEL_MISTRAL_7B        = "mistralai/Mistral-7B-v0.1"
# MODEL_MISTRAL_INSTRUCT  = "mistralai/Mistral-7B-Instruct-v0.1"

# mps
MODEL_PHI2   = "microsoft/phi-2"           
MODEL_TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  

# hyperparamters
MAX_NEW_TOKENS = 300   
MAX_INPUT_LEN  = 1024  
DO_SAMPLE      = False
TEMPERATURE    = 1.0  # greedy decoding


if torch.cuda.is_available():
    DEVICE= torch.device('cuda')
elif torch.backends.mps.is_availabe():
    DEVICE = torch.device('mps')
else:
    logger.warning('No GPU/MPS - falling back to CPU')
    DEVICE = torch.device('cpu')

