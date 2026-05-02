
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


import multiprocessing

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


from drafter_pipeline import run


# no optimization(default)
VLLM_AVAILABLE = False
BNB_AVAILABLE = False
INT8_Q = False


# Model names
MODEL_MISTRAL_7B        = "mistralai/Mistral-7B-v0.1"
MODEL_MISTRAL_INSTRUCT  = "mistralai/Mistral-7B-Instruct-v0.1"

# smoke test models
MODEL_PHI2   = "microsoft/phi-2"           
MODEL_TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  


PROFILE_RUN = False
PROFILE_BASE_DIR = Path('./profiler_traces')
DRAFTER_OUTPUT_PATH = Path('./drafter_output')

# hyperparamters
M  = 5    # number of drafts per question
K_DOCS = 2    # documents per subset (subset size)
TOP_K = 10

def run_test():
    '''
    Run n_samples without profiler - smoke test
    Model name : MODEL_TINYLLAMA
    No optimization
    '''
    logger.info('Experiment : no profiling | no optimization')
    run(m = M, 
        k_docs = K_DOCS, 
        profile_run = PROFILE_RUN,  
        output_path = DRAFTER_OUTPUT_PATH / 'test.json', 
        model_name = MODEL_TINYLLAMA, 
        use_vllm = VLLM_AVAILABLE,
        use_bnb_nf4 = BNB_AVAILABLE, 
        use_int8 = INT8_Q)

def run_test_profiler():
    '''
    Run n_samples with profiler - smoke test
    Model name : MODEL_TINYLLAMA
    No optimization
    '''
    logger.info('Experiment : profiling| no optimization')
    run(m = M, 
        k_docs = K_DOCS, 
        profile_run = True,  
        output_path = DRAFTER_OUTPUT_PATH / 'test_profiler.json', 
        model_name = MODEL_TINYLLAMA, 
        use_vllm = VLLM_AVAILABLE,
        use_bnb_nf4 = BNB_AVAILABLE, 
        use_int8 = INT8_Q,
        profile_dir = PROFILE_BASE_DIR/'test')


def run_no_opt():
    '''
    test = True(default) : Run n_samples
    test = False : Run complete TriviaQA split with profiler
    M  = 5    # number of drafts per question
    K_DOCS = 2     # documents per subset (subset size)
    Model name : MODEL_MISTRAL_INSTRUCT 
    No optimization
    '''
    logger.info('Experiment : profiling| no optimization')
    run(m = M, 
        k_docs = K_DOCS, 
        profile_run = True,  
        output_path = DRAFTER_OUTPUT_PATH / 'no_opt.json', 
        model_name = MODEL_MISTRAL_INSTRUCT, 
        use_vllm = VLLM_AVAILABLE,
        use_bnb_nf4 = BNB_AVAILABLE, 
        use_int8 = INT8_Q,
        profile_dir = PROFILE_BASE_DIR/'no_opt',
        test = False)


def run_nf4():
    '''
    test = True(default) : Run n_samples
    test = False : Run complete TriviaQA split with profiler
    Model name : MODEL_MISTRAL_INSTRUCT 
    M  = 5    # number of drafts per question
    K_DOCS = 2     # documents per subset (subset size)
    NF4 4-bit quantization
    '''
    logger.info('Experiment : profiling| NF4 4-bit quantization')
    run(m = M, 
        k_docs = K_DOCS, 
        profile_run = True,  
        output_path = DRAFTER_OUTPUT_PATH / 'bnb.json', 
        model_name = MODEL_MISTRAL_INSTRUCT, 
        use_vllm = VLLM_AVAILABLE,
        use_bnb_nf4 = True, 
        use_int8 = INT8_Q,
        profile_dir = PROFILE_BASE_DIR/'bnb',
        test = False
        )


def run_int8():
    '''
    test = True(default) : Run n_samples
    test = False : Run complete TriviaQA split with profiler
    Model name : MODEL_MISTRAL_INSTRUCT 
    M  = 5    # number of drafts per question
    K_DOCS = 2     # documents per subset (subset size)
    INT8 quantization 
    '''
    logger.info('Experiment : profiling| INT8 quantization')
    run(m = M, 
        k_docs = K_DOCS, 
        profile_run = True,  
        output_path = DRAFTER_OUTPUT_PATH / 'int8.json', 
        model_name = MODEL_MISTRAL_INSTRUCT, 
        use_vllm = VLLM_AVAILABLE,
        use_bnb_nf4 = BNB_AVAILABLE, 
        use_int8 = True,
        profile_dir = PROFILE_BASE_DIR/'int8',
        test = False)


def run_vllm():
    '''
    test = True(default) : Run n_samples
    test = False : Run complete TriviaQA split with profiler
    Model name : MODEL_MISTRAL_INSTRUCT 
    M  = 5    # number of drafts per question
    K_DOCS = 2     # documents per subset (subset size)
    vLLM for continous batching 
    '''
    logger.info('Experiment : profiling| vLLM')
    run(m = M, 
        k_docs = K_DOCS, 
        profile_run = True,  
        output_path = DRAFTER_OUTPUT_PATH / 'vllm_m5_k2.json', 
        model_name = MODEL_MISTRAL_INSTRUCT, 
        use_vllm = True,
        use_bnb_nf4 = BNB_AVAILABLE, 
        use_int8 = INT8_Q,
        profile_dir = PROFILE_BASE_DIR/'vllm_m5_k2',
        test = False
        )


def run_m():
    '''
    test = True(default) : Run n_samples without profiler
    test = False : Run complete TriviaQA split without profiler
    Model name : MODEL_MISTRAL_INSTRUCT 
    Set profile_run = True for profiling
    Effect of varying m - number of drafts per question
    K_DOCS = 2     # documents per subset (subset size)
    vLLM for continous batching 
    '''
    logger.info('Experiment : Varying m - number of drafts per question')
    for m in [5,10,15,20]:
        run(m = m, 
            k_docs = K_DOCS, 
            profile_run = False,  
            output_path = DRAFTER_OUTPUT_PATH / f'vllm_m{m}.json', 
            model_name = MODEL_MISTRAL_INSTRUCT, 
            use_vllm = True,
            use_bnb_nf4 = BNB_AVAILABLE, 
            use_int8 = INT8_Q,
            test = False)


def run_k():
    '''
    test = True(default) : Run n_samples without profiler
    test = False : Run complete TriviaQA split without profiler
    Model name : MODEL_MISTRAL_INSTRUCT 
    Set profile_run = True for profiling
    M  = 5    # number of drafts per question
    Effect of varying K_DOCS  # documents per subset (subset size)
    vLLM for continous batching 
    '''
    logger.info('Experiment : Varying k - documents per subset')
    for i in [2,4,6,10]:
        run(m = M, 
            k_docs = i, 
            profile_run = False,  
            output_path = DRAFTER_OUTPUT_PATH / f'vllm_k{i}.json', 
            model_name = MODEL_MISTRAL_INSTRUCT, 
            use_vllm = True,
            use_bnb_nf4 = BNB_AVAILABLE, 
            use_int8 = INT8_Q,
            test = False)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    import argparse

    RUN_MAP = {
        "test"    : run_test,
        "test_p"  : run_test_profiler,   
        "no_opt"  : run_no_opt,
        "nf4"     : run_nf4,
        "int8"    : run_int8,
        "run_vllm": run_vllm,
        "run_m"   : run_m,
        "run_k"   : run_k,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        choices=list(RUN_MAP.keys()),
        default="test",
        help="Which experiment to run"
    )
    args = parser.parse_args()
    RUN_MAP[args.run]()




