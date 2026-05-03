
'''Load vLLM for PagedAttention and continuous batching'''

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger(__name__)
import torch.cuda.nvtx as nvtx 


class VLLM:
    def __init__(self):
        pass
        
    def load_vllm(self, model_name, max_new_tokens, max_input_len, temperature):
        '''
        Load model via vLLM for continous batching
        -All m prompts share one scheduling window
        -Paged Attention
        -tensor parallel size = 1 (1 GPU)
        
        '''
        from vllm import LLM, SamplingParams
        self.vllm_llm = LLM(
            model = model_name,
            dtype = 'bfloat16',
            tensor_parallel_size = 1,
            max_model_len = max_new_tokens + max_input_len, 
            gpu_memory_utilization=0.85,
        )
        self.vllm_sampling = SamplingParams(
            temperature = temperature,
            max_tokens = max_new_tokens,
            logprobs = 1, # collect top-1 logprob
        )
        logger.info('vLLM engine loaded: %s', model_name)
        return self.vllm_llm, self.vllm_sampling

 


    def generate_vllm(self, prompts, vllm_llm, vllm_sampling, profile_run):
        '''
        Send all m prompts to vLLM engine in one call(continous batch)
        all m requests scheduled simulatenously
        m answer drafts generated
    
        Returns list of (completion_text, draft_logprob)
        '''
        if profile_run : nvtx.range_push("vllm.generate")
        outputs = vllm_llm.generate(prompts, vllm_sampling) # len = m-> one output per draft/prompt
        if profile_run : 
            nvtx.range_pop()
            nvtx.range_push("vllm.logprob_extraction")
        results = []
        for req_out in outputs:
            completion = req_out.outputs[0].text
            # Sum per-token logprobs
            lp = 0.0
            # logprobs -> list of dicts, one dict per generated token
            if req_out.outputs[0].logprobs:
                for tok_lp_dict in req_out.outputs[0].logprobs:
                    lp += list(tok_lp_dict.values())[0].logprob
            n_in  = len(req_out.prompt_token_ids)     #  prompt tokens
            n_out = len(req_out.outputs[0].token_ids)      #  generated tokens
            results.append((completion, lp))
        if profile_run : nvtx.range_pop()
        return results

    
    

    

