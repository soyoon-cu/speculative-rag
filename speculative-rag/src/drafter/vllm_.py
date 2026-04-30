#!/usr/bin/env python
# coding: utf-8

# In[5]:



# In[ ]:


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
        )
        self.vllm_sampling = SamplingParams(
            temperature = temperature,
            max_tokens = max_new_tokens,
            logprobs = 1, # collect top-1 logprob
        )
        logger.info('vLLM engine loaded: %s', model_name)
        return self.vllm_llm, self.vllm_sampling




    def generate_vllm(self, prompts, vllm_llm, vllm_sampling):
        '''
        Send all m prompts to vLLM engine in one call(continous batch)
        all m requests scheduled simulatenously
        m answer drafts generated

        Returns list of (completion_text, draft_logprob)
        '''
        outputs = vllm_llm.generate(prompts, vllm_sampling) # len = m-> one output per draft/prompt
        results = []
        for req_out in outputs:
            completion = req_out.outputs[0].text
            # Sum per-token logprobs
            lp = 0.0
            # logprobs -> list of dicts, one dict per generated token
            if req_out.outputs[0].logprobs:
                for tok_lp_dict in req_out.outputs[0].logprobs:
                    lp += list(tok_lp_dict.values())[0].logprob
            results.append((completion, lp))
        return results






