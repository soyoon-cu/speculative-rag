#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('jupyter nbconvert --to script vllm_.ipynb')


# In[ ]:


import drafter_pipeline as dp


# In[ ]:






# In[ ]:





# In[ ]:


class VLLM:
    def __init__(self):
        # self.model_name = model_name
        # self.load_vllm(self.model_name)

    def load_vllm(self, model_name):
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
            max_model_len = dp.MAX_NEW_TOKENS + dp.MAX_INPUT_LEN, 
        )
        self.vllm_sampling = SamplingParams(
            temperature = dp.TEMPERATURE,
            max_tokens = dp.MAX_NEW_TOKENS,
            logprobs = 1, # collect top-1 logprob
        )
        logger.info('vLLM engine loaded: %s', self.model_name)
        return self.vllm_llm, self.vllm_sampling




    def generate_vllm(self, prompts):
        '''
        Send all m prompts to vLLM engine in one call(continous batch)
        all m requests scheduled simulatenously
        m answer drafts generated

        Returns list of (completion_text, draft_logprob)
        '''
        outputs = self.vllm_llm.generate(prompts, self.vllm_sampling) # len = m-> one output per draft/prompt
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






