# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch
# Expects to be executed in folder: https://github.com/facebookresearch/llama-recipes/tree/main/src/llama_recipes/inference 

import fire
import torch
import os
import sys
import time
import json 
from typing import List

from transformers import LlamaTokenizer, LlamaForCausalLM
from safety_utils import get_safety_checker
from model_utils import load_model, load_peft_model

BASE_PROMPT = """Below is an interaction between a human and an AI fluent in English and Amharic, providing reliable and informative answers.
Human: {}
Assistant [Amharic] : """

def main(
    model_name: str="",
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =400, #The maximum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=1, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_saleforce_content_safety: bool=False, # Enable safety check woth Saleforce safety flan t5
    **kwargs
):    
    
    print("***Note: model is not set up for chat use case, history is reset after each response.")
    print("***Ensure that you have replaced the default LLAMA2 tokenizer with the Amharic tokenizer")
    
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    MAIN_PATH = '/path/to/llama2'
    peft_model = '/path/to/checkpoint'
    model_name = MAIN_PATH

    model = load_model(model_name, quantization)

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    embedding_size = model.get_input_embeddings().weight.shape[0]

    if len(tokenizer) != embedding_size:
        print("resize the embedding size by the size of the tokenizer")
        model.resize_token_embeddings(len(tokenizer))

    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()

    while True:

        
        user_query = input('Type question in Amharic or English: ')
        user_prompt = BASE_PROMPT.format(user_query)
        batch = tokenizer(user_prompt, return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs 
            )
        e2e_inference_time = (time.perf_counter()-start)*1000
        print(f"the inference time is {e2e_inference_time} ms")
        
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("MODEL_OUTPUT: {}".format(output_text))
        #user_prompt += output_text

if __name__ == "__main__":
    fire.Fire(main)
