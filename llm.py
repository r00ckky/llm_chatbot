from transformers import LlamaForCasualLM, LlamaTokenizer, pipeline, GenerationConfig
import torch
import numpy as np
import openai

class Llama:
    def __init__(self, key, model, system_prompt=None) -> None:
        self.tokenizer = LlamaTokenizer(
            model, 
            token = key
        )
        self.model = LlamaForCasualLM.from_pretrained(
            model,
            token = key,
        )
        self.system_prompt = system_prompt
            
    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
    
    def format_prompt(self, prompt):
        return f"""
        {self.system_prompt}
        {prompt}
    """.strip()
    
    def generate_response(self, prompt: str, max_new_tokens: int = 128) -> str: 
        encoding = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode(): 
            outputs = self.model.generate(
                **encoding,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                generation_config=self.generation_config,
            )
        answer_tokens = outputs[:, encoding.input_ids.shape[1] :]
        return self.tokenizer.decode(answer_tokens[0], skip_special_tokens=True)

class GPT:
    def __init__(self, key, model, system_prompt=None) -> None:
        self.model = model
        openai.api_key = key
        self.system_prompt = system_prompt
    
    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt

    def generate_response(self, prompt:str, temperature:float=0.6) -> str:
        response = openai.ChatCompletion.create(
            model= self.model,
            messages = [
                {
                    'role':'system',
                    'content':self.system_prompt
                } if self.system_prompt else None,
                {
                    'role':'user',
                    'content':prompt
                }
            ],
            temperature=temperature
        )
        return response.choices[0].text.strip()
    