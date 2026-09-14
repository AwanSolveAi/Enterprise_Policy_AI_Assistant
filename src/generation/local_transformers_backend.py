"""Optional CPU-only Transformers generation backend.

The module exposes the same ``prompt -> text`` callable expected by the
provider-neutral benchmark runner. Model loading is lazy and local-files-only
by default so ordinary imports/tests never download assets.
"""
from __future__ import annotations
import os
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

class LocalTransformersGenerator:
    def __init__(self, model_name: str | None=None, tokenizer: Any=None, model: Any=None,
                 max_new_tokens: int=220, local_files_only: bool=True, threads: int | None=None):
        self.model_name=model_name or os.getenv("POLICY_LLM_MODEL",DEFAULT_MODEL)
        self.tokenizer=tokenizer; self.model=model; self.max_new_tokens=max_new_tokens
        self.local_files_only=local_files_only; self.threads=threads or int(os.getenv("POLICY_LLM_THREADS","4"))

    def _load(self):
        if self.model is not None and self.tokenizer is not None: return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(max(1,self.threads))
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_name,local_files_only=self.local_files_only)
        self.model=AutoModelForCausalLM.from_pretrained(self.model_name,local_files_only=self.local_files_only,
                                                        torch_dtype=torch.float32,low_cpu_mem_usage=False)
        self.model.eval()

    def __call__(self, prompt: str) -> str:
        self._load()
        import torch
        messages=[{"role":"system","content":"You are a precise HR policy assistant. Follow the supplied evidence and citation rules exactly."},
                  {"role":"user","content":prompt}]
        rendered=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.tokenizer(rendered,return_tensors="pt",add_special_tokens=False)
        input_length=inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output=self.model.generate(**inputs,max_new_tokens=self.max_new_tokens,do_sample=False,
                                       pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0,input_length:],skip_special_tokens=True).strip()

generate=LocalTransformersGenerator()
