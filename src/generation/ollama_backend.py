"""Provider-neutral callable adapter for a locally installed Ollama model."""
from __future__ import annotations
import os
from typing import Any
import requests

class OllamaGenerator:
    def __init__(self, model: str | None=None, endpoint: str | None=None, session: Any=None,
                 timeout: int=300, max_tokens: int=120):
        self.model=model or os.getenv("POLICY_OLLAMA_MODEL","llama3.2:3b")
        self.endpoint=endpoint or os.getenv("POLICY_OLLAMA_ENDPOINT","http://127.0.0.1:11434/api/generate")
        self.session=session or requests.Session(); self.timeout=timeout; self.max_tokens=max_tokens

    def __call__(self, prompt: str) -> str:
        response=self.session.post(self.endpoint,json={"model":self.model,"prompt":prompt,"stream":False,
            "keep_alive":"15m","options":{"temperature":0,"seed":0,"num_predict":self.max_tokens}},timeout=self.timeout)
        response.raise_for_status()
        answer=response.json().get("response","").strip()
        if not answer: raise RuntimeError("Ollama returned an empty response")
        return answer

generate=OllamaGenerator()
