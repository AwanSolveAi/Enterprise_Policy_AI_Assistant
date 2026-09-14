import sys
from types import SimpleNamespace
import torch
from src.generation.local_transformers_backend import LocalTransformersGenerator

class FakeTokenizer:
    eos_token_id=0
    def apply_chat_template(self,messages,**kwargs):
        assert messages[-1]["content"]=="grounded prompt"
        return "rendered"
    def __call__(self,text,**kwargs): return {"input_ids":torch.tensor([[10,11]])}
    def decode(self,tokens,**kwargs):
        assert tokens.tolist()==[42,43]
        return "Grounded answer [S1]."

class FakeModel:
    def generate(self,**kwargs):
        assert kwargs["do_sample"] is False
        return torch.tensor([[10,11,42,43]])

def test_local_backend_uses_chat_template_and_decodes_only_new_tokens():
    backend=LocalTransformersGenerator(tokenizer=FakeTokenizer(),model=FakeModel())
    assert backend("grounded prompt")=="Grounded answer [S1]."

def test_local_backend_is_lazy_when_dependencies_are_injected():
    backend=LocalTransformersGenerator(tokenizer=FakeTokenizer(),model=FakeModel(),local_files_only=True)
    assert backend.model_name and backend.local_files_only
