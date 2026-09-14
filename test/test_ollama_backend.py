from src.generation.ollama_backend import OllamaGenerator

class Response:
    def raise_for_status(self): pass
    def json(self): return {"response":" Grounded answer [S1]. "}
class Session:
    def post(self,url,**kwargs): self.url=url; self.kwargs=kwargs; return Response()

def test_ollama_backend_sends_exact_prompt_with_deterministic_options():
    session=Session(); backend=OllamaGenerator(model="local",session=session)
    assert backend("EXACT GROUNDED PROMPT")=="Grounded answer [S1]."
    assert session.kwargs["json"]["prompt"]=="EXACT GROUNDED PROMPT"
    assert session.kwargs["json"]["options"]["temperature"]==0
    assert session.kwargs["json"]["stream"] is False
