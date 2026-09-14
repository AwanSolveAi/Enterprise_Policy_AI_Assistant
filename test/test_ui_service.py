import pytest

from app.service import PolicyAssistantService, backend_label, load_generator
from src.generation.rag_engine import INSUFFICIENT

def policy_result(cid="policy-1"):
    return {"rank":1,"chunk_id":cid,"rerank_score":4.2,"chunk":{"chunk_id":cid,"document_id":"doc-1",
        "title":"Workplace Conduct Policy","category":"Workplace Conduct","country":"Global",
        "source_url":"https://example.test/policy","chunk_index":1,"chunk_text":"Employees should report concerns to HR."}}

class Retriever:
    def __init__(self): self.calls=0
    def search(self,question,top_k=20):
        self.calls+=1
        return {"query":question,"expanded_query":question+" reporting","routes":["conduct"],"hybrid_results":[policy_result()]}

def test_service_reuses_one_retrieval_and_maps_cited_source_metadata():
    retriever=Retriever(); service=PolicyAssistantService(retriever,lambda prompt:"Report concerns to HR [S1].")
    response=service.ask("How do I report a concern?")
    assert retriever.calls==1 and response.grounded
    assert response.sources[0].title=="Workplace Conduct Policy"
    assert response.sources[0].category=="Workplace Conduct" and response.sources[0].country=="Global"
    assert response.sources[0].evidence=="Employees should report concerns to HR."
    assert response.debug["context_chunk_ids"]==["policy-1"]
    assert "prompt" not in response.debug and "context" not in response.debug

def test_service_preserves_insufficient_evidence_behavior_without_backend():
    response=PolicyAssistantService(Retriever(),None).ask("Unknown policy?")
    assert not response.grounded and response.answer==INSUFFICIENT and response.sources==()

def test_service_rejects_empty_question():
    with pytest.raises(ValueError): PolicyAssistantService(Retriever(),None).ask("  ")

def test_backend_configuration_is_provider_neutral():
    assert load_generator(None) is None and backend_label(None)=="Not configured"
    with pytest.raises(ValueError): load_generator("missing_separator")
