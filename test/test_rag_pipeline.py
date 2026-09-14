import numpy as np
from src.generation.context_builder import build_context
from src.generation.rag_engine import INSUFFICIENT, RAGEngine, canonicalize_citations, has_route_evidence, validate_citations
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion, validate_chunks
from src.retrieval.query_processing import expand_query
from src.retrieval.reranker import CrossEncoderReranker

def chunk(cid, text="evidence", title="Policy"):
    return {"chunk_id":cid,"chunk_text":text,"title":title,"source_url":"https://example.test"}

def test_rrf_joins_by_chunk_id_not_array_position():
    vector=[{"rank":1,"score":.9,"chunk_id":"a","chunk":chunk("a")},{"rank":2,"score":.8,"chunk_id":"b","chunk":chunk("b")}]
    bm25=[{"rank":1,"score":4,"chunk_id":"b","chunk":chunk("b")},{"rank":2,"score":3,"chunk_id":"a","chunk":chunk("a")}]
    fused=reciprocal_rank_fusion((vector,bm25))
    assert {r["chunk_id"] for r in fused}=={"a","b"}
    assert next(r for r in fused if r["chunk_id"]=="a")["bm25_rank"]==2

def test_duplicate_ids_fail_fast():
    try: validate_chunks([chunk("a"),chunk("a")],"test")
    except ValueError as exc: assert "duplicate" in str(exc)
    else: raise AssertionError("duplicate ID accepted")

def test_query_routes_are_light_and_specific():
    expanded,routes=expand_query("What happens when an employee leaves GitLab?")
    assert routes==["offboarding"] and "termination" in expanded
    expanded,routes=expand_query("Can GitLab hire family members?")
    assert routes==["related_person_hiring"] and "nepotism" in expanded

def test_context_is_bounded_deduplicated_and_cited():
    bundle=build_context([{"chunk_id":"a","chunk":chunk("a","one two three")},{"chunk_id":"a","chunk":chunk("a","duplicate")}],max_chars=100)
    assert len(bundle.sources)==1 and bundle.text.startswith("[S1]") and len(bundle.text)<=100

class FakeCrossEncoder:
    def predict(self,pairs,**kwargs): return np.array([0.1,0.9])
def test_cross_encoder_reorders_candidates():
    results=[{"chunk_id":"a","chunk":chunk("a"),"rrf_score":.03},{"chunk_id":"b","chunk":chunk("b"),"rrf_score":.02}]
    assert CrossEncoderReranker(model=FakeCrossEncoder()).rerank("q",results)[0]["chunk_id"]=="b"

class FakeRetriever:
    def search(self,*args,**kwargs): return {"hybrid_results":[{"chunk_id":"a","chunk":chunk("a","Employees return equipment.")}]}
def test_answer_accepts_valid_sources_and_rejects_bad_citations():
    ok=RAGEngine(FakeRetriever(),lambda prompt:"Employees return equipment [S1].").answer("What happens?")
    assert ok.grounded and ok.sources[0]["chunk_id"]=="a"
    bad=RAGEngine(FakeRetriever(),lambda prompt:"Unsupported [S99].").answer("What happens?")
    assert not bad.grounded and bad.text==INSUFFICIENT
    assert validate_citations("Claim [S1].",build_context([{"chunk_id":"a","chunk":chunk("a")}]))

def test_related_person_policy_requires_direct_hiring_evidence():
    leave={"chunk_id":"a","chunk":chunk("a","Family members may receive carers leave.")}
    policy={"chunk_id":"b","chunk":chunk("b","Hiring a family member requires review.")}
    assert not has_route_evidence(["related_person_hiring"],[leave])
    assert has_route_evidence(["related_person_hiring"],[policy])

def test_safe_equivalent_citation_wrappers_are_canonicalized():
    context=build_context([{"chunk_id":"a","chunk":chunk("a")}])
    for citation in ("\u3010S1\u3011","\uff3bS1\uff3d","(S1)"):
        assert canonicalize_citations(citation)=="[S1]"
        assert validate_citations("Claim "+citation,context)
    assert not validate_citations("Claim \u3010S99\u3011",context)
    assert not validate_citations("Claim \u3010source1\u3011",context)
    assert canonicalize_citations("ordinary (policy) text")=="ordinary (policy) text"
