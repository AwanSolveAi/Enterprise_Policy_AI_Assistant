"""Answer-quality benchmark independent of any paid LLM provider.

``deterministic`` mode is an evidence fixture: it verifies retrieval-to-context,
citation, scoring, and abstention mechanics. It is not an LLM quality score.
Use ``--backend package.module:function`` for a real local/provider generator.
"""
from __future__ import annotations
import argparse, importlib, json, re, statistics, time, unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from src.generation.context_builder import ContextBundle, build_context, _resolve_jurisdictions
from src.generation.rag_engine import INSUFFICIENT, RAGEngine, canonicalize_citations, validate_citations
from src.retrieval.hybrid_retriever import HybridRetriever

BENCHMARK_PATH = Path(__file__).with_name("answer_benchmark.json")
STOPWORDS={"a","an","and","are","as","at","be","by","for","from","in","is","it","of","on","or","that","the","their","to","what","when","where","which","who","with"}
EVAL_STOPWORDS=STOPWORDS|{"all","also","any","can","does","each","if","into","may","must","not","off","only","should","so","such","than","then","they","this","those","through","up","was","were","will","would","your"}
TOKEN_ALIASES={"employee":"team","employees":"team","employment":"contract","member":"team","members":"team",
               "boarding":"offboarding",
               "supplied":"provide","provides":"provide","provided":"provide","providing":"provide",
               "facilitated":"coordinate","facilitates":"coordinate","coordinated":"coordinate","coordinates":"coordinate",
               "reported":"report","reporting":"report","reports":"report","conducted":"conduct","conducts":"conduct",
               "investigation":"investigate","investigations":"investigate","returned":"return","returns":"return",
               "record":"enter","recorded":"enter","recording":"enter","entered":"enter","entitlements":"entitlement",
               "immediately":"without_delay","compliance":"comply","compliant":"comply",
               "involuntary":"terminate","termination":"terminate","terminated":"terminate"}
SEMANTIC_SUPPORT_THRESHOLD=0.42
_SEMANTIC_MODEL=None

def load_benchmark(path: Path=BENCHMARK_PATH) -> list[dict[str,Any]]:
    cases=json.loads(path.read_text(encoding="utf-8")); ids=set()
    if not 25 <= len(cases) <= 30: raise ValueError("benchmark must contain 25-30 cases")
    for case in cases:
        required={"id","group","question","answerable","evidence_terms"}
        if not required <= case.keys(): raise ValueError(f"missing fields in {case.get('id','unknown')}")
        if case["id"] in ids: raise ValueError(f"duplicate case id: {case['id']}")
        ids.add(case["id"])
        if not case["answerable"] and not case.get("expected_abstention"): raise ValueError(f"unanswerable case must expect abstention: {case['id']}")
    return cases

def load_backend(spec: str) -> Callable[[str],str]:
    module_name, separator, function_name=spec.partition(":")
    if not separator: raise ValueError("backend must be package.module:function")
    function=getattr(importlib.import_module(module_name),function_name)
    if not callable(function): raise TypeError("backend is not callable")
    return function

def _blocks(context: ContextBundle) -> list[tuple[str,str]]:
    output=[]
    for raw in context.text.split("\n\n"):
        match=re.match(r"\[(S\d+)\]",raw)
        if match: output.append((match.group(1),raw))
    return output

def deterministic_generator(case: dict[str,Any], context: ContextBundle, chunks_by_id: dict[str,dict] | None=None) -> str:
    """Emit only benchmark facts found verbatim in context, with their labels."""
    if not case["answerable"]: return INSUFFICIENT
    claims=[]; sources={source["citation"]:source for source in context.sources}
    for term in case["evidence_terms"]:
        candidates=[]
        for label,text in _blocks(context):
            if term.lower() not in text.lower(): continue
            chunk=(chunks_by_id or {}).get(sources[label]["chunk_id"],{})
            source_match=(not case.get("expected_document_id") or chunk.get("document_id")==case["expected_document_id"]) and (not case.get("expected_category") or chunk.get("category")==case["expected_category"])
            candidates.append((source_match,label,text))
        match=next(((label,text) for source_match,label,text in candidates if source_match),None) or next(((label,text) for _,label,text in candidates),None)
        if match is None: return INSUFFICIENT
        claims.append(f"{term} [{match[0]}]")
    return "; ".join(claims)+"."

def evidence_coverage(answer: str, terms: list[str]) -> float:
    if not terms: return 1.0
    answer_tokens={_singular(token) for token in _tokens(answer)}
    covered=0
    for term in terms:
        if {_singular(token) for token in _tokens(term)}<=answer_tokens:
            covered+=1; continue
        score,_=_semantic_score(term,[answer])
        covered+=score>=SEMANTIC_SUPPORT_THRESHOLD
    return covered/len(terms)

def _normalize(text: str) -> str:
    text=unicodedata.normalize("NFKC",text).lower()
    text=re.sub(r"\bany\s+time\b","anytime",text)
    text=re.sub(r"\bpto\b","paid time off",text)
    text=re.sub(r"\bwithout\s+delay\b","without_delay",text)
    text=re.sub(r"\bdirected\s+to\b|\breach(?:ed|es|ing)?\s+out\s+to\b","contact",text)
    text=re.sub(r"\bnot\s+disclosed\b|\bcannot\s+be\s+shared\b","withheld",text)
    text=re.sub(r"\bnot\s+(?:be\s+)?shared\b","withheld",text)
    text=re.sub(r"[\u2010-\u2015\u2212-]+"," ",text)
    text=re.sub(r"[*_~`]","",text)
    text=re.sub(r"\[\s*s\s*(\d+)\s*\]|\(\s*s\s*(\d+)\s*\)"," ",text,flags=re.I)
    return " ".join(re.findall(r"[a-z0-9@.']+",text))

def _labels(text: str) -> list[str]:
    text=canonicalize_citations(text)
    return [f"S{a or b}" for a,b in re.findall(r"\[\s*S\s*(\d+)\s*\]|\(\s*S\s*(\d+)\s*\)",text,re.I)]

def _tokens(text: str) -> set[str]:
    words=[]
    for word in _normalize(text).split():
        word=word.strip(".'")
        if word in EVAL_STOPWORDS or len(word)<=2: continue
        words.append(TOKEN_ALIASES.get(word,word))
    return set(words)

def _claim_units(answer: str) -> list[tuple[str,list[str]]]:
    answer=canonicalize_citations(answer)
    units=[]
    for paragraph in re.split(r"\n\s*\n",answer):
        paragraph_labels=_labels(paragraph)
        for line in paragraph.splitlines():
            line=re.sub(r"^\s*(?:[-*]|\d+[.)])\s*","",line).strip()
            if not line or re.fullmatch(r"[|:\-\s]+",line): continue
            if re.match(r"^\s*(?:\[\s*S\s*\d+\s*\]|\(\s*S\s*\d+\s*\))",line,re.I):
                if units:
                    claim,labels=units[-1]; units[-1]=(claim,list(dict.fromkeys(labels+_labels(line))))
                continue
            structural=re.sub(r"[*_~`]","",line).strip().rstrip(":\u2010\u2011\u2012\u2013\u2014- ")
            if line.rstrip().endswith(":") or (line.rstrip().endswith(("-","\u2010","\u2011","\u2012","\u2013","\u2014")) and len(_tokens(structural))<=5):
                continue
            protected=re.sub(r"\b(?:[A-Z]\.){2,}",lambda match:match.group().replace(".","<DOT>"),line)
            for sentence in re.split(r"(?<=[.!?])\s+|\s*;\s*",protected):
                sentence=sentence.replace("<DOT>",".")
                if sentence.strip() and _normalize(sentence): units.append((sentence.strip(),_labels(sentence) or paragraph_labels))
    if len(units)>1 and _normalize(units[0][0]).strip(".") in {"yes","no"}:
        response,_=units.pop(0); claim,labels=units[0]; units[0]=(response+" "+claim,labels)
    return units

def _semantic_model():
    global _SEMANTIC_MODEL
    if _SEMANTIC_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SEMANTIC_MODEL=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",local_files_only=True)
    return _SEMANTIC_MODEL

def _evidence_sentences(text: str) -> list[str]:
    protected=re.sub(r"\b(?:[A-Z]\.){2,}",lambda match:match.group().replace(".","<DOT>"),text)
    return [part.replace("<DOT>",".").strip() for part in re.split(r"(?<=[.!?])\s+|\n+",protected) if part.strip()]

def _semantic_score(claim: str, evidence_blocks: list[str], adjacent: bool=True) -> tuple[float,str]:
    candidates=_evidence_windows(evidence_blocks,adjacent)
    if not candidates: return 0.0,""
    vectors=_semantic_model().encode([claim,*candidates],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
    scores=vectors[1:]@vectors[0]; index=int(scores.argmax())
    return float(scores[index]),candidates[index]

def _evidence_windows(evidence_blocks: list[str], adjacent: bool=True) -> list[str]:
    candidates=[]
    for evidence in evidence_blocks:
        sentences=_evidence_sentences(evidence)
        candidates.extend(sentences)
        candidates.extend(part.strip() for sentence in sentences for part in re.split(r",\s*",sentence) if part.strip())
        if adjacent:
            candidates.extend(" ".join(sentences[index:index+2]) for index in range(len(sentences)-1))
    return candidates

def _singular(word: str) -> str:
    if re.search(r"[^aeiou]ies$",word): return word[:-3]+"y"
    if re.search(r"(?:ches|shes|xes|zes)$",word): return word[:-2]
    if word.endswith("s") and not word.endswith(("ss","us","is")): return word[:-1]
    return word

def _meaningful_phrase(text: str) -> str:
    return " ".join(_singular(word) for word in _normalize(text).split() if word not in EVAL_STOPWORDS and len(word)>2)

def _explicit_global_scope(text: str) -> bool:
    return bool(re.search(
        r"\b(?:all locations|worldwide|global(?:ly)?|company-wide|all team members (?:regardless|irrespective) of (?:their )?(?:work )?location)\b",
        text,re.I,
    ))

def _jurisdictions(text: str) -> set[str]:
    return _resolve_jurisdictions(text)

def _explicitly_broadens_scope(claim: str) -> bool:
    return bool(re.search(r"\b(?:all locations|worldwide|global(?:ly)?|company-wide|every country|regardless of location)\b",claim,re.I))

def _jurisdiction_generalization(claim: str, evidence: str, discourse_scope: set[str]) -> bool:
    jurisdiction=re.compile(
        r"\b(?:United States(?: of America)?|U\.?S\.?(?:A\.?)?|American|Australia|Australian|"
        r"France|French|Ireland|Irish|India|Indian|United Kingdom|U\.?K\.?|British|"
        r"United Arab Emirates|U\.?A\.?E\.?|Emirati)\b",re.I)
    markers={match.group(0) for match in jurisdiction.finditer(evidence)}
    if not markers or _explicit_global_scope(evidence): return False
    claim_scope=_jurisdictions(claim)
    if claim_scope or (discourse_scope and not _explicitly_broadens_scope(claim)): return False
    generalized=re.search(r"\b(all|any|company-wide|every|the process|team members?|employees?|where applicable)\b",claim,re.I)
    return bool(generalized)

def _scope_generalization(claim: str, evidence: str) -> bool:
    universal=re.compile(r"\b(all|every|equally|company-wide|universal(?:ly)?)\b",re.I)
    return bool(universal.search(claim) and not universal.search(evidence))

def _causal_parts(text: str) -> tuple[set[str],set[str]] | None:
    result_pattern=re.compile(r"\b(?:may |can |will )?(?:lead(?:s)? to|result(?:s)? in|trigger(?:s)?|be subject to)\b",re.I)
    reason_pattern=re.compile(r"\b(?:because|since|due to)\b",re.I)
    if match:=result_pattern.search(text):
        return _tokens(text[:match.start()]),_tokens(text[match.end():])
    if match:=reason_pattern.search(text):
        return _tokens(text[match.end():]),_tokens(text[:match.start()])
    return None

def _segment_match(left: set[str],right: set[str],lexical_threshold: float) -> bool:
    if not left or not right: return False
    generic={"equipment","team","employee","company","policy","person","individual","failure","non"}
    if (left&right)-generic: return True
    lexical=len(left&right)/max(1,min(len(left),len(right)))
    if lexical>=lexical_threshold: return True
    score,_=_semantic_score(" ".join(sorted(left)),[" ".join(sorted(right))],adjacent=False)
    return score>=0.7

def _causal_relationship_supported(claim: str, evidence_blocks: list[str]) -> bool:
    claim_parts=_causal_parts(claim)
    if not claim_parts: return True
    for sentence in _evidence_windows(evidence_blocks,adjacent=False):
        evidence_parts=_causal_parts(sentence)
        if not evidence_parts: continue
        cause,effect=claim_parts; evidence_cause,evidence_effect=evidence_parts
        score,_=_semantic_score(claim,[sentence],adjacent=False)
        if score>=SEMANTIC_SUPPORT_THRESHOLD and _segment_match(cause,evidence_cause,0.65) and _segment_match(effect,evidence_effect,0.5):
            return True
    return False

def grounding_evaluations(answer: str, context: ContextBundle, question: str="") -> list[dict[str,Any]]:
    """Classify factual claims against only their cited evidence."""
    if answer==INSUFFICIENT: return []
    source_text={label:text for label,text in _blocks(context)}
    answer_labels=_labels(answer); evaluations=[]
    causal=re.compile(r"\b(because|since|due to|caus\w*|ensure\w*|failure|lead\w* to|result\w* in|so that|trigger\w*)\b",re.I)
    discourse_scope=_jurisdictions(question)
    prior_claim=""
    for claim,explicit_labels in _claim_units(answer):
        labels=explicit_labels or answer_labels
        valid_labels=[label for label in labels if label in source_text]
        if not labels:
            evaluations.append({"claim":claim,"status":"UNCITED","citations":[],"score":0.0}); continue
        evidence_blocks=[source_text[label] for label in valid_labels]
        evidence=" ".join(evidence_blocks)
        evaluated_claim=re.sub(r"^\s*\*\*[^*]{1,60}\*\*\s*[\u2010-\u2015-]\s*","",claim)
        comparison_claim=(prior_claim+" "+evaluated_claim) if prior_claim and re.match(r"^\s*(?:they|them|their|employees?|team members?)\b",evaluated_claim,re.I) else evaluated_claim
        phrase=_meaningful_phrase(evaluated_claim)
        directly_supported=bool(phrase) and any(phrase in _meaningful_phrase(block) for block in evidence_blocks)
        claim_words=_tokens(evaluated_claim)
        bounded_lexical_support=any(
            len(claim_words & _tokens(window))/len(claim_words)>=0.65
            for window in _evidence_windows(evidence_blocks)
        ) if claim_words else False
        score,best_sentence=_semantic_score(comparison_claim,evidence_blocks)
        supported=bool(valid_labels) and (directly_supported or bounded_lexical_support or score>=SEMANTIC_SUPPORT_THRESHOLD)
        if supported and _jurisdiction_generalization(claim,evidence,discourse_scope): supported=False
        if supported and _scope_generalization(claim,evidence): supported=False
        if supported and causal.search(claim):
            supported=_causal_relationship_supported(evaluated_claim,evidence_blocks)
        evaluations.append({"claim":claim,"status":"SUPPORTED" if supported else "UNSUPPORTED",
                            "citations":valid_labels,"score":round(score,4)})
        claim_scope=_jurisdictions(claim)
        if claim_scope: discourse_scope=claim_scope
        prior_claim=evaluated_claim
    return evaluations

def unsupported_claims(answer: str, context: ContextBundle, question: str="") -> list[str]:
    """Return claims classified as unsupported or uncited."""
    return [item["claim"] for item in grounding_evaluations(answer,context,question) if item["status"]!="SUPPORTED"]

def citation_source_correct(answer: str, context: ContextBundle, chunks_by_id: dict[str,dict], case: dict) -> bool | None:
    if not case["answerable"]: return None
    labels=set(re.findall(r"\[(S\d+)\]",answer)); sources={s["citation"]:s for s in context.sources}
    if not labels: return False
    cited=[chunks_by_id[sources[label]["chunk_id"]] for label in labels if label in sources]
    if len(cited)!=len(labels): return False
    document=case.get("expected_document_id"); category=case.get("expected_category")
    return all((not document or c.get("document_id")==document) and (not category or c.get("category")==category) for c in cited)

def evaluate_case(case: dict, retriever: Any, backend: Callable[[str],str] | None=None) -> dict[str,Any]:
    started=time.perf_counter(); retrieval=retriever.search(case["question"],top_k=20); retrieval_ms=(time.perf_counter()-started)*1000
    results=retrieval["hybrid_results"]; context=build_context(results,query=case["question"])
    chunks={r["chunk_id"]:r["chunk"] for r in results}
    generator=backend or (lambda prompt: deterministic_generator(case,context,chunks)); captured=[]
    def capture(prompt):
        output=generator(prompt); captured.append(output); return output
    answer_started=time.perf_counter(); answer=RAGEngine(retriever,capture).answer_from_retrieval(case["question"],retrieval); generation_ms=(time.perf_counter()-answer_started)*1000
    raw_answer=captured[0] if captured else answer.text
    retrieved_docs={c.get("document_id") for c in chunks.values()}; retrieved_categories={c.get("category") for c in chunks.values()}
    source_ok=(case.get("expected_document_id") in retrieved_docs and case.get("expected_category") in retrieved_categories) if case["answerable"] else None
    abstained=not answer.grounded; citation_ok=validate_citations(answer.text,context) if answer.grounded else (True if not case["answerable"] else False)
    source_citation_ok=citation_source_correct(answer.text,context,chunks,case) if answer.grounded else (None if not case["answerable"] else False)
    coverage=evidence_coverage(answer.text,case["evidence_terms"]); unsupported=unsupported_claims(raw_answer,context,case["question"])
    answerability_ok=(answer.grounded==case["answerable"]); abstention_ok=(abstained==case.get("expected_abstention",False))
    relevance_ok=(coverage==1.0) if case["answerable"] else abstained
    grounded_accuracy=(answerability_ok and relevance_ok and not unsupported)
    passed=grounded_accuracy and citation_ok and (source_citation_ok is not False) and (source_ok is not False) and abstention_ok
    return {"id":case["id"],"group":case["group"],"answerable":case["answerable"],"raw_model_answer":raw_answer,"answer":answer.text,
            "retrieval_source_correct":source_ok,"grounded_answer_accuracy":grounded_accuracy,"citation_valid":citation_ok,
            "citation_source_correct":source_citation_ok,"abstention_correct":abstention_ok,"answer_relevant":relevance_ok,
            "unsupported_claims":unsupported,"evidence_coverage":coverage,"completeness":coverage,"guard_intervened":raw_answer.strip()!=answer.text.strip(),"retrieval_ms":round(retrieval_ms,2),
            "generation_ms":round(generation_ms,2),"total_ms":round(retrieval_ms+generation_ms,2),"passed":passed,
            "top_chunk_ids":[r["chunk_id"] for r in results],"context_chunk_ids":list(context.chunk_ids)}

def summarize(rows: list[dict], mode: str) -> dict[str,Any]:
    def rate(field, applicable=lambda row: True):
        values=[bool(row[field]) for row in rows if applicable(row) and row[field] is not None]
        return round(sum(values)/len(values),4) if values else None
    latencies=[row["total_ms"] for row in rows]; completeness=[r["completeness"] for r in rows if r["answerable"]]
    return {"mode":mode,"cases":len(rows),"distribution":dict(sorted(Counter(r["group"] for r in rows).items())),
            "retrieval_source_accuracy":rate("retrieval_source_correct",lambda r:r["answerable"]),
            "grounded_answer_accuracy":rate("grounded_answer_accuracy"),"citation_validity":rate("citation_valid",lambda r:r["answerable"]),
            "citation_source_accuracy":rate("citation_source_correct",lambda r:r["answerable"]),
            "abstention_accuracy":rate("abstention_correct",lambda r:not r["answerable"]),"answer_relevance":rate("answer_relevant"),
            "mean_completeness":round(statistics.mean(completeness),4) if completeness else None,
            "unsupported_claim_free_rate":round(sum(not r["unsupported_claims"] for r in rows)/len(rows),4),
            "latency_ms":{"mean":round(statistics.mean(latencies),2),"median":round(statistics.median(latencies),2),"max":round(max(latencies),2)},
            "passed":sum(r["passed"] for r in rows),"failed":sum(not r["passed"] for r in rows),"failed_cases":[r["id"] for r in rows if not r["passed"]]}

def run(retriever, cases, backend=None, mode="deterministic", on_case=None):
    rows=[]
    for index,case in enumerate(cases,1):
        rows.append(evaluate_case(case,retriever,backend))
        if on_case: on_case(index,len(cases),rows)
    return {"summary":summarize(rows,mode),"cases":rows}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--backend",help="optional package.module:function generator"); parser.add_argument("--no-reranker",action="store_true"); parser.add_argument("--output",type=Path)
    args=parser.parse_args(); backend=load_backend(args.backend) if args.backend else None; mode="real_llm" if backend else "deterministic_logic"
    def checkpoint(index,total,rows):
        partial={"summary":summarize(rows,mode),"cases":rows,"complete":index==total}
        if args.output: args.output.write_text(json.dumps(partial,indent=2,ensure_ascii=False),encoding="utf-8")
        print(f"Completed {index}/{total}: {rows[-1]['id']}",flush=True)
    report=run(HybridRetriever(enable_reranker=not args.no_reranker),load_benchmark(),backend,mode,checkpoint); report["complete"]=True; rendered=json.dumps(report,indent=2,ensure_ascii=False)
    if args.output: args.output.write_text(rendered,encoding="utf-8")
    print(json.dumps(report["summary"],indent=2))
if __name__=="__main__": main()
