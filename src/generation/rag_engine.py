"""Grounded answer orchestration with citations and fail-closed guardrails."""
import re
from dataclasses import dataclass
from typing import Any, Callable
from .context_builder import ContextBundle, build_context
INSUFFICIENT = "I couldn't find enough policy evidence in the indexed sources to answer that question reliably."
@dataclass(frozen=True)
class Answer:
    text: str
    sources: tuple[dict[str,str], ...]
    grounded: bool

_JURISDICTION_PATTERNS={
    "United States":re.compile(r"(?<!\w)(?:United States(?: of America)?|U\.?\s*S\.?(?:\s*A\.?)?|USA|American)(?!\w)",re.I),
    "Australia":re.compile(r"(?<!\w)(?:Australia|Australian)(?!\w)",re.I),
    "France":re.compile(r"(?<!\w)(?:France|French)(?!\w)",re.I),
    "Ireland":re.compile(r"(?<!\w)(?:Ireland|Irish)(?!\w)",re.I),
    "India":re.compile(r"(?<!\w)(?:India|Indian)(?!\w)",re.I),
}
_NARROW_SCOPE_PATTERNS={
    "United States":re.compile(r"(?:US|USA)[- ]specific|US(?:-based)? team members|GitLab Inc \(USA\)",re.I),
    "Australia":re.compile(r"Australia[- ]specific|Australian team members|GitLab PTY Ltd\. Australia",re.I),
    "France":re.compile(r"GitLab France|France[- ]specific|French (?:employees|team members|law)",re.I),
    "Ireland":re.compile(r"GitLab Ireland|Ireland[- ]specific|Irish (?:employees|team members|law)",re.I),
    "India":re.compile(r"GitLab India|India[- ]specific|Indian (?:employees|team members|law)",re.I),
}

def _scope_guidance(question: str, context: ContextBundle) -> str:
    requested={country for country,pattern in _JURISDICTION_PATTERNS.items() if pattern.search(question)}
    narrower={country for country,pattern in _NARROW_SCOPE_PATTERNS.items() if pattern.search(context.text)}
    unrequested=sorted(narrower-requested)
    if not unrequested:
        return ""
    countries=", ".join(unrequested)
    return (f"The context contains provisions explicitly limited to: {countries}. "
            "Because the question does not request those jurisdictions, either omit those provisions or explicitly qualify every such statement with its narrower jurisdiction.\n")

def build_prompt(question: str, context: ContextBundle) -> str:
    scope_guidance=_scope_guidance(question,context)
    return f"""Answer only from POLICY CONTEXT. Cite every factual statement with its supporting [S#] source.
Every factual introductory or summary sentence must carry its own directly associated [S#] citation; citations on later bullets do not support an uncited summary sentence.
State only facts explicitly supported by the supplied source context.
Answer the question directly and concisely, including only material necessary to answer it.
For questions asking who to contact, where to report, what options exist, or which channels or routes are available, include every materially relevant alternative supported by the selected evidence unless the question explicitly requests only one.
For a generic question, do not include any jurisdiction-specific procedure, exception, or legal requirement unless it is necessary for an accurate answer; include it only when the user requests that jurisdiction.
Do not add inferred summary or outcome sentences such as "These steps ensure...".
Do not infer, generalize, combine, or invent procedures, consequences, reasons, outcomes, applicability, or policy scope.
Do not turn a policy for one jurisdiction or group into a company-wide rule.
For a generic question, prefer generally applicable or global evidence. Use jurisdiction-specific evidence only when the question requests that jurisdiction or clearly identify it as jurisdiction-specific.
Do not create causal or procedural links between separate source passages unless the context explicitly states the link.
State a consequence, penalty, sanction, requirement, deadline, or causal outcome only when the same cited passage explicitly attaches it to the same action or condition.
Do not present a provision as universal or global unless its cited passage explicitly establishes that scope. If a passage names a narrower jurisdiction, entity, or subsection, preserve that scope; if scope is unclear, omit the scoped detail rather than universalizing it.
{scope_guidance}If evidence is narrower than the question, state that narrower scope explicitly in the answer; never silently universalize it.
If a requested detail is not explicitly supported, omit it rather than completing it from general knowledge.
Prefer a short, incomplete but supported answer over a comprehensive answer containing inference.
If the context does not directly answer the question, reply exactly: {INSUFFICIENT}
Do not use outside knowledge or infer policy. Be concise and distinguish voluntary resignation from involuntary termination.

QUESTION: {question}

POLICY CONTEXT:
{context.text}"""
def canonicalize_citations(answer: str) -> str:
    pattern=r"(?:\u3010|\uff3b|\()\s*[sS]\s*([1-9]\d*)\s*(?:\u3011|\uff3d|\))"
    return re.sub(pattern,lambda match:f"[S{match.group(1)}]",answer)

def validate_citations(answer: str, context: ContextBundle) -> bool:
    answer=canonicalize_citations(answer)
    cited=set(re.findall(r"\[(S\d+)\]", answer)); allowed={source["citation"] for source in context.sources}
    return bool(cited) and cited <= allowed

def has_route_evidence(routes, results) -> bool:
    """Require direct co-occurring evidence for routes prone to false matches."""
    if "related_person_hiring" not in routes:
        return True
    people_terms=("family member","relative","significant other","nepotism")
    hiring_terms=("hire","hiring","employment","employ")
    for result in results:
        chunk=result.get("chunk",result)
        text=(str(chunk.get("title",""))+" "+str(chunk.get("chunk_text",chunk.get("text","")))).lower()
        if any(term in text for term in people_terms) and any(term in text for term in hiring_terms):
            return True
    return False
class RAGEngine:
    def __init__(self, retriever: Any, generator: Callable[[str],str]|None=None, min_rerank_score=-8.0, context_candidate_k=20):
        self.retriever, self.generator, self.min_rerank_score, self.context_candidate_k = retriever, generator, min_rerank_score, context_candidate_k
    def answer(self, question: str, top_k=5) -> Answer:
        retrieval=self.retriever.search(question, top_k=max(top_k,self.context_candidate_k))
        return self.answer_from_retrieval(question, retrieval)
    def answer_from_retrieval(self, question: str, retrieval: dict) -> Answer:
        """Generate from a supplied retrieval result (useful for evaluation/UI tracing)."""
        results=retrieval["hybrid_results"]
        if not has_route_evidence(retrieval.get("routes",[]), results): return Answer(INSUFFICIENT, (), False)
        if not results or ("rerank_score" in results[0] and results[0]["rerank_score"] < self.min_rerank_score): return Answer(INSUFFICIENT, (), False)
        context=build_context(results,query=question)
        if not context.text or self.generator is None: return Answer(INSUFFICIENT, context.sources, False)
        answer=canonicalize_citations(self.generator(build_prompt(question, context)).strip())
        if answer == INSUFFICIENT or not validate_citations(answer, context): return Answer(INSUFFICIENT, context.sources, False)
        return Answer(answer, context.sources, True)
