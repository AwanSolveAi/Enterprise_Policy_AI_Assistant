"""UI-facing orchestration over the frozen retrieval and RAG interfaces."""
from __future__ import annotations

import importlib
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from src.generation.context_builder import build_context
from src.generation.rag_engine import Answer, INSUFFICIENT, RAGEngine
from src.retrieval.hybrid_retriever import HybridRetriever

BACKEND_ENV = "POLICY_GENERATOR_BACKEND"

@dataclass(frozen=True)
class SourceView:
    citation: str
    chunk_id: str
    title: str
    category: str
    country: str
    source_url: str
    evidence: str

@dataclass(frozen=True)
class AssistantResponse:
    question: str
    answer: str
    grounded: bool
    latency_ms: float
    sources: tuple[SourceView, ...]
    debug: dict[str, Any]

def load_generator(spec: str | None) -> Callable[[str], str] | None:
    """Load a replaceable ``prompt -> answer`` callable from module:attribute."""
    if not spec:
        return None
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Generator backend must use module.path:callable format")
    generator = getattr(importlib.import_module(module_name), attribute)
    if not callable(generator):
        raise TypeError(f"Configured generator is not callable: {spec}")
    return generator

def backend_label(spec: str | None) -> str:
    return spec or "Not configured"

class PolicyAssistantService:
    def __init__(self, retriever: Any, generator: Callable[[str], str] | None):
        self.retriever = retriever
        self.generator = generator
        self.rag = RAGEngine(retriever, generator)

    @classmethod
    def from_environment(cls) -> "PolicyAssistantService":
        return cls(HybridRetriever(), load_generator(os.getenv(BACKEND_ENV)))

    def ask(self, question: str) -> AssistantResponse:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        started = time.perf_counter()
        retrieval = self.retriever.search(question, top_k=self.rag.context_candidate_k)
        answer: Answer = self.rag.answer_from_retrieval(question, retrieval)
        latency_ms = (time.perf_counter() - started) * 1000

        results = retrieval.get("hybrid_results", [])
        by_id = {result["chunk_id"]: result.get("chunk", {}) for result in results}
        sources = tuple(
            SourceView(
                citation=source["citation"],
                chunk_id=source["chunk_id"],
                title=source["title"],
                category=str(by_id.get(source["chunk_id"], {}).get("category", "Not specified")),
                country=str(by_id.get(source["chunk_id"], {}).get("country", "Not specified")),
                source_url=source["source_url"],
                evidence=str(by_id.get(source["chunk_id"], {}).get("chunk_text", "")).strip(),
            )
            for source in answer.sources
            if source["citation"] in set(re.findall(r"\[(S\d+)\]", answer.text))
        )
        context = build_context(results, query=question)
        debug_results = [
            {
                "rank": result.get("rank"),
                "chunk_id": result.get("chunk_id"),
                "title": result.get("chunk", {}).get("title"),
                "category": result.get("chunk", {}).get("category"),
                "reranker_score": result.get("rerank_score"),
                "selected_for_context": result.get("chunk_id") in context.chunk_ids,
            }
            for result in results
        ]
        debug = {
            "routes": retrieval.get("routes", []),
            "expanded_query": retrieval.get("expanded_query", question),
            "context_chunk_ids": list(context.chunk_ids),
            "retrieved_sources": debug_results,
        }
        return AssistantResponse(question, answer.text, answer.grounded, round(latency_ms, 2), sources, debug)

    @property
    def available(self) -> bool:
        return self.generator is not None
