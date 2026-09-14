"""Reusable FAISS + BM25 + RRF retrieval with optional CPU reranking."""
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any, Iterable
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT / "data/vector_store/policy_faiss.index"
METADATA_PATH = PROJECT_ROOT / "data/vector_store/policy_metadata.json"
CHUNKS_PATH = PROJECT_ROOT / "data/processed/policy_chunks.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9']+\b", str(text).lower())

def normalize_data(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list): return data
    if isinstance(data, dict):
        for key in ("chunks", "documents", "data", "items", "metadata"):
            if isinstance(data.get(key), list): return data[key]
        if data and all(isinstance(value, dict) for value in data.values()): return list(data.values())
    raise ValueError("Unsupported JSON structure")

def get_field(item: dict[str, Any], *names: str, default: str = "") -> str:
    for name in names:
        if item.get(name) is not None: return str(item[name])
    return default

def validate_chunks(chunks: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    by_id = {}
    for chunk in chunks:
        chunk_id = get_field(chunk, "chunk_id", "id").strip()
        if not chunk_id: raise ValueError(f"{label} contains a record without chunk_id")
        if chunk_id in by_id: raise ValueError(f"{label} contains duplicate chunk_id: {chunk_id}")
        by_id[chunk_id] = chunk
    return by_id

def reciprocal_rank_fusion(rankings: Iterable[Iterable[dict[str, Any]]], rrf_k: int = 60) -> list[dict[str, Any]]:
    """Fuse rankings using chunk_id, independent of source ordering."""
    fused: dict[str, dict[str, Any]] = {}
    source_names = ("vector", "bm25", "expansion")
    for source_number, ranking in enumerate(rankings):
        source = source_names[source_number] if source_number < len(source_names) else f"source_{source_number}"
        seen = set()
        for fallback_rank, result in enumerate(ranking, 1):
            chunk = result.get("chunk", result)
            chunk_id = get_field(result, "chunk_id") or get_field(chunk, "chunk_id", "id")
            if not chunk_id or chunk_id in seen: continue
            seen.add(chunk_id); rank = int(result.get("rank", fallback_rank))
            entry = fused.setdefault(chunk_id, {"chunk_id": chunk_id, "chunk": chunk, "rrf_score": 0.0})
            entry["rrf_score"] += 1.0 / (rrf_k + rank)
            entry[f"{source}_rank"] = rank; entry[f"{source}_score"] = result.get("score")
    output = sorted(fused.values(), key=lambda item: (-item["rrf_score"], item["chunk_id"]))
    for rank, result in enumerate(output, 1): result["rank"] = rank
    return output

class HybridRetriever:
    def __init__(self, index_path=INDEX_PATH, metadata_path=METADATA_PATH, chunks_path=CHUNKS_PATH,
                 embedding_model=None, reranker=None, enable_reranker: bool = True) -> None:
        import faiss
        from rank_bm25 import BM25Okapi
        self.index = faiss.read_index(str(index_path))
        self.metadata = normalize_data(json.loads(Path(metadata_path).read_text(encoding="utf-8")))
        self.chunks = normalize_data(json.loads(Path(chunks_path).read_text(encoding="utf-8")))
        if self.index.ntotal != len(self.metadata): raise ValueError("FAISS vector count does not match metadata count")
        self.metadata_by_id = validate_chunks(self.metadata, "metadata")
        self.chunks_by_id = validate_chunks(self.chunks, "chunks")
        if set(self.metadata_by_id) != set(self.chunks_by_id):
            raise ValueError("FAISS metadata and BM25 chunks contain different chunk_id sets; rebuild the index")
        if embedding_model is None:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        self.embedding_model = embedding_model
        self.bm25 = BM25Okapi([tokenize(self._search_text(chunk)) for chunk in self.chunks])
        if reranker is None and enable_reranker:
            from .reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker()
        self.reranker = reranker

    @staticmethod
    def _search_text(chunk):
        return " ".join(filter(None, [get_field(chunk, "title"), get_field(chunk, "category"), get_field(chunk, "chunk_text", "text", "content")]))

    def vector_search(self, query: str, top_k: int = 20):
        embedding = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, positions = self.index.search(np.asarray(embedding, dtype="float32"), min(top_k, self.index.ntotal))
        results = []
        for rank, (score, position) in enumerate(zip(scores[0], positions[0]), 1):
            if position < 0: continue
            chunk = self.metadata[int(position)]
            results.append({"rank": rank, "score": float(score), "index": int(position), "chunk_id": chunk["chunk_id"], "chunk": chunk})
        return results

    def bm25_search(self, query: str, top_k: int = 20):
        scores = self.bm25.get_scores(tokenize(query)); positions = np.argsort(scores)[::-1]; results = []
        for position in positions[:min(top_k, len(positions))]:
            if scores[position] <= 0: continue
            chunk = self.chunks[int(position)]
            results.append({"rank": len(results)+1, "score": float(scores[position]), "index": int(position), "chunk_id": chunk["chunk_id"], "chunk": chunk})
        return results

    def search(self, query: str, top_k: int = 10, candidate_k: int = 20):
        from .query_processing import expand_query
        query = query.strip()
        if not query: return {"query": query, "expanded_query": query, "routes": [], "vector_results": [], "bm25_results": [], "hybrid_results": []}
        expanded_query, routes = expand_query(query)
        vector = self.vector_search(expanded_query, candidate_k); bm25 = self.bm25_search(expanded_query, candidate_k)
        candidates = reciprocal_rank_fusion((vector, bm25))[:candidate_k]
        candidates = self.reranker.rerank(query, candidates, top_k) if self.reranker else candidates[:top_k]
        for rank, result in enumerate(candidates, 1): result["rank"] = rank
        return {"query": query, "expanded_query": expanded_query, "routes": routes, "vector_results": vector, "bm25_results": bm25, "hybrid_results": candidates}

def display_results(output):
    for result in output["hybrid_results"]:
        print(f"{result['rank']}. {result['chunk_id']} | {get_field(result['chunk'], 'title')} | score={result.get('rerank_score', result['rrf_score']):.4f}")

def interactive_mode():
    retriever = HybridRetriever()
    while True:
        question = input("Question (or exit): ").strip()
        if question.lower() in {"exit", "quit", "q"}: break
        display_results(retriever.search(question))

if __name__ == "__main__": interactive_mode()
