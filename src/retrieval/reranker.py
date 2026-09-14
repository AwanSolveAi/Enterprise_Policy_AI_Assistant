"""Lazy, CPU-only CrossEncoder reranking."""
from typing import Any
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
class CrossEncoderReranker:
    def __init__(self, model_name=DEFAULT_MODEL, model: Any=None, batch_size=8, max_length=512):
        self.model_name, self.model, self.batch_size, self.max_length = model_name, model, batch_size, max_length
    def _load(self):
        if self.model is None:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device="cpu", max_length=self.max_length)
    def rerank(self, query, results, top_k=10):
        if not results: return []
        self._load(); pairs=[]
        for result in results:
            chunk=result["chunk"]
            pairs.append((query, "\n".join(filter(None, [str(chunk.get("title", "")), str(chunk.get("category", "")), str(chunk.get("chunk_text", chunk.get("text", "")))]))))
        scores=self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        for result, score in zip(results, scores): result["rerank_score"] = float(score)
        return sorted(results, key=lambda item: (-item["rerank_score"], -item["rrf_score"], item["chunk_id"]))[:top_k]
