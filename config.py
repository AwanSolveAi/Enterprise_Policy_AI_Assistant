"""Central defaults for application composition."""
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVAL_TOP_K = 5
RETRIEVAL_CANDIDATE_K = 20
MAX_CONTEXT_CHARS = 7000
