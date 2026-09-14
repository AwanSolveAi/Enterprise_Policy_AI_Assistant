# Enterprise Policy AI

CPU-oriented policy RAG pipeline that preserves the existing structured chunks,
MiniLM embeddings, FAISS index, BM25 search, and reciprocal-rank fusion. Fusion
uses `chunk_id`; an optional MiniLM CrossEncoder reranks the fused candidates.

```powershell
python -m pip install -r requirements.txt
python main.py "What happens when an employee leaves GitLab?"
python -m evaluation.evaluate --no-reranker
python -m evaluation.answer_quality --no-reranker
python -m pytest -q
```

The CrossEncoder is lazy-loaded on first use and may download once. Use
`--no-reranker` for an offline/fast retrieval pass. Answer generation is
provider-neutral: pass a callable to `RAGEngine`. If no generator is configured,
or returned citations are invalid, the engine fails closed rather than presenting
an unsupported answer.

The 28-case answer benchmark defaults to deterministic evidence-fixture mode;
its scores validate grounding/citation/abstention mechanics, not LLM prose
quality. A real local or provider-backed callable can be evaluated with
`--backend package.module:function`. The callable receives one prompt and must
return answer text; no paid provider is required by the project.

See [AUDIT.md](AUDIT.md) for the architecture audit and known corpus gap.

## Local Web UI

Configure any existing `prompt -> answer` callable, then launch Streamlit:

```powershell
$env:POLICY_GENERATOR_BACKEND="src.generation.ollama_backend:generate"
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

The backend value is replaceable and is not coupled to the UI. Without a
configured backend, the interface opens in a safe disabled state and explains
how to configure generation. Retrieval diagnostics are available only through
the optional developer sidebar toggle; raw prompts are never displayed.
