# Architecture audit

## Preserved

- Structured policy documents and stable chunk IDs produced by the current ingestion pipeline.
- Word-safe chunking, overlap, metadata, normalized MiniLM embeddings, FAISS inner-product search, BM25, and RRF.
- Existing vector index and metadata; no destructive rebuild was needed.

## Findings and targeted changes

- Hybrid fusion used array positions, so differing FAISS/BM25 ordering could silently join unrelated chunks. `hybrid_retriever.py` now validates ID uniqueness/set equality and fuses by `chunk_id`.
- Natural-language vocabulary mismatches needed small deterministic routing. `query_processing.py` adds only offboarding and related-person hiring expansions.
- No second-stage relevance model existed. `reranker.py` adds a lazy CPU MiniLM CrossEncoder over at most 20 candidates.
- Generation, root CLI, configuration, README, and tests were empty. Context construction and provider-neutral grounded generation now enforce bounded evidence, source labels, valid citations, and fail-closed behavior.
- Runtime dependencies were undeclared. Requirements now match imports and include pytest.
- There was no repeatable quality measurement. `evaluation/` contains labeled retrieval cases and answer guardrail checks.

## Corpus limitation

The indexed corpus contains the full GitLab Offboarding page. It does **not**
contain policy text about hiring family members. The raw “Hiring & Talent
Acquisition Handbook” record is a short landing page with no family/relative
language, and the current curation rules exclude generic `handbook` titles.
The system must therefore abstain on that answer until ingestion includes the
authoritative page; query expansion cannot replace missing evidence.
