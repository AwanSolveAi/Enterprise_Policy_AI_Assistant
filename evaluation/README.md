# Answer-quality evaluation

`answer_benchmark.json` contains 28 versioned cases. The default runner uses a
deterministic evidence fixture. It only emits expected evidence terms when they
occur in the built context, and otherwise abstains. Consequently its output
tests retrieval-to-context flow, citation enforcement, evidence gates,
unsupported-claim detection, and metric calculations—it is not a substitute
for judging natural-language answers from an LLM.

Metrics:

- `retrieval_source_accuracy`: expected document and category occur in top 5.
- `grounded_answer_accuracy`: answerability, evidence coverage, and no flagged unsupported claim.
- `citation_validity`: cited labels exist in the context bundle.
- `citation_source_accuracy`: cited chunks match the expected document/category.
- `abstention_accuracy`: unsupported cases return the standard abstention.
- `answer_relevance`: all expected evidence terms are represented, or an unsupported case abstains.
- `unsupported_claim_free_rate`: claims have citations and meaningful-token support in their cited blocks.

Run the reproducible CPU evaluation:

```powershell
$env:HF_HUB_OFFLINE='1'
python -m evaluation.answer_quality --output evaluation/results_deterministic.json
```

For actual LLM evaluation, supply a callable accepting a prompt and returning
text. The resulting report is labeled `real_llm`:

```powershell
python -m evaluation.answer_quality --backend my_local_backend:generate
```

For an already-installed Ollama model:

```powershell
python -m evaluation.answer_quality --backend src.generation.ollama_backend:generate --output evaluation/results_real_llm.json
```

The included CPU Transformers adapter is local-files-only by default:

```powershell
python -m evaluation.answer_quality --backend src.generation.local_transformers_backend:generate --output evaluation/results_real_llm.json
```
