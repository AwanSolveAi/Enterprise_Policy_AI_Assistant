"""Repeatable retrieval evaluation: Recall@k, MRR, route and category checks."""
import argparse, json
from pathlib import Path
from src.retrieval.hybrid_retriever import HybridRetriever

CASES = Path(__file__).with_name("cases.json")
def evaluate(retriever, cases):
    rows=[]
    for case in cases:
        output=retriever.search(case["query"], top_k=case.get("k",5)); results=output["hybrid_results"]
        ids=[r["chunk_id"] for r in results]; expected=case.get("expected_chunk_ids",[])
        ranks=[ids.index(chunk_id)+1 for chunk_id in expected if chunk_id in ids]
        categories={r["chunk"].get("category") for r in results}
        documents={r["chunk"].get("document_id") for r in results}
        rows.append({"id":case["id"],"recall_at_k":len(ranks)/len(expected) if expected else None,
                     "mrr":1/min(ranks) if ranks else (None if not expected else 0.0),
                     "category_hit":case.get("expected_category") in categories if case.get("expected_category") else None,
                     "document_hit":case.get("expected_document_id") in documents if case.get("expected_document_id") else None,
                     "routes":output["routes"],"top_chunk_ids":ids})
    return rows
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--no-reranker",action="store_true"); args=parser.parse_args()
    cases=json.loads(CASES.read_text(encoding="utf-8")); rows=evaluate(HybridRetriever(enable_reranker=not args.no_reranker),cases)
    print(json.dumps(rows,indent=2))
if __name__ == "__main__": main()
