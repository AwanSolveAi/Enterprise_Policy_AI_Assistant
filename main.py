"""Minimal CLI that preserves local, CPU-friendly retrieval."""
import argparse
from src.retrieval.hybrid_retriever import HybridRetriever, display_results

def main():
    parser=argparse.ArgumentParser(description="Search indexed enterprise policies")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--no-reranker", action="store_true", help="Skip CrossEncoder (fast/offline mode)")
    parser.add_argument("--top-k", type=int, default=5)
    args=parser.parse_args()
    retriever=HybridRetriever(enable_reranker=not args.no_reranker)
    query=args.query or input("Question: ")
    display_results(retriever.search(query, top_k=args.top_k))
if __name__ == "__main__": main()
