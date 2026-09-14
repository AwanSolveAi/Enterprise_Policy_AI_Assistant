"""Run the fixed five-case llama.cpp generator smoke test."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from evaluation.answer_quality import evaluate_case, load_benchmark
from src.generation.llama_cpp_backend import LlamaCppGenerator
from src.retrieval.hybrid_retriever import HybridRetriever


CASE_IDS = (
    "harassment_report",
    "offboarding_plain",
    "timeoff_sick",
    "france_remote_equipment",
    "family_hiring",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    benchmark = {case["id"]: case for case in load_benchmark()}
    cases = [benchmark[case_id] for case_id in CASE_IDS]
    backend = LlamaCppGenerator(args.executable, args.model, threads=args.threads)
    retriever = HybridRetriever()
    rows = []
    for case in cases:
        row = evaluate_case(case, retriever, backend)
        rows.append(row)
        print(f"Completed: {case['id']}", flush=True)

    latencies = [row["generation_ms"] for row in rows]
    answerable = [row for row in rows if row["answerable"]]
    unsupported = next(row for row in rows if not row["answerable"])
    report = {
        "runtime": "llama.cpp",
        "model_file": str(args.model.resolve()),
        "model_size_bytes": args.model.stat().st_size,
        "cases": rows,
        "summary": {
            "cases": len(rows),
            "generation_latency_ms": {
                "mean": round(statistics.mean(latencies), 2),
                "median": round(statistics.median(latencies), 2),
                "max": round(max(latencies), 2),
            },
            "answerable_passed": sum(row["passed"] for row in answerable),
            "answerable_total": len(answerable),
            "citation_compliance": round(
                sum(row["citation_valid"] for row in answerable) / len(answerable), 4
            ),
            "unsupported_abstention_correct": unsupported["abstention_correct"],
        },
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
