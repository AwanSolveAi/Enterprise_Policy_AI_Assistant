"""Rate-safe, resumable Groq runner for the real-LLM benchmark only."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable

from groq import Groq

from evaluation.answer_quality import evaluate_case, load_benchmark, summarize
from src.retrieval.hybrid_retriever import HybridRetriever

DEFAULT_OUTPUT = Path("evaluation/results_groq_rate_safe.json")
DEFAULT_PACING_SECONDS = 15.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 30.0


def retry_after_seconds(error: Exception, now: Callable[[], datetime] | None = None) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            current = (now or (lambda: datetime.now(timezone.utc)))()
            parsed = parsedate_to_datetime(str(value))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(0.0, (parsed-current).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def is_rate_limit(error: Exception) -> bool:
    return getattr(error, "status_code", None) == 429


def run_with_retries(operation: Callable[[], dict[str, Any]], stats: dict[str, int], *,
                     max_retries: int = DEFAULT_MAX_RETRIES,
                     sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    retries = 0
    while True:
        try:
            return operation()
        except Exception as error:
            if not is_rate_limit(error):
                raise
            stats["rate_limit_attempts"] += 1
            if retries >= max_retries:
                raise
            delay = retry_after_seconds(error)
            if delay is None:
                delay = min(MAX_BACKOFF_SECONDS, DEFAULT_BACKOFF_SECONDS*(2**retries))
            retries += 1
            stats["retries"] += 1
            sleep(delay)


def _checkpoint(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if data.get("runner") == "groq_rate_safe_v1" else {}


def run_cases(cases: list[dict[str, Any]], evaluate: Callable[[dict[str, Any]], dict[str, Any]],
              output: Path, *, pacing_seconds: float = DEFAULT_PACING_SECONDS,
              max_retries: int = DEFAULT_MAX_RETRIES,
              sleep: Callable[[float], None] = time.sleep,
              stats: dict[str, int] | None = None) -> dict[str, Any]:
    previous = _load_checkpoint(output)
    rows = list(previous.get("cases", []))
    completed = set(previous.get("completed_case_ids", []))
    failed = list(previous.get("permanently_failed_requests", []))
    counters = stats if stats is not None else dict(previous.get("request_stats", {}))
    for key in ("successful_evaluated_cases", "api_attempts", "rate_limit_attempts", "retries", "permanently_failed_requests"):
        counters.setdefault(key, 0)
    pending = [case for case in cases if case["id"] not in completed]
    for index, case in enumerate(pending):
        attempts_before = counters["api_attempts"]
        try:
            row = run_with_retries(lambda: evaluate(case), counters, max_retries=max_retries, sleep=sleep)
        except Exception as error:
            counters["permanently_failed_requests"] += 1
            failed.append({"case_id": case["id"], "status_code": getattr(error, "status_code", None), "error": str(error)})
            _checkpoint(output, {"runner":"groq_rate_safe_v1","complete":False,"completed_case_ids":sorted(completed),
                                 "request_stats":counters,"permanently_failed_requests":failed,"cases":rows})
            continue
        rows.append(row); completed.add(case["id"]); counters["successful_evaluated_cases"] += 1
        report={"runner":"groq_rate_safe_v1","complete":len(completed)==len(cases),"completed_case_ids":sorted(completed),
                "request_stats":counters,"permanently_failed_requests":failed,"cases":rows}
        _checkpoint(output,report)
        if counters["api_attempts"] > attempts_before and index < len(pending)-1 and pacing_seconds > 0:
            sleep(pacing_seconds)
    final={"runner":"groq_rate_safe_v1","complete":len(completed)==len(cases),"completed_case_ids":sorted(completed),
           "request_stats":counters,"permanently_failed_requests":failed,"cases":rows}
    if rows: final["summary"]=summarize(rows,"groq_real_llm")
    _checkpoint(output,final)
    return final


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--pacing-seconds",type=float,default=DEFAULT_PACING_SECONDS)
    parser.add_argument("--max-retries",type=int,default=DEFAULT_MAX_RETRIES)
    args=parser.parse_args()
    client=Groq(api_key=os.environ["GROQ_API_KEY"],max_retries=0)
    retriever=HybridRetriever(enable_reranker=True)
    stats={"successful_evaluated_cases":0,"api_attempts":0,"rate_limit_attempts":0,"retries":0,"permanently_failed_requests":0}
    def evaluate(case: dict[str, Any]) -> dict[str, Any]:
        def generate(prompt: str) -> str:
            stats["api_attempts"] += 1
            response=client.chat.completions.create(model="openai/gpt-oss-20b",messages=[{"role":"user","content":prompt}],
                                                    temperature=0,max_completion_tokens=2048)
            return response.choices[0].message.content or ""
        row=evaluate_case(case,retriever,generate); row["question"]=case["question"]
        return row
    report=run_cases(load_benchmark(),evaluate,args.output,pacing_seconds=args.pacing_seconds,
                     max_retries=args.max_retries,stats=stats)
    print(json.dumps({"complete":report["complete"],"completed":len(report["completed_case_ids"]),
                      "request_stats":report["request_stats"],"output":str(args.output)},indent=2))


if __name__ == "__main__":
    main()
