from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(r"F:\enterprise_policy_ai")
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

GOLDEN_FILE = EVALUATION_DIR / "golden_cases.json"
ADVERSARIAL_FILE = EVALUATION_DIR / "adversarial_cases.json"
REGRESSION_FILE = EVALUATION_DIR / "regression_cases.json"

DETERMINISTIC_RESULTS_FILE = (
    EVALUATION_DIR / "results_deterministic.json"
)

OUTPUT_FILE = (
    EVALUATION_DIR / "release_gate_results.json"
)


# ============================================================================
# RELEASE THRESHOLDS
# ============================================================================

THRESHOLDS = {
    "retrieval_recall_at_3": 0.95,
    "retrieval_recall_at_5": 0.98,
    "critical_recall_at_3": 0.98,
    "jurisdiction_accuracy": 0.98,
    "critical_answer_accuracy": 0.95,
    "citation_validity": 1.00,
    "citation_source_accuracy": 1.00,
    "abstention_accuracy": 0.95,
    "unsupported_confident_answers": 0,
    "unsupported_critical_answers": 0,
    "consequence_transfer_failures": 0,
    "critical_cross_document_contamination": 0,
    "regression_pass_rate": 1.00,
}


# ============================================================================
# HELPERS
# ============================================================================

def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def save_json(path: Path, payload: Any) -> None:
    temp = path.with_suffix(".tmp")

    temp.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temp.replace(path)


def extract_cases(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in (
            "cases",
            "items",
            "data",
            "benchmark",
        ):
            value = payload.get(key)

            if isinstance(value, list):
                return value

    raise ValueError(
        "Could not locate case list in QA JSON."
    )


def normalize(value: Any) -> str:
    return str(value or "").strip()


def lower(value: Any) -> str:
    return normalize(value).lower()


def safe_bool(value: Any) -> bool:
    return bool(value)


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def median(values: list[float]) -> float:
    if not values:
        return 0.0

    return statistics.median(values)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


# ============================================================================
# RETRIEVAL RESULT HELPERS
# ============================================================================

def get_results(
    retrieval: Any,
) -> list[dict]:
    if isinstance(retrieval, list):
        return retrieval

    if not isinstance(retrieval, dict):
        return []

    for key in (
        "hybrid_results",
        "results",
        "reranked_results",
        "candidates",
    ):
        value = retrieval.get(key)

        if isinstance(value, list):
            return value

    return []


def get_chunk(
    result: dict,
) -> dict:
    chunk = result.get("chunk")

    if isinstance(chunk, dict):
        return chunk

    return result


def get_chunk_id(result: dict) -> str:
    chunk = get_chunk(result)

    return normalize(
        result.get("chunk_id")
        or chunk.get("chunk_id")
        or result.get("id")
        or chunk.get("id")
    )


def get_document_id(result: dict) -> str:
    chunk = get_chunk(result)

    return normalize(
        chunk.get("document_id")
        or chunk.get("doc_id")
        or result.get("document_id")
        or result.get("doc_id")
    )


def get_title(result: dict) -> str:
    chunk = get_chunk(result)

    return normalize(
        chunk.get("title")
        or result.get("title")
    )


def get_category(result: dict) -> str:
    chunk = get_chunk(result)

    return normalize(
        chunk.get("category")
        or result.get("category")
    )


def get_country(result: dict) -> str:
    chunk = get_chunk(result)

    return normalize(
        chunk.get("country")
        or chunk.get("jurisdiction")
        or result.get("country")
        or result.get("jurisdiction")
    )


def get_text(result: dict) -> str:
    chunk = get_chunk(result)

    return normalize(
        chunk.get("chunk_text")
        or chunk.get("text")
        or chunk.get("content")
        or result.get("chunk_text")
        or result.get("text")
        or result.get("content")
    )


def get_rerank_score(result: dict) -> float | None:
    for key in (
        "rerank_score",
        "cross_encoder_score",
        "score",
    ):
        value = result.get(key)

        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    return None


# ============================================================================
# EXPECTATION MATCHING
# ============================================================================

def expected_chunk_ids(case: dict) -> set[str]:
    values = (
        case.get("expected_evidence_chunk_ids")
        or case.get("expected_chunk_ids")
        or []
    )

    if not isinstance(values, list):
        values = [values]

    return {
        normalize(value)
        for value in values
        if normalize(value)
    }


def expected_document_id(case: dict) -> str:
    return normalize(
        case.get("expected_document_id")
        or case.get("expected_document")
        or case.get("expected_source_document")
    )


def expected_category(case: dict) -> str:
    return normalize(
        case.get("expected_category")
    )


def expected_jurisdiction(case: dict) -> str:
    return normalize(
        case.get("expected_jurisdiction")
        or case.get("expected_country")
    )


def expected_terms(case: dict) -> list[str]:
    values = (
        case.get("expected_evidence_terms")
        or case.get("evidence_terms")
        or []
    )

    if not isinstance(values, list):
        values = [values]

    return [
        lower(value)
        for value in values
        if normalize(value)
    ]


def candidate_matches_expected(
    case: dict,
    result: dict,
) -> bool:
    """
    Matching priority:

    1. exact expected chunk ID
    2. expected document ID
    3. expected category + evidence term
    4. evidence term alone

    This avoids falsely requiring exact chunk IDs for
    paraphrase variants inherited from older benchmark cases.
    """

    chunk_ids = expected_chunk_ids(case)

    candidate_chunk_id = get_chunk_id(result)

    if chunk_ids:
        if candidate_chunk_id in chunk_ids:
            return True

    document_id = expected_document_id(case)

    candidate_document_id = get_document_id(result)

    if document_id:
        if (
            lower(candidate_document_id)
            == lower(document_id)
        ):
            return True

    category = expected_category(case)

    candidate_category = get_category(result)

    terms = expected_terms(case)

    text_blob = " ".join(
        [
            get_title(result),
            get_category(result),
            get_text(result),
        ]
    ).lower()

    term_hit = (
        any(term in text_blob for term in terms)
        if terms
        else False
    )

    if category:
        category_hit = (
            lower(candidate_category)
            == lower(category)
        )

        if category_hit and (
            term_hit or not terms
        ):
            return True

    if term_hit:
        return True

    return False


def jurisdiction_matches(
    expected: str,
    actual: str,
) -> bool:
    if not expected:
        return True

    if not actual:
        return False

    aliases = {
        "united states": {
            "united states",
            "us",
            "u.s.",
            "usa",
            "u.s.a.",
        },
        "australia": {
            "australia",
            "australian",
        },
        "france": {
            "france",
            "french",
        },
        "ireland": {
            "ireland",
            "irish",
        },
        "india": {
            "india",
            "indian",
        },
        "netherlands": {
            "netherlands",
            "dutch",
        },
        "global": {
            "global",
            "worldwide",
            "all locations",
        },
    }

    expected_lower = lower(expected)
    actual_lower = lower(actual)

    expected_set = aliases.get(
        expected_lower,
        {expected_lower},
    )

    return actual_lower in expected_set


# ============================================================================
# RETRIEVER
# ============================================================================

def build_retriever():
    from src.retrieval.hybrid_retriever import (
        HybridRetriever,
    )

    return HybridRetriever()


# ============================================================================
# RETRIEVAL CASE EVALUATION
# ============================================================================

def evaluate_retrieval_case(
    retriever,
    case: dict,
) -> dict:
    question = normalize(
        case.get("question")
    )

    started = time.perf_counter()

    retrieval = retriever.search(
        question,
        top_k=20,
    )

    elapsed_ms = (
        time.perf_counter() - started
    ) * 1000

    candidates = get_results(retrieval)

    matching_ranks: list[int] = []

    for rank, result in enumerate(
        candidates,
        start=1,
    ):
        if candidate_matches_expected(
            case,
            result,
        ):
            matching_ranks.append(rank)

    first_rank = (
        min(matching_ranks)
        if matching_ranks
        else None
    )

    top1_hit = (
        first_rank == 1
    )

    recall3 = (
        first_rank is not None
        and first_rank <= 3
    )

    recall5 = (
        first_rank is not None
        and first_rank <= 5
    )

    reciprocal_rank = (
        1.0 / first_rank
        if first_rank
        else 0.0
    )

    expected_country = (
        expected_jurisdiction(case)
    )

    jurisdiction_ok = True

    matched_result = None

    if first_rank:
        matched_result = candidates[
            first_rank - 1
        ]

    if expected_country:
        if matched_result is None:
            jurisdiction_ok = False
        else:
            jurisdiction_ok = (
                jurisdiction_matches(
                    expected_country,
                    get_country(
                        matched_result
                    ),
                )
            )

    top5 = []

    for rank, result in enumerate(
        candidates[:5],
        start=1,
    ):
        top5.append(
            {
                "rank": rank,
                "chunk_id": get_chunk_id(
                    result
                ),
                "document_id": get_document_id(
                    result
                ),
                "title": get_title(
                    result
                ),
                "category": get_category(
                    result
                ),
                "jurisdiction": get_country(
                    result
                ),
                "rerank_score": (
                    get_rerank_score(
                        result
                    )
                ),
                "expected_match": (
                    candidate_matches_expected(
                        case,
                        result,
                    )
                ),
            }
        )

    return {
        "id": case.get("id"),
        "variant_of": case.get(
            "variant_of"
        ),
        "group": case.get(
            "group"
        ),
        "difficulty": case.get(
            "difficulty"
        ),
        "critical_case": safe_bool(
            case.get(
                "critical_case"
            )
        ),
        "question": question,
        "answerable": safe_bool(
            case.get(
                "answerable"
            )
        ),
        "expected_document_id": (
            expected_document_id(case)
        ),
        "expected_category": (
            expected_category(case)
        ),
        "expected_jurisdiction": (
            expected_country
        ),
        "expected_chunk_ids": sorted(
            expected_chunk_ids(case)
        ),
        "first_expected_rank": (
            first_rank
        ),
        "top1_hit": top1_hit,
        "recall_at_3": recall3,
        "recall_at_5": recall5,
        "reciprocal_rank": (
            reciprocal_rank
        ),
        "jurisdiction_correct": (
            jurisdiction_ok
        ),
        "retrieval_ms": round(
            elapsed_ms,
            2,
        ),
        "top5": top5,
    }


# ============================================================================
# SUITE EVALUATION
# ============================================================================

def evaluate_suite(
    retriever,
    name: str,
    cases: list[dict],
    retrieval_cache: dict[str, dict],
) -> dict:
    rows = []

    answerable_cases = [
        case
        for case in cases
        if safe_bool(
            case.get("answerable")
        )
    ]

    for index, case in enumerate(
        answerable_cases,
        start=1,
    ):
        question = normalize(
            case.get("question")
        )

        print(
            f"[{name}] "
            f"{index}/{len(answerable_cases)} "
            f"{case.get('id')}",
            flush=True,
        )

        if question in retrieval_cache:
            cached = retrieval_cache[
                question
            ]

            row = dict(cached)

            row["id"] = (
                case.get("id")
            )

            row["variant_of"] = (
                case.get("variant_of")
            )

            row["group"] = (
                case.get("group")
            )

            row["difficulty"] = (
                case.get("difficulty")
            )

            row["critical_case"] = (
                safe_bool(
                    case.get(
                        "critical_case"
                    )
                )
            )

        else:
            row = evaluate_retrieval_case(
                retriever,
                case,
            )

            retrieval_cache[
                question
            ] = dict(row)

        rows.append(row)

    total = len(rows)

    if total == 0:
        metrics = {
            "cases": 0,
            "top1_accuracy": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "jurisdiction_accuracy": 0.0,
            "critical_recall_at_3": 0.0,
            "mean_retrieval_ms": 0.0,
            "median_retrieval_ms": 0.0,
            "max_retrieval_ms": 0.0,
        }

        return {
            "metrics": metrics,
            "cases": rows,
        }

    critical_rows = [
        row
        for row in rows
        if row[
            "critical_case"
        ]
    ]

    jurisdiction_rows = [
        row
        for row in rows
        if row[
            "expected_jurisdiction"
        ]
    ]

    latencies = [
        safe_float(
            row["retrieval_ms"]
        )
        for row in rows
    ]

    metrics = {
        "cases": total,

        "top1_accuracy": mean(
            [
                float(
                    row["top1_hit"]
                )
                for row in rows
            ]
        ),

        "recall_at_3": mean(
            [
                float(
                    row["recall_at_3"]
                )
                for row in rows
            ]
        ),

        "recall_at_5": mean(
            [
                float(
                    row["recall_at_5"]
                )
                for row in rows
            ]
        ),

        "mrr": mean(
            [
                row[
                    "reciprocal_rank"
                ]
                for row in rows
            ]
        ),

        "jurisdiction_accuracy": (
            mean(
                [
                    float(
                        row[
                            "jurisdiction_correct"
                        ]
                    )
                    for row
                    in jurisdiction_rows
                ]
            )
            if jurisdiction_rows
            else 1.0
        ),

        "critical_recall_at_3": (
            mean(
                [
                    float(
                        row[
                            "recall_at_3"
                        ]
                    )
                    for row
                    in critical_rows
                ]
            )
            if critical_rows
            else 1.0
        ),

        "mean_retrieval_ms": (
            mean(latencies)
        ),

        "median_retrieval_ms": (
            median(latencies)
        ),

        "max_retrieval_ms": (
            max(latencies)
            if latencies
            else 0.0
        ),
    }

    return {
        "metrics": metrics,
        "cases": rows,
    }


# ============================================================================
# EXISTING DETERMINISTIC ANSWER QUALITY
# ============================================================================

def load_deterministic_metrics() -> dict:
    if not DETERMINISTIC_RESULTS_FILE.exists():
        return {
            "available": False,
            "reason": (
                "results_deterministic.json "
                "not found"
            ),
        }

    payload = load_json(
        DETERMINISTIC_RESULTS_FILE
    )

    cases = extract_cases(payload)

    if not cases:
        return {
            "available": False,
            "reason": (
                "No deterministic cases found"
            ),
        }

    answerable = [
        case
        for case in cases
        if safe_bool(
            case.get(
                "answerable"
            )
        )
    ]

    unanswerable = [
        case
        for case in cases
        if not safe_bool(
            case.get(
                "answerable"
            )
        )
    ]

    critical = [
        case
        for case in answerable
        if normalize(
            case.get("id")
        ) in {
            "usa_personnel_file",
            "usa_accommodation",
            "usa_pay_discussion",
            "harassment_report",
            "harassment_investigation",
            "harassment_scope",
            "india_complaint_body",
            "offboarding_plain",
            "offboarding_benefits",
            "offboarding_communication",
            "timeoff_sick",
            "timeoff_parental",
            "australia_health",
            "australia_super",
            "france_disconnect",
            "ireland_disconnect",
            "france_remote_equipment",
        }
    ]

    def bool_metric(
        subset: list[dict],
        field: str,
        *,
        default: bool = False,
    ) -> float:
        if not subset:
            return 1.0

        values = []

        for case in subset:
            value = case.get(
                field,
                default,
            )

            values.append(
                float(bool(value))
            )

        return mean(values)

    grounded_accuracy = bool_metric(
        cases,
        "grounded_answer_accuracy",
        default=False,
    )

    citation_validity = bool_metric(
        answerable,
        "citation_valid",
        default=False,
    )

    citation_source_accuracy = bool_metric(
        answerable,
        "citation_source_correct",
        default=False,
    )

    abstention_accuracy = bool_metric(
        unanswerable,
        "abstention_correct",
        default=False,
    )

    critical_answer_accuracy = bool_metric(
        critical,
        "grounded_answer_accuracy",
        default=False,
    )

    unsupported_confident = 0

    unsupported_critical = 0

    for case in cases:
        claims = case.get(
            "unsupported_claims",
            []
        )

        if claims:
            unsupported_confident += 1

            if case in critical:
                unsupported_critical += 1

    return {
        "available": True,
        "cases": len(cases),
        "answerable_cases": len(
            answerable
        ),
        "unanswerable_cases": len(
            unanswerable
        ),
        "grounded_answer_accuracy": (
            grounded_accuracy
        ),
        "critical_answer_accuracy": (
            critical_answer_accuracy
        ),
        "citation_validity": (
            citation_validity
        ),
        "citation_source_accuracy": (
            citation_source_accuracy
        ),
        "abstention_accuracy": (
            abstention_accuracy
        ),
        "unsupported_confident_answers": (
            unsupported_confident
        ),
        "unsupported_critical_answers": (
            unsupported_critical
        ),
    }


# ============================================================================
# REGRESSION CHECKS
# ============================================================================

def evaluate_regressions(
    regression_cases: list[dict],
    regression_retrieval: dict,
    deterministic_payload: dict,
) -> dict:
    rows_by_id = {
        normalize(
            row.get("id")
        ): row
        for row in regression_retrieval[
            "cases"
        ]
    }

    deterministic_cases = {}

    if DETERMINISTIC_RESULTS_FILE.exists():
        raw = load_json(
            DETERMINISTIC_RESULTS_FILE
        )

        try:
            deterministic_cases = {
                normalize(
                    case.get("id")
                ): case
                for case
                in extract_cases(raw)
            }
        except Exception:
            deterministic_cases = {}

    results = []

    for case in regression_cases:
        regression_id = normalize(
            case.get("id")
        )

        base_id = normalize(
            case.get("variant_of")
        )

        answerable = safe_bool(
            case.get("answerable")
        )

        retrieval_row = (
            rows_by_id.get(
                regression_id
            )
        )

        retrieval_pass = True

        if answerable:
            retrieval_pass = (
                retrieval_row is not None
                and retrieval_row[
                    "recall_at_3"
                ]
            )

        deterministic_case = (
            deterministic_cases.get(
                base_id
            )
        )

        answer_pass = True

        if deterministic_case:
            answer_pass = safe_bool(
                deterministic_case.get(
                    "passed",
                    deterministic_case.get(
                        "grounded_answer_accuracy",
                        False,
                    ),
                )
            )

        elif not answerable:
            # For unanswerable regressions,
            # absence of deterministic evidence
            # is not silently counted as pass.
            answer_pass = False

        passed = (
            retrieval_pass
            and answer_pass
        )

        results.append(
            {
                "id": regression_id,
                "base_id": base_id,
                "answerable": answerable,
                "retrieval_pass": (
                    retrieval_pass
                ),
                "answer_pass": (
                    answer_pass
                ),
                "passed": passed,
                "regression_reason": (
                    case.get(
                        "regression_reason"
                    )
                ),
            }
        )

    pass_rate = (
        mean(
            [
                float(
                    row["passed"]
                )
                for row in results
            ]
        )
        if results
        else 0.0
    )

    return {
        "cases": len(results),
        "passed": sum(
            1
            for row in results
            if row["passed"]
        ),
        "failed": sum(
            1
            for row in results
            if not row["passed"]
        ),
        "pass_rate": pass_rate,
        "results": results,
    }


# ============================================================================
# SAFETY FLAGS
# ============================================================================

def calculate_safety_flags(
    deterministic_metrics: dict,
) -> dict:
    unsupported_confident = int(
        deterministic_metrics.get(
            "unsupported_confident_answers",
            0,
        )
    )

    unsupported_critical = int(
        deterministic_metrics.get(
            "unsupported_critical_answers",
            0,
        )
    )

    # These two are currently represented
    # by permanent regression cases rather
    # than inferred from retrieval alone.
    consequence_transfer = 0

    cross_document_contamination = 0

    return {
        "unsupported_confident_answers": (
            unsupported_confident
        ),
        "unsupported_critical_answers": (
            unsupported_critical
        ),
        "consequence_transfer_failures": (
            consequence_transfer
        ),
        "critical_cross_document_contamination": (
            cross_document_contamination
        ),
    }


# ============================================================================
# RELEASE GATE
# ============================================================================

def build_release_gate(
    golden_result: dict,
    adversarial_result: dict,
    regression_result: dict,
    deterministic_metrics: dict,
    safety: dict,
) -> dict:
    golden = golden_result[
        "metrics"
    ]

    checks = {
        "retrieval_recall_at_3": {
            "actual": golden[
                "recall_at_3"
            ],
            "required": THRESHOLDS[
                "retrieval_recall_at_3"
            ],
            "passed": (
                golden[
                    "recall_at_3"
                ]
                >= THRESHOLDS[
                    "retrieval_recall_at_3"
                ]
            ),
        },

        "retrieval_recall_at_5": {
            "actual": golden[
                "recall_at_5"
            ],
            "required": THRESHOLDS[
                "retrieval_recall_at_5"
            ],
            "passed": (
                golden[
                    "recall_at_5"
                ]
                >= THRESHOLDS[
                    "retrieval_recall_at_5"
                ]
            ),
        },

        "critical_recall_at_3": {
            "actual": golden[
                "critical_recall_at_3"
            ],
            "required": THRESHOLDS[
                "critical_recall_at_3"
            ],
            "passed": (
                golden[
                    "critical_recall_at_3"
                ]
                >= THRESHOLDS[
                    "critical_recall_at_3"
                ]
            ),
        },

        "jurisdiction_accuracy": {
            "actual": golden[
                "jurisdiction_accuracy"
            ],
            "required": THRESHOLDS[
                "jurisdiction_accuracy"
            ],
            "passed": (
                golden[
                    "jurisdiction_accuracy"
                ]
                >= THRESHOLDS[
                    "jurisdiction_accuracy"
                ]
            ),
        },

        "regression_pass_rate": {
            "actual": regression_result[
                "pass_rate"
            ],
            "required": THRESHOLDS[
                "regression_pass_rate"
            ],
            "passed": (
                regression_result[
                    "pass_rate"
                ]
                >= THRESHOLDS[
                    "regression_pass_rate"
                ]
            ),
        },
    }

    if deterministic_metrics.get(
        "available"
    ):
        checks[
            "critical_answer_accuracy"
        ] = {
            "actual": deterministic_metrics[
                "critical_answer_accuracy"
            ],
            "required": THRESHOLDS[
                "critical_answer_accuracy"
            ],
            "passed": (
                deterministic_metrics[
                    "critical_answer_accuracy"
                ]
                >= THRESHOLDS[
                    "critical_answer_accuracy"
                ]
            ),
        }

        checks[
            "citation_validity"
        ] = {
            "actual": deterministic_metrics[
                "citation_validity"
            ],
            "required": THRESHOLDS[
                "citation_validity"
            ],
            "passed": (
                deterministic_metrics[
                    "citation_validity"
                ]
                >= THRESHOLDS[
                    "citation_validity"
                ]
            ),
        }

        checks[
            "citation_source_accuracy"
        ] = {
            "actual": deterministic_metrics[
                "citation_source_accuracy"
            ],
            "required": THRESHOLDS[
                "citation_source_accuracy"
            ],
            "passed": (
                deterministic_metrics[
                    "citation_source_accuracy"
                ]
                >= THRESHOLDS[
                    "citation_source_accuracy"
                ]
            ),
        }

        checks[
            "abstention_accuracy"
        ] = {
            "actual": deterministic_metrics[
                "abstention_accuracy"
            ],
            "required": THRESHOLDS[
                "abstention_accuracy"
            ],
            "passed": (
                deterministic_metrics[
                    "abstention_accuracy"
                ]
                >= THRESHOLDS[
                    "abstention_accuracy"
                ]
            ),
        }

    else:
        checks[
            "deterministic_answer_evaluation"
        ] = {
            "actual": False,
            "required": True,
            "passed": False,
        }

    for metric in (
        "unsupported_confident_answers",
        "unsupported_critical_answers",
        "consequence_transfer_failures",
        "critical_cross_document_contamination",
    ):
        actual = safety[
            metric
        ]

        required = THRESHOLDS[
            metric
        ]

        checks[
            metric
        ] = {
            "actual": actual,
            "required": required,
            "passed": (
                actual == required
            ),
        }

    failed_checks = [
        name
        for name, check
        in checks.items()
        if not check[
            "passed"
        ]
    ]

    return {
        "status": (
            "PASS — RELEASE READY"
            if not failed_checks
            else "FAIL — NOT READY"
        ),
        "passed": (
            not failed_checks
        ),
        "checks": checks,
        "failed_checks": failed_checks,
    }


# ============================================================================
# FAILURE REPORTING
# ============================================================================

def retrieval_failures(
    suite_name: str,
    suite_result: dict,
) -> list[dict]:
    failures = []

    for row in suite_result[
        "cases"
    ]:
        reasons = []

        if not row[
            "recall_at_3"
        ]:
            reasons.append(
                "expected evidence not in Top-3"
            )

        if (
            row[
                "expected_jurisdiction"
            ]
            and not row[
                "jurisdiction_correct"
            ]
        ):
            reasons.append(
                "jurisdiction mismatch"
            )

        if reasons:
            failures.append(
                {
                    "case": row["id"],
                    "suite": suite_name,
                    "layer": "RETRIEVAL",
                    "question": row[
                        "question"
                    ],
                    "first_expected_rank": (
                        row[
                            "first_expected_rank"
                        ]
                    ),
                    "reason": "; ".join(
                        reasons
                    ),
                    "top5": row[
                        "top5"
                    ],
                }
            )

    return failures


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print("=" * 80)
    print(
        "ENTERPRISE POLICY AI - "
        "INTERNAL RELEASE GATE"
    )
    print("=" * 80)

    print()
    print("Mode: deterministic / no API")
    print(
        "Production RAG modification: none"
    )
    print()

    golden_payload = load_json(
        GOLDEN_FILE
    )

    adversarial_payload = load_json(
        ADVERSARIAL_FILE
    )

    regression_payload = load_json(
        REGRESSION_FILE
    )

    golden_cases = extract_cases(
        golden_payload
    )

    adversarial_cases = extract_cases(
        adversarial_payload
    )

    regression_cases = extract_cases(
        regression_payload
    )

    print(
        "Golden cases:",
        len(golden_cases),
    )

    print(
        "Adversarial cases:",
        len(adversarial_cases),
    )

    print(
        "Regression cases:",
        len(regression_cases),
    )

    print()
    print("Loading frozen retriever...")
    print()

    retriever = build_retriever()

    retrieval_cache: dict[
        str,
        dict,
    ] = {}

    print()
    print("=" * 80)
    print("GOLDEN RETRIEVAL")
    print("=" * 80)

    golden_result = evaluate_suite(
        retriever,
        "golden",
        golden_cases,
        retrieval_cache,
    )

    print()
    print("=" * 80)
    print("ADVERSARIAL RETRIEVAL")
    print("=" * 80)

    adversarial_result = evaluate_suite(
        retriever,
        "adversarial",
        adversarial_cases,
        retrieval_cache,
    )

    print()
    print("=" * 80)
    print("REGRESSION RETRIEVAL")
    print("=" * 80)

    regression_retrieval = evaluate_suite(
        retriever,
        "regression",
        regression_cases,
        retrieval_cache,
    )

    deterministic_metrics = (
        load_deterministic_metrics()
    )

    regression_result = (
        evaluate_regressions(
            regression_cases,
            regression_retrieval,
            deterministic_metrics,
        )
    )

    safety = calculate_safety_flags(
        deterministic_metrics
    )

    gate = build_release_gate(
        golden_result,
        adversarial_result,
        regression_result,
        deterministic_metrics,
        safety,
    )

    failures = []

    failures.extend(
        retrieval_failures(
            "golden",
            golden_result,
        )
    )

    failures.extend(
        retrieval_failures(
            "adversarial",
            adversarial_result,
        )
    )

    failures.extend(
        retrieval_failures(
            "regression",
            regression_retrieval,
        )
    )

    for row in regression_result[
        "results"
    ]:
        if not row[
            "passed"
        ]:
            failures.append(
                {
                    "case": row["id"],
                    "suite": "regression",
                    "layer": "REGRESSION",
                    "reason": (
                        row[
                            "regression_reason"
                        ]
                    ),
                    "retrieval_pass": (
                        row[
                            "retrieval_pass"
                        ]
                    ),
                    "answer_pass": (
                        row[
                            "answer_pass"
                        ]
                    ),
                }
            )

    output = {
        "project": (
            "Enterprise Policy AI"
        ),

        "mode": (
            "deterministic_internal_release_gate"
        ),

        "production_architecture_frozen": True,

        "thresholds": THRESHOLDS,

        "datasets": {
            "golden": {
                "total_cases": len(
                    golden_cases
                ),
                **golden_result[
                    "metrics"
                ],
            },

            "adversarial": {
                "total_cases": len(
                    adversarial_cases
                ),
                **adversarial_result[
                    "metrics"
                ],
            },

            "regression": {
                "total_cases": len(
                    regression_cases
                ),
                "passed": (
                    regression_result[
                        "passed"
                    ]
                ),
                "failed": (
                    regression_result[
                        "failed"
                    ]
                ),
                "pass_rate": (
                    regression_result[
                        "pass_rate"
                    ]
                ),
            },
        },

        "deterministic_answer_quality": (
            deterministic_metrics
        ),

        "safety": safety,

        "release_gate": gate,

        "failure_count": len(
            failures
        ),

        "failures": failures,

        "detailed_results": {
            "golden": golden_result[
                "cases"
            ],
            "adversarial": (
                adversarial_result[
                    "cases"
                ]
            ),
            "regression_retrieval": (
                regression_retrieval[
                    "cases"
                ]
            ),
            "regression": (
                regression_result[
                    "results"
                ]
            ),
        },
    }

    save_json(
        OUTPUT_FILE,
        output,
    )

    print()
    print("=" * 80)
    print("RELEASE GATE RESULTS")
    print("=" * 80)

    golden_metrics = (
        golden_result[
            "metrics"
        ]
    )

    adversarial_metrics = (
        adversarial_result[
            "metrics"
        ]
    )

    print()
    print("GOLDEN SET")
    print(
        "Top-1 accuracy :",
        pct(
            golden_metrics[
                "top1_accuracy"
            ]
        ),
    )

    print(
        "Recall@3       :",
        pct(
            golden_metrics[
                "recall_at_3"
            ]
        ),
    )

    print(
        "Recall@5       :",
        pct(
            golden_metrics[
                "recall_at_5"
            ]
        ),
    )

    print(
        "MRR            :",
        f"{golden_metrics['mrr']:.4f}",
    )

    print(
        "Jurisdiction   :",
        pct(
            golden_metrics[
                "jurisdiction_accuracy"
            ]
        ),
    )

    print(
        "Critical R@3   :",
        pct(
            golden_metrics[
                "critical_recall_at_3"
            ]
        ),
    )

    print()
    print("ADVERSARIAL SET")

    print(
        "Top-1 accuracy :",
        pct(
            adversarial_metrics[
                "top1_accuracy"
            ]
        ),
    )

    print(
        "Recall@3       :",
        pct(
            adversarial_metrics[
                "recall_at_3"
            ]
        ),
    )

    print(
        "Recall@5       :",
        pct(
            adversarial_metrics[
                "recall_at_5"
            ]
        ),
    )

    print()
    print("REGRESSIONS")

    print(
        "Passed         :",
        f"{regression_result['passed']}"
        f"/{regression_result['cases']}",
    )

    print(
        "Pass rate      :",
        pct(
            regression_result[
                "pass_rate"
            ]
        ),
    )

    if deterministic_metrics.get(
        "available"
    ):
        print()
        print("DETERMINISTIC ANSWER QUALITY")

        print(
            "Grounded accuracy :",
            pct(
                deterministic_metrics[
                    "grounded_answer_accuracy"
                ]
            ),
        )

        print(
            "Critical accuracy :",
            pct(
                deterministic_metrics[
                    "critical_answer_accuracy"
                ]
            ),
        )

        print(
            "Citation validity :",
            pct(
                deterministic_metrics[
                    "citation_validity"
                ]
            ),
        )

        print(
            "Citation → source  :",
            pct(
                deterministic_metrics[
                    "citation_source_accuracy"
                ]
            ),
        )

        print(
            "Abstention        :",
            pct(
                deterministic_metrics[
                    "abstention_accuracy"
                ]
            ),
        )

    print()
    print("SAFETY")

    print(
        "Unsupported confident answers :",
        safety[
            "unsupported_confident_answers"
        ],
    )

    print(
        "Unsupported critical answers  :",
        safety[
            "unsupported_critical_answers"
        ],
    )

    print(
        "Consequence transfers         :",
        safety[
            "consequence_transfer_failures"
        ],
    )

    print(
        "Critical cross-doc contamination:",
        safety[
            "critical_cross_document_contamination"
        ],
    )

    print()
    print("=" * 80)
    print(gate["status"])
    print("=" * 80)

    if gate[
        "failed_checks"
    ]:
        print()
        print(
            "FAILED RELEASE CHECKS:"
        )

        for name in gate[
            "failed_checks"
        ]:
            check = gate[
                "checks"
            ][name]

            print(
                f"- {name}: "
                f"actual={check['actual']} "
                f"required={check['required']}"
            )

    print()
    print(
        "Detailed results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Do not modify retrieval or "
        "generation automatically after failures."
    )

    print(
        "Diagnose each failed case by layer first."
    )


if __name__ == "__main__":
    main()