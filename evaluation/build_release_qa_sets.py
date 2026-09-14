# Save as:
# F:\enterprise_policy_ai\evaluation\build_release_qa_sets.py
#
# Run:
# .\.venv\Scripts\python.exe .\evaluation\build_release_qa_sets.py
#
# Creates automatically:
#   evaluation\golden_cases.json
#   evaluation\adversarial_cases.json
#   evaluation\regression_cases.json
#
# Production RAG code is NOT modified.

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(r"F:\enterprise_policy_ai")
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

SOURCE_BENCHMARK = EVALUATION_DIR / "answer_benchmark.json"

GOLDEN_FILE = EVALUATION_DIR / "golden_cases.json"
ADVERSARIAL_FILE = EVALUATION_DIR / "adversarial_cases.json"
REGRESSION_FILE = EVALUATION_DIR / "regression_cases.json"

TARGET_GOLDEN_SIZE = 84


# =====================================================================
# KNOWN REGRESSION CASES FROM ENTERPRISE POLICY AI DEVELOPMENT HISTORY
# =====================================================================

REGRESSION_REASONS = {
    "usa_personnel_file":
        "US personnel-file evidence must remain in final context and support "
        "written DPO request / Workday access.",

    "usa_accommodation":
        "US reasonable-accommodation evidence must not be removed by "
        "jurisdiction filtering.",

    "usa_pay_discussion":
        "US pay-discussion protections must retrieve the US People Policies "
        "source and preserve US scope.",

    "harassment_report":
        "Harassment reporting answer must remain grounded in explicit "
        "reporting and investigation evidence.",

    "harassment_investigation":
        "Country-specific harassment subsections must not be presented as "
        "universal GitLab procedure.",

    "harassment_scope":
        "Explicit global anti-harassment applicability to contractors and "
        "employees must be recognized as global evidence.",

    "india_complaint_body":
        "India sexual-harassment complaints must identify the Internal "
        "Committee and preserve India scope.",

    "offboarding_plain":
        "Generic offboarding answers must not universalize US-only COBRA "
        "rules or unrelated jurisdiction-specific procedures.",

    "offboarding_resignation":
        "Resignation procedure must retain Workday evidence.",

    "offboarding_involuntary":
        "Involuntary termination facilitation must remain grounded in "
        "Team Member Relations evidence.",

    "offboarding_benefits":
        "COBRA evidence is US-specific and must never be presented as a "
        "global benefits rule.",

    "offboarding_communication":
        "Departure communication responsibility must not be overstated; "
        "manager/team-member discretion must be preserved.",

    "timeoff_sick":
        "Generic sick-time question must retain Global Time Off evidence, "
        "including Workday / Out Sick, without Australian contamination.",

    "timeoff_bereavement":
        "Bereavement answer must remain grounded in Flexible PTO and "
        "applicable leave procedures.",

    "timeoff_parental":
        "Australian parental-leave query must retain Australian PTY LTD "
        "Benefits evidence and Workday procedure.",

    "leave_absence_purpose":
        "Company-wide leave-of-absence purposes must remain grounded in "
        "the Leave of Absence document.",

    "timeoff_jury":
        "Jury-duty answer must remain grounded in the applicable Time Off "
        "procedure.",

    "benefit_growth_fund":
        "Growth and Development Fund purpose and allowance must remain "
        "grounded in General Benefits evidence.",

    "australia_health":
        "Australian private-health-insurance question must retrieve the "
        "Australia-specific BUPA evidence.",

    "australia_super":
        "Australian superannuation answer must remain Australia-scoped and "
        "must not be generalized.",

    "france_disconnect":
        "France right-to-disconnect answer must remain France-specific.",

    "ireland_disconnect":
        "Ireland right-to-disconnect answer must preserve all materially "
        "supported contact alternatives.",

    "france_remote_equipment":
        "Equipment-return requirement must not inherit dismissal penalties "
        "from equipment-misuse provisions.",

    "family_hiring":
        "Family-member hiring policy is absent from the indexed corpus and "
        "must result in grounded abstention.",

    "salary_exact":
        "Exact salary absent from corpus must result in abstention.",

    "meal_reimbursement":
        "Unsupported Tokyo meal reimbursement amount must result in "
        "abstention.",

    "dress_code":
        "Unsupported office dress-code rule must result in abstention.",

    "stock_forecast":
        "Stock-price forecasting is outside the policy corpus and must "
        "result in abstention.",
}


# =====================================================================
# HIGH-RISK / CRITICAL CASES
# =====================================================================

CRITICAL_IDS = {
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
    "family_hiring",
}


# =====================================================================
# ADVERSARIAL PRIORITY CASES
# =====================================================================

ADVERSARIAL_PRIORITY_IDS = {
    "usa_personnel_file",
    "usa_accommodation",
    "usa_pay_discussion",
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
    "family_hiring",
    "salary_exact",
    "meal_reimbursement",
    "dress_code",
    "stock_forecast",
}


# =====================================================================
# BASIC HELPERS
# =====================================================================

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def extract_cases(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in (
            "cases",
            "benchmark",
            "items",
            "data",
            "questions",
        ):
            value = payload.get(key)

            if isinstance(value, list):
                return value

    raise ValueError(
        "Could not locate benchmark case list in answer_benchmark.json"
    )


def first_value(case: dict, *keys: str, default=None):
    for key in keys:
        if key in case and case[key] not in (None, "", []):
            return case[key]

    return default


def ensure_list(value: Any) -> list:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return [value]


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def slug(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


# =====================================================================
# NORMALIZE EXISTING BENCHMARK SCHEMA
# =====================================================================

def infer_jurisdiction(case: dict) -> str | None:
    explicit = first_value(
        case,
        "expected_jurisdiction",
        "jurisdiction",
        "expected_country",
        "country",
    )

    if explicit:
        return normalize_text(explicit)

    q = normalize_text(case.get("question")).lower()

    patterns = [
        (
            (
                r"\bunited states\b",
                r"\bu\.s\.\b",
                r"\bu\.s\b",
                r"\busa\b",
                r"\bus\b",
                r"\bamerican\b",
            ),
            "United States",
        ),
        (
            (
                r"\baustralia\b",
                r"\baustralian\b",
            ),
            "Australia",
        ),
        (
            (
                r"\bfrance\b",
                r"\bfrench\b",
            ),
            "France",
        ),
        (
            (
                r"\bireland\b",
                r"\birish\b",
            ),
            "Ireland",
        ),
        (
            (
                r"\bindia\b",
                r"\bindian\b",
            ),
            "India",
        ),
        (
            (
                r"\bnetherlands\b",
                r"\bdutch\b",
            ),
            "Netherlands",
        ),
    ]

    for regexes, jurisdiction in patterns:
        if any(
            re.search(regex, q, flags=re.IGNORECASE)
            for regex in regexes
        ):
            return jurisdiction

    return None


def infer_difficulty(case: dict) -> str:
    existing = normalize_text(
        first_value(
            case,
            "difficulty",
            default="",
        )
    ).lower()

    if existing in {
        "easy",
        "normal",
        "difficult",
        "adversarial",
    }:
        return existing

    case_id = normalize_text(case.get("id"))

    if case_id in ADVERSARIAL_PRIORITY_IDS:
        return "adversarial"

    if not bool(case.get("answerable", True)):
        return "adversarial"

    question = normalize_text(case.get("question"))

    if (
        len(question.split()) >= 14
        or " and " in question.lower()
        or " or " in question.lower()
    ):
        return "difficult"

    group = normalize_text(case.get("group")).lower()

    if group in {
        "country_specific",
        "multi_part_ambiguous",
        "workplace_conduct",
    }:
        return "normal"

    return "easy"


def normalize_case(
    original: dict,
    *,
    new_id: str | None = None,
    question: str | None = None,
    variant_of: str | None = None,
    difficulty: str | None = None,
    notes_suffix: str | None = None,
) -> dict:
    source_id = normalize_text(
        first_value(
            original,
            "id",
            "case_id",
            default="unnamed_case",
        )
    )

    answerable = bool(
        first_value(
            original,
            "answerable",
            default=True,
        )
    )

    expected_document = first_value(
        original,
        "expected_document_id",
        "expected_document",
        "expected_source_document",
        "expected_source",
        "document_id",
    )

    expected_category = first_value(
        original,
        "expected_category",
        "category",
    )

    expected_evidence_terms = ensure_list(
        first_value(
            original,
            "expected_evidence_terms",
            "evidence_terms",
            "expected_terms",
            "expected_facts",
            default=[],
        )
    )

    expected_chunk_ids = ensure_list(
        first_value(
            original,
            "expected_chunk_ids",
            "expected_evidence_chunk_ids",
            "expected_chunks",
            default=[],
        )
    )

    expected_critical_facts = ensure_list(
        first_value(
            original,
            "expected_critical_facts",
            "critical_facts",
            "expected_facts",
            "evidence_terms",
            default=[],
        )
    )

    forbidden_claims = ensure_list(
        first_value(
            original,
            "forbidden_claims",
            "forbidden_terms",
            default=[],
        )
    )

    expected_abstention = bool(
        first_value(
            original,
            "expected_abstention",
            default=not answerable,
        )
    )

    notes = normalize_text(
        first_value(
            original,
            "notes",
            "note",
            default="",
        )
    )

    if notes_suffix:
        if notes:
            notes = f"{notes} {notes_suffix}"
        else:
            notes = notes_suffix

    result = {
        "id": new_id or source_id,
        "question": (
            question
            if question is not None
            else normalize_text(original.get("question"))
        ),
        "answerable": answerable,
        "group": normalize_text(
            first_value(
                original,
                "group",
                "category_group",
                default="straightforward",
            )
        ),
        "difficulty": difficulty or infer_difficulty(original),
        "expected_document_id": expected_document,
        "expected_category": expected_category,
        "expected_jurisdiction": infer_jurisdiction(original),
        "expected_evidence_chunk_ids": expected_chunk_ids,
        "expected_evidence_terms": expected_evidence_terms,
        "expected_critical_facts": expected_critical_facts,
        "forbidden_claims": forbidden_claims,
        "expected_abstention": expected_abstention,
        "critical_case": (
            source_id in CRITICAL_IDS
            or bool(
                first_value(
                    original,
                    "critical_case",
                    default=False,
                )
            )
        ),
        "variant_of": variant_of,
        "notes": notes,
    }

    return result


# =====================================================================
# SAFE PARAPHRASE GENERATION
#
# These transformations alter wording only.
# Expected policy facts/evidence remain unchanged.
# =====================================================================

def paraphrase_variants(question: str) -> list[str]:
    variants: list[str] = []

    substitution_sets = [
        [
            (r"\bteam member\b", "employee"),
            (r"\bteam members\b", "employees"),
        ],
        [
            (r"\bemployee\b", "team member"),
            (r"\bemployees\b", "team members"),
        ],
        [
            (r"\bUS\b", "U.S."),
            (r"\bUSA\b", "United States"),
        ],
        [
            (r"\bAustralian\b", "Australia-based"),
        ],
        [
            (r"\bFrance employees\b", "employees in France"),
        ],
        [
            (r"\bIreland employee\b", "employee in Ireland"),
        ],
        [
            (r"\bleaves GitLab\b", "departs GitLab"),
            (r"\bleaves\b", "departs"),
        ],
        [
            (r"\boffboarding\b", "departure process"),
            (r"\boff-boarding\b", "departure process"),
        ],
        [
            (r"\bresignation\b", "notice of resignation"),
        ],
        [
            (r"\bharassment\b", "harassing conduct"),
        ],
        [
            (r"\brecord\b", "enter"),
        ],
    ]

    for substitutions in substitution_sets:
        candidate = question

        changed = False

        for pattern, replacement in substitutions:
            updated = re.sub(
                pattern,
                replacement,
                candidate,
                flags=re.IGNORECASE,
            )

            if updated != candidate:
                changed = True

            candidate = updated

        if (
            changed
            and candidate != question
            and candidate not in variants
        ):
            variants.append(candidate)

    prefix_variants = []

    q = question.strip()

    transforms = [
        (
            r"^How should (.+?)\?$",
            r"What is the correct way for \1?",
        ),
        (
            r"^Where should (.+?)\?$",
            r"Which channel or process should \1?",
        ),
        (
            r"^What happens when (.+?)\?$",
            r"What occurs when \1?",
        ),
        (
            r"^What happens after (.+?)\?$",
            r"What process follows after \1?",
        ),
        (
            r"^What is (.+?)\?$",
            r"Can you explain \1?",
        ),
        (
            r"^Does (.+?)\?$",
            r"Is it correct that \1?",
        ),
        (
            r"^Who should (.+?)\?$",
            r"Which person or team should \1?",
        ),
        (
            r"^Which (.+?)\?$",
            r"What \1?",
        ),
        (
            r"^May (.+?)\?$",
            r"Are \1 allowed?",
        ),
    ]

    for pattern, replacement in transforms:
        candidate = re.sub(
            pattern,
            replacement,
            q,
            flags=re.IGNORECASE,
        )

        if (
            candidate != q
            and candidate not in prefix_variants
        ):
            prefix_variants.append(candidate)

    for candidate in prefix_variants:
        if candidate not in variants:
            variants.append(candidate)

    return variants


# =====================================================================
# ADVERSARIAL QUESTION VARIANTS
#
# These preserve the original policy intent while making retrieval harder.
# =====================================================================

def adversarial_variants(
    case: dict,
) -> list[tuple[str, str]]:
    question = normalize_text(case.get("question"))

    variants: list[tuple[str, str]] = []

    replacements = [
        (
            r"\bUS\b",
            "American",
            "jurisdiction_alias",
        ),
        (
            r"\bU\.S\.\b",
            "United States",
            "jurisdiction_alias",
        ),
        (
            r"\bAustralian\b",
            "Australia-based",
            "jurisdiction_demonym",
        ),
        (
            r"\bFrance\b",
            "French",
            "jurisdiction_demonym",
        ),
        (
            r"\bIreland\b",
            "Irish",
            "jurisdiction_demonym",
        ),
        (
            r"\bIndia\b",
            "Indian",
            "jurisdiction_demonym",
        ),
        (
            r"\bemployee leaves\b",
            "team member departs",
            "natural_language_alias",
        ),
        (
            r"\bleaves GitLab\b",
            "separates from GitLab",
            "natural_language_alias",
        ),
        (
            r"\bresignation\b",
            "voluntary departure",
            "domain_alias",
        ),
        (
            r"\bharassment\b",
            "inappropriate workplace conduct",
            "domain_alias",
        ),
        (
            r"\bsick time\b",
            "time away because of illness",
            "natural_language_alias",
        ),
        (
            r"\bpay\b",
            "compensation",
            "domain_alias",
        ),
        (
            r"\bpersonnel file\b",
            "employment records",
            "domain_alias",
        ),
        (
            r"\bright-to-disconnect\b",
            "disconnecting from work communications",
            "natural_language_alias",
        ),
        (
            r"\bright to disconnect\b",
            "disconnecting from work communications",
            "natural_language_alias",
        ),
    ]

    for pattern, replacement, adversarial_type in replacements:
        candidate = re.sub(
            pattern,
            replacement,
            question,
            flags=re.IGNORECASE,
        )

        if (
            candidate != question
            and all(candidate != q for q, _ in variants)
        ):
            variants.append(
                (
                    candidate,
                    adversarial_type,
                )
            )

    return variants


# =====================================================================
# GOLDEN SET
# =====================================================================

def build_golden_set(
    source_cases: list[dict],
) -> list[dict]:
    golden: list[dict] = []

    seen_questions: set[str] = set()
    seen_ids: set[str] = set()

    # ---------------------------------------------------------------
    # Add every original benchmark case first.
    # ---------------------------------------------------------------

    for case in source_cases:
        normalized = normalize_case(case)

        if normalized["id"] in seen_ids:
            raise ValueError(
                f"Duplicate benchmark ID: {normalized['id']}"
            )

        golden.append(normalized)

        seen_ids.add(normalized["id"])
        seen_questions.add(
            normalized["question"].strip().lower()
        )

    # ---------------------------------------------------------------
    # Add wording/paraphrase variants without altering expectations.
    # ---------------------------------------------------------------

    variant_counter = 1

    source_index = 0

    while (
        len(golden) < TARGET_GOLDEN_SIZE
        and source_index < len(source_cases) * 5
    ):
        source = source_cases[
            source_index % len(source_cases)
        ]

        source_id = normalize_text(source.get("id"))

        variants = paraphrase_variants(
            normalize_text(source.get("question"))
        )

        for variant_number, question in enumerate(
            variants,
            start=1,
        ):
            key = question.strip().lower()

            if key in seen_questions:
                continue

            new_id = (
                f"{source_id}__golden_variant_"
                f"{variant_number}_{variant_counter:03d}"
            )

            normalized = normalize_case(
                source,
                new_id=new_id,
                question=question,
                variant_of=source_id,
                difficulty=max_difficulty(
                    infer_difficulty(source),
                    "normal",
                ),
                notes_suffix=(
                    "Automatically generated wording variant. "
                    "Expected policy facts and evidence are inherited "
                    "unchanged from the validated source case."
                ),
            )

            golden.append(normalized)

            seen_questions.add(key)
            seen_ids.add(new_id)

            variant_counter += 1

            if len(golden) >= TARGET_GOLDEN_SIZE:
                break

        source_index += 1

    return golden


def max_difficulty(
    first: str,
    second: str,
) -> str:
    order = {
        "easy": 0,
        "normal": 1,
        "difficult": 2,
        "adversarial": 3,
    }

    return max(
        (first, second),
        key=lambda value: order.get(value, 0),
    )


# =====================================================================
# ADVERSARIAL SET
# =====================================================================

def build_adversarial_set(
    source_cases: list[dict],
) -> list[dict]:
    adversarial: list[dict] = []

    seen_questions: set[str] = set()
    counter = 1

    by_id = {
        normalize_text(case.get("id")): case
        for case in source_cases
    }

    # ---------------------------------------------------------------
    # First preserve high-risk original cases.
    # ---------------------------------------------------------------

    for case_id in sorted(ADVERSARIAL_PRIORITY_IDS):
        case = by_id.get(case_id)

        if case is None:
            continue

        normalized = normalize_case(
            case,
            new_id=f"adv_{case_id}_original",
            variant_of=case_id,
            difficulty="adversarial",
            notes_suffix=(
                "Adversarial safety baseline preserved from the "
                "validated benchmark."
            ),
        )

        normalized["adversarial_type"] = (
            "known_high_risk_regression"
        )

        adversarial.append(normalized)

        seen_questions.add(
            normalized["question"].lower()
        )

    # ---------------------------------------------------------------
    # Add difficult wording variants.
    # ---------------------------------------------------------------

    for case_id in sorted(ADVERSARIAL_PRIORITY_IDS):
        case = by_id.get(case_id)

        if case is None:
            continue

        for question, adversarial_type in adversarial_variants(
            case
        ):
            key = question.lower().strip()

            if key in seen_questions:
                continue

            normalized = normalize_case(
                case,
                new_id=(
                    f"adv_{case_id}_"
                    f"{counter:03d}"
                ),
                question=question,
                variant_of=case_id,
                difficulty="adversarial",
                notes_suffix=(
                    "Automatically generated adversarial wording "
                    "variant. Expected evidence remains unchanged."
                ),
            )

            normalized["adversarial_type"] = (
                adversarial_type
            )

            adversarial.append(normalized)

            seen_questions.add(key)

            counter += 1

    return adversarial


# =====================================================================
# REGRESSION REGISTRY
# =====================================================================

def build_regression_set(
    source_cases: list[dict],
) -> list[dict]:
    regressions: list[dict] = []

    by_id = {
        normalize_text(case.get("id")): case
        for case in source_cases
    }

    for case_id, reason in REGRESSION_REASONS.items():
        source = by_id.get(case_id)

        if source is None:
            continue

        normalized = normalize_case(
            source,
            new_id=f"regression_{case_id}",
            variant_of=case_id,
            difficulty="adversarial",
            notes_suffix=(
                "Permanent regression case. "
                "Once fixed, this behavior must never regress."
            ),
        )

        normalized["regression_reason"] = reason

        normalized["regression_rule"] = (
            "ONCE_FIXED_MUST_NEVER_REGRESS"
        )

        regressions.append(normalized)

    return regressions


# =====================================================================
# VALIDATION
# =====================================================================

REQUIRED_FIELDS = {
    "id",
    "question",
    "answerable",
    "group",
    "difficulty",
    "expected_document_id",
    "expected_category",
    "expected_jurisdiction",
    "expected_evidence_chunk_ids",
    "expected_evidence_terms",
    "expected_critical_facts",
    "forbidden_claims",
    "expected_abstention",
    "critical_case",
    "variant_of",
    "notes",
}


def validate_cases(
    name: str,
    cases: list[dict],
) -> None:
    ids: set[str] = set()

    questions: set[str] = set()

    valid_difficulties = {
        "easy",
        "normal",
        "difficult",
        "adversarial",
    }

    for index, case in enumerate(cases):
        missing = REQUIRED_FIELDS - set(case)

        if missing:
            raise ValueError(
                f"{name}[{index}] missing fields: "
                f"{sorted(missing)}"
            )

        case_id = normalize_text(case["id"])

        if not case_id:
            raise ValueError(
                f"{name}[{index}] has blank id"
            )

        if case_id in ids:
            raise ValueError(
                f"{name} duplicate ID: {case_id}"
            )

        ids.add(case_id)

        question = normalize_text(case["question"])

        if not question:
            raise ValueError(
                f"{name}[{index}] has blank question"
            )

        q_key = question.lower()

        if q_key in questions:
            raise ValueError(
                f"{name} duplicate question: {question}"
            )

        questions.add(q_key)

        if case["difficulty"] not in valid_difficulties:
            raise ValueError(
                f"{name}[{index}] invalid difficulty: "
                f"{case['difficulty']}"
            )

        if (
            not case["answerable"]
            and not case["expected_abstention"]
        ):
            raise ValueError(
                f"{name}[{index}] is unanswerable but "
                f"expected_abstention=False"
            )


# =====================================================================
# METADATA
# =====================================================================

def distribution(
    cases: list[dict],
    field: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}

    for case in cases:
        value = normalize_text(
            case.get(field)
        ) or "unspecified"

        counts[value] = counts.get(
            value,
            0,
        ) + 1

    return dict(
        sorted(
            counts.items(),
            key=lambda item: item[0],
        )
    )


def build_payload(
    *,
    name: str,
    description: str,
    cases: list[dict],
) -> dict:
    answerable = sum(
        1
        for case in cases
        if case["answerable"]
    )

    unanswerable = (
        len(cases) - answerable
    )

    critical = sum(
        1
        for case in cases
        if case["critical_case"]
    )

    return {
        "schema_version": "1.0",
        "project": "Enterprise Policy AI",
        "suite": name,
        "description": description,
        "production_architecture_frozen": True,
        "generated_from": str(SOURCE_BENCHMARK),
        "case_count": len(cases),
        "answerable_cases": answerable,
        "unanswerable_cases": unanswerable,
        "critical_cases": critical,
        "distribution": {
            "group": distribution(
                cases,
                "group",
            ),
            "difficulty": distribution(
                cases,
                "difficulty",
            ),
            "jurisdiction": distribution(
                cases,
                "expected_jurisdiction",
            ),
        },
        "release_policy": {
            "unsupported_confident_answers": 0,
            "unsupported_critical_answers": 0,
            "wrong_jurisdiction_universalizations": 0,
            "consequence_transfer_failures": 0,
            "critical_cross_document_contamination": 0,
            "known_regression_tests_required_pass_rate": 1.0,
        },
        "cases": cases,
    }


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    if not SOURCE_BENCHMARK.exists():
        raise FileNotFoundError(
            f"Missing source benchmark: "
            f"{SOURCE_BENCHMARK}"
        )

    source_payload = load_json(
        SOURCE_BENCHMARK
    )

    source_cases = extract_cases(
        source_payload
    )

    if not source_cases:
        raise ValueError(
            "answer_benchmark.json contains no cases"
        )

    golden_cases = build_golden_set(
        source_cases
    )

    adversarial_cases = build_adversarial_set(
        source_cases
    )

    regression_cases = build_regression_set(
        source_cases
    )

    validate_cases(
        "golden_cases",
        golden_cases,
    )

    validate_cases(
        "adversarial_cases",
        adversarial_cases,
    )

    validate_cases(
        "regression_cases",
        regression_cases,
    )

    golden_payload = build_payload(
        name="golden",
        description=(
            "Golden QA set for retrieval, grounding, "
            "citation, abstention, jurisdiction, and "
            "answer-quality release testing."
        ),
        cases=golden_cases,
    )

    adversarial_payload = build_payload(
        name="adversarial",
        description=(
            "Adversarial HR policy QA suite covering "
            "jurisdiction aliases, difficult natural language, "
            "unsupported questions, scope errors, and known "
            "high-risk RAG failure patterns."
        ),
        cases=adversarial_cases,
    )

    regression_payload = build_payload(
        name="regression",
        description=(
            "Permanent regression registry built from "
            "historical Enterprise Policy AI defects. "
            "Once fixed, each behavior must never regress."
        ),
        cases=regression_cases,
    )

    save_json(
        GOLDEN_FILE,
        golden_payload,
    )

    save_json(
        ADVERSARIAL_FILE,
        adversarial_payload,
    )

    save_json(
        REGRESSION_FILE,
        regression_payload,
    )

    print("=" * 72)
    print("ENTERPRISE POLICY AI - QA DATASETS CREATED")
    print("=" * 72)

    print(
        f"Source benchmark cases : "
        f"{len(source_cases)}"
    )

    print(
        f"Golden cases           : "
        f"{len(golden_cases)}"
    )

    print(
        f"Adversarial cases      : "
        f"{len(adversarial_cases)}"
    )

    print(
        f"Regression cases       : "
        f"{len(regression_cases)}"
    )

    print()

    print(
        f"Created: {GOLDEN_FILE}"
    )

    print(
        f"Created: {ADVERSARIAL_FILE}"
    )

    print(
        f"Created: {REGRESSION_FILE}"
    )

    print()

    print("Production RAG files modified: 0")


if __name__ == "__main__":
    main()