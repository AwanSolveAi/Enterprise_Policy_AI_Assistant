"""Small deterministic HR vocabulary expansion; no extra model required."""
import re
ROUTES = {
    "offboarding": {"triggers": (r"\bleav(?:e|es|ing|er)\b", r"\bdepart(?:s|ing|ure)?\b", r"\bend(?:ing)? employment\b"),
                    "terms": ("offboarding", "resignation", "termination", "separation", "departure")},
    "related_person_hiring": {"triggers": (r"\bfamily members?\b", r"\brelatives?\b", r"\bsignificant others?\b", r"\bnepotism\b"),
                              "terms": ("family member", "relative", "significant other", "nepotism", "employment", "hiring")},
}
def expand_query(query: str) -> tuple[str, list[str]]:
    matched, additions = [], []
    for route, rule in ROUTES.items():
        if any(re.search(pattern, query, re.I) for pattern in rule["triggers"]):
            matched.append(route); additions.extend(term for term in rule["terms"] if term.lower() not in query.lower())
    return " ".join([query, *additions]), matched
