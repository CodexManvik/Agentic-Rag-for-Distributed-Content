import re
from functools import lru_cache


POLICY_PATTERNS = [
    r"\bprivate\b",
    r"\binternal\b",
    r"\bsecret\b",
    r"\bconfidential\b",
    r"\bproprietary\b",
    r"\bnon[- ]?public\b",
    r"\bemployee\b",
    r"\bsalary\b",
    r"\bpayroll\b",
    r"\bhr policy\b",
    r"\bhidden api key\b",
    r"\bapi keys?\b",
    r"\bleak\b",
    # Prompt-injection / adversarial patterns
    r"\bignore\s+(previous|prior|all|above|your)\b",
    r"\bbypass\s+(your|the|all|any)\b",
    r"\breveal\s+(internal|private|secret|hidden|confidential|key|password)\b",
    r"\bdump\s+(all|the|secret|key|hidden|document|data|doc)\b",
    r"\bsystem\s+prompt\b",
    r"\bprompt\s+injection\b",
    r"\bsalary\s+band\b",
    r"\bsalary\s+spreadsheet\b",
    r"\bhidden\s+(api|key|secret|document|data)\b",
    r"\bexpose\s+(key|secret|credential|password|internal)\b",
    r"\bextract\s+(secret|hidden|private|confidential)\b",
]


@lru_cache(maxsize=1)
def _compiled_policy_patterns() -> list[re.Pattern[str]]:
    return [re.compile(p, flags=re.IGNORECASE) for p in POLICY_PATTERNS]


def _reload_patterns() -> None:
    """Call after updating POLICY_PATTERNS at runtime to invalidate the cache."""
    _compiled_policy_patterns.cache_clear()


def detect_policy_scope_violation(query: str) -> tuple[bool, str | None, list[str]]:
    matches: list[str] = []
    for pattern in _compiled_policy_patterns():
        if pattern.search(query):
            matches.append(pattern.pattern)

    if not matches:
        return False, None, []

    reason = (
        "Policy scope guard triggered: query requests private/internal/confidential information "
        "outside approved public-source scope"
    )
    return True, reason, matches