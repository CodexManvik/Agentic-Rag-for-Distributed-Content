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

# Patterns for queries that are clearly outside the KB domain (personal/chit-chat/food/etc.)
# These get a friendlier abstain message rather than "policy violation".
OFF_TOPIC_PATTERNS = [
    r"\b(what('s| is| should) (i|we|my).{0,20}(eat|have|cook|lunch|dinner|breakfast|snack|drink))\b",
    r"\b(lunch|dinner|breakfast|snack|recipe|food|meal|restaurant)\b",
    r"\bweather\b",
    r"\b(tell me (a )?joke)\b",
    r"\b(how are you|how('s| is) it going|what('s| is) up)\b",
    r"\b(who are you|what are you)\b",
    r"\bmy (name|age|birthday|location|address)\b",
    r"\b(stock price|bitcoin|crypto|forex)\b",
    r"\b(movie|series|show|netflix|spotify|song|album|artist)\b",
    r"\b(sports|football|cricket|basketball|nba|ipl|nfl)\b",
]


@lru_cache(maxsize=1)
def _compiled_policy_patterns() -> list[re.Pattern[str]]:
    return [re.compile(p, flags=re.IGNORECASE) for p in POLICY_PATTERNS]


@lru_cache(maxsize=1)
def _compiled_off_topic_patterns() -> list[re.Pattern[str]]:
    return [re.compile(p, flags=re.IGNORECASE) for p in OFF_TOPIC_PATTERNS]


def _reload_patterns() -> None:
    """Call after updating patterns at runtime to invalidate the cache."""
    _compiled_policy_patterns.cache_clear()
    _compiled_off_topic_patterns.cache_clear()


def detect_policy_scope_violation(query: str) -> tuple[bool, str | None, list[str]]:
    # Check off-topic first — gives a cleaner user-facing message
    for pattern in _compiled_off_topic_patterns():
        if pattern.search(query):
            reason = (
                "This assistant only answers questions about its knowledge base topics "
                "(RAG, LangGraph, LangChain, Confluence, etc.). "
                "Your question appears to be outside that scope."
            )
            return True, reason, [pattern.pattern]

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