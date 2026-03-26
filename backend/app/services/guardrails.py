import re
from typing import TypedDict


CITATION_PATTERN = re.compile(r"\[(\d+)\]")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


class CitationValidationResult(TypedDict):
    valid: bool
    errors: list[str]
    cited_indices: list[int]


def validate_citations(answer: str, citation_count: int) -> CitationValidationResult:
    stripped = answer.strip()
    if not stripped:
        return {"valid": False, "errors": ["Empty answer"], "cited_indices": []}

    if stripped.startswith("I do not have sufficient information"):
        return {"valid": True, "errors": [], "cited_indices": []}

    sentences = [s.strip() for s in SENTENCE_PATTERN.split(stripped) if s.strip()]
    errors: list[str] = []

    for sentence in sentences:
        matches = CITATION_PATTERN.findall(sentence)
        if not matches:
            errors.append(f"Missing citation in sentence: {sentence[:80]}")

    all_indices = [int(idx) for idx in CITATION_PATTERN.findall(stripped)]
    invalid = sorted({idx for idx in all_indices if idx < 1 or idx > citation_count})
    if invalid:
        errors.append(f"Invalid citation indices: {invalid}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "cited_indices": sorted(set(all_indices)),
    }
