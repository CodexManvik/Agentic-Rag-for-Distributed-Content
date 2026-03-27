import re
from typing import TypedDict


CITATION_PATTERN = re.compile(r"\[(\d+)\]")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
BULLET_PREFIX_PATTERN = re.compile(r"^\s*[-*•]\s+")
VERB_HINT_PATTERN = re.compile(
    r"\b(is|are|was|were|be|been|being|has|have|had|do|does|did|supports|requires|uses|provides|enables|allows|means)\b",
    flags=re.IGNORECASE,
)
ENTITY_HINT_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
NUMBER_HINT_PATTERN = re.compile(r"\d")
CONNECTIVE_ONLY = {
    "and",
    "or",
    "but",
    "however",
    "therefore",
    "thus",
    "also",
    "then",
    "next",
    "finally",
    "for example",
    "for instance",
}


class CitationValidationResult(TypedDict):
    valid: bool
    errors: list[str]
    cited_indices: list[int]
    error_categories: list[str]


def _tokenize_for_overlap(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9']+", text.lower()) if len(token) > 2}


def citation_semantically_valid(claim_sentence: str, cited_chunk_text: str) -> bool:
    claim_tokens = _tokenize_for_overlap(claim_sentence)
    chunk_tokens = _tokenize_for_overlap(cited_chunk_text)
    overlap = len(claim_tokens & chunk_tokens) / max(len(claim_tokens), 1)
    return overlap >= 0.12


def _split_units(answer: str) -> list[str]:
    units: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if BULLET_PREFIX_PATTERN.match(stripped):
            bullet_text = BULLET_PREFIX_PATTERN.sub("", stripped).strip()
            if bullet_text:
                units.append(bullet_text)
            continue
        parts = [s.strip() for s in SENTENCE_PATTERN.split(stripped) if s.strip()]
        units.extend(parts or [stripped])
    return units


def _should_require_citation(unit: str) -> bool:
    unit_wo_citations = CITATION_PATTERN.sub("", unit).strip(" .,:;()[]{}")
    if not unit_wo_citations:
        return False
    lowered = unit_wo_citations.lower()
    if lowered in CONNECTIVE_ONLY:
        return False

    words = re.findall(r"[a-zA-Z0-9']+", unit_wo_citations)
    if len(words) <= 3:
        return False
    if len(words) >= 4:
        return True

    has_signal = bool(
        NUMBER_HINT_PATTERN.search(unit_wo_citations)
        or VERB_HINT_PATTERN.search(unit_wo_citations)
        or ENTITY_HINT_PATTERN.search(unit_wo_citations)
    )
    return has_signal


def validate_citations(answer: str, citation_count: int, citation_snippets: dict[int, str] | None = None) -> CitationValidationResult:
    stripped = answer.strip()
    if not stripped:
        return {
            "valid": False,
            "errors": ["empty_answer: Empty answer"],
            "cited_indices": [],
            "error_categories": ["empty_answer"],
        }

    if stripped.startswith("I do not have sufficient information"):
        return {"valid": True, "errors": [], "cited_indices": [], "error_categories": []}

    sentences = _split_units(stripped)
    errors: list[str] = []
    categories: set[str] = set()

    for sentence in sentences:
        if not _should_require_citation(sentence):
            continue
        matches = CITATION_PATTERN.findall(sentence)
        if not matches:
            categories.add("missing_citation")
            errors.append(f"missing_citation: {sentence[:80]}")
            continue

        if citation_snippets:
            for raw_idx in matches:
                idx = int(raw_idx)
                chunk_text = citation_snippets.get(idx, "")
                if not chunk_text:
                    continue
                if not citation_semantically_valid(sentence, chunk_text):
                    categories.add("weak_semantic_citation")
                    errors.append(f"weak_semantic_citation: sentence='{sentence[:60]}' index={idx}")

    all_indices = [int(idx) for idx in CITATION_PATTERN.findall(stripped)]
    invalid = sorted({idx for idx in all_indices if idx < 1 or idx > citation_count})
    if invalid:
        categories.add("invalid_index")
        errors.append(f"invalid_index: {invalid}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "cited_indices": sorted(set(all_indices)),
        "error_categories": sorted(categories),
    }
