from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.guardrails import validate_citations


def test_validate_citations_passes_with_valid_indices() -> None:
    answer = "LangGraph controls state transitions [1]. It supports conditional routing [2]."
    result = validate_citations(answer, 2)
    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_citations_fails_missing_and_invalid() -> None:
    answer = "LangGraph controls state transitions. It supports conditional routing [9]."
    result = validate_citations(answer, 2)
    assert result["valid"] is False
    assert any("missing_citation" in e for e in result["errors"])
    assert any("invalid_index" in e for e in result["errors"])


def test_validate_citations_handles_bullets_and_connectives() -> None:
    answer = """
- LangGraph manages agent state transitions [1]
- Therefore
- It supports conditional edges [2]
"""
    result = validate_citations(answer, 2)
    assert result["valid"] is True
