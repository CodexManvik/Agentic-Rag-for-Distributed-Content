from pathlib import Path
import sys
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from app.graph.state import NavigatorState
from app.config import settings
from app.graph.nodes import _run_synthesis
from app.services.ingestion import ingest_web_page


def test_ingestion_rejects_disallowed_domain() -> None:
    previous = settings.public_sources_only
    settings.public_sources_only = True
    with pytest.raises(ValueError):
        ingest_web_page("https://example.org/not-allowlisted")
    settings.public_sources_only = previous


def test_synthesis_json_parse_failure_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def _invalid_response(prompt: str, purpose: str, timeout_seconds: float | None = None) -> str:
        return "not-json-response"

    monkeypatch.setattr("app.graph.nodes.invoke_chat_with_timeout", _invalid_response)

    state = cast(NavigatorState, {
        "original_query": "What is RAG?",
        "sub_queries": [],
        "retrieved_chunks": [
            {"chunk_id": "1", "source": "docs", "content": "RAG combines retrieval and generation.", "score": 0.8, "metadata": {}},
            {"chunk_id": "2", "source": "docs2", "content": "It grounds answers in external context.", "score": 0.7, "metadata": {}},
        ],
        "final_response": "",
        "citations": [],
        "retrieval_quality": {
            "max_score": 0.0,
            "avg_score": 0.0,
            "source_diversity": 0,
            "chunk_count": 0,
            "adequate": False,
            "reason": "Not evaluated",
        },
        "retries_used": 0,
        "validation_retries_used": 0,
        "validation_errors": [],
        "abstained": False,
        "abstain_reason": None,
        "confidence": 0.0,
        "cited_indices": [],
        "synthesis_output": {"answer": "", "cited_indices": [], "confidence": 0.0, "abstain_reason": None},
        "trace": [],
    })

    output = _run_synthesis(state, strict=False)
    assert output["confidence"] == 0.0
    assert output["cited_indices"] == []
    assert output["abstain_reason"] == "Invalid synthesis JSON"
