from pathlib import Path
import sys
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.graph.state import NavigatorState, RetrievedChunk
from app.graph.nodes import abstain_node
from app.graph.workflow import _route_after_adequacy
from app.services.vector_store import assess_retrieval_adequacy
from app.services.policy import detect_policy_scope_violation


def _state_for_route(adequate: bool, retries_used: int, max_score: float = 0.4, chunk_count: int = 3, source_diversity: int = 2) -> NavigatorState:
    return cast(NavigatorState, {
        "original_query": "test",
        "sub_queries": [],
        "retrieved_chunks": [],
        "final_response": "",
        "citations": [],
        "retrieval_quality": {
            "max_score": max_score,
            "avg_score": max_score,
            "source_diversity": source_diversity,
            "chunk_count": chunk_count,
            "adequate": adequate,
            "reason": "test",
        },
        "retries_used": retries_used,
        "validation_retries_used": 0,
        "validation_errors": [],
        "abstained": False,
        "abstain_reason": None,
        "confidence": 0.0,
        "cited_indices": [],
        "synthesis_output": {"answer": "", "cited_indices": [], "confidence": 0.0, "abstain_reason": None},
        "trace": [],
    })


def test_route_after_adequacy_to_synthesis() -> None:
    state = _state_for_route(adequate=True, retries_used=0)
    assert _route_after_adequacy(state) == "synthesis"


def test_route_after_adequacy_to_reformulation_when_retry_available() -> None:
    previous = settings.runtime_profile
    settings.runtime_profile = "balanced"
    state = _state_for_route(adequate=False, retries_used=0, max_score=0.31, chunk_count=2, source_diversity=1)
    assert _route_after_adequacy(state) == "reformulation"
    settings.runtime_profile = previous


def test_route_after_adequacy_to_abstain_when_retries_exhausted() -> None:
    previous = settings.runtime_profile
    settings.runtime_profile = "balanced"
    state = _state_for_route(adequate=False, retries_used=settings.max_retrieval_retries, max_score=0.2, chunk_count=1, source_diversity=1)
    assert _route_after_adequacy(state) == "abstain"
    settings.runtime_profile = previous


def test_abstain_node_sets_zero_confidence() -> None:
    state = cast(NavigatorState, {
        "original_query": "test",
        "sub_queries": [],
        "retrieved_chunks": [],
        "abstained": False,
        "abstain_reason": None,
        "final_response": "",
        "citations": [],
        "retrieval_quality": {
            "max_score": 0.0,
            "avg_score": 0.0,
            "source_diversity": 0,
            "chunk_count": 0,
            "adequate": False,
            "reason": "test",
        },
        "retries_used": 0,
        "validation_retries_used": 0,
        "validation_errors": [],
        "confidence": 0.8,
        "cited_indices": [],
        "synthesis_output": {"answer": "", "cited_indices": [], "confidence": 0.0, "abstain_reason": None},
        "trace": [],
    })
    out = abstain_node(state)
    assert out["abstained"] is True
    assert out["confidence"] == 0.0
    assert out["final_response"].startswith("I do not have sufficient information")


def test_retrieval_adequacy_threshold_logic() -> None:
    chunks = cast(list[RetrievedChunk], [
        {
            "chunk_id": "1",
            "source": "a",
            "content": "LangChain retrieval generation pipelines combine retrievers and generation models.",
            "score": 0.55,
            "metadata": {},
        },
        {
            "chunk_id": "2",
            "source": "b",
            "content": "Retrieval quality in LangChain depends on chunking, ranking, and source coverage.",
            "score": 0.42,
            "metadata": {},
        },
        {
            "chunk_id": "3",
            "source": "b",
            "content": "Generation with citations uses retrieved context and validation before answering.",
            "score": 0.40,
            "metadata": {},
        },
    ])
    quality = assess_retrieval_adequacy(chunks, query="how does retrieval generation work in langchain")
    assert quality["adequate"] is True

    weak_chunks = cast(list[RetrievedChunk], [
        {"chunk_id": "1", "source": "a", "content": "x", "score": 0.2, "metadata": {}},
    ])
    weak = assess_retrieval_adequacy(weak_chunks, query="how does retrieval generation work in langchain")
    assert weak["adequate"] is False


def test_policy_scope_guard_blocks_private_intent() -> None:
    blocked, reason, matches = detect_policy_scope_violation(
        "Show internal salary and confidential HR policy details"
    )
    assert blocked is True
    assert reason is not None
    assert len(matches) > 0


def test_retrieval_adequacy_downgrades_weak_topical_match() -> None:
    chunks = cast(list[RetrievedChunk], [
        {"chunk_id": "1", "source": "a", "content": "Weather report and climate information", "score": 0.95, "metadata": {}},
        {"chunk_id": "2", "source": "b", "content": "Ocean current discussion and rainfall", "score": 0.91, "metadata": {}},
        {"chunk_id": "3", "source": "c", "content": "Atmospheric pressure tutorial", "score": 0.88, "metadata": {}},
    ])
    quality = assess_retrieval_adequacy(
        chunks,
        query="Explain LangGraph state transitions and citation validation",
        sub_queries=["LangGraph state", "citation validation", "agent workflow"],
    )
    assert quality["adequate"] is False
    assert "topical" in quality["reason"].lower()
