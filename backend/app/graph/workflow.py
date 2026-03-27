import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from langgraph.graph import END, StateGraph

from app.config import settings
from app.graph.nodes import (
    abstain_node,
    adequacy_check_agent,
    citation_validation_agent,
    finalize_node,
    normalize_query_node,
    planning_agent,
    reformulation_agent,
    retrieval_agent,
    synthesis_agent,
)
from app.graph.state import NavigatorState


def _timed_node(name: str, fn: Callable[[NavigatorState], NavigatorState]) -> Callable[[NavigatorState], NavigatorState]:
    def wrapped(state: NavigatorState) -> NavigatorState:
        started_at = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        out = fn(state)
        elapsed_ms = (time.perf_counter() - start) * 1000
        finished_at = datetime.now(timezone.utc).isoformat()
        timings = out.get("stage_timings", {})
        timings[name] = round(float(timings.get(name, 0.0)) + elapsed_ms, 2)
        out["stage_timings"] = timings
        timestamps = out.get("stage_timestamps", {})
        timestamps[name] = {
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": round(elapsed_ms, 2),
        }
        out["stage_timestamps"] = timestamps
        if out.get("trace"):
            last = out["trace"][-1]
            if last.get("node") == name:
                last["duration_ms"] = round(elapsed_ms, 2)
        return out

    return wrapped


def _route_after_adequacy(state: NavigatorState) -> str:
    if state["abstained"]:
        return "abstain"

    quality = state["retrieval_quality"]
    if quality["adequate"]:
        return "synthesis"

    moderate_support = (
        quality["max_score"] >= max(0.30, settings.retrieval_min_score * 0.75)
        and quality["chunk_count"] >= max(2, settings.retrieval_min_chunks - 1)
        and quality["source_diversity"] >= 1
    )

    if settings.normalized_runtime_profile == "low_latency":
        very_weak = (
            quality["max_score"] < max(0.20, settings.retrieval_min_score * 0.6)
            or quality["chunk_count"] <= 1
            or quality["source_diversity"] == 0
        )
        if not very_weak:
            return "synthesis"

    if moderate_support and state["retries_used"] >= settings.max_retrieval_retries:
        return "synthesis"

    if state["retries_used"] < settings.max_retrieval_retries:
        return "reformulation"
    return "abstain"


def _route_after_validation(state: NavigatorState) -> str:
    return "abstain" if state["abstained"] else "finalize"


def build_graph():
    graph = StateGraph(NavigatorState)
    graph.add_node("normalize_query", _timed_node("normalize_query", normalize_query_node))
    graph.add_node("planning", _timed_node("planning", planning_agent))
    graph.add_node("retrieval", _timed_node("retrieval", retrieval_agent))
    graph.add_node("adequacy", _timed_node("adequacy", adequacy_check_agent))
    graph.add_node("reformulation", _timed_node("reformulation", reformulation_agent))
    graph.add_node("synthesis", _timed_node("synthesis", synthesis_agent))
    graph.add_node("citation_validation", _timed_node("citation_validation", citation_validation_agent))
    graph.add_node("abstain", _timed_node("abstain", abstain_node))
    graph.add_node("finalize", _timed_node("finalize", finalize_node))

    graph.set_entry_point("normalize_query")
    graph.add_edge("normalize_query", "planning")
    graph.add_edge("planning", "retrieval")
    graph.add_edge("retrieval", "adequacy")
    graph.add_conditional_edges(
        "adequacy",
        _route_after_adequacy,
        {
            "synthesis": "synthesis",
            "reformulation": "reformulation",
            "abstain": "abstain",
        },
    )
    graph.add_edge("reformulation", "retrieval")
    graph.add_edge("synthesis", "citation_validation")
    graph.add_conditional_edges(
        "citation_validation",
        _route_after_validation,
        {
            "finalize": "finalize",
            "abstain": "abstain",
        },
    )
    graph.add_edge("finalize", END)
    graph.add_edge("abstain", END)
    return graph.compile()


workflow = build_graph()


def _append_stage_latency_log(query: str, state: NavigatorState) -> None:
    resources_dir = Path(__file__).resolve().parents[2] / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    log_path = resources_dir / "stage_latency.jsonl"
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "profile": settings.normalized_runtime_profile,
        "abstained": bool(state.get("abstained", False)),
        "stage_timings": state.get("stage_timings", {}),
        "stage_timestamps": state.get("stage_timestamps", {}),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def run_workflow(query: str) -> NavigatorState:
    initial_state: NavigatorState = {
        "query": query,
        "original_query": query,
        "sub_queries": [],
        "retrieved_chunks": [],
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
        "synthesis_output": {
            "answer": "",
            "cited_indices": [],
            "confidence": 0.0,
            "abstain_reason": None,
        },
        "trace": [],
        "stage_timings": {},
        "stage_timestamps": {},
    }
    result = workflow.invoke(initial_state)
    _append_stage_latency_log(query, result)
    return result
