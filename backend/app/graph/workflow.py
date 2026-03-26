import time
from typing import Callable

from langgraph.graph import END, StateGraph

from app.config import settings
from app.graph.nodes import (
    abstain_node,
    adequacy_check_agent,
    citation_validation_agent,
    finalize_node,
    planning_agent,
    reformulation_agent,
    retrieval_agent,
    synthesis_agent,
)
from app.graph.state import NavigatorState


def _timed_node(name: str, fn: Callable[[NavigatorState], NavigatorState]) -> Callable[[NavigatorState], NavigatorState]:
    def wrapped(state: NavigatorState) -> NavigatorState:
        start = time.perf_counter()
        out = fn(state)
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings = out.get("stage_timings", {})
        timings[name] = round(float(timings.get(name, 0.0)) + elapsed_ms, 2)
        out["stage_timings"] = timings
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

    if settings.normalized_runtime_profile == "low_latency":
        very_weak = (
            quality["max_score"] < max(0.20, settings.retrieval_min_score * 0.6)
            or quality["chunk_count"] <= 1
            or quality["source_diversity"] == 0
        )
        if not very_weak:
            return "abstain"

    if state["retries_used"] < settings.max_retrieval_retries:
        return "reformulation"
    return "abstain"


def _route_after_validation(state: NavigatorState) -> str:
    return "abstain" if state["abstained"] else "finalize"


def build_graph():
    graph = StateGraph(NavigatorState)
    graph.add_node("planning", _timed_node("planning", planning_agent))
    graph.add_node("retrieval", _timed_node("retrieval", retrieval_agent))
    graph.add_node("adequacy", _timed_node("adequacy", adequacy_check_agent))
    graph.add_node("reformulation", _timed_node("reformulation", reformulation_agent))
    graph.add_node("synthesis", _timed_node("synthesis", synthesis_agent))
    graph.add_node("citation_validation", _timed_node("citation_validation", citation_validation_agent))
    graph.add_node("abstain", _timed_node("abstain", abstain_node))
    graph.add_node("finalize", _timed_node("finalize", finalize_node))

    graph.set_entry_point("planning")
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


def run_workflow(query: str) -> NavigatorState:
    initial_state: NavigatorState = {
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
    }
    return workflow.invoke(initial_state)
