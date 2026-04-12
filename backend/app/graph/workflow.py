from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph

from app.agents.executor import AgentExecutor
from app.agents.registry import AgentRegistry
from app.config import settings
from app.graph.nodes import FALLBACK_ABSTAIN_TEXT, retrieval_node, synthesis_node
from app.graph.state import AgentState, Citation, NavigatorState, RetrievedChunk
from app.supervisor.config import SupervisorConfig
from app.supervisor.execution_engine import ExecutionEngine
from app.supervisor.supervisor_agent import SupervisorAgent


def _route_from_supervisor(state: AgentState) -> str:
    next_step = str(state.get("next_step", "FINISH"))
    if next_step in {"RetrievalAgent", "SynthesisAgent", "FINISH"}:
        return next_step
    if next_step == "AdequacyAgent":
        return "SynthesisAgent" if state.get("retrieved_chunks") else "RetrievalAgent"
    return "FINISH"


_manifest_dir = Path(__file__).resolve().parents[1] / "agents" / "manifests"
_agent_registry = AgentRegistry(_manifest_dir)
_agent_registry.load_agents()
_agent_executor = AgentExecutor(_agent_registry)
_supervisor = SupervisorAgent(
    registry=_agent_registry,
    execution_engine=ExecutionEngine(_agent_executor),
    config=SupervisorConfig(
        planning_enabled=True,
        fallback_agent="retrieval",
        enable_short_circuit=settings.enable_short_circuit_routing,
        short_circuit_confidence_threshold=settings.short_circuit_confidence_threshold,
    ),
)


def supervisor_node(state: AgentState) -> dict[str, Any]:
    return _supervisor.route_state(state)


builder = StateGraph(AgentState)
builder.add_node("Supervisor", supervisor_node)
builder.add_node("RetrievalAgent", retrieval_node)
builder.add_node("SynthesisAgent", synthesis_node)

builder.set_entry_point("Supervisor")

builder.add_conditional_edges(
    "Supervisor",
    lambda state: _route_from_supervisor(cast(AgentState, state)),
    {
        "RetrievalAgent": "RetrievalAgent",
        "SynthesisAgent": "SynthesisAgent",
        "FINISH": END,
    },
)

builder.add_edge("RetrievalAgent", "Supervisor")
builder.add_edge("SynthesisAgent", "Supervisor")

graph = builder.compile()
workflow = graph


def _build_citation_objects(chunks: list[RetrievedChunk], cited_indices: list[int]) -> list[Citation]:
    citations: list[Citation] = []
    for new_idx, cited_idx in enumerate(cited_indices, start=1):
        if cited_idx < 1 or cited_idx > len(chunks):
            continue
        chunk = chunks[cited_idx - 1]
        metadata = cast(dict[str, Any], chunk.get("metadata", {}))
        citations.append(
            {
                "index": new_idx,
                "source": str(chunk.get("source", "unknown")),
                "url": cast(str | None, metadata.get("url") or metadata.get("path")),
                "snippet": str(chunk.get("content", ""))[:420],
                "source_type": cast(str | None, metadata.get("source_type")),
                "section": cast(str | None, metadata.get("section")),
                "page_number": cast(int | None, metadata.get("page_number")),
            }
        )
    return citations


def _build_retrieval_quality(chunks: list[RetrievedChunk]) -> dict[str, Any]:
    scores = [float(c.get("score", 0.0) or 0.0) for c in chunks]
    sources = {str(c.get("source", "unknown")) for c in chunks}
    max_score = max(scores) if scores else 0.0
    avg_score = (sum(scores) / len(scores)) if scores else 0.0
    return {
        "max_score": max_score,
        "avg_score": avg_score,
        "source_diversity": len(sources),
        "chunk_count": len(chunks),
        "adequate": len(chunks) > 0,
        "reason": "Retrieved chunks" if chunks else "No chunks retrieved",
    }


def run_workflow(query: str, model: str | None = None) -> NavigatorState:
    selected_model = model or settings.ollama_chat_model
    initial_state: AgentState = {
        "query": query,
        "original_query": query,
        "selected_model": selected_model,
        "messages": [],
        "retrieved_chunks": [],
        "final_response": "",
        "citations": [],
        "cited_indices": [],
        "confidence": 0.0,
        "sub_queries": [query],
        "trace": [],
        "next_step": "RetrievalAgent",
        "short_circuited": False,
        "retries_used": 0,
        "validation_retries_used": 0,
        "abstained": False,
        "abstain_reason": None,
        "used_deterministic_fallback": False,
        "retrieval_quality": {
            "max_score": 0.0,
            "avg_score": 0.0,
            "source_diversity": 0,
            "chunk_count": 0,
            "adequate": False,
            "reason": "Not evaluated",
        },
        "stage_timings": {},
        "stage_timestamps": {},
        "validation_errors": [],
        "synthesis_output": {
            "answer": "",
            "cited_indices": [],
            "confidence": 0.0,
            "abstain_reason": None,
        },
    }

    try:
        result = cast(AgentState, graph.invoke(initial_state, config={"recursion_limit": 25}))
    except GraphRecursionError as exc:
        timeout_trace = list(initial_state.get("trace", []))
        timeout_trace.append(
            {
                "node": "Supervisor",
                "status": "failed",
                "detail": f"recursion_limit_reached: {exc}",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
        result = cast(
            AgentState,
            {
                **initial_state,
                "trace": timeout_trace,
                "abstained": True,
                "abstain_reason": "Recursion limit reached before completion",
            },
        )
    final_state: dict[str, Any] = dict(result)

    chunks = cast(list[RetrievedChunk], list(final_state.get("retrieved_chunks", [])))
    raw_citations = list(final_state.get("citations", []))

    cited_indices: list[int] = []
    if raw_citations and all(isinstance(item, int) for item in raw_citations):
        cited_indices = sorted(
            {
                int(item)
                for item in raw_citations
                if isinstance(item, int) and 1 <= int(item) <= len(chunks)
            }
        )
        final_state["citations"] = _build_citation_objects(chunks, cited_indices)
    elif raw_citations and all(isinstance(item, dict) for item in raw_citations):
        cited_indices = sorted(
            {
                int(item.get("index", 0))
                for item in raw_citations
                if isinstance(item.get("index"), int)
            }
        )
    else:
        final_state["citations"] = []

    final_state["cited_indices"] = cited_indices
    final_state.setdefault("retrieval_quality", _build_retrieval_quality(chunks))
    final_state.setdefault("trace", [])
    final_state.setdefault("stage_timings", {})
    final_state.setdefault("sub_queries", [query])
    final_state.setdefault("short_circuited", False)

    final_response = str(final_state.get("final_response", "")).strip()
    final_state["final_response"] = final_response
    is_abstain_answer = final_response in {"", FALLBACK_ABSTAIN_TEXT}
    final_state["abstained"] = is_abstain_answer
    if not is_abstain_answer:
        final_state["abstain_reason"] = None
    else:
        final_state.setdefault("abstain_reason", "No sufficient grounded answer available")
    final_state.setdefault("confidence", 0.0)
    final_state.setdefault(
        "synthesis_output",
        {
            "answer": final_response,
            "cited_indices": cited_indices,
            "confidence": float(final_state.get("confidence", 0.0) or 0.0),
            "abstain_reason": final_state.get("abstain_reason"),
        },
    )

    return cast(NavigatorState, final_state)
