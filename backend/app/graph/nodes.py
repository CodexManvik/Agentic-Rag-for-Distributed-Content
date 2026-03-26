import json
import re
from typing import Any

from app.config import settings
from app.graph.state import Citation, NavigatorState, SynthesisOutput
from app.services.guardrails import validate_citations
from app.services.llm import ModelInvocationError, invoke_chat_with_timeout
from app.services.vector_store import assess_retrieval_adequacy, query_chunks


def _message_to_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _trace(state: NavigatorState, node: str, status: str, detail: str) -> None:
    state["trace"].append({"node": node, "status": status, "detail": detail})


def _unique_queries(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        candidate = line.strip().strip(" -0123456789.")
        if not candidate:
            continue
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def planning_agent(state: NavigatorState) -> NavigatorState:
    query = state["original_query"].strip()
    if settings.normalized_runtime_profile == "low_latency":
        state["sub_queries"] = [query]
        _trace(state, "planning", "ok", "Low-latency fast path used single query")
        return state

    prompt = f"""
You are a retrieval planner. Output concise, independent search queries.
Rules:
- One query per line.
- No numbering or explanations.
- Preserve key entities/acronyms.
- Maximum {settings.planner_max_subqueries} lines.

USER QUERY:
{query}
"""
    try:
        response_text = _message_to_text(
            invoke_chat_with_timeout(
                prompt,
                purpose="planning",
                timeout_seconds=min(settings.model_request_timeout_seconds, 8.0),
            )
        )
    except ModelInvocationError as exc:
        _trace(state, "planning", "failed", str(exc))
        state["sub_queries"] = [query]
        return state
    candidates = _unique_queries(response_text.splitlines())
    state["sub_queries"] = (candidates[: settings.planner_max_subqueries] if candidates else [query])
    _trace(state, "planning", "ok", f"Generated {len(state['sub_queries'])} sub-queries")
    return state


def retrieval_agent(state: NavigatorState) -> NavigatorState:
    chunks = query_chunks(state["sub_queries"])
    state["retrieved_chunks"] = chunks
    _trace(state, "retrieval", "ok", f"Retrieved {len(chunks)} chunks")
    return state


def adequacy_check_agent(state: NavigatorState) -> NavigatorState:
    quality = assess_retrieval_adequacy(state["retrieved_chunks"])
    state["retrieval_quality"] = quality
    status = "ok" if quality["adequate"] else "weak"
    _trace(state, "adequacy", status, quality["reason"])
    return state


def reformulation_agent(state: NavigatorState) -> NavigatorState:
    state["retries_used"] += 1
    prompt = f"""
You are the query reformulation agent.
Initial question: {state['original_query']}
Current sub-queries:
{chr(10).join(state['sub_queries'])}

Retrieval quality:
{state['retrieval_quality']}

Return up to {settings.planner_max_subqueries} sharper, non-duplicate retrieval queries.
One line per query.
"""
    try:
        text = _message_to_text(
            invoke_chat_with_timeout(
                prompt,
                purpose="reformulation",
                timeout_seconds=min(settings.model_request_timeout_seconds, 8.0),
            )
        )
    except ModelInvocationError as exc:
        _trace(state, "reformulation", "failed", str(exc))
        return state
    revised = _unique_queries(text.splitlines())
    if revised:
        state["sub_queries"] = revised[: settings.planner_max_subqueries]
    _trace(state, "reformulation", "ok", f"Retries used: {state['retries_used']}")
    return state


def _build_citations(chunks: list[dict[str, Any]]) -> list[Citation]:
    citations: list[Citation] = []
    for idx, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        citations.append(
            {
                "index": idx,
                "source": chunk.get("source", "unknown"),
                "url": metadata.get("url") or metadata.get("path"),
                "snippet": str(chunk.get("content", ""))[:260],
                "source_type": metadata.get("source_type"),
                "section": metadata.get("section"),
                "page_number": metadata.get("page_number"),
            }
        )
    return citations


def _extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else text


def _synthesis_prompt(state: NavigatorState, context: str, strict: bool) -> str:
    strict_block = "" if not strict else "Each factual sentence must include [n] citations."
    return f"""
You are a grounded synthesis agent.
Use only the context below. If evidence is insufficient, abstain.

CONTEXT:
{context}

Rules:
1. Grounded only in provided context.
2. Every factual sentence needs [n] citation.
3. Return JSON only with keys: answer, cited_indices, confidence, abstain_reason.
4. {strict_block}

USER QUESTION:
{state['original_query']}
"""


def _select_context_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    limit = settings.context_chunk_limit
    char_limit = settings.context_chunk_char_limit
    chosen: list[dict[str, Any]] = []
    seen_sources: set[str] = set()

    for chunk in chunks:
        source = str(chunk.get("source", "unknown"))
        content = str(chunk.get("content", ""))[:char_limit]
        if not content.strip():
            continue
        if source not in seen_sources:
            chunk_copy = dict(chunk)
            chunk_copy["content"] = content
            chosen.append(chunk_copy)
            seen_sources.add(source)
        if len(chosen) >= limit:
            return chosen

    for chunk in chunks:
        if len(chosen) >= limit:
            break
        content = str(chunk.get("content", ""))[:char_limit]
        if not content.strip():
            continue
        chunk_copy = dict(chunk)
        chunk_copy["content"] = content
        if chunk_copy not in chosen:
            chosen.append(chunk_copy)
    return chosen[:limit]


def _run_synthesis(state: NavigatorState, strict: bool) -> SynthesisOutput:
    chunks = _select_context_chunks(state["retrieved_chunks"])
    if not chunks:
        return {
            "answer": "I do not have sufficient information in the retrieved documents to answer this query.",
            "cited_indices": [],
            "confidence": 0.0,
            "abstain_reason": "No evidence retrieved",
        }

    context_lines: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        context_lines.append(
            f"[{idx}] SOURCE: {chunk['source']}\nCONTENT: {chunk['content']}"
        )
    context = "\n\n".join(context_lines)

    try:
        synth_timeout = settings.model_request_timeout_seconds
        if settings.normalized_runtime_profile == "low_latency":
            synth_timeout = min(synth_timeout, 12.0)
        raw_text = _message_to_text(
            invoke_chat_with_timeout(
                _synthesis_prompt(state, context, strict),
                purpose="synthesis",
                timeout_seconds=synth_timeout,
            )
        ).strip()
    except ModelInvocationError:
        return {
            "answer": "I do not have sufficient information in the retrieved documents to answer this query.",
            "cited_indices": [],
            "confidence": 0.0,
            "abstain_reason": "Model unavailable during synthesis",
        }
    try:
        parsed = json.loads(_extract_json(raw_text))
        answer = str(parsed.get("answer", "")).strip()
        cited_indices_raw = parsed.get("cited_indices", [])
        cited_indices = [int(i) for i in cited_indices_raw if isinstance(i, int) or str(i).isdigit()]
        confidence = float(parsed.get("confidence", 0.0))
        abstain_reason = parsed.get("abstain_reason")
        abstain_reason_str = str(abstain_reason) if isinstance(abstain_reason, str) else None
        return {
            "answer": answer,
            "cited_indices": cited_indices,
            "confidence": max(0.0, min(1.0, confidence)),
            "abstain_reason": abstain_reason_str,
        }
    except Exception:
        return {
            "answer": "I do not have sufficient information in the retrieved documents to answer this query.",
            "cited_indices": [],
            "confidence": 0.0,
            "abstain_reason": "Invalid synthesis JSON",
        }


def synthesis_agent(state: NavigatorState) -> NavigatorState:
    selected = _select_context_chunks(state["retrieved_chunks"])
    state["citations"] = _build_citations(selected)
    synth = _run_synthesis(state, strict=False)
    state["synthesis_output"] = synth
    state["final_response"] = synth["answer"]
    state["cited_indices"] = synth["cited_indices"]
    state["confidence"] = synth["confidence"]
    state["abstain_reason"] = synth["abstain_reason"]
    _trace(state, "synthesis", "ok", "Generated structured synthesis")
    return state


def citation_validation_agent(state: NavigatorState) -> NavigatorState:
    validation = validate_citations(state["final_response"], len(state["citations"]))
    invalid_list = [idx for idx in state["cited_indices"] if idx < 1 or idx > len(state["citations"])]
    if invalid_list:
        validation["valid"] = False
        validation["errors"].append(f"Structured cited_indices out of range: {invalid_list}")

    if validation["valid"]:
        quality = state["retrieval_quality"]
        if (
            settings.normalized_runtime_profile == "low_latency"
            and not quality["adequate"]
            and state["confidence"] < 0.45
        ):
            state["abstained"] = True
            state["abstain_reason"] = "Low-confidence answer under weak evidence"
            _trace(state, "citation_validation", "failed", "quality_guard:auto_abstain")
            return state
        _trace(state, "citation_validation", "ok", "Citation validation passed")
        return state

    state["validation_errors"] = validation["errors"]
    if state["validation_retries_used"] < settings.max_validation_retries:
        state["validation_retries_used"] += 1
        strict = _run_synthesis(state, strict=True)
        state["synthesis_output"] = strict
        state["final_response"] = strict["answer"]
        state["cited_indices"] = strict["cited_indices"]
        state["confidence"] = strict["confidence"]
        state["abstain_reason"] = strict["abstain_reason"]
        second = validate_citations(state["final_response"], len(state["citations"]))
        if second["valid"]:
            state["validation_errors"] = []
            _trace(state, "citation_validation", "ok", "Passed after one regeneration")
            return state
        state["validation_errors"] = second["errors"]
        if second.get("error_categories"):
            _trace(
                state,
                "citation_validation",
                "failed",
                "categories:" + ",".join(second["error_categories"]),
            )

    state["abstained"] = True
    if not state["abstain_reason"]:
        state["abstain_reason"] = "Citation validation failed"
    _trace(state, "citation_validation", "failed", "; ".join(state["validation_errors"]))
    return state


def abstain_node(state: NavigatorState) -> NavigatorState:
    state["abstained"] = True
    if not state["abstain_reason"]:
        state["abstain_reason"] = "Evidence is insufficient or unverifiable"
    state["final_response"] = (
        "I do not have sufficient information in the retrieved documents to answer this query."
    )
    state["confidence"] = 0.0
    _trace(state, "abstain", "ok", state["abstain_reason"])
    return state


def finalize_node(state: NavigatorState) -> NavigatorState:
    if state["abstained"] and not state["abstain_reason"]:
        state["abstain_reason"] = "Unspecified abstention"
    _trace(state, "finalize", "ok", "Completed workflow")
    return state
