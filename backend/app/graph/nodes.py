import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.graph.state import Citation, NavigatorState, SynthesisOutput
from app.services.guardrails import validate_citations
from app.services.policy import detect_policy_scope_violation
from app.services.llm import ModelInvocationError, invoke_chat_with_timeout
from app.services.vector_store import assess_retrieval_adequacy, query_chunks


FALLBACK_ABSTAIN_TEXT = "I do not have sufficient information in the retrieved documents to answer this query."


def is_fallback_abstain_answer(answer: str) -> bool:
    return answer.strip().lower() == FALLBACK_ABSTAIN_TEXT.lower()


def _policy_blocked(state: NavigatorState) -> bool:
    quality = state.get("retrieval_quality", {})
    return str(quality.get("reason", "")) == "Policy scope violation"


def _no_evidence(state: NavigatorState) -> bool:
    quality = state.get("retrieval_quality", {})
    return len(state.get("retrieved_chunks", [])) == 0 or int(quality.get("chunk_count", 0)) == 0


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
    state["trace"].append(
        {
            "node": node,
            "status": status,
            "detail": detail,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    )


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


def normalize_query_node(state: NavigatorState) -> NavigatorState:
    """Lightweight rule-based query normalization prior to planning."""
    original_query = state.get("original_query", "")
    query = state.get("query", original_query)
    normalized = " ".join(str(query).split())
    normalized = normalized.strip(".,!?")
    normalized = normalized.lower()
    abbreviations = {
        "rag": "retrieval augmented generation",
        "llm": "large language model",
        "kb": "knowledge base",
    }
    for abbr, expansion in abbreviations.items():
        normalized = re.sub(rf"\b{re.escape(abbr)}\b", expansion, normalized)
    state["query"] = normalized or original_query
    state["original_query"] = original_query
    _trace(state, "normalize_query", "ok", "Applied rule-based query normalization")
    return state


def planning_agent(state: NavigatorState) -> NavigatorState:
    query = state.get("query", state["original_query"]).strip()
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
                timeout_seconds=settings.effective_planner_request_timeout_seconds,
                max_output_tokens=settings.effective_planner_max_output_tokens,
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
    blocked, reason, matches = detect_policy_scope_violation(state["original_query"])
    if blocked:
        state["abstained"] = True
        state["abstain_reason"] = reason
        state["retrieval_quality"] = {
            "max_score": 0.0,
            "avg_score": 0.0,
            "source_diversity": 0,
            "chunk_count": len(state["retrieved_chunks"]),
            "adequate": False,
            "reason": "Policy scope violation",
        }
        _trace(state, "adequacy", "failed", f"policy_block: matched={len(matches)}")
        return state

    quality = assess_retrieval_adequacy(
        state["retrieved_chunks"],
        query=state["original_query"],
        sub_queries=state["sub_queries"],
    )
    state["retrieval_quality"] = quality
    status = "ok" if quality["adequate"] else "weak"
    _trace(state, "adequacy", status, quality["reason"])
    return state


def reformulation_agent(state: NavigatorState) -> NavigatorState:
    state["retries_used"] += 1
    active_query = state.get("query", state["original_query"])
    prompt = f"""
You are the query reformulation agent.
Initial question: {active_query}
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
                timeout_seconds=settings.effective_reformulation_request_timeout_seconds,
                max_output_tokens=settings.effective_reformulation_max_output_tokens,
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


def _build_citations(chunks: Sequence[Mapping[str, Any]]) -> list[Citation]:
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
    low_latency_block = ""
    if settings.normalized_runtime_profile == "low_latency":
        low_latency_block = "Answer format must be 3-5 concise bullet points."
    return f"""
You are a grounded synthesis agent.
Use only the context below.

CONTEXT:
{context}

Rules:
1. Grounded only in provided context.
2. Every factual sentence needs [n] citation.
3. You MUST only cite a source if that specific source's text directly contains the information
    in the sentence you are writing. Do not cite a source because it is generally relevant.
    Each citation [n] must be traceable word-for-word to chunk n.
4. If evidence is adequate and citations exist, produce a concise grounded answer.
5. Abstain only when one of these is true:
    - policy violation
    - no relevant evidence
    - unresolved evidence contradiction
6. Never output the exact fallback sentence unless abstain_reason is explicit.
7. Return JSON only with keys: answer, cited_indices, confidence, abstain_reason.
8. {strict_block}
9. {low_latency_block}

USER QUESTION:
{state['original_query']}
"""


def _select_context_chunks(chunks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
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
            "answer": FALLBACK_ABSTAIN_TEXT,
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
        synth_timeout = settings.effective_synthesis_request_timeout_seconds
        raw_text = _message_to_text(
            invoke_chat_with_timeout(
                _synthesis_prompt(state, context, strict),
                purpose="synthesis",
                timeout_seconds=synth_timeout,
                max_output_tokens=settings.effective_synthesis_max_output_tokens,
            )
        ).strip()
    except ModelInvocationError:
        return {
            "answer": FALLBACK_ABSTAIN_TEXT,
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
            "answer": FALLBACK_ABSTAIN_TEXT,
            "cited_indices": [],
            "confidence": 0.0,
            "abstain_reason": "Invalid synthesis JSON",
        }


def synthesis_agent(state: NavigatorState) -> NavigatorState:
    selected = _select_context_chunks(state["retrieved_chunks"])
    state["citations"] = _build_citations(selected)
    synth = _run_synthesis(state, strict=False)

    # Enforce output contract: a non-abstained synthesis must not return the fallback template.
    if is_fallback_abstain_answer(str(synth.get("answer", ""))) and not synth.get("abstain_reason"):
        retry = _run_synthesis(state, strict=True)
        if retry.get("answer") and not is_fallback_abstain_answer(str(retry.get("answer", ""))):
            synth = retry
            _trace(state, "synthesis", "ok", "Recovered from abstain-template contract violation")
        else:
            synth = retry
            synth["abstain_reason"] = str(
                synth.get("abstain_reason") or "Synthesis produced abstain template without abstain reason"
            )

    # Keep abstain language mutually exclusive under adequate cited evidence.
    if (
        state.get("retrieval_quality", {}).get("adequate", False)
        and len(state["citations"]) > 0
        and is_fallback_abstain_answer(str(synth.get("answer", "")))
        and not _policy_blocked(state)
        and not _no_evidence(state)
    ):
        retry = _run_synthesis(state, strict=True)
        if retry.get("answer") and not is_fallback_abstain_answer(str(retry.get("answer", ""))):
            synth = retry
            _trace(state, "synthesis", "ok", "Recovered grounded answer under adequate evidence")
        else:
            synth = retry
            synth["abstain_reason"] = "Contradiction: adequate evidence present but synthesis abstained"
            state["validation_errors"].append(
                "synthesis_contradiction:adequate_with_citations_and_abstain_template"
            )

    state["synthesis_output"] = synth
    state["final_response"] = synth["answer"]
    state["cited_indices"] = synth["cited_indices"]
    state["confidence"] = synth["confidence"]
    state["abstain_reason"] = synth["abstain_reason"]
    state["abstained"] = bool(synth.get("abstain_reason")) or is_fallback_abstain_answer(state["final_response"])

    if state["abstain_reason"] == "Model unavailable during synthesis":
        _trace(state, "synthesis", "failed", "Model unavailable during synthesis")
    elif state["abstained"]:
        _trace(state, "synthesis", "failed", str(state["abstain_reason"] or "Synthesis abstained"))
    else:
        _trace(state, "synthesis", "ok", "Generated structured synthesis")
    return state


def citation_validation_agent(state: NavigatorState) -> NavigatorState:
    citation_snippets = {
        int(c.get("index", 0)): str(c.get("snippet", ""))
        for c in state["citations"]
        if isinstance(c.get("index"), int)
    }
    validation = validate_citations(
        state["final_response"],
        len(state["citations"]),
        citation_snippets=citation_snippets,
    )
    invalid_list = [idx for idx in state["cited_indices"] if idx < 1 or idx > len(state["citations"])]
    if invalid_list:
        validation["valid"] = False
        validation["errors"].append(f"Structured cited_indices out of range: {invalid_list}")

    if validation["valid"]:
        if (
            len(state["citations"]) > 0
            and state["retrieval_quality"]["adequate"]
            and is_fallback_abstain_answer(state["final_response"])
            and not _policy_blocked(state)
            and not _no_evidence(state)
        ):
            state["abstained"] = True
            state["abstain_reason"] = "Contradiction: abstain template blocked under adequate evidence"
            state["validation_errors"].append("abstain_template_blocked_under_adequate_evidence")
            _trace(state, "citation_validation", "failed", "abstain_template:blocked")
            return state
        quality = state["retrieval_quality"]
        if (
            settings.normalized_runtime_profile == "low_latency"
            and not quality["adequate"]
            and state["confidence"] < 0.25
            and quality["max_score"] < max(0.25, settings.retrieval_min_score * 0.6)
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
        second = validate_citations(
            state["final_response"],
            len(state["citations"]),
            citation_snippets=citation_snippets,
        )
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
    state["final_response"] = FALLBACK_ABSTAIN_TEXT
    state["confidence"] = 0.0
    _trace(state, "abstain", "ok", state["abstain_reason"])
    return state


def finalize_node(state: NavigatorState) -> NavigatorState:
    if (
        len(state.get("citations", [])) > 0
        and bool(state.get("retrieval_quality", {}).get("adequate", False))
        and not _policy_blocked(state)
        and is_fallback_abstain_answer(state.get("final_response", ""))
    ):
        contradiction = "finalize_contradiction:adequate_evidence_with_abstain_template"
        if contradiction not in state["validation_errors"]:
            state["validation_errors"].append(contradiction)
        state["abstained"] = True
        state["abstain_reason"] = "Contradiction error: adequate evidence exists but response was abstain template"
        state["final_response"] = (
            "Contradiction detected: adequate cited evidence exists, but synthesis produced an abstention template. "
            "Check trace and validation_errors for debugging."
        )
    if state["abstained"] and not state["abstain_reason"]:
        state["abstain_reason"] = "Unspecified abstention"
    _trace(state, "finalize", "ok", "Completed workflow")
    return state
