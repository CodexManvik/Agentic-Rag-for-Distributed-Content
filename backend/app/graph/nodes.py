import json
import re
from typing import Any

from app.config import settings
from app.graph.state import Citation, NavigatorState, SynthesisOutput
from app.services.guardrails import validate_citations
from app.services.llm import get_chat_model
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
    model = get_chat_model()
    prompt = f"""
You are the Planning Agent for an enterprise knowledge retrieval system.
Your objective is to decompose the user's complex query into 3 to 5 highly specific, independent search queries optimized for a vector database.

RULES:
1. Isolate distinct entities, acronyms, and operational concepts.
2. If the query asks for a comparison, create separate search queries for each subject.
3. Do not include conversational filler, explanations, or numbering.
4. Output exactly ONE search query per line.

USER QUERY:
{query}

OUTPUT FORMAT:
[Query 1]
[Query 2]
[Query 3]
"""
    response_text = _message_to_text(model.invoke(prompt))
    candidates = _unique_queries(response_text.splitlines())
    state["sub_queries"] = (candidates[:5] if candidates else [query])
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
    model = get_chat_model()
    prompt = f"""
You are the Query Reformulation Agent.
Initial question: {state['original_query']}
Current sub-queries:
{chr(10).join(state['sub_queries'])}

Retrieval quality:
{state['retrieval_quality']}

Return up to 4 sharper, non-duplicate retrieval queries. One line per query.
"""
    text = _message_to_text(model.invoke(prompt))
    revised = _unique_queries(text.splitlines())
    if revised:
        state["sub_queries"] = revised[:4]
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
    strict_block = "" if not strict else "If validation previously failed, be stricter: each sentence must end with [n] or [n][m]."
    return f"""
You are the Synthesis Agent for an enterprise knowledge assistant.
Your objective is to answer the user's question using ONLY the provided context retrieved from the system.

CONTEXT DATA:
{context}

RULES FOR SYNTHESIS:
1. STRICT GROUNDING: If the context does not contain the answer, you must state: "I do not have sufficient information in the retrieved documents to answer this query." Do not attempt to guess or use external knowledge.
2. MANDATORY CITATIONS: Every factual claim, statistic, or specific procedure you state MUST be immediately followed by its corresponding source index in brackets. Example: "The deployment threshold is 95% [1]."
3. MULTIPLE CITATIONS: If multiple sources support a claim, combine them. Example: "Both nodes must be restarted [1][3]."
4. NO SOURCE NAMING IN TEXT: Do not write "According to Source [1]...". State the fact directly, followed by the bracketed citation.
5. RETURN JSON ONLY with fields: answer (string), cited_indices (array of integers), confidence (float 0-1), abstain_reason (string or null).
6. {strict_block}

USER QUESTION:
{state['original_query']}

Generate the final response following all rules above:
"""


def _run_synthesis(state: NavigatorState, strict: bool) -> SynthesisOutput:
    chunks = state["retrieved_chunks"][:8]
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

    model = get_chat_model()
    raw_text = _message_to_text(model.invoke(_synthesis_prompt(state, context, strict))).strip()
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
    state["citations"] = _build_citations(state["retrieved_chunks"][:8])
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
