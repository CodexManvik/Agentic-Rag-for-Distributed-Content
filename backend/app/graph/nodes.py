import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.graph.state import Citation, NavigatorState, SynthesisOutput
from app.services.guardrails import validate_citations
from app.services.policy import detect_policy_scope_violation
from app.services.llm import ModelInvocationError, invoke_chat_with_timeout, invoke_synthesis
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
    """Lightweight rule-based query normalization prior to planning.

    Preserves the original casing in ``original_query`` so that entity
    extraction (which uses capitalisation patterns) remains effective downstream.
    The lowercased, expanded form is stored in ``query`` and is used only for
    retrieval sub-query generation.
    """
    # Capture true original before any transformation.
    original_query = str(state.get("original_query", "") or "").strip()
    query = str(state.get("query", original_query) or "").strip() or original_query

    normalized = " ".join(query.split()).strip(".,!?")
    normalized_lower = normalized.lower()

    abbreviations = {
        "rag": "retrieval augmented generation",
        "llm": "large language model",
        "kb": "knowledge base",
    }
    for abbr, expansion in abbreviations.items():
        normalized_lower = re.sub(rf"\b{re.escape(abbr)}\b", expansion, normalized_lower)

    # Store normalized form for retrieval planning; never overwrite original_query.
    state["query"] = normalized_lower or original_query
    # Guarantee original_query always holds the user's raw, cased input.
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
    """Extract the first JSON object from a model response.

    Handles common model output patterns:
    - Raw JSON
    - Markdown code fences (```json ... ```)
    - <think>...</think> preamble blocks (qwen3 reasoning models)
    - Leading/trailing prose before or after the JSON object
    """
    # Strip <think>...</think> reasoning tokens emitted by qwen3.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try to extract from a markdown code fence first.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)

    # Depth-matched brace extraction to avoid greedy regex issues.
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return text


def _synthesis_prompt(state: NavigatorState, context: str, strict: bool) -> str:
    citation_instruction = (
        "Every sentence that states a fact MUST end with a [n] citation marker "
        "where n is the chunk number from the CONTEXT."
        if strict
        else "Add [n] citation markers (e.g. [1], [2]) to factual sentences, "
             "where n matches the chunk number in the CONTEXT."
    )
    return (
        "You are a knowledge synthesis agent. Answer the USER QUESTION using ONLY "
        "the information in the CONTEXT below.\n\n"
        f"CONTEXT (each section is prefixed with its chunk number):\n{context}\n\n"
        "INSTRUCTIONS:\n"
        f"- {citation_instruction}\n"
        "- Base your answer strictly on the CONTEXT. Do not use outside knowledge.\n"
        "- Write a concise answer of 2-5 sentences when the context is relevant.\n"
        "- Set confidence to a float between 0.0 and 1.0 (e.g. 0.85 if well-supported).\n"
        "- Set abstain_reason to null unless you genuinely cannot answer from the context.\n"
        "- cited_indices is a JSON array of chunk numbers you referenced, e.g. [1, 2].\n\n"
        'OUTPUT FORMAT — respond with ONLY valid JSON, no markdown fences, no extra text:\n'
        '{"answer": "<your answer>", "cited_indices": [1], "confidence": 0.8, "abstain_reason": null}\n\n'
        f"USER QUESTION: {state['original_query']}\n"
    )


def _select_context_chunks(chunks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select up to ``context_chunk_limit`` chunks for the synthesis context.

    Phase 1 – anchor: one best-scoring chunk per unique source to preserve
    source diversity.
    Phase 2 – fill: remaining slots filled by descending score from any source.
    This ensures high-relevance chunks from the same source are not silently
    discarded when the corpus is small or single-source.
    """
    limit = settings.context_chunk_limit
    char_limit = settings.context_chunk_char_limit

    def _clamp(chunk: Mapping[str, Any]) -> dict[str, Any]:
        c = dict(chunk)
        c["content"] = str(c.get("content", ""))[:char_limit]
        return c

    ranked: list[dict[str, Any]] = []
    for chunk in chunks:
        if str(chunk.get("content", "")).strip():
            ranked.append(_clamp(chunk))
    ranked.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)

    if not ranked:
        return []

    chosen: list[dict[str, Any]] = []
    chosen_ids: set[int] = set()
    seen_sources: set[str] = set()

    # Phase 1: one anchor per source.
    for chunk in ranked:
        source = str(chunk.get("source", "unknown"))
        if source not in seen_sources:
            seen_sources.add(source)
            chosen.append(chunk)
            chosen_ids.add(id(chunk))
        if len(chosen) >= limit:
            return chosen

    # Phase 2: fill remaining slots by score.
    for chunk in ranked:
        if len(chosen) >= limit:
            break
        if id(chunk) not in chosen_ids:
            chosen.append(chunk)
            chosen_ids.add(id(chunk))

    return chosen


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
        max_tokens = settings.effective_synthesis_max_output_tokens
        raw_text = invoke_synthesis(
            _synthesis_prompt(state, context, strict),
            timeout_seconds=synth_timeout,
            max_output_tokens=max_tokens,
        )
        print("====== LLM RAW OUTPUT ======\n", repr(raw_text), "\n============================")
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
        raw_confidence = float(parsed.get("confidence", 0.0))
        # Normalise: model may emit percentage integers (e.g. 85 instead of 0.85).
        confidence = raw_confidence / 100.0 if raw_confidence > 1.0 else raw_confidence
        abstain_reason = parsed.get("abstain_reason")
        # Treat JSON null / None as no abstain reason.
        abstain_reason_str = str(abstain_reason) if isinstance(abstain_reason, str) else None
        if not answer:
            # Parser succeeded but answer field was empty — treat as parse failure.
            raise ValueError("Parsed JSON has empty answer field")
        return {
            "answer": answer,
            "cited_indices": cited_indices,
            "confidence": max(0.0, min(1.0, confidence)),
            "abstain_reason": abstain_reason_str,
        }
    except Exception:
        # JSON parsing failed. Before giving up, check whether the raw text itself
        # looks like a usable plain-text answer (model forgot to wrap in JSON).
        salvaged = raw_text.strip()
        # Strip any <think> blocks that leaked through.
        salvaged = re.sub(r"<think>.*?</think>", "", salvaged, flags=re.DOTALL).strip()
        if salvaged and not is_fallback_abstain_answer(salvaged) and len(salvaged) > 20:
            return {
                "answer": salvaged,
                "cited_indices": [],
                "confidence": 0.3,
                "abstain_reason": None,
            }
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

    answer_is_abstain = is_fallback_abstain_answer(str(synth.get("answer", "")))
    model_declared_abstain = bool(synth.get("abstain_reason"))

    # Retry ONLY when the model silently produced the fallback template without
    # setting an abstain_reason. That indicates a formatting/parse failure,
    # NOT a deliberate model decision. If the model set abstain_reason, trust it.
    if answer_is_abstain and not model_declared_abstain:
        retry = _run_synthesis(state, strict=True)
        retry_abstain = is_fallback_abstain_answer(str(retry.get("answer", "")))
        retry_declared = bool(retry.get("abstain_reason"))
        if retry.get("answer") and not retry_abstain:
            synth = retry
            _trace(state, "synthesis", "ok", "Recovered grounded answer (silent abstain retry)")
        else:
            synth = retry
            # If the second attempt is still abstain but now has a reason, treat
            # it as a legitimate model abstain. Otherwise mark as format failure.
            if not retry_declared:
                synth["abstain_reason"] = "Synthesis produced abstain template without reason after retry"

    state["synthesis_output"] = synth
    state["final_response"] = synth["answer"]
    state["cited_indices"] = synth["cited_indices"]
    state["confidence"] = synth["confidence"]
    state["abstain_reason"] = synth["abstain_reason"]
    state["abstained"] = (
        bool(synth.get("abstain_reason")) or is_fallback_abstain_answer(state["final_response"])
    )

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
        # If the model already declared an abstain_reason, the abstain is intentional.
        # Do not block it — let it flow to the abstain node cleanly.
        if state.get("abstained") and state.get("abstain_reason"):
            _trace(state, "citation_validation", "ok", "Legitimate model abstain — passed through")
            return state

        # Guard: forbid the silent fallback template when evidence is clearly adequate
        # AND the model produced NO abstain_reason (i.e. synthesis silently failed).
        if (
            len(state["citations"]) > 0
            and state["retrieval_quality"]["adequate"]
            and is_fallback_abstain_answer(state["final_response"])
            and not state.get("abstain_reason")
            and not _policy_blocked(state)
            and not _no_evidence(state)
        ):
            state["abstained"] = True
            state["abstain_reason"] = "Synthesis silently produced abstain template under adequate evidence"
            state["validation_errors"].append("silent_abstain_template_blocked")
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
    # Only flag a contradiction when the response is the silent fallback template
    # AND no abstain_reason was declared. A declared abstain_reason means the model
    # intentionally abstained — that is a valid outcome, not a contradiction.
    if (
        len(state.get("citations", [])) > 0
        and bool(state.get("retrieval_quality", {}).get("adequate", False))
        and not _policy_blocked(state)
        and is_fallback_abstain_answer(state.get("final_response", ""))
        and not state.get("abstain_reason")
    ):
        contradiction = "finalize_contradiction:silent_abstain_under_adequate_evidence"
        if contradiction not in state["validation_errors"]:
            state["validation_errors"].append(contradiction)
        state["abstained"] = True
        state["abstain_reason"] = "Silent abstain under adequate evidence: synthesis failed to produce a grounded answer"
        state["final_response"] = (
            "The system retrieved relevant documents but could not generate a grounded answer. "
            "Try rephrasing your question or ingesting more specific source documents."
        )
    if state["abstained"] and not state["abstain_reason"]:
        state["abstain_reason"] = "Unspecified abstention"
    _trace(state, "finalize", "ok", "Completed workflow")
    return state
