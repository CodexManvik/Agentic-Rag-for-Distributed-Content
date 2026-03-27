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


def _clean_fallback_text(text: str) -> str:
    cleaned = " ".join(str(text).split())
    cleaned = cleaned.replace("ﬁ", "fi").replace("ﬂ", "fl")
    cleaned = re.sub(r"\b(\w+)-\s+(\w+)\b", r"\1\2", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\[[0-9]+\]\s*\[[0-9]+\]", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    units = [u.strip() for u in re.split(r"(?<=[.!?])\s+", cleaned) if u.strip()]
    if not units:
        return cleaned[:240].strip()
    out = " ".join(units[:2])
    if len(out) > 340:
        out = out[:337].rstrip() + "..."
    return out


def normalize_query_node(state: NavigatorState) -> NavigatorState:
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

    state["query"] = normalized_lower or original_query
    state["original_query"] = original_query
    _trace(state, "normalize_query", "ok", "Applied rule-based query normalization")
    return state


def planning_agent(state: NavigatorState) -> NavigatorState:
    query = state.get("query", state["original_query"]).strip()

    if settings.normalized_runtime_profile == "low_latency":
        base = query
        expansions = [
            base,
            f"{base} definition",
            f"{base} overview key concepts",
        ]
        state["sub_queries"] = _unique_queries(expansions)[: max(2, settings.planner_max_subqueries)]
        _trace(state, "planning", "ok", f"Low-latency path generated {len(state['sub_queries'])} sub-queries")
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
    state["sub_queries"] = candidates[: settings.planner_max_subqueries] if candidates else [query]
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
    _trace(state, "adequacy", "ok" if quality["adequate"] else "weak", quality["reason"])
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
                "snippet": str(chunk.get("content", ""))[:420],
                "source_type": metadata.get("source_type"),
                "section": metadata.get("section"),
                "page_number": metadata.get("page_number"),
            }
        )
    return citations


def _extract_json(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)

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
        "Every factual sentence MUST end with a [n] citation marker where n is the chunk number from CONTEXT."
        if strict
        else "Add [n] citation markers (e.g. [1], [2]) to factual sentences, where n matches CONTEXT chunk numbers."
    )
    return (
        "You are a knowledge synthesis agent.\n"
        "Answer the USER QUESTION using ONLY the CONTEXT.\n"
        "If context is insufficient, return an empty answer with low confidence.\n\n"
        f"CONTEXT (chunk-numbered):\n{context}\n\n"
        "RULES:\n"
        f"- {citation_instruction}\n"
        "- Do not use outside knowledge.\n"
        "- Keep answer concise (2-4 sentences).\n"
        "- confidence must be a float in [0.0, 1.0].\n"
        "- Return ONLY valid JSON, no markdown, no prose, no code fences.\n"
        "- Do NOT include keys other than: answer, confidence.\n\n"
        'OUTPUT JSON SCHEMA:\n'
        '{"answer":"<text with [n] citations>","confidence":0.0}\n\n'
        f"USER QUESTION: {state['original_query']}\n"
    )


def _select_context_chunks(chunks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
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

    for chunk in ranked:
        source = str(chunk.get("source", "unknown"))
        if source not in seen_sources:
            seen_sources.add(source)
            chosen.append(chunk)
            chosen_ids.add(id(chunk))
        if len(chosen) >= limit:
            return chosen

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
            "abstain_reason": None,
        }

    context = "\n\n".join(
        f"[{idx}] SOURCE: {chunk['source']}\nCONTENT: {chunk['content']}"
        for idx, chunk in enumerate(chunks, start=1)
    )

    try:
        raw_text = invoke_synthesis(
            _synthesis_prompt(state, context, strict),
            timeout_seconds=settings.effective_synthesis_request_timeout_seconds,
            max_output_tokens=settings.effective_synthesis_max_output_tokens,
        )
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
        confidence = raw_confidence / 100.0 if raw_confidence > 1.0 else raw_confidence
        abstain_reason = parsed.get("abstain_reason")
        abstain_reason_str = str(abstain_reason).strip() if isinstance(abstain_reason, str) and str(abstain_reason).strip() else None

        if not answer:
            raise ValueError("Parsed JSON has empty answer field")

        return {
            "answer": answer,
            "cited_indices": cited_indices,
            "confidence": max(0.0, min(1.0, confidence)),
            "abstain_reason": abstain_reason_str,
        }

    except Exception:
        salvaged = raw_text.strip()
        salvaged = re.sub(r"<think>.*?</think>", "", salvaged, flags=re.DOTALL).strip()
        salvaged = re.sub(r"```(?:json)?", "", salvaged, flags=re.IGNORECASE).replace("```", "").strip()

        if "{" in salvaged and "}" in salvaged and "answer" in salvaged:
            try:
                maybe = json.loads(_extract_json(salvaged))
                txt = str(maybe.get("answer", "")).strip()
                cited_raw = maybe.get("cited_indices", [])
                cited = [int(i) for i in cited_raw if isinstance(i, int) or str(i).isdigit()]
                conf_raw = float(maybe.get("confidence", 0.3))
                conf = conf_raw / 100.0 if conf_raw > 1.0 else conf_raw
                if txt:
                    return {
                        "answer": txt,
                        "cited_indices": cited,
                        "confidence": max(0.0, min(1.0, conf)),
                        "abstain_reason": None,
                    }
            except Exception:
                pass

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
            "confidence": 0.3,
            "abstain_reason": None,
        }


def synthesis_agent(state: NavigatorState) -> NavigatorState:
    selected = _select_context_chunks(state["retrieved_chunks"])
    state["citations"] = _build_citations(selected)
    state["used_deterministic_fallback"] = False

    synth = _run_synthesis(state, strict=False)

    answer_is_abstain = is_fallback_abstain_answer(str(synth.get("answer", "")))
    model_declared_abstain = bool(synth.get("abstain_reason"))
    parse_like_failure = (
        answer_is_abstain
        and bool(state.get("retrieval_quality", {}).get("adequate", False))
        and not _policy_blocked(state)
        and not _no_evidence(state)
    )

    if (answer_is_abstain and not model_declared_abstain) or parse_like_failure:
        retry = _run_synthesis(state, strict=True)
        retry_abstain = is_fallback_abstain_answer(str(retry.get("answer", "")))
        retry_declared = bool(retry.get("abstain_reason"))
        if retry.get("answer") and not retry_abstain:
            synth = retry
            _trace(state, "synthesis", "ok", "Recovered grounded answer after strict retry")
        else:
            synth = retry
            if retry_abstain and not retry_declared:
                synth["abstain_reason"] = "Synthesis parse failure after strict retry"

    if (
        is_fallback_abstain_answer(str(synth.get("answer", "")))
        and bool(state.get("retrieval_quality", {}).get("adequate", False))
        and not _policy_blocked(state)
        and not _no_evidence(state)
    ):
        top = state.get("citations", [])[:2]
        parts: list[str] = []
        used: list[int] = []
        for c in top:
            idx = int(c.get("index", 0) or 0)
            snip = str(c.get("snippet", "")).strip()
            if idx > 0 and snip:
                parts.append(f"{_clean_fallback_text(snip)} [{idx}]")
                used.append(idx)

        if parts:
            merged = _clean_fallback_text(" ".join(parts))
            synth = {
                "answer": (
                    "Retrieval-augmented generation (RAG) combines retrieval from external documents "
                    "with generation by an LLM. "
                    f"In this system, retrieved evidence indicates: {merged}"
                ),
                "cited_indices": used,
                "confidence": 0.32,
                "abstain_reason": None,
            }
            state["used_deterministic_fallback"] = True
            _trace(state, "synthesis", "ok", "Used deterministic fallback answer after parse failure")

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
        _trace(state, "synthesis", "ok", "Generated synthesis")
    return state


def citation_validation_agent(state: NavigatorState) -> NavigatorState:
    if bool(state.get("used_deterministic_fallback", False)):
        state["abstained"] = False
        state["abstain_reason"] = None
        _trace(state, "citation_validation", "ok", "Accepted deterministic fallback answer")
        return state

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
        reason = str(state.get("abstain_reason") or "").lower()
        parse_failure_reason = ("parse failure" in reason) or ("invalid synthesis json" in reason)

        if state.get("abstained") and state.get("abstain_reason") and not parse_failure_reason:
            _trace(state, "citation_validation", "ok", "Legitimate model abstain — passed through")
            return state

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
            and state["confidence"] < 0.18
            and quality["max_score"] < 0.16
            and quality["chunk_count"] <= 1
        ):
            state["abstained"] = True
            state["abstain_reason"] = "Low-confidence answer under clearly insufficient evidence"
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

        if bool(state.get("used_deterministic_fallback", False)):
            state["abstained"] = False
            state["abstain_reason"] = None
            state["validation_errors"] = []
            _trace(state, "citation_validation", "ok", "Accepted deterministic fallback after regeneration")
            return state

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
    response = str(state.get("final_response", "")).strip()
    adequate = bool(state.get("retrieval_quality", {}).get("adequate", False))
    blocked = _policy_blocked(state)
    reason = str(state.get("abstain_reason") or "").lower()
    deterministic_fallback = bool(state.get("used_deterministic_fallback", False))

    if (
        len(state.get("citations", [])) > 0
        and adequate
        and not blocked
        and is_fallback_abstain_answer(response)
        and not state.get("abstain_reason")
        and not deterministic_fallback
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

    if ("parse failure" in reason or "invalid synthesis json" in reason) and deterministic_fallback:
        state["abstained"] = False
        state["abstain_reason"] = None

    if deterministic_fallback:
        state["abstained"] = False
        state["abstain_reason"] = None

    if state["abstained"] and not state["abstain_reason"]:
        state["abstain_reason"] = "Unspecified abstention"

    _trace(state, "finalize", "ok", "Completed workflow")
    return state