import json
import re
import sys
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.graph.state import Citation, NavigatorState, SynthesisOutput
from app.services.guardrails import validate_citations
from app.services.policy import detect_policy_scope_violation
from app.services.llm import ModelInvocationError, invoke_chat_with_timeout, invoke_synthesis
from app.services.vector_store import assess_retrieval_adequacy, query_chunks


# Global lock for thread-safe log file writes (prevents interleaved writes on Windows NTFS)
_LOG_FILE_LOCK = threading.Lock()

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
    # Remove bracketed paper references like [160] from source text so they are
    # not confused with system citation indices [1], [2], ...
    cleaned = re.sub(r"\[[0-9]+\]", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    units = [u.strip() for u in re.split(r"(?<=[.!?])\s+", cleaned) if u.strip()]
    if not units:
        return cleaned[:240].strip()
    out = " ".join(units[:2])
    if len(out) > 340:
        out = out[:337].rstrip() + "..."
    return out


def _first_sentence(text: str) -> str:
    parts = [u.strip() for u in re.split(r"(?<=[.!?])\s+", text) if u.strip()]
    if not parts:
        return ""
    first = parts[0]
    if first and first[-1] not in ".!?":
        first += "."
    return first


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
        expansions = [base]

        q_lower = base.lower()
        if any(k in q_lower for k in ["compare", "difference", "versus", "vs"]):
            # comparison: add each side as a standalone sub-query
            # Use more robust splitting that avoids edge cases like empty strings
            parts = re.split(r"\b(?:compare|versus|vs|difference between|and)\b", q_lower, maxsplit=1)
            for p in parts:
                p = p.strip()
                # Only add if it's a meaningful non-empty string
                if p and len(p) > 6 and not p.isspace():
                    expansions.append(p)
        elif any(k in q_lower for k in ["how", "steps", "procedure", "guide"]):
            expansions.append(f"{base} tutorial steps")
            expansions.append(f"{base} best practices")
        elif any(k in q_lower for k in ["what is", "define", "explain", "describe"]):
            # extract the noun phrase after the question word
            noun = re.sub(r"^(?:what\s+is|define|explain|describe)\s+", "", q_lower, flags=re.IGNORECASE).strip()
            if noun and len(noun) > 3:  # Ensure noun is meaningful
                expansions.append(f"{noun} overview")
                expansions.append(f"{noun} use cases examples")
        else:
            # Generic: add a "use case" and "guide" variant
            expansions.append(f"{base} use cases")
            expansions.append(f"{base} guide")

        # _unique_queries filters empty strings, so this is safe, but let's be explicit
        filtered_expansions = [e for e in expansions if e and e.strip()]
        state["sub_queries"] = _unique_queries(filtered_expansions)[: max(2, settings.planner_max_subqueries)]
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
                model_name=state.get("selected_model"),
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
                model_name=state.get("selected_model"),
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


def _synthesis_prompt(state: NavigatorState, context: str, num_chunks: int, strict: bool) -> list[dict[str, str]]:
    """Return a messages list (system + user) instead of a plain string.

    llama3.2:3b reliably follows instructions when given a strong system role
    and a clean user turn. A flat text prompt causes it to echo the FACTS block
    instead of answering, because the small model treats the whole string as
    continuation context to complete rather than an instruction to follow.
    """
    system = (
        "You are a helpful assistant. Answer questions using ONLY the provided sources. "
        "Be concise (2-3 sentences). Always end with a citation like [1] or [2]."
    )
    user = (
        f"Sources:\n{context}\n\n"
        f"Question: {state['original_query']}\n\n"
        f"Answer in 2-3 sentences using only the sources above. "
        f"Cite with [1] or [2] (only numbers 1 to {num_chunks})."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_BIBLIOGRAPHY_PATTERN = re.compile(
    r"(?:arXiv|arxiv)\s*preprint|"
    r"\[\d{2,3}\]\s+[A-Z]|"
    r"(?:doi|DOI):\s*10\.\d{4}|"
    r"(?:pp\.|pages?)\s+\d+[-–]\d+",
    re.IGNORECASE,
)


def _is_bibliography_chunk(content: str) -> bool:
    return len(_BIBLIOGRAPHY_PATTERN.findall(content)) >= 2


def _is_fragment_chunk(content: str) -> bool:
    """Return True if the chunk starts mid-sentence (broken PDF boundary)."""
    stripped = content.strip()
    if not stripped:
        return True
    first_char = stripped[0]
    # Starts with lowercase = mid-sentence fragment from a bad chunk boundary
    return first_char.islower()


def _content_fingerprint(text: str) -> str:
    """First 80 chars of normalized text — used to detect near-duplicate chunks."""
    return re.sub(r"\s+", " ", text.strip().lower())[:80]


def _select_context_chunks(chunks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    limit = settings.context_chunk_limit
    char_limit = settings.context_chunk_char_limit

    def _clamp(chunk: Mapping[str, Any]) -> dict[str, Any]:
        c = dict(chunk)
        c["content"] = str(c.get("content", ""))[:char_limit]
        return c

    ranked: list[dict[str, Any]] = []
    for chunk in chunks:
        content = str(chunk.get("content", "")).strip()
        if content and not _is_bibliography_chunk(content) and not _is_fragment_chunk(content):
            ranked.append(_clamp(chunk))
    ranked.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)

    if not ranked:
        return []

    chosen: list[dict[str, Any]] = []
    chosen_ids: set[int] = set()
    seen_sources: set[str] = set()
    seen_fingerprints: set[str] = set()

    for chunk in ranked:
        source = str(chunk.get("source", "unknown"))
        fp = _content_fingerprint(str(chunk.get("content", "")))
        if source not in seen_sources and fp not in seen_fingerprints:
            seen_sources.add(source)
            seen_fingerprints.add(fp)
            chosen.append(chunk)
            chosen_ids.add(id(chunk))
        if len(chosen) >= limit:
            return chosen

    for chunk in ranked:
        if len(chosen) >= limit:
            break
        fp = _content_fingerprint(str(chunk.get("content", "")))
        if id(chunk) not in chosen_ids and fp not in seen_fingerprints:
            seen_fingerprints.add(fp)
            chosen.append(chunk)
            chosen_ids.add(id(chunk))

    return chosen


def _run_synthesis(state: NavigatorState, selected_chunks: list[dict[str, Any]], strict: bool) -> SynthesisOutput:
    """Run synthesis with pre-selected chunks (avoids redundant chunk selection)."""
    chunks = selected_chunks
    if not chunks:
        return {
            "answer": FALLBACK_ABSTAIN_TEXT,
            "cited_indices": [],
            "confidence": 0.0,
            "abstain_reason": None,
        }

    # Keep synthesis prompt compact to avoid model timeouts on low-resource hardware.
    compact_chunks = chunks[:3]

    def _clean_content(text: str) -> str:
        cleaned = re.sub(r"\[\d{2,}\]", "", text)
        return re.sub(r"\s{2,}", " ", cleaned).strip()

    context = "\n\n".join(
        f"[{idx}] SOURCE: {chunk['source']}\nCONTENT: {_clean_content(str(chunk['content']))[:320]}"
        for idx, chunk in enumerate(compact_chunks, start=1)
    )

    max_tokens = settings.effective_synthesis_max_output_tokens
    if settings.normalized_runtime_profile == "low_latency":
        max_tokens = min(max_tokens, 280)

    def _grounded_fallback_output() -> SynthesisOutput | None:
        top = compact_chunks[:2]
        fallback_parts: list[str] = []
        cited: list[int] = []
        for idx, chunk in enumerate(top, start=1):
            snippet = _clean_fallback_text(str(chunk.get("content", "")))
            sentence = _first_sentence(snippet)
            if sentence:
                # Keep one cited sentence per chunk so citation validation can map claims.
                fallback_parts.append(f"{sentence} [{idx}]")
                cited.append(idx)
        if not fallback_parts:
            return None
        return {
            "answer": " ".join(fallback_parts),
            "cited_indices": cited,
            "confidence": 0.25,
            "abstain_reason": None,
        }

    max_valid_idx = max(1, len(compact_chunks))

    def _normalize_citation_markers(answer_text: str, cited: list[int]) -> tuple[str, list[int]]:
        # Keep only [1..N] markers where N is the number of chunks used in synthesis.
        valid = sorted({i for i in cited if 1 <= i <= max_valid_idx})

        def _strip_invalid(match: re.Match[str]) -> str:
            idx = int(match.group(1))
            return match.group(0) if 1 <= idx <= max_valid_idx else ""

        normalized = re.sub(r"\[(\d+)\]", _strip_invalid, answer_text)
        normalized = re.sub(r"\s{2,}", " ", normalized).strip()
        return normalized, valid

    try:
        # Invoke synthesis with enough time and tokens for proper answer generation
        raw_text = invoke_synthesis(
            _synthesis_prompt(state, context, len(compact_chunks), strict),
            timeout_seconds=settings.effective_synthesis_request_timeout_seconds,
            max_output_tokens=max_tokens,
            model_name=state.get("selected_model"),
        )
        # Log raw LLM output to file — use lock to prevent interleaved writes from concurrent threads on Windows NTFS
        _log_path = Path(__file__).resolve().parents[2] / "resources" / "llm_raw_output.log"
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_FILE_LOCK:
            with _log_path.open("a", encoding="utf-8") as _f:
                _f.write(f"\n{'='*60}\n")
                _f.write(f"[{datetime.now(timezone.utc).isoformat()}] QUERY: {state.get('original_query', '')}\n")
                _f.write(f"RAW OUTPUT:\n{raw_text}\n")
                _f.write(f"{'='*60}\n")
        sys.stderr.write(f"[LLM] raw output written to {_log_path}\n")
        sys.stderr.flush()
    except ModelInvocationError as exc:
        # Grounded fallback on synthesis timeout/error.
        fallback = _grounded_fallback_output()
        if fallback:
            return fallback

        return {
            "answer": FALLBACK_ABSTAIN_TEXT,
            "cited_indices": [],
            "confidence": 0.0,
            "abstain_reason": str(exc),
        }

    try:
        # Try JSON parsing first (in case model outputs it anyway)
        try:
            parsed = json.loads(_extract_json(raw_text))
            answer = str(parsed.get("answer", "")).strip()
            if answer:
                if is_fallback_abstain_answer(answer):
                    fallback = _grounded_fallback_output()
                    if fallback:
                        return fallback
                inline_cited = sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", answer)))
                answer, inline_cited = _normalize_citation_markers(answer, inline_cited)
                return {
                    "answer": answer,
                    "cited_indices": inline_cited,
                    "confidence": 0.85 if inline_cited else 0.6,
                    "abstain_reason": None,
                }
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: treat raw text as answer, extract [n] citations
        answer = raw_text.strip()
        if answer.startswith("```"):
            answer = re.sub(r"^```json?\n?", "", answer).rsplit("```", 1)[0].strip()
        
        # Extract cited indices from [n] markers (can be empty)
        cited_indices = sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", answer)))
        answer, cited_indices = _normalize_citation_markers(answer, cited_indices)
        
        # Accept ANY non-empty answer, even without citations
        if answer:
            if is_fallback_abstain_answer(answer):
                fallback = _grounded_fallback_output()
                if fallback:
                    return fallback
            return {
                "answer": answer,
                "cited_indices": cited_indices,
                "confidence": 0.85 if cited_indices else 0.5,
                "abstain_reason": None,
            }

        raise ValueError("Answer is empty")


    except Exception as e:
        # Last resort: use raw text directly, any length
        salvaged = raw_text.strip()
        # Remove markdown fences if present
        salvaged = re.sub(r"^```.*?\n?", "", salvaged, flags=re.IGNORECASE).rsplit("```", 1)[0].strip()
        # Remove think tags
        salvaged = re.sub(r"<think>.*?</think>", "", salvaged, flags=re.DOTALL).strip()
        
        inline_citations = sorted(set(int(m) for m in re.findall(r"\[(\d+)\]", salvaged)))
        salvaged, inline_citations = _normalize_citation_markers(salvaged, inline_citations)
        
        if salvaged and not is_fallback_abstain_answer(salvaged):
            return {
                "answer": salvaged,
                "cited_indices": inline_citations,
                "confidence": 0.7,
                "abstain_reason": None,
            }

        fallback = _grounded_fallback_output()
        if fallback:
            return fallback

        # True fallback: can't extract anything useful
        return {
            "answer": FALLBACK_ABSTAIN_TEXT,
            "cited_indices": [],
            "confidence": 0.0,
            "abstain_reason": None,
        }


def synthesis_agent(state: NavigatorState) -> NavigatorState:
    selected = _select_context_chunks(state["retrieved_chunks"])
    state["citations"] = _build_citations(selected)
    state["_selected_context_chunks"] = selected  # Store for potential retries
    state["used_deterministic_fallback"] = False

    # Run synthesis once with the selected chunks (no re-selection inside _run_synthesis)
    synth = _run_synthesis(state, selected, strict=False)

    # Prune citations list to only those actually referenced in the answer.
    # Unreferenced citation objects inflate citation FP in evaluation.
    cited_set = set(synth.get("cited_indices", []))
    if cited_set and not is_fallback_abstain_answer(str(synth.get("answer", ""))):
        kept_citations = [c for c in state["citations"] if c.get("index") in cited_set]
        
        # BUILD MAPPING: old index → new index
        old_to_new: dict[int, int] = {}
        for new_idx, c in enumerate(kept_citations, start=1):
            old_idx = int(c.get("index", 0) or 0)
            old_to_new[old_idx] = new_idx
            c["index"] = new_idx
        
        # REMAP ANSWER TEXT: [old] → [new] using regex substitution to avoid conflicts
        # This prevents substring matching issues (e.g., [2] corrupting [12])
        answer_text = str(synth.get("answer", ""))
        
        def replace_citation(match: re.Match[str]) -> str:
            old_idx_str = match.group(1)
            try:
                old_idx = int(old_idx_str)
                new_idx = old_to_new.get(old_idx)
                if new_idx is not None:
                    return f"[{new_idx}]"
            except (ValueError, KeyError):
                pass
            return match.group(0)
        
        # Replace all citation markers using a callback to handle all replacements atomically
        answer_text = re.sub(r"\[(\d+)\]", replace_citation, answer_text)
        
        synth["answer"] = answer_text
        state["citations"] = kept_citations
        # Rebuild cited_indices to match new sequential indices
        synth["cited_indices"] = list(range(1, len(kept_citations) + 1))

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
    # REMOVED: blanket skip for deterministic fallback
    # Instead, validate ALL answers including fallback ones
    # if bool(state.get("used_deterministic_fallback", False)):
    #     state["abstained"] = False
    #     state["abstain_reason"] = None
    #     _trace(state, "citation_validation", "ok", "Accepted deterministic fallback answer")
    #     return state

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
        # Use pre-selected chunks from synthesis_agent to avoid re-selection
        selected_chunks = state.get("_selected_context_chunks", [])
        strict = _run_synthesis(state, selected_chunks, strict=True)
        state["synthesis_output"] = strict
        state["final_response"] = strict["answer"]
        state["cited_indices"] = strict["cited_indices"]
        state["confidence"] = strict["confidence"]
        state["abstain_reason"] = strict["abstain_reason"]

        # Re-prune citations to those actually referenced in the new answer
        new_cited_set = set(strict.get("cited_indices", []))
        if new_cited_set and not is_fallback_abstain_answer(str(strict.get("answer", ""))):
            kept_citations = [c for c in state["citations"] if c.get("index") in new_cited_set]
            
            # BUILD MAPPING: old index → new index
            old_to_new: dict[int, int] = {}
            for new_idx, c in enumerate(kept_citations, start=1):
                old_idx = int(c.get("index", 0) or 0)
                old_to_new[old_idx] = new_idx
                c["index"] = new_idx
            
            # REMAP answer text to new indices using regex to avoid substring conflicts
            answer_text = str(strict.get("answer", ""))
            
            def replace_citation(match: re.Match[str]) -> str:
                old_idx_str = match.group(1)
                try:
                    old_idx = int(old_idx_str)
                    new_idx = old_to_new.get(old_idx)
                    if new_idx is not None:
                        return f"[{new_idx}]"
                except (ValueError, KeyError):
                    pass
                return match.group(0)
            
            answer_text = re.sub(r"\[(\d+)\]", replace_citation, answer_text)
            
            strict["answer"] = answer_text
            state["citations"] = kept_citations
            state["final_response"] = answer_text
            
            # Rebuild citation_snippets with NEW sequential indices
            citation_snippets = {
                int(c.get("index", 0)): str(c.get("snippet", ""))
                for c in state["citations"]
                if isinstance(c.get("index"), int)
            }

        # REMOVED blanket acceptance of deterministic fallback
        # Validate it like any other answer (BUG #5 fix)
        
        second = validate_citations(
            state["final_response"],
            len(state["citations"]),
            citation_snippets=citation_snippets,
        )
        if second["valid"]:
            state["validation_errors"] = []
            _trace(state, "citation_validation", "ok", "Passed after one regeneration")
            return state
        
        # Check if deterministic fallback failed validation (BUG #5 fix)
        if bool(state.get("used_deterministic_fallback", False)):
            state["abstained"] = True
            state["abstain_reason"] = "Fallback answer failed citation validation"
            state["validation_errors"] = second["errors"]
            _trace(state, "citation_validation", "failed", "fallback_validation_failed")
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
    chunk_count = int(state.get("retrieval_quality", {}).get("chunk_count", 0))

    # CATCH: Fallback answer + actual citations + no explicit abstain reason
    # = Silent synthesis failure that should be flagged (BUG #10 fix)
    if (
        is_fallback_abstain_answer(response)
        and chunk_count >= 2
        and not state.get("abstain_reason")
        and not blocked
        and not deterministic_fallback
    ):
        # Silent abstain despite having evidence
        state["abstained"] = True
        state["abstain_reason"] = (
            "Synthesis was unable to generate a grounded answer despite having "
            f"retrieved {chunk_count} relevant chunks. Try rephrasing or ingesting more specific docs."
        )
        state["validation_errors"].append("silent_abstain_with_evidence")
        _trace(state, "finalize", "warning", state["abstain_reason"])
        return state

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