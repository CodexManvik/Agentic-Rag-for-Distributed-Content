from datetime import datetime, timezone
from typing import Any, cast

from langchain_ollama import ChatOllama
from loguru import logger

from app.agents.schemas import SupervisorDecision, SynthesisOutput
from app.config import settings
from app.graph.state import AgentState, RetrievedChunk, TraceEvent
from app.services.retrieval import get_retriever


FALLBACK_ABSTAIN_TEXT = "I do not have sufficient information in the retrieved documents to answer this query."


def _append_trace(state: AgentState, node: str, status: str, detail: str) -> list[TraceEvent]:
    trace = list(state.get("trace", []))
    trace.append(
        {
            "node": node,
            "status": status,
            "detail": detail,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    )
    return trace


def _resolve_query(state: AgentState) -> str:
    return str(state.get("query") or state.get("original_query") or "").strip()


def _extract_chunk(source_node: Any, index: int) -> RetrievedChunk:
    node = getattr(source_node, "node", None)
    score = float(getattr(source_node, "score", 0.0) or 0.0)

    metadata_raw = getattr(node, "metadata", {}) if node is not None else {}
    metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}

    if node is not None and hasattr(node, "get_content"):
        try:
            content = str(node.get_content(metadata_mode="none"))
        except TypeError:
            content = str(node.get_content())
    else:
        content = str(getattr(node, "text", "") or "")

    source = str(
        metadata.get("source")
        or metadata.get("file_name")
        or metadata.get("title")
        or metadata.get("path")
        or "unknown"
    )
    chunk_id = str(
        (getattr(node, "node_id", "") if node is not None else "")
        or metadata.get("doc_id")
        or f"chunk-{index}"
    )

    return {
        "chunk_id": chunk_id,
        "source": source,
        "content": content,
        "score": score,
        "metadata": metadata,
    }


def _compute_retrieval_quality(chunks: list[RetrievedChunk]) -> dict[str, Any]:
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


def retrieval_node(state: AgentState) -> dict[str, Any]:
    logger.info(
        f"🔎 Retrieval Node executing. Chunks currently in state: {len(state.get('retrieved_chunks', []))}"
    )
    retrieval_attempts = int(state.get("retries_used", 0)) + 1
    query = _resolve_query(state)
    if not query:
        return {
            "retrieved_chunks": [],
            "retrieval_quality": _compute_retrieval_quality([]),
            "retries_used": retrieval_attempts,
            "trace": _append_trace(state, "RetrievalAgent", "failed", "Empty query"),
        }

    try:
        retriever = get_retriever()
        
        # Debug the actual database size (LanceDB)
        try:
            from app.vectorstore.lancedb_store import LanceDBVectorStore
            store = LanceDBVectorStore(db_path="./lancedb_data")
            db_size = store.count_documents(knowledge_base="hackathon_demo")
            logger.info(f"📊 Total chunks currently in LanceDB (hackathon_demo): {db_size}")
        except Exception as e:
            logger.info(f"📊 Could not count LanceDB size: {e}")
        
        # retrieve() directly returns a list of NodeWithScore objects (no LLM call)
        source_nodes = retriever.retrieve(query)
        formatted_chunks = [_extract_chunk(node, idx) for idx, node in enumerate(source_nodes, start=1)]

        return {
            "retrieved_chunks": formatted_chunks,
            "sub_queries": [query],
            "retries_used": retrieval_attempts,
            "retrieval_quality": _compute_retrieval_quality(formatted_chunks),
            "trace": _append_trace(
                state,
                "RetrievalAgent",
                "ok",
                f"Retrieved {len(formatted_chunks)} chunks from LanceDB",
            ),
        }
    except Exception as exc:
        return {
            "retrieved_chunks": [],
            "retrieval_quality": _compute_retrieval_quality([]),
            "retries_used": retrieval_attempts,
            "trace": _append_trace(state, "RetrievalAgent", "failed", str(exc)),
            "abstain_reason": str(exc),
        }


def _build_synthesis_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    serialized_chunks = []
    for idx, chunk in enumerate(chunks, start=1):
        serialized_chunks.append(
            f"[{idx}] SOURCE: {chunk.get('source', 'unknown')}\nCONTENT: {str(chunk.get('content', ''))[:1200]}"
        )

    context = "\n\n".join(serialized_chunks)
    return (
        "You are a grounded synthesis agent.\n"
        "Use only the provided chunks.\n"
        "Return a concise answer with citations by chunk index.\n\n"
        "CRITICAL: You must output ONLY a valid JSON object. Do not output markdown, schema definitions, or introductory text. Example valid output:\n"
        '{\n'
        '  "answer": "The retrieved documents state that RAG combines retrieval with generation.",\n'
        '  "citations": [1, 2],\n'
        '  "confidence": 0.95\n'
        '}\n\n'
        f"USER QUERY:\n{query}\n\n"
        f"RETRIEVED CHUNKS:\n{context}\n"
    )


def _build_grounded_narrative_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    serialized_chunks = []
    for idx, chunk in enumerate(chunks[:6], start=1):
        serialized_chunks.append(
            f"[{idx}] SOURCE: {chunk.get('source', 'unknown')}\nCONTENT: {str(chunk.get('content', ''))[:900]}"
        )
    context = "\n\n".join(serialized_chunks)
    return (
        "You are a precise RAG answer writer.\n"
        "Use only the retrieved chunks as evidence.\n"
        "Write a direct, readable answer that synthesizes information, not raw excerpts.\n"
        "If the user asks for a brief explanation, keep it to 4-7 sentences.\n"
        "Do not say 'Based on the retrieved documents'.\n"
        "Do not invent facts not present in the chunks.\n\n"
        f"USER QUERY:\n{query}\n\n"
        f"RETRIEVED CHUNKS:\n{context}\n\n"
        "FINAL ANSWER:"
    )


def _llm_grounded_narrative_answer(
    query: str,
    chunks: list[RetrievedChunk],
    selected_model: str,
) -> str:
    try:
        prompt = _build_grounded_narrative_prompt(query, chunks)
        llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=selected_model,
            temperature=0,
        )
        response = llm.invoke(prompt)
        content = getattr(response, "content", response)
        return str(content).strip()
    except Exception:
        return ""


def _deterministic_grounded_answer(chunks: list[RetrievedChunk]) -> tuple[str, list[int]]:
    """Build a grounded fallback answer from retrieved chunks without abstaining."""
    if not chunks:
        return "", []

    top_k = min(3, len(chunks))
    cited_indices = list(range(1, top_k + 1))
    bullet_points: list[str] = []
    for idx in cited_indices:
        chunk = chunks[idx - 1]
        text = " ".join(str(chunk.get("content", "")).split())
        if not text:
            continue
        snippet = text[:260].rstrip(" ,.;:")
        source = str(chunk.get("source", "unknown"))
        bullet_points.append(f"- {snippet} (source: {source})")

    if not bullet_points:
        return "", cited_indices

    answer = (
        "Based on the retrieved documents, here is what can be stated:\n"
        + "\n".join(bullet_points)
    )
    return answer, cited_indices


def synthesis_node(state: AgentState) -> dict[str, Any]:
    logger.info(
        f"🧠 Synthesis Node executing. Chunks currently in state: {len(state.get('retrieved_chunks', []))}"
    )
    query = _resolve_query(state)
    chunks = list(state.get("retrieved_chunks", []))

    if not chunks:
        return {
            "final_response": FALLBACK_ABSTAIN_TEXT,
            "citations": [],
            "cited_indices": [],
            "confidence": 0.0,
            "abstained": True,
            "abstain_reason": "No retrieved chunks for synthesis",
            "trace": _append_trace(state, "SynthesisAgent", "failed", "No chunks available"),
        }

    prompt = _build_synthesis_prompt(query, cast(list[RetrievedChunk], chunks))
    selected_model = str(state.get("selected_model") or settings.ollama_chat_model)

    try:
        llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=selected_model,
            temperature=0,
        )
        structured_llm = llm.with_structured_output(SynthesisOutput)
        response = cast(SynthesisOutput, structured_llm.invoke(prompt))
        
        if response is None:
            raise ValueError("LLM failed to return structured JSON. Response was None.")

        max_idx = len(chunks)
        cited_indices = sorted(
            {
                int(idx)
                for idx in response.citations
                if isinstance(idx, int) and 1 <= int(idx) <= max_idx
            }
        )
        answer = response.answer.strip()
        confidence = float(response.confidence)

        if answer in {"", FALLBACK_ABSTAIN_TEXT}:
            narrative_answer = _llm_grounded_narrative_answer(query, cast(list[RetrievedChunk], chunks), selected_model)
            if narrative_answer and narrative_answer != FALLBACK_ABSTAIN_TEXT:
                fallback_citations = cited_indices if cited_indices else [1, 2, 3][: len(chunks)]
                return {
                    "final_response": narrative_answer,
                    "citations": fallback_citations,
                    "cited_indices": fallback_citations,
                    "confidence": 0.65,
                    "abstained": False,
                    "abstain_reason": None,
                    "used_deterministic_fallback": False,
                    "trace": _append_trace(
                        state,
                        "SynthesisAgent",
                        "degraded",
                        "Structured output was empty/abstain; used grounded narrative synthesis fallback",
                    ),
                }

            fallback_answer, fallback_citations = _deterministic_grounded_answer(cast(list[RetrievedChunk], chunks))
            if fallback_answer:
                return {
                    "final_response": fallback_answer,
                    "citations": fallback_citations,
                    "cited_indices": fallback_citations,
                    "confidence": 0.55,
                    "abstained": False,
                    "abstain_reason": None,
                    "used_deterministic_fallback": True,
                    "trace": _append_trace(
                        state,
                        "SynthesisAgent",
                        "degraded",
                        "Model produced empty/abstain answer; used deterministic grounded fallback",
                    ),
                }

            answer = FALLBACK_ABSTAIN_TEXT

        return {
            "final_response": answer,
            "citations": cited_indices,
            "cited_indices": cited_indices,
            "confidence": confidence,
            "abstained": answer == FALLBACK_ABSTAIN_TEXT,
            "abstain_reason": None if answer != FALLBACK_ABSTAIN_TEXT else "Model returned empty answer",
            "used_deterministic_fallback": False,
            "trace": _append_trace(state, "SynthesisAgent", "ok", "Generated structured synthesis"),
        }
    except Exception as exc:
        narrative_answer = _llm_grounded_narrative_answer(query, cast(list[RetrievedChunk], chunks), selected_model)
        if narrative_answer and narrative_answer != FALLBACK_ABSTAIN_TEXT:
            fallback_citations = [1, 2, 3][: len(chunks)]
            return {
                "final_response": narrative_answer,
                "citations": fallback_citations,
                "cited_indices": fallback_citations,
                "confidence": 0.6,
                "abstained": False,
                "abstain_reason": None,
                "used_deterministic_fallback": False,
                "trace": _append_trace(
                    state,
                    "SynthesisAgent",
                    "degraded",
                    f"Structured synthesis failed ({exc}); used grounded narrative synthesis fallback",
                ),
            }

        fallback_answer, fallback_citations = _deterministic_grounded_answer(cast(list[RetrievedChunk], chunks))
        if fallback_answer:
            return {
                "final_response": fallback_answer,
                "citations": fallback_citations,
                "cited_indices": fallback_citations,
                "confidence": 0.45,
                "abstained": False,
                "abstain_reason": None,
                "used_deterministic_fallback": True,
                "trace": _append_trace(
                    state,
                    "SynthesisAgent",
                    "degraded",
                    f"Structured synthesis failed ({exc}); used deterministic grounded fallback",
                ),
            }

        return {
            "final_response": FALLBACK_ABSTAIN_TEXT,
            "citations": [],
            "cited_indices": [],
            "confidence": 0.0,
            "abstained": True,
            "abstain_reason": str(exc),
            "trace": _append_trace(state, "SynthesisAgent", "failed", str(exc)),
        }


def _fallback_next_step(state: AgentState) -> str:
    chunks = list(state.get("retrieved_chunks", []))
    final_response = str(state.get("final_response", "")).strip()
    if final_response:
        return "FINISH"
    if not chunks:
        return "RetrievalAgent"
    return "SynthesisAgent"


def supervisor_node(state: AgentState) -> dict[str, Any]:
    logger.info(
        f"🛡️ Supervisor Node executing. Chunks currently in state: {len(state.get('retrieved_chunks', []))}"
    )
    query = _resolve_query(state)
    chunks = list(state.get("retrieved_chunks", []))
    final_response = str(state.get("final_response", "")).strip()
    retrieval_attempts = int(state.get("retries_used", 0))

    # Hard stop to prevent retrieval loops when retrieval has already run and produced no chunks.
    if final_response:
        return {
            "next_step": "FINISH",
            "trace": _append_trace(state, "Supervisor", "ok", "Routed to FINISH (response already generated)"),
        }
    if len(chunks) > 0:
        return {
            "next_step": "SynthesisAgent",
            "trace": _append_trace(state, "Supervisor", "ok", "Routed to SynthesisAgent (chunks available)"),
        }
    if retrieval_attempts >= 1:
        return {
            "next_step": "FINISH",
            "abstained": True,
            "abstain_reason": "No chunks retrieved after retrieval attempt",
            "trace": _append_trace(state, "Supervisor", "ok", "Routed to FINISH (no chunks after retrieval)"),
        }

    prompt = (
        "You are a routing supervisor.\n"
        "Rule 1: If the user's query requires information and 'retrieved_chunks' is empty, you MUST choose 'RetrievalAgent'.\n"
        "Rule 2: If 'retrieved_chunks' has 1 or more items, you MUST choose 'SynthesisAgent'. Do NOT retrieve again.\n"
        "Rule 3: If 'final_response' has been generated, you MUST choose 'FINISH'.\n\n"
        f"Query: {query}\n"
        f"retrieved_chunks_count: {len(chunks)}\n"
        f"final_response_generated: {bool(final_response)}\n"
        "Return a valid supervisor decision.\n\n"
        "CRITICAL: You must output ONLY a valid JSON object. Do not output schema definitions. Example valid output:\n"
        "{\n"
        '  "reasoning": "I need to search the knowledge base to answer the user\'s query.",\n'
        '  "next_agent": "RetrievalAgent"\n'
        "}"
    )

    selected_model = str(state.get("selected_model") or settings.ollama_chat_model)
    fallback_step = _fallback_next_step(state)

    try:
        llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=selected_model,
            temperature=0,
        )
        structured_llm = llm.with_structured_output(SupervisorDecision)
        decision = cast(SupervisorDecision, structured_llm.invoke(prompt))
        next_step = decision.next_agent

        if next_step == "AdequacyAgent":
            next_step = "RetrievalAgent"
        next_step = "RetrievalAgent"

        return {
            "next_step": next_step,
            "trace": _append_trace(state, "Supervisor", "ok", f"Routed to {next_step}"),
        }
    except Exception as exc:
        return {
            "next_step": fallback_step,
            "trace": _append_trace(
                state,
                "Supervisor",
                "failed",
                f"Fallback route to {fallback_step}: {exc}",
            ),
        }
