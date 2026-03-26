from typing import Any
import re

from app.graph.state import Citation, NavigatorState
from app.services.llm import get_chat_model
from app.services.vector_store import query_chunks


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


def planning_agent(state: NavigatorState) -> NavigatorState:
    query = state["original_query"].strip()
    model = get_chat_model()
    sub_queries: list[str]

    if model:
        prompt = (
            "You are a planning agent. Generate 3 to 5 concise retrieval queries "
            "that decompose the user question. Return one query per line without numbering.\n\n"
            f"User question: {query}"
        )
        response_text = _message_to_text(model.invoke(prompt))
        sub_queries = [line.strip(" -0123456789.") for line in response_text.splitlines()]
        sub_queries = [q for q in sub_queries if q]
    else:
        tokens = re.findall(r"[A-Za-z0-9]+", query)
        keywords = " ".join(tokens[:8])
        sub_queries = [query, keywords] if keywords and keywords != query else [query]

    state["sub_queries"] = sub_queries[:5]
    return state


def retrieval_agent(state: NavigatorState) -> NavigatorState:
    chunks = query_chunks(state["sub_queries"])
    state["retrieved_chunks"] = chunks
    return state


def synthesis_agent(state: NavigatorState) -> NavigatorState:
    chunks = state["retrieved_chunks"][:8]
    citations: list[Citation] = []

    for idx, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        citations.append(
            {
                "index": idx,
                "source": chunk["source"],
                "url": metadata.get("url"),
                "snippet": chunk["content"][:240],
            }
        )

    if not chunks:
        state["final_response"] = "No relevant sources were found for this query."
        state["citations"] = []
        return state

    context_lines: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        context_lines.append(
            f"[{idx}] SOURCE: {chunk['source']}\nCONTENT: {chunk['content']}"
        )
    context = "\n\n".join(context_lines)

    model = get_chat_model()
    answer: str
    if model:
        prompt = (
            "You are a synthesis agent for an enterprise knowledge assistant. "
            "Answer only from the provided context. Every factual statement must "
            "include one or more bracket citations like [1], [2]. If evidence is "
            "insufficient, state that clearly.\n\n"
            f"Question: {state['original_query']}\n\n"
            f"Context:\n{context}\n\n"
            "Return concise, complete prose."
        )
        answer = _message_to_text(model.invoke(prompt)).strip()
    else:
        points: list[str] = []
        for idx, chunk in enumerate(chunks[:3], start=1):
            sentence = chunk["content"].strip().replace("\n", " ")
            points.append(f"{sentence[:220]} [{idx}]")
        answer = " ".join(points)

    state["final_response"] = answer
    state["citations"] = citations
    return state
