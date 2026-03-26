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
        prompt = f"""
You are the Synthesis Agent for an enterprise knowledge assistant.
Your objective is to answer the user's question using ONLY the provided context retrieved from the system.

CONTEXT DATA:
{context}

RULES FOR SYNTHESIS:
1. STRICT GROUNDING: If the context does not contain the answer, you must state: "I do not have sufficient information in the retrieved documents to answer this query." Do not attempt to guess or use external knowledge.
2. MANDATORY CITATIONS: Every factual claim, statistic, or specific procedure you state MUST be immediately followed by its corresponding source index in brackets. Example: "The deployment threshold is 95% [1]."
3. MULTIPLE CITATIONS: If multiple sources support a claim, combine them. Example: "Both nodes must be restarted [1][3]."
4. NO SOURCE NAMING IN TEXT: Do not write "According to Source [1]...". State the fact directly, followed by the bracketed citation.

USER QUESTION:
{state['original_query']}

Generate the final response following all rules above:
"""
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
