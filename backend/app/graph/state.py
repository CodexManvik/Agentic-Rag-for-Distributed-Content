from typing import Any
from typing_extensions import TypedDict


class RetrievedChunk(TypedDict):
    chunk_id: str
    source: str
    content: str
    score: float
    metadata: dict[str, Any]


class Citation(TypedDict):
    index: int
    source: str
    url: str | None
    snippet: str


class NavigatorState(TypedDict):
    original_query: str
    sub_queries: list[str]
    retrieved_chunks: list[RetrievedChunk]
    final_response: str
    citations: list[Citation]
