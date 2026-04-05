from typing import Any
from typing_extensions import NotRequired, TypedDict


class RetrievedChunk(TypedDict):
    chunk_id: str
    source: str
    content: str
    score: float
    metadata: dict[str, Any]
    matched_subqueries: NotRequired[list[str]]
    relevance_components: NotRequired[dict[str, float]]


class Citation(TypedDict):
    index: int
    source: str
    url: str | None
    snippet: str
    source_type: NotRequired[str | None]
    section: NotRequired[str | None]
    page_number: NotRequired[int | None]


class TraceEvent(TypedDict):
    node: str
    status: str
    detail: str
    ts: NotRequired[str]
    duration_ms: NotRequired[float]


class RetrievalQuality(TypedDict):
    max_score: float
    avg_score: float
    source_diversity: int
    chunk_count: int
    adequate: bool
    reason: str


class SynthesisOutput(TypedDict):
    answer: str
    cited_indices: list[int]
    confidence: float
    abstain_reason: str | None


class NavigatorState(TypedDict):
    query: str
    original_query: str
    selected_model: str  # Ollama model to use for this workflow execution
    sub_queries: list[str]
    retrieved_chunks: list[RetrievedChunk]
    final_response: str
    citations: list[Citation]
    retrieval_quality: RetrievalQuality
    retries_used: int
    validation_retries_used: int
    validation_errors: list[str]
    used_deterministic_fallback: bool
    abstained: bool
    abstain_reason: str | None
    confidence: float
    cited_indices: list[int]
    synthesis_output: SynthesisOutput
    trace: list[TraceEvent]
    stage_timings: dict[str, float]
    stage_timestamps: dict[str, dict[str, Any]]
