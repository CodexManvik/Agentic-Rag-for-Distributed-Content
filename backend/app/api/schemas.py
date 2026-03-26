from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)


class Citation(BaseModel):
    index: int
    source: str
    url: str | None = None
    snippet: str
    source_type: str | None = None
    section: str | None = None
    page_number: int | None = None


class TraceEvent(BaseModel):
    node: str
    status: str
    detail: str


class RetrievalQuality(BaseModel):
    max_score: float
    avg_score: float
    source_diversity: int
    chunk_count: int
    adequate: bool
    reason: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    sub_queries: list[str]
    confidence: float
    abstained: bool
    abstain_reason: str | None = None
    trace: list[TraceEvent]
    retrieval_quality: RetrievalQuality
