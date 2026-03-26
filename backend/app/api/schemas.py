from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)


class Citation(BaseModel):
    index: int
    source: str
    url: str | None = None
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    sub_queries: list[str]
