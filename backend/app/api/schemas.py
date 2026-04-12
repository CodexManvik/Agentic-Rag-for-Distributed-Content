from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Message roles for conversation history."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    model: str | None = Field(default=None, description="Optional Ollama model name.")
    hide_reasoning: bool = Field(default=True, description="If True, strips <think> blocks from the output.")
    session_id: str | None = Field(default=None, description="Optional session ID for conversation continuity.")


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
    ts: str | None = None
    duration_ms: float | None = None


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
    stage_timings: dict[str, float] = Field(default_factory=dict)
    short_circuited: bool = Field(default=False, description="Whether supervisor short-circuit routing was used.")
    session_id: str | None = Field(default=None, description="Session ID if using session-based chat.")


# Session Management Schemas
class ConversationMessageSchema(BaseModel):
    """Schema for individual conversation messages."""
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelConfigSchema(BaseModel):
    """Schema for model configuration."""
    backend: str
    model_name: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0, le=32768)
    context_length: int = Field(default=8192, gt=0, le=65536)
    additional_params: Dict[str, Any] = Field(default_factory=dict)


class SessionStateSchema(BaseModel):
    """Schema for complete session state."""
    session_id: str
    user_id: str | None = None
    created_at: datetime
    last_active: datetime
    conversation_history: List[ConversationMessageSchema] = Field(default_factory=list)
    active_agents: List[str] = Field(default_factory=list)
    knowledge_base_id: str | None = None
    model_configuration: ModelConfigSchema | None = None
    session_metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateSessionRequest(BaseModel):
    """Request to create a new session."""
    user_id: str | None = None
    knowledge_base_id: str | None = None
    model_configuration: ModelConfigSchema | None = None
    session_metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateSessionResponse(BaseModel):
    """Response for session creation."""
    session_id: str
    created_at: datetime
    message: str = "Session created successfully"


class UpdateSessionRequest(BaseModel):
    """Request to update session configuration."""
    knowledge_base_id: str | None = None
    model_configuration: ModelConfigSchema | None = None
    session_metadata: Dict[str, Any] | None = None


class SessionListResponse(BaseModel):
    """Response for listing sessions."""
    sessions: List[SessionStateSchema]
    total: int
    limit: int
    offset: int


class SessionStatsResponse(BaseModel):
    """Response for session statistics."""
    total_sessions: int
    active_sessions: int
    messages_today: int
    avg_session_length: float
    top_knowledge_bases: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)

