from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QueryIntent(str, Enum):
    FACT_LOOKUP = "fact_lookup"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    PROCEDURAL = "procedural"
    ANALYSIS = "analysis"
    UNKNOWN = "unknown"


class QueryComplexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    MULTI_STEP = "multi_step"
    SIMPLE_LOOKUP = "simple_lookup"


class QueryAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: QueryIntent
    complexity: QueryComplexity
    required_capabilities: list[str] = Field(default_factory=list)
    suggested_agents: list[str] = Field(default_factory=list)
    requires_planning: bool = False
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class RoutingStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str = Field(min_length=1)
    order: int = Field(ge=1)
    parallel_group: str | None = None


class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_agents: list[str] = Field(default_factory=list)
    execution_order: list[RoutingStep] = Field(default_factory=list)
    reason: str = Field(default="", min_length=0)
    is_short_circuited: bool = False


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1)
    agent: str = Field(min_length=1)
    input_transform: str = "identity"
    output_transform: str = "identity"
    dependencies: list[str] = Field(default_factory=list)
    parallel_group: str | None = None


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(min_length=1)
    steps: list[PlanStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool = True
    state: dict[str, Any] = Field(default_factory=dict)
    executed_steps: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class SupervisorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis: QueryAnalysis
    routing: RoutingDecision
    plan: ExecutionPlan | None = None
    execution: ExecutionResult
    final_state: dict[str, Any] = Field(default_factory=dict)
