from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SupervisorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(min_length=1)
    next_agent: Literal["RetrievalAgent", "SynthesisAgent", "AdequacyAgent", "FINISH"]


class SynthesisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
