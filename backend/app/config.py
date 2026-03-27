from functools import cached_property
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3.5:0.8b"
    ollama_embedding_model: str = "nomic-embed-text:latest"

    chroma_collection_name: str = "knowledge_base"
    chroma_persist_directory: str = "./chroma_data"
    retrieval_top_k: int = 6
    retrieval_per_query_k: int = 4
    retrieval_bm25_k: int = 6
    hybrid_retrieval_enabled: bool = True
    bm25_cache_enabled: bool = True
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    retrieval_min_score: float = 0.28
    retrieval_min_chunks: int = 2
    retrieval_min_source_diversity: int = 1
    retrieval_hard_query_min_score_boost: float = 0.08
    retrieval_hard_query_min_source_diversity: int = 3
    retrieval_query_overlap_min: float = 0.15
    retrieval_entity_overlap_min: float = 0.10
    retrieval_chunk_min_term_overlap: int = 1
    retrieval_workflow_chunk_min_term_overlap: int = 2
    retrieval_pdf_penalty_for_workflow: float = 0.55
    retrieval_workflow_source_boost: float = 1.35
    retrieval_langgraph_source_boost: float = 1.2
    rerank_enabled: bool = True
    low_latency_skip_overlap_check: bool = False
    debug_trace_enabled: bool = False
    max_retrieval_retries: int = 1
    max_validation_retries: int = 1

    chunk_size: int = 1200
    chunk_overlap: int = 200

    runtime_profile: str = "low_latency"
    model_temperature: float = 0.0
    model_top_p: float = 0.9
    model_top_k: int = 20
    model_min_p: float = 0.0
    model_presence_penalty: float = 0.0
    model_repetition_penalty: float = 1.0
    model_request_timeout_seconds: float = 45.0
    low_latency_model_request_timeout_seconds: float = 25.0
    model_max_output_tokens: int = 800
    low_latency_max_output_tokens: int = 512
    planner_request_timeout_seconds: float = 12.0
    low_latency_planner_request_timeout_seconds: float = 6.0
    planner_max_output_tokens: int = 96
    low_latency_planner_max_output_tokens: int = 64
    reformulation_request_timeout_seconds: float = 12.0
    low_latency_reformulation_request_timeout_seconds: float = 6.0
    reformulation_max_output_tokens: int = 96
    low_latency_reformulation_max_output_tokens: int = 64
    synthesis_request_timeout_seconds: float = 120.0
    low_latency_synthesis_request_timeout_seconds: float = 70.0
    synthesis_max_output_tokens: int = 700
    low_latency_synthesis_max_output_tokens: int = 400
    model_stop_sequences: str = ""
    fail_fast_on_startup: bool = True

    allowed_source_domains: str = (
        "support.atlassian.com,confluence.atlassian.com,docs.langchain.com,"
        "python.langchain.com,langchain.com,atlassian.com,www.atlassian.com,www.langchain.com,"
        "arxiv.org,cdn.openai.com,openai.com,www.openai.com,anthropic.com,www.anthropic.com,"
        "ai.google.dev,developers.google.com,research.google,paperswithcode.com,langchain-ai.github.io,github.io"
    )
    public_sources_only: bool = True

    model_config = SettingsConfigDict(
        # Resolve .env relative to the repo root (two levels above this file),
        # regardless of the working directory uvicorn is started from.
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
    )

    @cached_property
    def allowed_domains(self) -> list[str]:
        return [d.strip().lower() for d in self.allowed_source_domains.split(",") if d.strip()]

    @property
    def stop_sequences(self) -> list[str]:
        return [s.strip() for s in self.model_stop_sequences.split(",") if s.strip()]

    @property
    def normalized_runtime_profile(self) -> str:
        profile = self.runtime_profile.strip().lower()
        return profile if profile in {"low_latency", "balanced", "high_quality"} else "balanced"

    def _runtime_knobs(self) -> dict[str, int]:
        profile = self.normalized_runtime_profile
        if profile == "low_latency":
            return {
                "planner_max_subqueries": 3,
                "retrieval_per_query_k": min(self.retrieval_per_query_k, 3),
                "retrieval_top_k": min(max(self.retrieval_top_k, 5), 6),
                "context_chunk_limit": 4,
                "context_chunk_char_limit": 420,
            }
        if profile == "high_quality":
            return {
                "planner_max_subqueries": 5,
                "retrieval_per_query_k": max(self.retrieval_per_query_k, 4),
                "retrieval_top_k": max(self.retrieval_top_k, 6),
                "context_chunk_limit": 8,
                "context_chunk_char_limit": 850,
            }
        return {
            "planner_max_subqueries": 4,
            "retrieval_per_query_k": self.retrieval_per_query_k,
            "retrieval_top_k": self.retrieval_top_k,
            "context_chunk_limit": 6,
            "context_chunk_char_limit": 650,
        }

    @property
    def planner_max_subqueries(self) -> int:
        return self._runtime_knobs()["planner_max_subqueries"]

    @property
    def effective_retrieval_per_query_k(self) -> int:
        return self._runtime_knobs()["retrieval_per_query_k"]

    @property
    def effective_retrieval_top_k(self) -> int:
        return self._runtime_knobs()["retrieval_top_k"]

    @property
    def context_chunk_limit(self) -> int:
        return self._runtime_knobs()["context_chunk_limit"]

    @property
    def context_chunk_char_limit(self) -> int:
        return self._runtime_knobs()["context_chunk_char_limit"]

    @property
    def effective_model_request_timeout_seconds(self) -> float:
        if self.normalized_runtime_profile == "low_latency":
            return self.low_latency_model_request_timeout_seconds
        return self.model_request_timeout_seconds

    @property
    def effective_model_max_output_tokens(self) -> int:
        if self.normalized_runtime_profile == "low_latency":
            return self.low_latency_max_output_tokens
        return self.model_max_output_tokens

    @property
    def effective_planner_request_timeout_seconds(self) -> float:
        if self.normalized_runtime_profile == "low_latency":
            return self.low_latency_planner_request_timeout_seconds
        return self.planner_request_timeout_seconds

    @property
    def effective_planner_max_output_tokens(self) -> int:
        if self.normalized_runtime_profile == "low_latency":
            return self.low_latency_planner_max_output_tokens
        return self.planner_max_output_tokens

    @property
    def effective_reformulation_request_timeout_seconds(self) -> float:
        if self.normalized_runtime_profile == "low_latency":
            return self.low_latency_reformulation_request_timeout_seconds
        return self.reformulation_request_timeout_seconds

    @property
    def effective_reformulation_max_output_tokens(self) -> int:
        if self.normalized_runtime_profile == "low_latency":
            return self.low_latency_reformulation_max_output_tokens
        return self.reformulation_max_output_tokens

    @property
    def effective_synthesis_request_timeout_seconds(self) -> float:
        if self.normalized_runtime_profile == "low_latency":
            return self.low_latency_synthesis_request_timeout_seconds
        return self.synthesis_request_timeout_seconds

    @property
    def effective_synthesis_max_output_tokens(self) -> int:
        if self.normalized_runtime_profile == "low_latency":
            return self.low_latency_synthesis_max_output_tokens
        return self.synthesis_max_output_tokens

    @property
    def resolved_chroma_persist_directory(self) -> str:
        configured = Path(self.chroma_persist_directory)
        if configured.is_absolute():
            return str(configured)
        repo_root = Path(__file__).resolve().parents[2]
        return str((repo_root / configured).resolve())


settings = Settings()
