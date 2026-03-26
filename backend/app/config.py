from functools import cached_property

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3.5:0.8b"
    ollama_embedding_model: str = "nomic-embed-text:latest"

    chroma_collection_name: str = "knowledge_base"
    chroma_persist_directory: str = "./chroma_data"
    retrieval_top_k: int = 4
    retrieval_per_query_k: int = 4
    retrieval_bm25_k: int = 6
    hybrid_retrieval_enabled: bool = True
    bm25_cache_enabled: bool = True
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    retrieval_min_score: float = 0.35
    retrieval_min_chunks: int = 3
    retrieval_min_source_diversity: int = 2
    rerank_enabled: bool = True
    debug_trace_enabled: bool = False
    max_retrieval_retries: int = 1
    max_validation_retries: int = 1

    chunk_size: int = 1200
    chunk_overlap: int = 200

    runtime_profile: str = "balanced"
    model_temperature: float = 0.0
    model_top_p: float = 0.9
    model_top_k: int = 20
    model_min_p: float = 0.0
    model_presence_penalty: float = 0.0
    model_repetition_penalty: float = 1.0
    model_request_timeout_seconds: float = 18.0
    model_max_output_tokens: int = 180
    model_stop_sequences: str = ""
    fail_fast_on_startup: bool = True

    allowed_source_domains: str = (
        "support.atlassian.com,confluence.atlassian.com,docs.langchain.com,"
        "python.langchain.com,langchain.com,atlassian.com,www.atlassian.com,www.langchain.com"
    )
    public_sources_only: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

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
                "retrieval_top_k": min(max(self.retrieval_top_k, 4), 5),
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


settings = Settings()
