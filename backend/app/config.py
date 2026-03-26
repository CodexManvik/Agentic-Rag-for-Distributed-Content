from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3.5:4b"
    ollama_embedding_model: str = "nomic-embed-text"

    chroma_collection_name: str = "knowledge_base"
    chroma_persist_directory: str = "./chroma_data"
    retrieval_top_k: int = 4
    retrieval_per_query_k: int = 4
    retrieval_bm25_k: int = 6
    retrieval_min_score: float = 0.35
    retrieval_min_chunks: int = 3
    retrieval_min_source_diversity: int = 2
    rerank_enabled: bool = True
    debug_trace_enabled: bool = False
    max_retrieval_retries: int = 1
    max_validation_retries: int = 1

    chunk_size: int = 1200
    chunk_overlap: int = 200

    allowed_source_domains: str = "python.langchain.com,docs.langchain.com,langchain.com"
    public_sources_only: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def allowed_domains(self) -> list[str]:
        return [d.strip().lower() for d in self.allowed_source_domains.split(",") if d.strip()]


settings = Settings()
