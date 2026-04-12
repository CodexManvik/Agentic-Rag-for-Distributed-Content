"""Retrieval service using LanceDB instead of ChromaDB."""

from functools import lru_cache
import os

from llama_index.core import Settings as LlamaIndexSettings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from app.config import settings
from app.services.lancedb_retrieval import get_lancedb_retriever, clear_lancedb_caches

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


def _configure_llamaindex_settings() -> None:
    """Route all LlamaIndex model calls through local Ollama."""
    LlamaIndexSettings.llm = Ollama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_base_url,
        request_timeout=settings.effective_model_request_timeout_seconds,
        temperature=settings.model_temperature,
    )
    LlamaIndexSettings.embed_model = OllamaEmbedding(
        model_name=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )


def get_retriever():
    """Return a LanceDB retriever with LlamaIndex-compatible interface."""
    _configure_llamaindex_settings()
    return get_lancedb_retriever()


def clear_retrieval_caches() -> None:
    """Clear cached clients after ingestion or config changes."""
    clear_lancedb_caches()


# Legacy compatibility functions (kept for backward compatibility but now use LanceDB)
def get_vector_index():
    """Legacy function - now redirects to LanceDB retriever."""
    return get_retriever()


def _get_vector_store():
    """Legacy function - no longer used but kept for compatibility."""
    pass


def _get_index():
    """Legacy function - no longer used but kept for compatibility."""
    pass
