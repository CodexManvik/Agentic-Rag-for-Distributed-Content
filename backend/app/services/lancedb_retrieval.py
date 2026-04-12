"""LanceDB-based retrieval service compatible with existing LlamaIndex interface."""

from functools import lru_cache
from typing import Any, List
from dataclasses import dataclass

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover - fallback for older environments
    from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core import Settings as LlamaIndexSettings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import MetadataMode
from loguru import logger

from app.config import settings
from app.vectorstore.lancedb_store import LanceDBVectorStore


@dataclass
class LanceDBNode:
    """LanceDB node wrapper compatible with LlamaIndex TextNode."""
    id_: str
    text: str
    metadata: dict
    
    def get_content(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        """Get node content, compatible with LlamaIndex interface."""
        return self.text


@dataclass 
class LanceDBNodeWithScore:
    """LanceDB node with score, compatible with LlamaIndex NodeWithScore."""
    node: LanceDBNode
    score: float


class LanceDBRetriever:
    """LanceDB retriever compatible with LlamaIndex retriever interface."""
    
    def __init__(self, lancedb_store: LanceDBVectorStore, similarity_top_k: int = 4):
        self.lancedb_store = lancedb_store
        self.similarity_top_k = similarity_top_k
        
    def retrieve(self, query: str) -> List[LanceDBNodeWithScore]:
        """Retrieve documents using LanceDB, returning LlamaIndex-compatible format."""
        try:
            # Use hackathon_demo knowledge base from our ingestion
            results = self.lancedb_store.similarity_search(
                query=query,
                k=self.similarity_top_k,
                knowledge_base="hackathon_demo"
            )
            
            nodes_with_score = []
            for result in results:
                # Create LanceDB node
                metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
                node = LanceDBNode(
                    id_=str(result.get("id", "")),
                    text=str(result.get("text", "")),
                    metadata=metadata,
                )
                
                # LanceDB returns _distance (lower is better), convert to score (higher is better)
                distance = result.get("score", 1.0)
                score = 1.0 / (1.0 + float(distance)) if distance is not None else 0.5
                
                nodes_with_score.append(LanceDBNodeWithScore(node=node, score=score))
                
            logger.info(f"LanceDB retrieved {len(nodes_with_score)} chunks for query: {query[:50]}...")
            return nodes_with_score
            
        except Exception as e:
            logger.error(f"LanceDB retrieval failed: {e}")
            return []


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


@lru_cache(maxsize=1)
def _get_lancedb_store() -> LanceDBVectorStore:
    """Get configured LanceDB vector store."""
    # Configure LlamaIndex runtime models used elsewhere in the graph.
    _configure_llamaindex_settings()

    # IMPORTANT: LanceDB vectors were ingested with a local HuggingFace model (384-d).
    # Use the same model for query embeddings to avoid dimension mismatch errors.
    lancedb_embedding = HuggingFaceEmbeddings(model_name=settings.lancedb_embedding_model)

    store = LanceDBVectorStore(
        db_path="./lancedb_data",  # Default LanceDB path
        embedding_function=lancedb_embedding,
    )
    
    logger.info("Initialized LanceDB vector store")
    return store


@lru_cache(maxsize=1)
def get_lancedb_retriever():
    """Return a LanceDB retriever with LlamaIndex-compatible interface."""
    store = _get_lancedb_store()
    similarity_top_k = max(1, int(settings.effective_retrieval_top_k))
    
    retriever = LanceDBRetriever(
        lancedb_store=store,
        similarity_top_k=similarity_top_k
    )
    
    logger.info(f"Created LanceDB retriever with top_k={similarity_top_k}")
    return retriever


def clear_lancedb_caches() -> None:
    """Clear cached clients after ingestion or config changes."""
    _get_lancedb_store.cache_clear()
    get_lancedb_retriever.cache_clear()
    logger.info("Cleared LanceDB retrieval caches")
