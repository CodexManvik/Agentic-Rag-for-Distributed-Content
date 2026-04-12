"""
Ollama inference backend implementation.

This module wraps the existing Ollama integration to conform to the
unified InferenceBackend interface while maintaining backward compatibility.
"""

from typing import Any, AsyncIterator, Iterator, Optional
import asyncio

from loguru import logger

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("langchain-ollama not installed. OllamaBackend will not be available.")

from .base import (
    InferenceBackend,
    InferenceBackendType,
    ModelInfo,
    GenerationConfig,
)


class OllamaBackend(InferenceBackend):
    """
    Ollama inference backend.
    
    Wraps langchain-ollama to provide unified interface while maintaining
    backward compatibility with existing code.
    """
    
    def __init__(self, config: dict[str, Any]):
        """
        Initialize Ollama backend.
        
        Args:
            config: Configuration dictionary with optional keys:
                - base_url: Ollama server URL (default: http://localhost:11434)
                - timeout: Request timeout in seconds (default: 120)
        """
        if not OLLAMA_AVAILABLE:
            raise RuntimeError(
                "langchain-ollama is not installed. "
                "Install it with: pip install langchain-ollama"
            )
        
        super().__init__(config)
        self._chat_model: Optional[ChatOllama] = None
        
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.timeout = config.get("timeout", 120)
        self._current_model_name: Optional[str] = None
    
    def load_model(self, model_path: str, **kwargs: Any) -> ModelInfo:
        """
        Load an Ollama model.
        
        Args:
            model_path: Ollama model name (e.g., "llama2", "mistral", "qwen2.5:7b")
            **kwargs: Additional Ollama parameters
            
        Returns:
            ModelInfo with model metadata
        """
        logger.info(f"Initializing Ollama model: {model_path}")
        
        try:
            self._chat_model = ChatOllama(
                model=model_path,
                base_url=kwargs.get("base_url", self.base_url),
                timeout=kwargs.get("timeout", self.timeout),
            )
            
            self._current_model_name = model_path
            
            # Ollama models vary in context length; default to 4096
            # Could be enhanced by querying Ollama API for model details
            context_length = kwargs.get("num_ctx", 4096)
            
            self._model_info = ModelInfo(
                model_id=model_path,
                name=model_path,
                context_length=context_length,
                backend_type=InferenceBackendType.OLLAMA,
                is_loaded=True,
            )
            
            logger.info(f"Ollama model initialized: {model_path}")
            return self._model_info
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama model: {e}")
            raise RuntimeError(f"Ollama model initialization failed: {e}")
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Generate text from prompt (blocking)."""
        if not self.is_loaded() or self._chat_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        gen_config = config or GenerationConfig()
        
        options = {
            "temperature": gen_config.temperature,
            "top_p": gen_config.top_p,
            "top_k": gen_config.top_k,
            "num_predict": gen_config.max_tokens,
            "repeat_penalty": gen_config.repeat_penalty,
            "stop": gen_config.stop_sequences or None,
        }

        response = self._chat_model.invoke(prompt, **options)
        return response.content if hasattr(response, 'content') else str(response)
    
    def stream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> Iterator[str]:
        """Generate text with streaming (blocking iterator)."""
        if not self.is_loaded() or self._chat_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        gen_config = config or GenerationConfig()
        
        options = {
            "temperature": gen_config.temperature,
            "top_p": gen_config.top_p,
            "top_k": gen_config.top_k,
            "num_predict": gen_config.max_tokens,
            "repeat_penalty": gen_config.repeat_penalty,
            "stop": gen_config.stop_sequences or None,
        }

        for chunk in self._chat_model.stream(prompt, **options):
            if hasattr(chunk, 'content'):
                yield chunk.content
            else:
                yield str(chunk)
    
    async def agenerate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Generate text (async)."""
        if not self.is_loaded() or self._chat_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        gen_config = config or GenerationConfig()
        
        options = {
            "temperature": gen_config.temperature,
            "top_p": gen_config.top_p,
            "top_k": gen_config.top_k,
            "num_predict": gen_config.max_tokens,
            "repeat_penalty": gen_config.repeat_penalty,
            "stop": gen_config.stop_sequences or None,
        }

        response = await self._chat_model.ainvoke(prompt, **options)
        return response.content if hasattr(response, 'content') else str(response)
    
    async def astream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:
        """Generate text with streaming (async iterator)."""
        if not self.is_loaded() or self._chat_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        gen_config = config or GenerationConfig()
        
        options = {
            "temperature": gen_config.temperature,
            "top_p": gen_config.top_p,
            "top_k": gen_config.top_k,
            "num_predict": gen_config.max_tokens,
            "repeat_penalty": gen_config.repeat_penalty,
            "stop": gen_config.stop_sequences or None,
        }

        async for chunk in self._chat_model.astream(prompt, **options):
            if hasattr(chunk, 'content'):
                yield chunk.content
            else:
                yield str(chunk)
    
    def embed(self, text: str) -> list[float]:
        """
        Generate embeddings using Ollama.
        
        Note: Requires separate embedding model to be loaded in Ollama.
        This is a placeholder for future implementation.
        """
        raise NotImplementedError(
            "Ollama embeddings require separate embedding model. "
            "Use OllamaEmbeddings from langchain-ollama for embedding support."
        )
    
    def unload_model(self) -> None:
        """Unload model."""
        if self._chat_model is not None:
            logger.info(f"Unloading Ollama model: {self._current_model_name}")
            self._chat_model = None
            self._current_model_name = None
            self._model_info = None
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """Get current model info."""
        return self._model_info
