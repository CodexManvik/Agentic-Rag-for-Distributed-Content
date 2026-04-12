"""
Local model inference backend.

This module provides support for loading local model files directly,
useful for custom models or models not served by Ollama.
"""

from typing import Any, AsyncIterator, Iterator, Optional
from pathlib import Path

from loguru import logger

from .base import (
    InferenceBackend,
    InferenceBackendType,
    ModelInfo,
    GenerationConfig,
)

# Try to use llama.cpp as the underlying engine for local models
try:
    from .llama_cpp_backend import LlamaCppBackend
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


class LocalModelBackend(InferenceBackend):
    """
    Local model file inference backend.
    
    Loads models directly from user-specified file paths. Currently uses
    llama.cpp as the underlying engine for GGUF files.
    
    Future: Could be extended to support other formats (safetensors, etc.)
    """
    
    def __init__(self, config: dict[str, Any]):
        """
        Initialize local model backend.
        
        Args:
            config: Configuration dictionary (same as LlamaCppBackend for now)
        """
        super().__init__(config)
        
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError(
                "llama-cpp-python is required for local model loading. "
                "Install it with: pip install llama-cpp-python"
            )
        
        # Use llama.cpp as underlying engine
        self._engine = LlamaCppBackend(config)
    
    def load_model(self, model_path: str, **kwargs: Any) -> ModelInfo:
        """
        Load a local model file.
        
        Args:
            model_path: Absolute or relative path to model file
            **kwargs: Engine-specific parameters
            
        Returns:
            ModelInfo with model metadata
        """
        # Resolve and validate path
        path = Path(model_path).resolve()
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        if not path.is_file():
            raise ValueError(f"Path must point to a file, not a directory: {path}")
        
        # Check file extension
        supported_extensions = ['.gguf', '.bin']
        if path.suffix.lower() not in supported_extensions:
            logger.warning(
                f"Model file extension '{path.suffix}' may not be supported. "
                f"Supported: {supported_extensions}"
            )
        
        logger.info(f"Loading local model from: {path}")
        
        # Use llama.cpp backend to load
        model_info = self._engine.load_model(str(path), **kwargs)
        
        # Override backend type to indicate this is a user-loaded local model
        model_info.backend_type = InferenceBackendType.LOCAL
        
        self._model_info = model_info
        return model_info
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Generate text from prompt (blocking)."""
        return self._engine.generate(prompt, config)
    
    def stream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> Iterator[str]:
        """Generate text with streaming (blocking iterator)."""
        return self._engine.stream_generate(prompt, config)
    
    async def agenerate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Generate text (async)."""
        return await self._engine.agenerate(prompt, config)
    
    async def astream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:
        """Generate text with streaming (async iterator)."""
        async for chunk in self._engine.astream_generate(prompt, config):
            yield chunk
    
    def embed(self, text: str) -> list[float]:
        """Generate embeddings for input text."""
        return self._engine.embed(text)
    
    def unload_model(self) -> None:
        """Unload model and free memory."""
        self._engine.unload_model()
        self._model_info = None
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """Get current model info."""
        return self._model_info
