"""
Unified LLM interface for LangChain compatibility.

Provides a single entry point for inference that works seamlessly with
LangChain while supporting multiple backends.
"""

from typing import Any, Optional, Iterator, AsyncIterator
from dataclasses import dataclass

from loguru import logger

from .base import (
    InferenceBackend,
    GenerationConfig,
    ModelInfo,
    InferenceBackendType
)
from .backend_factory import BackendFactory


@dataclass
class LLMConfig:
    """Configuration for the unified LLM."""
    
    model_id: str
    backend_type: Optional[InferenceBackendType] = None
    model_path: Optional[str] = None
    
    # Backend-specific options
    backend_kwargs: dict[str, Any] = None
    
    def __post_init__(self):
        if self.backend_kwargs is None:
            self.backend_kwargs = {}


class UnifiedLLM:
    """
    Unified LLM interface that wraps multiple inference backends.
    
    Provides a simple, consistent API for text generation while supporting
    llama.cpp, Ollama, and local model backends. Compatible with LangChain.
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize unified LLM.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        self.backend: Optional[InferenceBackend] = None
        
        # Create backend
        self._initialize_backend()

        resolved_backend = getattr(self.backend, "backend_type", None)
        logger.info(f"Initialized UnifiedLLM with {resolved_backend} backend")
    
    def _initialize_backend(self):
        """Initialize the inference backend."""
        factory = BackendFactory()
        
        if self.config.backend_type:
            # Explicit backend type
            self.backend = factory.create_backend(
                backend_type=self.config.backend_type,
                **self.config.backend_kwargs
            )
        else:
            # Auto-select based on model_id or model_path
            self.backend = factory.auto_select_backend(
                model_id=self.config.model_id,
                model_path=self.config.model_path,
                **self.config.backend_kwargs
            )
        
        # Load model
        self.backend.load_model(
            model_id=self.config.model_id,
            model_path=self.config.model_path
        )
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if not self.backend:
            raise RuntimeError("Backend not initialized")
        
        # Merge config with kwargs
        if config is None:
            config = GenerationConfig()
        
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return self.backend.generate(prompt=prompt, config=config)
    
    def stream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Stream text generation.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            **kwargs: Additional generation parameters
            
        Yields:
            Generated text chunks
        """
        if not self.backend:
            raise RuntimeError("Backend not initialized")
        
        # Merge config
        if config is None:
            config = GenerationConfig()
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        yield from self.backend.stream_generate(prompt=prompt, config=config)
    
    async def agenerate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> str:
        """
        Generate text asynchronously.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if not self.backend:
            raise RuntimeError("Backend not initialized")
        
        # Merge config
        if config is None:
            config = GenerationConfig()
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return await self.backend.agenerate(prompt=prompt, config=config)
    
    async def astream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream text generation asynchronously.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            **kwargs: Additional generation parameters
            
        Yields:
            Generated text chunks
        """
        if not self.backend:
            raise RuntimeError("Backend not initialized")
        
        # Merge config
        if config is None:
            config = GenerationConfig()
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        async for chunk in self.backend.astream_generate(prompt=prompt, config=config):
            yield chunk
    
    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not self.backend:
            raise RuntimeError("Backend not initialized")
        
        return self.backend.embed(text)
    
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the loaded model.
        
        Returns:
            Model information
        """
        if not self.backend:
            raise RuntimeError("Backend not initialized")
        
        return self.backend.get_model_info()
    
    def unload(self):
        """Unload the model and free resources."""
        if self.backend:
            self.backend.unload_model()
            logger.info("Unloaded model")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
    
    # LangChain compatibility methods
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Call interface for LangChain compatibility.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Generated text
        """
        return self.generate(prompt=prompt, **kwargs)
    
    @property
    def model_name(self) -> str:
        """Get model name (for LangChain)."""
        return self.config.model_id
