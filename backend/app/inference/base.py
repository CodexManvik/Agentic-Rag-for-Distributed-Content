"""
Abstract base class for inference backends.

This module defines the unified interface that all inference backends
(llama.cpp, Ollama, local models) must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterator, Optional
from dataclasses import dataclass
from enum import Enum


class InferenceBackendType(str, Enum):
    """Supported inference backend types."""
    LLAMA_CPP = "llama_cpp"
    OLLAMA = "ollama"
    LOCAL = "local"


@dataclass
class ModelInfo:
    """Metadata about a loaded model."""
    model_id: str
    name: str
    context_length: int
    parameters_billion: Optional[float] = None
    quantization: Optional[str] = None
    vram_usage_gb: Optional[float] = None
    backend_type: InferenceBackendType = InferenceBackendType.LLAMA_CPP
    is_loaded: bool = False


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    repeat_penalty: float = 1.1
    stop_sequences: Optional[list[str]] = None
    stream: bool = False


class InferenceBackend(ABC):
    """
    Abstract base class for inference backends.
    
    All concrete backends (llama.cpp, Ollama, local) must implement this interface
    to ensure consistent behavior across the application.
    """
    
    def __init__(self, config: dict[str, Any]):
        """
        Initialize the backend with configuration.
        
        Args:
            config: Backend-specific configuration dictionary
        """
        self.config = config
        self._model_info: Optional[ModelInfo] = None
    
    @abstractmethod
    def load_model(self, model_path: str, **kwargs: Any) -> ModelInfo:
        """
        Load a model from the specified path.
        
        Args:
            model_path: Path to the model file or model identifier
            **kwargs: Backend-specific loading parameters
            
        Returns:
            ModelInfo object with model metadata
            
        Raises:
            RuntimeError: If model loading fails
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate text from a prompt (blocking).
        
        Args:
            prompt: Input text prompt
            config: Generation configuration
            
        Returns:
            Generated text string
            
        Raises:
            RuntimeError: If generation fails or model not loaded
        """
        pass
    
    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> Iterator[str]:
        """
        Generate text from a prompt with streaming (blocking iterator).
        
        Args:
            prompt: Input text prompt
            config: Generation configuration
            
        Yields:
            Text chunks as they are generated
            
        Raises:
            RuntimeError: If generation fails or model not loaded
        """
        pass
    
    @abstractmethod
    async def agenerate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate text from a prompt (async).
        
        Args:
            prompt: Input text prompt
            config: Generation configuration
            
        Returns:
            Generated text string
            
        Raises:
            RuntimeError: If generation fails or model not loaded
        """
        pass
    
    @abstractmethod
    async def astream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:
        """
        Generate text from a prompt with streaming (async iterator).
        
        Args:
            prompt: Input text prompt
            config: Generation configuration
            
        Yields:
            Text chunks as they are generated
            
        Raises:
            RuntimeError: If generation fails or model not loaded
        """
        pass
    
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Generate embeddings for input text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
            
        Raises:
            RuntimeError: If embedding fails or model not loaded
            NotImplementedError: If backend doesn't support embeddings
        """
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """
        Unload the current model from memory.
        
        Raises:
            RuntimeError: If unloading fails
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Optional[ModelInfo]:
        """
        Get information about the currently loaded model.
        
        Returns:
            ModelInfo if model is loaded, None otherwise
        """
        pass
    
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model_info is not None and self._model_info.is_loaded
