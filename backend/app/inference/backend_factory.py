"""
Backend factory for creating inference backends.

Provides centralized backend instantiation with configuration management.
"""

from typing import Any, Optional
from enum import Enum

from loguru import logger

from .base import InferenceBackend, InferenceBackendType
from .llama_cpp_backend import LlamaCppBackend
from .ollama_backend import OllamaBackend
from .local_backend import LocalModelBackend


class BackendFactory:
    """
    Factory for creating inference backends.
    
    Handles backend instantiation and provides a unified interface
    for selecting and configuring different inference engines.
    """
    
    _backends: dict[InferenceBackendType, type[InferenceBackend]] = {
        InferenceBackendType.LLAMA_CPP: LlamaCppBackend,
        InferenceBackendType.OLLAMA: OllamaBackend,
        InferenceBackendType.LOCAL: LocalModelBackend,
    }
    
    @classmethod
    def create_backend(
        cls,
        backend_type: InferenceBackendType | str,
        config: Optional[dict[str, Any]] = None
    ) -> InferenceBackend:
        """
        Create an inference backend instance.
        
        Args:
            backend_type: Type of backend to create (enum or string)
            config: Backend-specific configuration dictionary
            
        Returns:
            Instantiated InferenceBackend
            
        Raises:
            ValueError: If backend type is not supported
        """
        # Convert string to enum if needed
        if isinstance(backend_type, str):
            try:
                backend_type = InferenceBackendType(backend_type)
            except ValueError:
                raise ValueError(
                    f"Invalid backend type: {backend_type}. "
                    f"Supported types: {[t.value for t in InferenceBackendType]}"
                )
        
        backend_class = cls._backends.get(backend_type)
        
        if backend_class is None:
            raise ValueError(f"Backend type not registered: {backend_type}")
        
        config = config or {}
        
        logger.info(f"Creating {backend_type.value} backend")
        
        try:
            return backend_class(config)
        except Exception as e:
            logger.error(f"Failed to create {backend_type.value} backend: {e}")
            raise
    
    @classmethod
    def register_backend(
        cls,
        backend_type: InferenceBackendType,
        backend_class: type[InferenceBackend]
    ) -> None:
        """
        Register a custom backend type.
        
        Allows extending the factory with custom backend implementations.
        
        Args:
            backend_type: Enum value for the backend type
            backend_class: Backend class (must inherit from InferenceBackend)
        """
        if not issubclass(backend_class, InferenceBackend):
            raise TypeError(
                f"Backend class must inherit from InferenceBackend, "
                f"got {backend_class}"
            )
        
        cls._backends[backend_type] = backend_class
        logger.info(f"Registered custom backend: {backend_type.value}")
    
    @classmethod
    def get_available_backends(cls) -> list[str]:
        """Get list of available backend types."""
        return [backend_type.value for backend_type in cls._backends.keys()]
    
    @classmethod
    def auto_select_backend(
        cls,
        model_path: str,
    ) -> InferenceBackendType:
        """
        Automatically select the best backend for a model.
        
        Args:
            model_path: Model file path or identifier
            
        Returns:
            Recommended backend type
        """
        from pathlib import Path
        
        # Check if it's a file path
        is_file = Path(model_path).exists() and Path(model_path).is_file()
        
        if is_file:
            # Local file - use llama.cpp or local backend
            if model_path.endswith('.gguf'):
                return InferenceBackendType.LLAMA_CPP
            else:
                return InferenceBackendType.LOCAL
        else:
            # Model identifier - assume Ollama
            return InferenceBackendType.OLLAMA
