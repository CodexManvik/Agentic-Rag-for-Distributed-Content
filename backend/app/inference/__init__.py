"""
Unified inference layer for multiple backends.

This package provides a unified interface for working with different
LLM inference backends (llama.cpp, Ollama, local models).
"""

from .base import (
    InferenceBackend,
    InferenceBackendType,
    ModelInfo,
    GenerationConfig,
)
from .backend_factory import BackendFactory
from .model_registry import ModelRegistry, ModelMetadata
from .unified_llm import UnifiedLLM, LLMConfig
from .hf_downloader import HuggingFaceDownloader

__all__ = [
    "InferenceBackend",
    "InferenceBackendType",
    "ModelInfo",
    "GenerationConfig",
    "BackendFactory",
    "ModelRegistry",
    "ModelMetadata",
    "UnifiedLLM",
    "LLMConfig",
    "HuggingFaceDownloader",
]
