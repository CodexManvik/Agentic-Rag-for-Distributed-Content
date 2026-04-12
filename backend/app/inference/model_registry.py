"""
Model registry for tracking available models and their metadata.

Provides a centralized database of models with hardware requirements,
capabilities, and download information.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import json

from loguru import logger


@dataclass
class ModelMetadata:
    """Metadata for a model."""
    model_id: str
    name: str
    description: str
    
    # Size and requirements
    parameters_billion: float
    context_length: int
    quantization: str
    vram_requirement_gb: float
    
    # Source information
    huggingface_repo: Optional[str] = None
    filename: Optional[str] = None
    
    # Capabilities
    supports_embeddings: bool = False
    supports_streaming: bool = True
    supports_function_calling: bool = False
    
    # Performance hints
    recommended_batch_size: int = 1
    recommended_n_ctx: int = 4096
    
    # Local status
    local_path: Optional[str] = None
    is_downloaded: bool = False
    
    # Tags for filtering
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "description": self.description,
            "parameters_billion": self.parameters_billion,
            "context_length": self.context_length,
            "quantization": self.quantization,
            "vram_requirement_gb": self.vram_requirement_gb,
            "huggingface_repo": self.huggingface_repo,
            "filename": self.filename,
            "supports_embeddings": self.supports_embeddings,
            "supports_streaming": self.supports_streaming,
            "supports_function_calling": self.supports_function_calling,
            "recommended_batch_size": self.recommended_batch_size,
            "recommended_n_ctx": self.recommended_n_ctx,
            "local_path": self.local_path,
            "is_downloaded": self.is_downloaded,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ModelMetadata":
        """Create from dictionary."""
        return cls(**data)


class ModelRegistry:
    """
    Registry of available models with metadata.
    
    Tracks models, their requirements, and download status.
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to JSON file for persistent storage
        """
        self.registry_path = registry_path
        self._models: dict[str, ModelMetadata] = {}
        
        # Load default models
        self._register_default_models()
        
        # Load from file if exists
        if registry_path and registry_path.exists():
            self.load()
    
    def register_model(self, metadata: ModelMetadata) -> None:
        """Register a model in the registry."""
        self._models[metadata.model_id] = metadata
        logger.debug(f"Registered model: {metadata.model_id}")
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID."""
        return self._models.get(model_id)
    
    def list_models(
        self,
        max_vram_gb: Optional[float] = None,
        tags: Optional[list[str]] = None,
        downloaded_only: bool = False
    ) -> list[ModelMetadata]:
        """
        List models matching criteria.
        
        Args:
            max_vram_gb: Maximum VRAM requirement
            tags: Filter by tags (any match)
            downloaded_only: Only show downloaded models
            
        Returns:
            List of matching models
        """
        results = []
        
        for model in self._models.values():
            # VRAM filter
            if max_vram_gb and model.vram_requirement_gb > max_vram_gb:
                continue
            
            # Downloaded filter
            if downloaded_only and not model.is_downloaded:
                continue
            
            # Tags filter
            if tags:
                if not any(tag in model.tags for tag in tags):
                    continue
            
            results.append(model)
        
        return results
    
    def mark_downloaded(self, model_id: str, local_path: str) -> None:
        """Mark a model as downloaded."""
        model = self.get_model(model_id)
        if model:
            model.is_downloaded = True
            model.local_path = local_path
            logger.info(f"Marked model as downloaded: {model_id} -> {local_path}")
    
    def save(self) -> None:
        """Save registry to file."""
        if not self.registry_path:
            logger.warning("No registry path set, cannot save")
            return
        
        data = {
            model_id: model.to_dict()
            for model_id, model in self._models.items()
        }
        
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved model registry to {self.registry_path}")
    
    def load(self) -> None:
        """Load registry from file."""
        if not self.registry_path or not self.registry_path.exists():
            return
        
        with open(self.registry_path, 'r') as f:
            data = json.load(f)
        
        for model_id, model_data in data.items():
            try:
                metadata = ModelMetadata.from_dict(model_data)
                self._models[model_id] = metadata
            except Exception as e:
                logger.warning(f"Failed to load model {model_id}: {e}")
        
        logger.info(f"Loaded {len(self._models)} models from registry")
    
    def _register_default_models(self) -> None:
        """Register default recommended models."""
        
        # Qwen2.5 7B - Good balance of performance and size
        self.register_model(ModelMetadata(
            model_id="qwen2.5-7b-q4",
            name="Qwen 2.5 7B (Q4_K_M)",
            description="Balanced 7B model with strong reasoning, good for general RAG",
            parameters_billion=7.0,
            context_length=32768,
            quantization="Q4_K_M",
            vram_requirement_gb=5.2,
            huggingface_repo="Qwen/Qwen2.5-7B-Instruct-GGUF",
            filename="qwen2.5-7b-instruct-q4_k_m.gguf",
            supports_streaming=True,
            recommended_n_ctx=8192,
            tags=["general", "reasoning", "7b"],
        ))
        
        # Mistral 7B - Popular and efficient
        self.register_model(ModelMetadata(
            model_id="mistral-7b-q4",
            name="Mistral 7B v0.3 (Q4_K_M)",
            description="Efficient 7B model with good instruction following",
            parameters_billion=7.0,
            context_length=32768,
            quantization="Q4_K_M",
            vram_requirement_gb=5.0,
            huggingface_repo="mistralai/Mistral-7B-Instruct-v0.3-GGUF",
            filename="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
            supports_streaming=True,
            recommended_n_ctx=8192,
            tags=["general", "efficient", "7b"],
        ))
        
        # Phi-3 Mini - Very small, good for low-end hardware
        self.register_model(ModelMetadata(
            model_id="phi3-mini-q4",
            name="Phi-3 Mini (Q4_K_M)",
            description="Compact 3.8B model for resource-constrained environments",
            parameters_billion=3.8,
            context_length=4096,
            quantization="Q4_K_M",
            vram_requirement_gb=3.0,
            huggingface_repo="microsoft/Phi-3-mini-4k-instruct-gguf",
            filename="Phi-3-mini-4k-instruct-q4.gguf",
            supports_streaming=True,
            recommended_n_ctx=4096,
            tags=["small", "efficient", "3b"],
        ))
        
        logger.debug("Registered 3 default models")
