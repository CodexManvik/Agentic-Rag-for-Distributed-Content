"""
Adaptive configuration based on hardware capabilities.

Automatically configures optimal settings for models and backends
based on detected hardware.
"""

from typing import Optional
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ..system.hardware import detect_hardware, estimate_vram_requirement
from ..inference.base import InferenceBackendType
from ..inference.model_registry import ModelRegistry


@dataclass
class AdaptiveConfig:
    """Adaptive configuration result."""
    
    # Recommended backend
    backend_type: InferenceBackendType
    
    # Model selection
    recommended_model: str
    max_model_size_gb: float
    
    # Hardware-specific settings
    n_gpu_layers: int
    n_threads: int
    context_length: int
    batch_size: int
    
    # Memory settings
    use_mmap: bool
    use_mlock: bool
    
    # Performance tuning
    parallel_requests: int
    
    # Hardware info
    hardware_info: dict


class AdaptiveConfigGenerator:
    """
    Generate optimal configuration based on hardware.
    
    Analyzes system capabilities and recommends appropriate settings
    for models, backends, and inference parameters.
    """
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        """
        Initialize adaptive config generator.
        
        Args:
            registry: Model registry for model selection (optional)
        """
        self.registry = registry or ModelRegistry()
    
    def generate_config(
        self,
        prefer_local: bool = True,
        min_context_length: int = 2048
    ) -> AdaptiveConfig:
        """
        Generate adaptive configuration.
        
        Args:
            prefer_local: Prefer local inference over remote (Ollama)
            min_context_length: Minimum context length required
            
        Returns:
            Adaptive configuration
        """
        # Detect hardware
        hw = detect_hardware()
        
        logger.info(f"Hardware detected: {hw['gpu_count']} GPU(s), "
                   f"{hw['total_vram_gb']:.1f}GB VRAM, "
                   f"{hw['cpu_cores']} CPU cores, "
                   f"{hw['ram_gb']:.1f}GB RAM")
        
        # Determine backend
        backend_type = self._select_backend(hw, prefer_local)
        
        # Select appropriate model
        available_vram = hw['total_vram_gb'] * 0.8  # 80% safety margin
        
        model = self._select_model(
            max_vram=available_vram,
            min_context_length=min_context_length
        )
        
        # Calculate GPU layers
        n_gpu_layers = self._calculate_gpu_layers(hw, model)
        
        # Calculate thread count
        n_threads = self._calculate_threads(hw)
        
        # Determine context length
        context_length = self._determine_context_length(hw, model)
        
        # Determine batch size
        batch_size = self._determine_batch_size(hw)
        
        # Memory settings
        use_mmap = True  # Always beneficial
        use_mlock = hw['ram_gb'] > 16  # Only with sufficient RAM
        
        # Parallel requests
        parallel_requests = self._calculate_parallel_requests(hw)
        
        config = AdaptiveConfig(
            backend_type=backend_type,
            recommended_model=model['model_id'] if model else "qwen2.5-7b-instruct-q4",
            max_model_size_gb=available_vram,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            context_length=context_length,
            batch_size=batch_size,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            parallel_requests=parallel_requests,
            hardware_info=hw
        )
        
        logger.info(f"Generated adaptive config: backend={backend_type}, "
                   f"model={config.recommended_model}, "
                   f"gpu_layers={n_gpu_layers}, "
                   f"context={context_length}")
        
        return config
    
    def _select_backend(
        self,
        hw: dict,
        prefer_local: bool
    ) -> InferenceBackendType:
        """Select appropriate backend based on hardware."""
        
        # Check for GPU
        has_gpu = hw['gpu_count'] > 0 and hw['total_vram_gb'] > 4
        
        if has_gpu and prefer_local:
            # Local inference with GPU
            return InferenceBackendType.LLAMA_CPP
        elif not prefer_local:
            # Remote inference via Ollama
            return InferenceBackendType.OLLAMA
        else:
            # Fallback to CPU-based local inference
            logger.warning("No GPU detected, using CPU inference (slow)")
            return InferenceBackendType.LLAMA_CPP
    
    def _select_model(
        self,
        max_vram: float,
        min_context_length: int
    ) -> Optional[dict]:
        """Select appropriate model based on constraints."""
        
        models = self.registry.list_models(
            max_vram_gb=max_vram,
            min_context_length=min_context_length
        )
        
        if not models:
            logger.warning(f"No models found for VRAM={max_vram}GB, "
                          f"context={min_context_length}")
            return None
        
        # Sort by parameter count (larger is generally better)
        models.sort(key=lambda m: m.parameters_b, reverse=True)
        
        return models[0]
    
    def _calculate_gpu_layers(self, hw: dict, model: Optional[dict]) -> int:
        """Calculate number of GPU layers to offload."""
        
        if hw['gpu_count'] == 0:
            return 0
        
        if model is None:
            # Conservative default
            return 20
        
        # Estimate based on VRAM
        vram_available = hw['total_vram_gb'] * 0.8
        model_vram = model.get('vram_gb', 5.0)
        
        if vram_available >= model_vram:
            # Full offload
            return -1  # -1 means all layers
        elif vram_available >= model_vram * 0.5:
            # Partial offload (estimate layers)
            # Assuming typical 32-layer model
            ratio = vram_available / model_vram
            return int(32 * ratio)
        else:
            # Minimal offload
            return 5
    
    def _calculate_threads(self, hw: dict) -> int:
        """Calculate optimal thread count."""
        
        cores = hw['cpu_cores']
        
        # Leave some cores for system
        if cores <= 4:
            return cores
        elif cores <= 8:
            return cores - 1
        else:
            return cores - 2
    
    def _determine_context_length(
        self,
        hw: dict,
        model: Optional[dict]
    ) -> int:
        """Determine optimal context length."""
        
        if model:
            max_context = model.get('context_length', 4096)
        else:
            max_context = 4096
        
        # Adjust based on VRAM
        if hw['total_vram_gb'] < 8:
            # Limit context for low VRAM
            return min(2048, max_context)
        elif hw['total_vram_gb'] < 12:
            return min(4096, max_context)
        else:
            return max_context
    
    def _determine_batch_size(self, hw: dict) -> int:
        """Determine optimal batch size."""
        
        if hw['gpu_count'] == 0:
            return 1  # CPU inference, no batching
        
        # GPU batch size based on VRAM
        vram = hw['total_vram_gb']
        
        if vram < 8:
            return 1
        elif vram < 12:
            return 2
        elif vram < 16:
            return 4
        else:
            return 8
    
    def _calculate_parallel_requests(self, hw: dict) -> int:
        """Calculate max parallel requests."""
        
        cores = hw['cpu_cores']
        
        # Conservative estimate
        if cores <= 4:
            return 2
        elif cores <= 8:
            return 4
        else:
            return 8
