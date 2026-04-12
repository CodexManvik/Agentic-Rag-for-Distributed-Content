"""
llama.cpp inference backend implementation.

This module provides integration with llama-cpp-python for local GGUF model inference
with KV cache support and optimized performance.
"""

from typing import Any, AsyncIterator, Iterator, Optional
import asyncio
from pathlib import Path

from loguru import logger

try:
    from llama_cpp import Llama, LlamaCache
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not installed. LlamaCppBackend will not be available.")

from .base import (
    InferenceBackend,
    InferenceBackendType,
    ModelInfo,
    GenerationConfig,
)


class LlamaCppBackend(InferenceBackend):
    """
    llama.cpp inference backend using llama-cpp-python.
    
    Supports GGUF model loading with KV cache management for efficient
    context handling within sessions.
    """
    
    def __init__(self, config: dict[str, Any]):
        """
        Initialize llama.cpp backend.
        
        Args:
            config: Configuration dictionary with optional keys:
                - n_ctx: Context window size (default: 4096)
                - n_gpu_layers: Number of layers to offload to GPU (default: -1, all)
                - n_threads: CPU threads for inference (default: None, auto-detect)
                - use_mmap: Use memory mapping (default: True)
                - use_mlock: Lock model in RAM (default: False)
                - seed: Random seed (default: -1)
        """
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError(
                "llama-cpp-python is not installed. "
                "Install it with: pip install llama-cpp-python"
            )
        
        super().__init__(config)
        self._llama: Optional[Llama] = None
        self._model_info: Optional[ModelInfo] = None
        
        # Default configuration
        self.n_ctx = config.get("n_ctx", 4096)
        self.n_gpu_layers = config.get("n_gpu_layers", -1)  # -1 = offload all to GPU
        self.n_threads = config.get("n_threads")
        self.use_mmap = config.get("use_mmap", True)
        self.use_mlock = config.get("use_mlock", False)
        self.seed = config.get("seed", -1)
    
    def load_model(self, model_path: str, **kwargs: Any) -> ModelInfo:
        """
        Load a GGUF model from file.
        
        Args:
            model_path: Path to GGUF model file
            **kwargs: Additional llama.cpp parameters
            
        Returns:
            ModelInfo with model metadata
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading GGUF model from {model_path}")
        
        try:
            effective_n_ctx = kwargs.get("n_ctx", self.n_ctx)
            self._llama = Llama(
                model_path=model_path,
                n_ctx=effective_n_ctx,
                n_gpu_layers=kwargs.get("n_gpu_layers", self.n_gpu_layers),
                n_threads=kwargs.get("n_threads", self.n_threads),
                use_mmap=kwargs.get("use_mmap", self.use_mmap),
                use_mlock=kwargs.get("use_mlock", self.use_mlock),
                seed=kwargs.get("seed", self.seed),
                verbose=False,
            )
            
            # Extract model info from metadata
            model_name = Path(model_path).stem
            
            self._model_info = ModelInfo(
                model_id=model_path,
                name=model_name,
                context_length=effective_n_ctx,
                quantization=self._extract_quantization(model_name),
                backend_type=InferenceBackendType.LLAMA_CPP,
                is_loaded=True,
            )
            
            logger.info(f"Model loaded successfully: {model_name}")
            return self._model_info
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Generate text from prompt (blocking)."""
        if not self.is_loaded() or self._llama is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        gen_config = config or GenerationConfig()
        
        response = self._llama(
            prompt,
            max_tokens=gen_config.max_tokens,
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            top_k=gen_config.top_k,
            repeat_penalty=gen_config.repeat_penalty,
            stop=gen_config.stop_sequences or [],
            echo=False,
        )
        
        return response["choices"][0]["text"]
    
    def stream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> Iterator[str]:
        """Generate text with streaming (blocking iterator)."""
        if not self.is_loaded() or self._llama is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        gen_config = config or GenerationConfig()
        
        stream = self._llama(
            prompt,
            max_tokens=gen_config.max_tokens,
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            top_k=gen_config.top_k,
            repeat_penalty=gen_config.repeat_penalty,
            stop=gen_config.stop_sequences or [],
            stream=True,
            echo=False,
        )
        
        for chunk in stream:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                text = chunk["choices"][0].get("text", "")
                if text:
                    yield text
    
    async def agenerate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Generate text (async wrapper around blocking call)."""
        return await asyncio.to_thread(self.generate, prompt, config)
    
    async def astream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:
        """Generate text with streaming (async iterator)."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        error_holder: dict[str, Exception] = {}

        def _worker() -> None:
            try:
                for chunk in self.stream_generate(prompt, config):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            except Exception as exc:
                error_holder["error"] = exc
                logger.exception("llama.cpp streaming worker failed")
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        worker_future = loop.run_in_executor(None, _worker)

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

        await worker_future

        if "error" in error_holder:
            raise error_holder["error"]
    
    def embed(self, text: str) -> list[float]:
        """
        Generate embeddings (if model supports it).
        
        Note: Not all GGUF models support embeddings. This is a placeholder
        for future implementation.
        """
        if not self.is_loaded() or self._llama is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # llama-cpp-python supports embeddings via llama.embed()
        try:
            embedding = self._llama.embed(text)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            raise NotImplementedError(f"Embedding not supported by this model: {e}")
    
    def unload_model(self) -> None:
        """Unload model and free memory."""
        if self._llama is not None:
            logger.info("Unloading llama.cpp model")
            del self._llama
            self._llama = None
            self._model_info = None
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """Get current model info."""
        return self._model_info
    
    @staticmethod
    def _extract_quantization(model_name: str) -> Optional[str]:
        """Extract quantization type from model filename."""
        # Common quantization patterns: Q4_0, Q4_K_M, Q5_0, Q5_K_M, Q8_0, F16
        import re
        patterns = [
            r'Q4_0', r'Q4_K_M', r'Q4_K_S',
            r'Q5_0', r'Q5_K_M', r'Q5_K_S',
            r'Q8_0', r'F16', r'F32'
        ]
        
        for pattern in patterns:
            if re.search(pattern, model_name, re.IGNORECASE):
                return pattern
        
        return None
