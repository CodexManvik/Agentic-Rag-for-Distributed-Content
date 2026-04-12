"""
Hardware detection module for adaptive configuration.

Detects GPU, CPU, and memory capabilities to optimize model selection
and inference parameters.
"""

from dataclasses import dataclass
from typing import Optional
import platform

from loguru import logger

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logger.warning("GPUtil not installed. GPU detection will be limited.")


@dataclass
class HardwareInfo:
    """Hardware capabilities information."""
    # GPU information
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    gpu_count: int = 0
    
    # CPU information
    cpu_cores: int = 0
    cpu_threads: int = 0
    
    # Memory information
    ram_gb: Optional[float] = None
    
    # Platform
    os_name: str = ""
    architecture: str = ""
    
    def __str__(self) -> str:
        """Human-readable hardware summary."""
        parts = [f"OS: {self.os_name} ({self.architecture})"]
        
        if self.gpu_available and self.gpu_name:
            if self.gpu_vram_gb is not None:
                parts.append(f"GPU: {self.gpu_name} ({self.gpu_vram_gb:.1f}GB VRAM)")
            else:
                parts.append(f"GPU: {self.gpu_name}")
        else:
            parts.append("GPU: None (CPU-only mode)")
        
        parts.append(f"CPU: {self.cpu_cores} cores / {self.cpu_threads} threads")
        
        if self.ram_gb:
            parts.append(f"RAM: {self.ram_gb:.1f}GB")
        
        return " | ".join(parts)


def detect_hardware() -> HardwareInfo:
    """
    Detect hardware capabilities.
    
    Returns:
        HardwareInfo object with detected capabilities
    """
    logger.info("Detecting hardware capabilities...")
    
    # Platform information
    os_name = platform.system()
    architecture = platform.machine()
    
    # CPU information
    try:
        import os
        cpu_cores = os.cpu_count() or 0
        # Assuming hyperthreading: threads = cores * 2 (rough estimate)
        cpu_threads = cpu_cores
    except Exception as e:
        logger.warning(f"Failed to detect CPU info: {e}")
        cpu_cores = cpu_threads = 0
    
    # RAM information
    ram_gb: Optional[float] = None
    try:
        import psutil
        ram_bytes = psutil.virtual_memory().total
        ram_gb = ram_bytes / (1024 ** 3)  # Convert to GB
    except ImportError:
        logger.warning("psutil not installed. RAM detection unavailable.")
    except Exception as e:
        logger.warning(f"Failed to detect RAM: {e}")
    
    # GPU information
    gpu_available = False
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    gpu_count = 0
    
    if GPUTIL_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_available = True
                gpu_count = len(gpus)
                
                # Use the first GPU for primary info
                primary_gpu = gpus[0]
                gpu_name = primary_gpu.name
                gpu_vram_gb = primary_gpu.memoryTotal / 1024  # Convert MB to GB
                
                logger.info(f"Detected {gpu_count} GPU(s): {gpu_name} ({gpu_vram_gb:.1f}GB)")
            else:
                logger.info("No GPUs detected")
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
    else:
        # Try alternative methods for NVIDIA GPUs
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                gpu_count = len(lines)
                
                if lines:
                    gpu_available = True
                    # Parse first GPU info
                    parts = [segment.strip() for segment in lines[0].split(',')]
                    if parts:
                        gpu_name = parts[0]
                    if len(parts) > 1:
                        vram_str = parts[1].replace(' MiB', '').replace('MiB', '').strip()
                        try:
                            gpu_vram_gb = float(vram_str) / 1024
                        except (ValueError, TypeError):
                            gpu_vram_gb = None
                    
                    logger.info(f"Detected {gpu_count} NVIDIA GPU(s) via nvidia-smi")
        except Exception as e:
            logger.debug(f"nvidia-smi detection failed: {e}")
    
    hardware = HardwareInfo(
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        gpu_count=gpu_count,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        ram_gb=ram_gb,
        os_name=os_name,
        architecture=architecture,
    )
    
    logger.info(f"Hardware detected: {hardware}")
    
    return hardware


def estimate_vram_requirement(
    params_billion: float,
    quantization: str = "Q4_K_M",
    context_length: int = 4096
) -> float:
    """
    Estimate VRAM requirement for a model.
    
    Args:
        params_billion: Model size in billions of parameters
        quantization: Quantization type (Q4_K_M, Q5_K_M, Q8_0, F16, etc.)
        context_length: Context window size
        
    Returns:
        Estimated VRAM in GB
    """
    # Bits per parameter for different quantizations
    bits_per_param = {
        "Q4_0": 4.5,
        "Q4_K_M": 4.85,
        "Q4_K_S": 4.5,
        "Q5_0": 5.5,
        "Q5_K_M": 5.5,
        "Q5_K_S": 5.5,
        "Q8_0": 8.5,
        "F16": 16,
        "F32": 32,
    }
    
    bits = bits_per_param.get(quantization, 4.85)  # Default to Q4_K_M
    
    # Model weights in GB
    model_gb = (params_billion * 1e9 * bits) / (8 * 1024 ** 3)
    
    # Context buffer (rough estimate: 2 bytes per token per layer)
    # Assuming 32 layers for mid-size models
    context_gb = (context_length * 32 * 2) / (1024 ** 3)
    
    # Add overhead for computation and buffers (20%)
    total_gb = (model_gb + context_gb) * 1.2
    
    return round(total_gb, 2)
