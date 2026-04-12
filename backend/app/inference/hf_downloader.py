"""
HuggingFace model downloader with progress tracking.

Downloads GGUF models from HuggingFace Hub with resume support
and progress callbacks.
"""

from pathlib import Path
from typing import Optional, Callable
import os

from loguru import logger

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.utils import HfHubHTTPError
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("huggingface-hub not installed. Model downloading unavailable.")


class HuggingFaceDownloader:
    """
    Download models from HuggingFace Hub.
    
    Handles single file downloads and repository snapshots with
    resume support and progress tracking.
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        token: Optional[str] = None
    ):
        """
        Initialize HuggingFace downloader.
        
        Args:
            cache_dir: Directory for cached downloads (default: ~/.cache/huggingface)
            token: HuggingFace API token for private repos (optional)
        """
        if not HF_HUB_AVAILABLE:
            raise RuntimeError(
                "huggingface-hub is not installed. "
                "Install it with: pip install huggingface-hub"
            )
        
        self.cache_dir = cache_dir
        self.token = token or os.getenv("HF_TOKEN")
    
    def download_model(
        self,
        repo_id: str,
        filename: str,
        local_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Path:
        """
        Download a single model file from HuggingFace.
        
        Args:
            repo_id: Repository ID (e.g., "Qwen/Qwen2.5-7B-Instruct-GGUF")
            filename: File to download (e.g., "qwen2.5-7b-instruct-q4_k_m.gguf")
            local_dir: Target directory (optional, uses cache if not specified)
            progress_callback: Optional callback(downloaded_bytes, total_bytes)
            
        Returns:
            Path to downloaded file
            
        Raises:
            RuntimeError: If download fails
        """
        logger.info(f"Downloading {filename} from {repo_id}")
        
        try:
            # Download to cache or specific directory
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                local_dir=str(local_dir) if local_dir else None,
                token=self.token,
                resume_download=True,  # Resume interrupted downloads
            )
            
            downloaded_path = Path(downloaded_path)
            logger.info(f"Downloaded to: {downloaded_path}")
            
            return downloaded_path
            
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                raise RuntimeError(
                    f"File not found: {filename} in {repo_id}. "
                    "Check that the repository and filename are correct."
                )
            elif e.response.status_code == 401:
                raise RuntimeError(
                    f"Access denied to {repo_id}. "
                    "You may need to provide a HuggingFace token for private repos."
                )
            else:
                raise RuntimeError(f"Download failed: {e}")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise RuntimeError(f"Download error: {e}")
    
    def download_repository(
        self,
        repo_id: str,
        local_dir: Path,
        allow_patterns: Optional[list[str]] = None,
        ignore_patterns: Optional[list[str]] = None
    ) -> Path:
        """
        Download entire repository or filtered files.
        
        Args:
            repo_id: Repository ID
            local_dir: Target directory
            allow_patterns: File patterns to include (e.g., ["*.gguf"])
            ignore_patterns: File patterns to exclude
            
        Returns:
            Path to local directory
        """
        logger.info(f"Downloading repository {repo_id} to {local_dir}")
        
        # Default: only download GGUF files
        if allow_patterns is None and ignore_patterns is None:
            allow_patterns = ["*.gguf", "*.json", "*.txt"]
        
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                token=self.token,
                resume_download=True,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
            
            logger.info(f"Repository downloaded to: {local_dir}")
            return local_dir
            
        except Exception as e:
            logger.error(f"Repository download failed: {e}")
            raise RuntimeError(f"Download error: {e}")
    
    def list_repo_files(
        self,
        repo_id: str,
        pattern: Optional[str] = None
    ) -> list[str]:
        """
        List files in a HuggingFace repository.
        
        Args:
            repo_id: Repository ID
            pattern: Optional file pattern filter (e.g., "*.gguf")
            
        Returns:
            List of file paths in the repository
        """
        from huggingface_hub import list_repo_files
        
        try:
            files = list_repo_files(repo_id=repo_id, token=self.token)
            
            if pattern:
                import fnmatch
                files = [f for f in files if fnmatch.fnmatch(f, pattern)]
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list repository files: {e}")
            raise RuntimeError(f"List error: {e}")


def get_model_download_path(model_name: str, base_dir: Path) -> Path:
    """
    Get standardized download path for a model.
    
    Args:
        model_name: Model identifier
        base_dir: Base models directory
        
    Returns:
        Path where model should be stored
    """
    # Sanitize model name for filesystem
    safe_name = model_name.replace('/', '_').replace('\\', '_')
    return base_dir / safe_name
