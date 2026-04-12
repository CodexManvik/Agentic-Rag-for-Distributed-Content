"""
Session-scoped KV cache for llama.cpp integration.

Provides session-aware caching of key-value states for efficient context reuse
within conversation sessions.
"""

import os
import pickle
import logging
import hmac
import hashlib
import re
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class SessionKVCache:
    """
    Session-scoped KV cache for llama.cpp integration.
    
    Manages key-value cache states that persist for the duration of a session,
    enabling efficient context reuse and faster response generation.
    """
    
    def __init__(self, session_id: str, cache_dir: str = "data/kv_cache"):
        """
        Initialize session KV cache.
        
        Args:
            session_id: Unique session identifier
            cache_dir: Directory to store cache files
        """
        if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
            raise ValueError("session_id contains unsafe characters; only letters, digits, '_' and '-' are allowed")

        self.session_id = session_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_file = self.cache_dir / f"session_{session_id}.cache"
        self._cache_data: Dict[str, Any] = {}
        self._context_tokens: List[int] = []
        self._is_loaded = False
        self._closed = False
        self._hmac_key = os.getenv("KV_CACHE_HMAC_KEY", "dev-only-kv-cache-key").encode("utf-8")
        
        # Load existing cache if available
        self._load_cache()
    
    def save_state(
        self,
        context_tokens: List[int],
        kv_state: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save current KV cache state.
        
        Args:
            context_tokens: Token sequence that generated this state
            kv_state: Serializable KV cache state (implementation-specific)
            metadata: Additional metadata about the state
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            self._cache_data = {
                "session_id": self.session_id,
                "context_tokens": context_tokens,
                "kv_state": kv_state,
                "metadata": metadata or {},
                "token_count": len(context_tokens)
            }
            self._context_tokens = context_tokens.copy()
            self._write_cache_to_disk()
            
            logger.debug(f"Saved KV cache state for session {self.session_id} ({len(context_tokens)} tokens)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save KV cache for session {self.session_id}: {e}")
            return False
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """
        Load KV cache state.
        
        Returns:
            Cache state dict if available, None otherwise
        """
        if not self._is_loaded:
            self._load_cache()
        
        return self._cache_data.copy() if self._cache_data else None
    
    def clear(self) -> bool:
        """
        Clear the KV cache state.
        
        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            self._cache_data.clear()
            self._context_tokens.clear()
            
            # Remove cache file
            if self.cache_file.exists():
                self.cache_file.unlink()
            
            logger.debug(f"Cleared KV cache for session {self.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear KV cache for session {self.session_id}: {e}")
            return False
    
    def get_context_tokens(self) -> List[int]:
        """
        Get the token sequence for current cached context.
        
        Returns:
            List of context tokens
        """
        return self._context_tokens.copy()
    
    def can_reuse_context(self, new_tokens: List[int], min_overlap: float = 0.8) -> bool:
        """
        Check if current cache can be reused for new token sequence.
        
        Args:
            new_tokens: New token sequence to check
            min_overlap: Minimum overlap ratio to consider reusable
            
        Returns:
            True if cache can be reused, False otherwise
        """
        if not self._context_tokens or not new_tokens:
            return False
        
        # Find longest common prefix
        common_length = 0
        min_length = min(len(self._context_tokens), len(new_tokens))
        
        for i in range(min_length):
            if self._context_tokens[i] == new_tokens[i]:
                common_length += 1
            else:
                break
        
        if common_length == 0:
            return False
        
        # Calculate overlap ratio based on cached context
        overlap_ratio = common_length / len(self._context_tokens)
        reusable = overlap_ratio >= min_overlap
        
        logger.debug(f"Context overlap: {common_length}/{len(self._context_tokens)} tokens "
                    f"({overlap_ratio:.1%}) - {'reusable' if reusable else 'not reusable'}")
        
        return reusable
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about current cache state.
        
        Returns:
            Cache information dictionary
        """
        cache_file_size = self._get_file_size()
        return {
            "session_id": self.session_id,
            "has_cache": bool(self._cache_data),
            "token_count": len(self._context_tokens),
            "cache_file_exists": cache_file_size > 0,
            "cache_file_size": cache_file_size,
            "metadata": self._cache_data.get("metadata", {}) if self._cache_data else {}
        }

    def _get_file_size(self) -> int:
        try:
            return self.cache_file.stat().st_size
        except FileNotFoundError:
            return 0

    def _write_cache_to_disk(self) -> None:
        payload = pickle.dumps(self._cache_data, protocol=pickle.HIGHEST_PROTOCOL)
        signature = hmac.new(self._hmac_key, payload, hashlib.sha256).hexdigest().encode("ascii")
        with open(self.cache_file, "wb") as f:
            f.write(signature + b"\n" + payload)
    
    def _load_cache(self) -> None:
        """Load cache from disk if available."""
        if self._get_file_size() <= 0:
            self._is_loaded = True
            return
        
        try:
            with open(self.cache_file, "rb") as f:
                raw = f.read()

            sig_bytes, payload = raw.split(b"\n", 1)
            expected_sig = hmac.new(self._hmac_key, payload, hashlib.sha256).hexdigest().encode("ascii")
            if not hmac.compare_digest(sig_bytes, expected_sig):
                raise ValueError("Cache file HMAC verification failed")

            self._cache_data = pickle.loads(payload)
            
            self._context_tokens = self._cache_data.get("context_tokens", [])
            self._is_loaded = True
            
            logger.debug(f"Loaded KV cache for session {self.session_id} "
                        f"({len(self._context_tokens)} tokens)")
            
        except Exception as e:
            logger.warning(f"Failed to load KV cache for session {self.session_id}: {e}")
            self._cache_data = {}
            self._context_tokens = []
            self._is_loaded = True

    def close(self) -> None:
        """Persist cache state deterministically and mark the cache as closed."""
        if self._closed:
            return
        try:
            if self._cache_data and self.cache_file.parent.exists():
                self._write_cache_to_disk()
        except Exception as exc:
            logger.exception(f"Failed to persist KV cache for session {self.session_id}: {exc}")
        finally:
            self._closed = True

    def __enter__(self) -> "SessionKVCache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
    
    @classmethod
    def cleanup_old_caches(cls, cache_dir: str = "data/kv_cache", days_old: int = 7) -> int:
        """
        Clean up old cache files.
        
        Args:
            cache_dir: Directory containing cache files
            days_old: Age threshold in days
            
        Returns:
            Number of files cleaned up
        """
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            return 0
        
        import time
        cutoff_time = time.time() - (days_old * 24 * 3600)
        cleaned_count = 0
        
        try:
            for cache_file in cache_path.glob("session_*.cache"):
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} old KV cache files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old cache files: {e}")
            return 0
    
    def __del__(self):
        """Best-effort cleanup; explicit close() should be preferred by callers."""
        try:
            self.close()
        except Exception:
            pass