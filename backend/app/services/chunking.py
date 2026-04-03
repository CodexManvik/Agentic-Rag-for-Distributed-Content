from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from loguru import logger

from app.config import settings


def build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _count_chunk_tokens(text: str) -> int:
    """Count tokens in chunk using tiktoken."""
    try:
        enc = tiktoken.get_encoding(settings.token_encoding_model)
        return len(enc.encode(text))
    except Exception as e:
        logger.warning(f"Token counting in chunk failed: {e}. Using word count fallback")
        return len(text.split())


def split_structured_text(text: str) -> list[str]:
    """
    Split text into chunks respecting both character and token limits.
    
    Uses RecursiveCharacterTextSplitter for semantic preservation, then validates
    against token limits to ensure chunks fit within LLM context windows.
    
    Args:
        text: Input text to chunk
        
    Returns:
        List of chunks, each under both chunk_size and chunk_size_tokens limits
    """
    splitter = build_splitter()
    cleaned = text.strip()
    if not cleaned:
        return []
    
    # Get character-based chunks first
    char_chunks = [chunk.strip() for chunk in splitter.split_text(cleaned) if chunk.strip()]
    
    # Filter chunks by token limit (additional validation)
    token_limited_chunks = []
    for chunk in char_chunks:
        chunk_tokens = _count_chunk_tokens(chunk)
        if chunk_tokens <= settings.chunk_size_tokens:
            token_limited_chunks.append(chunk)
        else:
            # Log oversized chunks for monitoring
            logger.warning(
                f"Chunk exceeded token limit: {chunk_tokens} tokens > {settings.chunk_size_tokens} max. "
                f"Chunk size: {len(chunk)} chars. Consider increasing chunk_size_tokens or reducing content complexity."
            )
    
    return token_limited_chunks
