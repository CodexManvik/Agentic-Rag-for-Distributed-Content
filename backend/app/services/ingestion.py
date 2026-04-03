import re
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, TypedDict, cast
from urllib.parse import urlparse
from functools import lru_cache
import time

import requests
from bs4 import BeautifulSoup
from chromadb.api.types import Metadata
import fitz  # pymupdf — better text extraction than pypdf for academic PDFs
import tiktoken
from loguru import logger

from app.config import settings
from app.services.compliance import is_url_allowlisted
from app.services.chunking import split_structured_text
from app.services.vector_store import get_collection, refresh_bm25_cache, reset_collection


class IngestionStats(TypedDict):
    documents_processed: int
    chunks_added: int
    skipped_duplicates: int
    errors: list[str]


def reset_index() -> None:
    reset_collection()
    refresh_bm25_cache()
    _existing_hashes_cached.cache_clear()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return sha256(normalized.encode("utf-8")).hexdigest()


def _count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken (OpenAI token counting)."""
    try:
        enc = tiktoken.get_encoding(encoding)
        return len(enc.encode(text))
    except Exception as e:
        logger.warning(f"Token counting failed: {e}. Estimating as {len(text.split())} tokens")
        return len(text.split())  # Fallback to word count


_HASH_CACHE_TTL_SECONDS = 60  # Cache hashes for 60 seconds to avoid repeated full scans


@lru_cache(maxsize=1)
def _existing_hashes_cached() -> tuple[set[str], float]:
    """Cached version returning set of hashes and timestamp."""
    collection = get_collection()
    payload = collection.get(include=["metadatas"])
    metadatas = payload.get("metadatas", [])
    hashes: set[str] = set()
    for item in metadatas:
        if isinstance(item, dict):
            value = item.get("content_hash")
            if isinstance(value, str):
                hashes.add(value)
    return hashes, time.time()


def _existing_hashes() -> set[str]:
    """Get existing content hashes with in-process caching to avoid O(n) queries on every chunk write."""
    hashes, cached_time = _existing_hashes_cached()
    
    # If cache is still fresh, return it
    if time.time() - cached_time < _HASH_CACHE_TTL_SECONDS:
        return hashes
    
    # Cache expired: clear and refetch
    _existing_hashes_cached.cache_clear()
    hashes, _ = _existing_hashes_cached()
    return hashes


def _ensure_allowed_domain(url: str) -> None:
    allowed, host = is_url_allowlisted(url)
    if not settings.public_sources_only:
        return
    if not allowed:
        raise ValueError(f"Domain '{host}' is not allowlisted for public ingestion")


def _extract_web_sections(url: str, html: str) -> tuple[str, list[tuple[str, str]]]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title = (soup.title.string or url).strip() if soup.title else url
    sections: list[tuple[str, str]] = []
    current_heading = "Overview"
    current_lines: list[str] = []

    for element in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        name = element.name.lower()
        text = " ".join(element.get_text(" ", strip=True).split())
        if not text:
            continue
        if name in {"h1", "h2", "h3"}:
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines)))
            current_heading = text
            current_lines = []
        else:
            current_lines.append(text)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines)))
    return title, sections


def _classify_web_source_type(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host == "confluence.atlassian.com":
        return "confluence"
    return "web"


def ingest_web_page(url: str) -> IngestionStats:
    _ensure_allowed_domain(url)
    stats: IngestionStats = {
        "documents_processed": 1,
        "chunks_added": 0,
        "skipped_duplicates": 0,
        "errors": [],
    }

    try:
        collection = get_collection()
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        title, sections = _extract_web_sections(url, response.text)
        source_type = _classify_web_source_type(url)
        hashes = _existing_hashes()

        ids: list[str] = []
        docs: list[str] = []
        metadatas: list[Metadata] = []

        for heading, text in sections:
            for idx, chunk in enumerate(split_structured_text(text)):
                content_hash = _content_hash(chunk)
                if content_hash in hashes:
                    stats["skipped_duplicates"] += 1
                    continue

                chunk_id = f"web::{content_hash[:24]}"
                metadata = cast(
                    Metadata,
                    {
                        "source": title,
                        "source_type": source_type,
                        "title": title,
                        "section": heading,
                        "url": url,
                        "anchor": f"{url}#{heading.lower().replace(' ', '-')}",
                        "ingested_at": _timestamp(),
                        "content_hash": content_hash,
                        "chunk_index": idx,
                    },
                )
                ids.append(chunk_id)
                docs.append(chunk)
                metadatas.append(metadata)
                hashes.add(content_hash)

        if docs:
            collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
            stats["chunks_added"] = len(docs)
            refresh_bm25_cache()
    except Exception as exc:
        stats["errors"].append(str(exc))

    return stats


def _extract_pdf_sections(path: Path) -> list[tuple[int, int, str]]:
    """Extract (page_number, section_idx, text) tuples from a PDF using pymupdf.

    pymupdf preserves reading order and sentence boundaries far better than pypdf
    for multi-column academic papers. We use 'text' mode with word sorting to get
    clean left-to-right, top-to-bottom text flow.
    """
    doc = fitz.open(str(path))
    results: list[tuple[int, int, str]] = []
    try:
        for page_number, page in enumerate(doc, start=1):
            # 'text' flag with preserve_whitespace=False gives cleaner output
            page_text = page.get_text("text", sort=True).strip()
            if not page_text:
                continue
            # Split on double newlines (section breaks) — pymupdf inserts these
            # reliably at paragraph/section boundaries unlike pypdf
            sections = [s.strip() for s in page_text.split("\n\n") if s.strip()]
            for section_idx, section in enumerate(sections):
                # Join hyphenated line-breaks that pymupdf sometimes leaves
                section = re.sub(r"-\n(\w)", r"\1", section)
                # Collapse single newlines within a paragraph to spaces
                section = re.sub(r"(?<!\n)\n(?!\n)", " ", section)
                section = re.sub(r"\s{2,}", " ", section).strip()
                if section:
                    results.append((page_number, section_idx, section))
    finally:
        doc.close()
    return results


def _is_fragment(text: str) -> bool:
    """Return True if the text starts mid-sentence (broken chunk boundary).
    
    Avoids false positives on:
    - Code examples (import, const, function, etc.)
    - URLs (http://, ftp://, etc.)
    - JSON or structured data
    - List items or bullet points
    - Acronyms and proper nouns (JSON, API, etc.)
    """
    stripped = text.strip()
    if not stripped:
        return False
    
    first_char = stripped[0]
    
    # Not a fragment if starts with uppercase, digit, or code-like pattern
    if first_char.isupper() or first_char.isdigit():
        return False
    
    # Not a fragment if starts with special characters
    if first_char in "({[<\"'`@#$%&*-*/+=|\\:;.,/?!~":
        return False
    
    # Not a fragment if starts with common code/markup keywords
    code_keywords = (
        "import ", "from ", "const ", "let ", "var ", "function ", "class ",
        "def ", "async ", "await ", "return ", "throw ", "if ", "else ",
        "for ", "while ", "try ", "catch ", "switch ", "case ",
        "interface ", "type ", "enum ", "export ", "import{", "require(",
        "package ", "namespace ", "using ", "public ", "private ", "protected ",
        "abstract ", "static ", "final ", "extends ", "implements "
    )
    lower_text = stripped.lower()
    if any(lower_text.startswith(kw) for kw in code_keywords):
        return False
    
    # Not a fragment if starts with URL/URI patterns
    url_prefixes = ("http://", "https://", "ftp://", "ftps://", "file://", "www.")
    if any(lower_text.startswith(prefix) for prefix in url_prefixes):
        return False
    
    # Not a fragment if starts with common acronyms (JSON, REST, API, etc.)
    # These are often single/double char followed by space or common continuation
    if len(stripped) > 1:
        first_word = stripped.split()[0] if stripped.split() else ""
        if len(first_word) <= 4 and first_word.isupper() and first_word.isalpha():
            return False
    
    # True fragment: starts with lowercase letter and no special context
    return first_char.islower()


# Matches common patterns in academic reference sections
_REFERENCE_ENTRY_PATTERN = re.compile(
    r"(?:arXiv|arxiv)\s*(?:preprint\s*)?(?:arXiv:)?[\d.]+|"   # arXiv IDs/preprints
    r"(?:doi|DOI):\s*10\.\d{4}|"                               # DOI
    r"\bCoRR\b,\s*abs/|"                                       # CoRR entries
    r"(?:pp\.|pages?)\s+\d+[-–]\d+|"                          # page ranges
    r"In\s+(?:Proceedings|Advances|Proc\.)\b|"                 # conference proceedings
    r"(?:abs/\d{4}\.\d{4,5})|"                                # arXiv abs/ links
    r"https?://arxiv\.org/",                                   # arxiv URLs
    re.IGNORECASE,
)

# Matches lines that are purely a list of author names (First Last, First Last, ...)
_AUTHOR_LIST_PATTERN = re.compile(
    r"^(?:[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-zA-Z\-]+,?\s*){3,}",
)


def _is_reference_chunk(text: str) -> bool:
    """Return True if this chunk is a bibliography / reference list entry."""
    stripped = text.strip()
    # Single reference entry: one strong signal is enough
    if _REFERENCE_ENTRY_PATTERN.search(stripped):
        return True
    # Pure author-list line (no actual content)
    if _AUTHOR_LIST_PATTERN.match(stripped) and len(stripped) < 300:
        return True
    return False


def ingest_pdf(file_path: str) -> IngestionStats:
    stats: IngestionStats = {
        "documents_processed": 1,
        "chunks_added": 0,
        "skipped_duplicates": 0,
        "errors": [],
    }
    path = Path(file_path)
    try:
        collection = get_collection()
        hashes = _existing_hashes()

        ids: list[str] = []
        docs: list[str] = []
        metadatas: list[Metadata] = []

        for page_number, section_idx, section in _extract_pdf_sections(path):
            for chunk_idx, chunk in enumerate(split_structured_text(section)):
                # Skip mid-sentence fragments and bibliography/reference entries
                if _is_fragment(chunk) or _is_reference_chunk(chunk):
                    continue

                content_hash = _content_hash(chunk)
                if content_hash in hashes:
                    stats["skipped_duplicates"] += 1
                    continue

                chunk_id = f"pdf::{content_hash[:24]}"
                metadata = cast(
                    Metadata,
                    {
                        "source": path.name,
                        "source_type": "pdf",
                        "title": path.stem,
                        "section": f"page-{page_number}-section-{section_idx + 1}",
                        "page_number": page_number,
                        "path": str(path),
                        "ingested_at": _timestamp(),
                        "content_hash": content_hash,
                        "chunk_index": chunk_idx,
                    },
                )
                ids.append(chunk_id)
                docs.append(chunk)
                metadatas.append(metadata)
                hashes.add(content_hash)

        if docs:
            collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
            stats["chunks_added"] = len(docs)
            refresh_bm25_cache()
    except Exception as exc:
        stats["errors"].append(str(exc))

    return stats


def ingest_text_file(file_path: str) -> IngestionStats:
    stats: IngestionStats = {
        "documents_processed": 1,
        "chunks_added": 0,
        "skipped_duplicates": 0,
        "errors": [],
    }
    path = Path(file_path)
    try:
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {path}")

        collection = get_collection()
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return stats

        hashes = _existing_hashes()
        ids: list[str] = []
        docs: list[str] = []
        metadatas: list[Metadata] = []

        for idx, chunk in enumerate(split_structured_text(text)):
            content_hash = _content_hash(chunk)
            if content_hash in hashes:
                stats["skipped_duplicates"] += 1
                continue

            chunk_id = f"text::{content_hash[:24]}"
            metadata = cast(
                Metadata,
                {
                    "source": path.name,
                    "source_type": "project_doc",
                    "title": path.stem,
                    "section": "document",
                    "path": str(path.resolve()),
                    "ingested_at": _timestamp(),
                    "content_hash": content_hash,
                    "chunk_index": idx,
                },
            )
            ids.append(chunk_id)
            docs.append(chunk)
            metadatas.append(metadata)
            hashes.add(content_hash)

        if docs:
            collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
            stats["chunks_added"] = len(docs)
            refresh_bm25_cache()
    except Exception as exc:
        stats["errors"].append(str(exc))

    return stats