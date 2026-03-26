from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, TypedDict, cast
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from chromadb.api.types import Metadata
from pypdf import PdfReader

from app.config import settings
from app.services.chunking import split_structured_text
from app.services.vector_store import get_collection, reset_collection


class IngestionStats(TypedDict):
    documents_processed: int
    chunks_added: int
    skipped_duplicates: int
    errors: list[str]


def reset_index() -> None:
    reset_collection()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return sha256(normalized.encode("utf-8")).hexdigest()


def _existing_hashes() -> set[str]:
    collection = get_collection()
    payload = collection.get(include=["metadatas"])
    metadatas = payload.get("metadatas", [])
    hashes: set[str] = set()
    for item in metadatas:
        if isinstance(item, dict):
            value = item.get("content_hash")
            if isinstance(value, str):
                hashes.add(value)
    return hashes


def _ensure_allowed_domain(url: str) -> None:
    if not settings.public_sources_only:
        return
    host = urlparse(url).netloc.lower()
    allowed = any(host == domain or host.endswith(f".{domain}") for domain in settings.allowed_domains)
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
                        "source_type": "web",
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
    except Exception as exc:
        stats["errors"].append(str(exc))

    return stats


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
        reader = PdfReader(str(path))
        hashes = _existing_hashes()

        ids: list[str] = []
        docs: list[str] = []
        metadatas: list[Metadata] = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            sections = [s for s in page_text.split("\n\n") if s.strip()]
            for section_idx, section in enumerate(sections):
                for chunk_idx, chunk in enumerate(split_structured_text(section)):
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
    except Exception as exc:
        stats["errors"].append(str(exc))

    return stats
