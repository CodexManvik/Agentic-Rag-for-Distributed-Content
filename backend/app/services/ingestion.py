from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import chromadb
import requests
from bs4 import BeautifulSoup
from llama_index.core import Document, SimpleDirectoryReader

from app.config import settings
from app.services.compliance import is_url_allowlisted
from app.services.retrieval import clear_retrieval_caches, get_vector_index

try:
    from chromadb.errors import NotFoundError as ChromaNotFoundError
except Exception:  # pragma: no cover - fallback for older chromadb variants
    ChromaNotFoundError = type("ChromaNotFoundError", (Exception,), {})


class IngestionStats(TypedDict):
    documents_processed: int
    chunks_added: int
    skipped_duplicates: int
    errors: list[str]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_collection_count() -> int:
    client = chromadb.PersistentClient(path=settings.resolved_chroma_persist_directory)
    collection = client.get_or_create_collection(name=settings.chroma_collection_name)
    return int(collection.count())


def _insert_documents(documents: list[Document]) -> int:
    if not documents:
        return 0

    index = get_vector_index()
    inserted_count = 0
    for document in documents:
        try:
            index.insert(document)
            inserted_count += 1
        except Exception:
            continue
    clear_retrieval_caches()
    return inserted_count


def reset_index() -> None:
    client = chromadb.PersistentClient(path=settings.resolved_chroma_persist_directory)
    try:
        client.delete_collection(name=settings.chroma_collection_name)
    except ChromaNotFoundError:
        pass
    client.get_or_create_collection(name=settings.chroma_collection_name)
    clear_retrieval_caches()


def _ensure_allowed_domain(url: str) -> None:
    allowed, host = is_url_allowlisted(url)
    if settings.public_sources_only and not allowed:
        raise ValueError(f"Domain '{host}' is not allowlisted for public ingestion")


def ingest_web_page(url: str) -> IngestionStats:
    _ensure_allowed_domain(url)
    stats: IngestionStats = {
        "documents_processed": 1,
        "chunks_added": 0,
        "skipped_duplicates": 0,
        "errors": [],
    }

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
            tag.decompose()

        title = (soup.title.string or url).strip() if soup.title else url
        text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())

        if not text:
            return stats

        document = Document(
            text=text,
            metadata={
                "source": title,
                "source_type": "web",
                "title": title,
                "url": url,
                "ingested_at": _timestamp(),
            },
        )
        stats["chunks_added"] = _insert_documents([document])
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
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        documents = SimpleDirectoryReader(input_files=[str(path)]).load_data()
        # Keeps API contract intact; for now duplicates are managed by vector DB behavior.
        stats["chunks_added"] = _insert_documents(documents)
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

        documents = SimpleDirectoryReader(input_files=[str(path)]).load_data()
        stats["chunks_added"] = _insert_documents(documents)
    except Exception as exc:
        stats["errors"].append(str(exc))

    return stats
