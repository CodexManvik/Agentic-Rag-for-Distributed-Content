from pathlib import Path
from typing import cast

import requests
from bs4 import BeautifulSoup
from chromadb import PersistentClient
from chromadb.api.types import Metadata
from pypdf import PdfReader

from app.config import settings
from app.services.llm import get_chroma_embedding_function


client = PersistentClient(path=settings.chroma_persist_directory)
embedding_function = get_chroma_embedding_function()
collection = client.get_or_create_collection(
    name=settings.chroma_collection_name,
    embedding_function=embedding_function,
)


def ingest_web_page(url: str) -> int:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    text = " ".join(soup.get_text(" ", strip=True).split())
    chunks = [text[i : i + 1200] for i in range(0, len(text), 1200)]

    ids = [f"web::{url}::{i}" for i in range(len(chunks))]
    metadatas: list[Metadata] = [
        cast(Metadata, {"source": url, "url": url, "type": "web"}) for _ in chunks
    ]
    if chunks:
        collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def ingest_pdf(file_path: str) -> int:
    path = Path(file_path)
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    chunks = [text[i : i + 1200] for i in range(0, len(text), 1200)]

    ids = [f"pdf::{path.name}::{i}" for i in range(len(chunks))]
    metadatas: list[Metadata] = [
        cast(Metadata, {"source": path.name, "type": "pdf", "path": str(path)})
        for _ in chunks
    ]
    if chunks:
        collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)
