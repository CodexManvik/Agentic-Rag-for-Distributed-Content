from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_structured_text(text: str) -> list[str]:
    splitter = build_splitter()
    cleaned = text.strip()
    if not cleaned:
        return []
    return [chunk.strip() for chunk in splitter.split_text(cleaned) if chunk.strip()]
