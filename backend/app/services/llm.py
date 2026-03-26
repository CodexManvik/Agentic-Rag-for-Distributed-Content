from functools import lru_cache
from typing import Any, Protocol, cast

import requests
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.config import settings


class ChatModel(Protocol):
    def invoke(self, input: str) -> Any:
        ...


class EmbeddingModel(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...


class ChromaLocalEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, embeddings_model: EmbeddingModel):
        self._embeddings_model = embeddings_model

    def __call__(self, input: Documents) -> Embeddings:
        return cast(Embeddings, self._embeddings_model.embed_documents(list(input)))


def get_chat_model() -> ChatModel | None:
    _ensure_model_available(settings.ollama_chat_model)
    model = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_chat_model,
        temperature=0,
    )
    return cast(ChatModel, model)


def get_embedding_model() -> EmbeddingModel:
    _ensure_model_available(settings.ollama_embedding_model)
    model = OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    )
    return cast(EmbeddingModel, model)


def get_chroma_embedding_function() -> EmbeddingFunction[Documents]:
    model = get_embedding_model()
    return ChromaLocalEmbeddingFunction(model)


@lru_cache(maxsize=1)
def _available_models() -> set[str]:
    tags_url = f"{settings.ollama_base_url.rstrip('/')}" + "/api/tags"
    try:
        response = requests.get(tags_url, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {settings.ollama_base_url}. Start Ollama before running the app."
        ) from exc

    models = payload.get("models", [])
    names: set[str] = set()
    for model in models:
        name = model.get("name")
        if isinstance(name, str):
            names.add(name)
    return names


def _ensure_model_available(model_name: str) -> None:
    available = _available_models()
    if model_name not in available:
        raise RuntimeError(
            f"Required Ollama model '{model_name}' is not available. Pull it with: ollama pull {model_name}"
        )
