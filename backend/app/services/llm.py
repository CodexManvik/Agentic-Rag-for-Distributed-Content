from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Protocol, cast

import requests
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
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


class ModelInvocationError(RuntimeError):
    pass


def get_chat_model(max_output_tokens: int | None = None) -> ChatModel:
    _ensure_model_available(settings.ollama_chat_model)
    resolved_tokens = max_output_tokens or settings.effective_model_max_output_tokens
    ollama_kwargs: dict[str, Any] = {
        "base_url": settings.ollama_base_url,
        "model": settings.ollama_chat_model,
        "temperature": settings.model_temperature,
        "top_p": settings.model_top_p,
        "top_k": settings.model_top_k,
        "repeat_penalty": settings.model_repetition_penalty,
        "num_predict": resolved_tokens,
    }
    model = ChatOllama(**ollama_kwargs)
    return cast(ChatModel, model)


def invoke_synthesis(
    prompt: str,
    timeout_seconds: float,
    max_output_tokens: int,
) -> str:
    """Invoke the chat model for synthesis using native Ollama HTTP API.

    Uses the /no_think directive (prepended to the user message) and the
    Ollama-native ``think: false`` parameter to disable chain-of-thought for
    Qwen3 thinking models.  This avoids empty ``.content`` fields that occur
    when LangChain ChatOllama splits thinking tokens away from the answer, and
    ensures the model does not emit ``<think>`` blocks that corrupt JSON output.
    """
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    payload = {
        "model": settings.ollama_chat_model,
        "messages": [
            # /no_think placed in the user message is the official Qwen3
            # mechanism to suppress chain-of-thought output.
            {"role": "user", "content": "/no_think\n" + prompt},
        ],
        "stream": False,
        # Ollama ≥ 0.7.0 native thinking control – suppresses <think> tokens
        # at the server level regardless of model defaults.
        "think": False,
        "options": {
            "temperature": settings.model_temperature,
            "top_p": settings.model_top_p,
            "top_k": settings.model_top_k,
            "repeat_penalty": settings.model_repetition_penalty,
            "num_predict": max_output_tokens,
        },
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.json()
        return str(data.get("message", {}).get("content", "")).strip()
    except requests.Timeout as exc:
        raise ModelInvocationError(
            f"Timed out during synthesis after {timeout_seconds:.1f}s"
        ) from exc
    except Exception as exc:
        raise ModelInvocationError(
            f"Model invocation failed during synthesis: {exc}"
        ) from exc


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


@lru_cache(maxsize=6)
def get_shared_chat_model(runtime_profile: str, max_output_tokens: int) -> ChatModel:
    return get_chat_model(max_output_tokens=max_output_tokens)


@lru_cache(maxsize=1)
def get_shared_embedding_model() -> EmbeddingModel:
    return get_embedding_model()


@lru_cache(maxsize=1)
def get_shared_chroma_embedding_function() -> EmbeddingFunction[Documents]:
    return ChromaLocalEmbeddingFunction(get_shared_embedding_model())


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


def refresh_model_registry() -> None:
    _available_models.cache_clear()


def check_ollama_readiness() -> tuple[bool, str]:
    tags_url = f"{settings.ollama_base_url.rstrip('/')}" + "/api/tags"
    try:
        response = requests.get(tags_url, timeout=8)
        response.raise_for_status()
    except Exception as exc:
        return False, (
            f"Cannot reach Ollama at {settings.ollama_base_url}. "
            "Start Ollama and verify network access."
        )

    refresh_model_registry()
    available = _available_models()

    # The chat model is required; without it queries cannot be answered.
    if not _is_model_available(settings.ollama_chat_model, available):
        return False, (
            f"Missing Ollama chat model: {settings.ollama_chat_model}. "
            f"Run: ollama pull {settings.ollama_chat_model}"
        )

    # The embedding model is preferred but not strictly required – the system
    # falls back to Chroma's bundled ONNX embedding when it is absent.
    if not _is_model_available(settings.ollama_embedding_model, available):
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "Embedding model '%s' not found in Ollama; falling back to "
            "Chroma's default embedding function (all-MiniLM-L6-v2). "
            "Retrieval quality may differ from production. "
            "Pull the model with: ollama pull %s",
            settings.ollama_embedding_model,
            settings.ollama_embedding_model,
        )

    return True, "ready"


def _ensure_model_available(model_name: str) -> None:
    available = _available_models()
    if not _is_model_available(model_name, available):
        raise RuntimeError(
            f"Required Ollama model '{model_name}' is not available. Pull it with: ollama pull {model_name}"
        )


def _is_model_available(model_name: str, available_models: set[str]) -> bool:
    if model_name in available_models:
        return True
    if ":" not in model_name and f"{model_name}:latest" in available_models:
        return True
    if ":" in model_name:
        base_name = model_name.split(":", 1)[0]
        if model_name.endswith(":latest") and base_name in available_models:
            return True
    return False


def invoke_chat_with_timeout(
    prompt: str | list[Any],
    purpose: str,
    timeout_seconds: float | None = None,
    max_output_tokens: int | None = None,
) -> Any:
    timeout = timeout_seconds or settings.effective_model_request_timeout_seconds
    profile = settings.normalized_runtime_profile
    max_tokens = max_output_tokens or settings.effective_model_max_output_tokens
    model = get_shared_chat_model(profile, max_tokens)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(model.invoke, prompt)
            return future.result(timeout=timeout)
    except FutureTimeoutError as exc:
        raise ModelInvocationError(
            f"Timed out during {purpose} after {timeout:.1f}s"
        ) from exc
    except Exception as exc:
        raise ModelInvocationError(
            f"Model invocation failed during {purpose}: {exc}"
        ) from exc
