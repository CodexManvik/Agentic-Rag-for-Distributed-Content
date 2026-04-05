from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Protocol, cast
import time

import requests
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.config import settings


class ChatModel(Protocol):
    def invoke(self, input: str | list[Any]) -> Any:
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


def get_chat_model(max_output_tokens: int | None = None, model_name: str | None = None) -> ChatModel:
    chat_model = model_name or settings.ollama_chat_model
    _ensure_model_available(chat_model)
    resolved_tokens = max_output_tokens or settings.effective_model_max_output_tokens
    ollama_kwargs: dict[str, Any] = {
        "base_url": settings.ollama_base_url,
        "model": chat_model,
        "temperature": settings.model_temperature,
        "top_p": settings.model_top_p,
        "top_k": settings.model_top_k,
        "repeat_penalty": settings.model_repetition_penalty,
        "num_predict": resolved_tokens,
    }
    model = ChatOllama(**ollama_kwargs)
    return cast(ChatModel, model)


def invoke_synthesis(
    prompt: str | list[dict[str, str]],
    timeout_seconds: float,
    max_output_tokens: int,
    model_name: str | None = None,
) -> str:
    """Invoke synthesis via ChatOllama using a proper chat messages list.

    Accepts either a plain string (legacy) or a list of {"role", "content"} dicts.
    Uses a dedicated ChatOllama instance with num_ctx=2048 so the small model
    doesn't time out trying to process a huge context window.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    chat_model = model_name or settings.ollama_chat_model
    _ensure_model_available(chat_model)

    # Build a dedicated model instance with synthesis-specific settings.
    # num_ctx=2048 keeps generation fast on small/CPU-bound hardware.
    model = ChatOllama(
        base_url=settings.ollama_base_url,
        model=chat_model,
        temperature=settings.model_temperature,
        top_p=settings.model_top_p,
        top_k=settings.model_top_k,
        repeat_penalty=settings.model_repetition_penalty,
        num_predict=max_output_tokens,
        num_ctx=2048,
    )

    # Convert messages list to LangChain message objects
    if isinstance(prompt, list):
        lc_messages: list[Any] = []
        for msg in prompt:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        input_payload: Any = lc_messages
    else:
        input_payload = prompt

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(model.invoke, input_payload)
            result = future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        raise ModelInvocationError(
            f"Timed out during synthesis after {timeout_seconds:.1f}s"
        ) from exc
    except Exception as exc:
        raise ModelInvocationError(
            f"Model invocation failed during synthesis: {exc}"
        ) from exc

    # Extract text from LangChain response
    if isinstance(result, str):
        return result.strip()
    content = getattr(result, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [item if isinstance(item, str) else item.get("text", "") for item in content]
        return "\n".join(p for p in parts if p).strip()
    return str(result).strip()


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


_MODEL_CACHE_TTL_SECONDS = 30  # Refresh model list every 30 seconds to pick up newly pulled models


@lru_cache(maxsize=1)
def _available_models_cached() -> tuple[set[str], float]:
    """Cached version returning model set and timestamp."""
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
    return names, time.time()


def _available_models() -> set[str]:
    """Get available models with automatic TTL-based refresh (30s).
    
    This allows the server to detect newly pulled models without requiring a restart.
    """
    try:
        models, cached_time = _available_models_cached()
    except RuntimeError:
        # If we haven't cached yet, let the exception propagate
        _available_models_cached.cache_clear()
        raise
    
    # If cache is fresh, return it
    if time.time() - cached_time < _MODEL_CACHE_TTL_SECONDS:
        return models
    
    # Cache expired: clear and refetch
    _available_models_cached.cache_clear()
    models, _ = _available_models_cached()
    return models


def refresh_model_registry() -> None:
    """Clear the model cache to force a refresh on next access."""
    _available_models_cached.cache_clear()


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
    missing: list[str] = []
    if not _is_model_available(settings.ollama_chat_model, available):
        missing.append(settings.ollama_chat_model)
    if not _is_model_available(settings.ollama_embedding_model, available):
        missing.append(settings.ollama_embedding_model)
    if missing:
        pulls = ", ".join(f"ollama pull {m}" for m in missing)
        return False, f"Missing Ollama models: {', '.join(missing)}. Run: {pulls}"
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
    model_name: str | None = None,
) -> Any:
    timeout = timeout_seconds or settings.effective_model_request_timeout_seconds
    profile = settings.normalized_runtime_profile
    max_tokens = max_output_tokens or settings.effective_model_max_output_tokens
    # If custom model specified, create instance directly; otherwise use cached
    if model_name:
        model = get_chat_model(max_tokens, model_name=model_name)
    else:
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