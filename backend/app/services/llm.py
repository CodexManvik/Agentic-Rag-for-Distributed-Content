from typing import Any, Protocol, cast

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import SecretStr

from app.config import settings


class ChatModel(Protocol):
    def invoke(self, input: str) -> Any:
        ...


class ChromaAzureEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, embeddings_model: AzureOpenAIEmbeddings):
        self._embeddings_model = embeddings_model

    def __call__(self, input: Documents) -> Embeddings:
        return cast(Embeddings, self._embeddings_model.embed_documents(list(input)))


def get_chat_model() -> ChatModel | None:
    if not (
        settings.azure_openai_endpoint
        and settings.azure_openai_api_key
        and settings.azure_openai_deployment
    ):
        return None

    model = AzureChatOpenAI(  # pyright: ignore[reportCallIssue]
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=SecretStr(settings.azure_openai_api_key),
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_deployment,
        temperature=0,
    )
    return cast(ChatModel, model)


def get_embedding_model() -> AzureOpenAIEmbeddings | None:
    embedding_deployment = (
        settings.azure_openai_embedding_deployment or settings.azure_openai_deployment
    )
    if not (
        settings.azure_openai_endpoint
        and settings.azure_openai_api_key
        and embedding_deployment
    ):
        return None

    return AzureOpenAIEmbeddings(  # pyright: ignore[reportCallIssue]
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=SecretStr(settings.azure_openai_api_key),
        api_version=settings.azure_openai_api_version,
        azure_deployment=embedding_deployment,
    )


def get_chroma_embedding_function() -> EmbeddingFunction[Documents] | None:
    model = get_embedding_model()
    if model is None:
        return None
    return ChromaAzureEmbeddingFunction(model)
