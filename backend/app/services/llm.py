from typing import Any, Protocol, cast

from langchain_openai import AzureChatOpenAI

from app.config import settings


class ChatModel(Protocol):
    def invoke(self, input: str) -> Any:
        ...


def get_chat_model() -> ChatModel | None:
    if not (
        settings.azure_openai_endpoint
        and settings.azure_openai_api_key
        and settings.azure_openai_deployment
    ):
        return None

    model = AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        openai_api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_deployment,
        temperature=0,
    )
    return cast(ChatModel, model)
