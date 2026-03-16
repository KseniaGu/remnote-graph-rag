from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from ollama._types import ResponseError
from openai import APIError as OpenAIAPIError

try:
    from google.genai.errors import APIError as GoogleAPIError
except ImportError:
    GoogleAPIError = Exception  # type: ignore[assignment,misc]

from backend.configs.constants import WORKFLOW_LOGGING, MAX_RETRIES
from backend.configs.enums import LLMProviderType
from backend.utils.common_funcs import get_logger

logger = get_logger(WORKFLOW_LOGGING)


class AgentsFactory:
    """Factory class for creating and configuring LLM agents for any supported provider."""

    @classmethod
    def get_llm_by_role(cls, model_settings: Any) -> Any:
        """Creates an LLM instance configured for a specific role and provider.

        Dispatches to the appropriate LangChain chat model based on model's provider:

        Args:
            model_settings: Model configuration settings including model name, API key,
                base_url, temperature, and other provider-specific parameters.

        Returns:
            Configured LangChain chat model instance.
        """
        provider = getattr(model_settings, "provider", LLMProviderType.ollama)
        api_key = model_settings.api_key.get_secret_value() if model_settings.api_key else None

        if provider == LLMProviderType.ollama:
            role_params = model_settings.model_dump(
                include={"temperature", "top_k", "top_p", "num_predict", "base_url", "num_ctx"}
            )
            return ChatOllama(
                model=model_settings.model_name,
                client_kwargs={
                    "headers": {"Authorization": f"Bearer {api_key}"}
                } if api_key else {},
                **role_params,
            )

        if provider == LLMProviderType.vllm:
            return ChatOpenAI(
                model=model_settings.model_name,
                base_url=getattr(model_settings, "base_url", None),
                api_key=api_key or "EMPTY",
                temperature=model_settings.temperature,
                max_tokens=getattr(model_settings, "num_predict", None),
            )

        if provider == LLMProviderType.openai:
            return ChatOpenAI(
                model=model_settings.model_name,
                api_key=api_key,
                temperature=model_settings.temperature,
                max_tokens=getattr(model_settings, "num_predict", None),
            )

        if provider == LLMProviderType.gemini:
            return ChatGoogleGenerativeAI(
                model=model_settings.model_name,
                google_api_key=api_key,
                temperature=model_settings.temperature,
            )

        raise ValueError(f"Unsupported LLM provider: {provider}")

    @classmethod
    def add_retry(cls, runnable: Any, provider: LLMProviderType = LLMProviderType.ollama) -> Any:
        """Applies retry logic to a runnable, scoped to provider-specific transient errors.

        Args:
            runnable: Runnable object to add retry logic to.
            provider: LLM provider type, used to select which exception types to retry on.

        Returns:
            Runnable with retry logic applied.
        """
        _retry_exceptions: dict[LLMProviderType, tuple] = {
            LLMProviderType.ollama: (ResponseError,),
            LLMProviderType.vllm: (OpenAIAPIError,),
            LLMProviderType.openai: (OpenAIAPIError,),
            LLMProviderType.gemini: (GoogleAPIError,),
        }
        retry_on = _retry_exceptions.get(provider, (Exception,))
        return runnable.with_retry(
            retry_if_exception_type=retry_on,
            wait_exponential_jitter=True,
            stop_after_attempt=MAX_RETRIES,
        )
