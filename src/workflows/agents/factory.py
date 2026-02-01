from typing import Any

from langchain_ollama import ChatOllama
from ollama._types import ResponseError

from configs.constants import WORKFLOW_LOGGING, MAX_RETRIES
from src.utils.common_funcs import get_logger

logger = get_logger(WORKFLOW_LOGGING)


class AgentsFactory:
    """Factory class for creating and configuring LLM agents."""

    @classmethod
    def get_llm_by_role(cls, model_settings: Any) -> ChatOllama:
        """Creates an LLM instance configured for a specific role.
        
        Args:
            model_settings: Model configuration settings including model name, API key, and parameters.
            
        Returns:
            Configured ChatOllama instance.
        """
        role_params = model_settings.model_dump(
            include={"temperature", "top_k", "top_p", "num_predict", "base_url"}
        )

        llm = ChatOllama(
            model=model_settings.model_name,
            client_kwargs={
                "headers": {'Authorization': 'Bearer ' + model_settings.api_key.get_secret_value()}
            },
            **role_params
        )

        return llm

    @classmethod
    def add_retry(cls, runnable: Any) -> Any:
        """Applies retry logic to a runnable.
        
        Args:
            runnable: Runnable object to add retry logic to.
            
        Returns:
            Runnable with retry logic applied.
        """
        return runnable.with_retry(
            retry_if_exception_type=(ResponseError,),
            wait_exponential_jitter=True,
            stop_after_attempt=MAX_RETRIES,
        )
