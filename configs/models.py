from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from configs.constants import ENV_PATH
from configs.enums import ModelRoleType


class BaseLLMSettings(BaseSettings):
    """Base settings for language model configurations.

    Provides common configuration options for all language model services.
    """
    model_name: str
    role: ModelRoleType
    temperature: float = 0.
    prompt_version: str | dict = "v1"
    tokenizer_model_name: str | None = None  # To use when tokenization needed before the generation

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def model_config_with_prefix(cls, prefix: str) -> SettingsConfigDict:
        return SettingsConfigDict(**cls.model_config) | {'env_prefix': prefix}


class OpenAISettings(BaseLLMSettings):
    """Configuration settings for OpenAI models.

    Extends base LLM settings with OpenAI-specific configurations.
    """
    model_config = BaseLLMSettings.model_config_with_prefix("OPENAI_")
    api_key: SecretStr | None = None


class OllamaSettings(BaseLLMSettings):
    """Configuration settings for Ollama models.

    Extends base LLM settings with Ollama-specific configurations.
    """
    model_config = BaseLLMSettings.model_config_with_prefix("OLLAMA_")
    api_key: SecretStr | None = None
    reasoning: bool = False
    num_ctx: int = 2048
    top_k: int = 40
    top_p: float = 0.5
    num_predict: int = 128
    base_url: str = "https://ollama.com"


class CohereSettings(BaseLLMSettings):
    """Configuration for Cohere models."""
    model_config = BaseLLMSettings.model_config_with_prefix("COHERE_")
    api_key: SecretStr | None = None
    top_n: int


# Settings for the models that run locally (downloaded from huggingface hub etc.)
class LocalModelSettings(BaseSettings):
    """Configuration for locally hosted models.

    Contains settings for models that are run locally rather than through an API.
    """
    role: ModelRoleType
    model_path: str = "nomic-ai/nomic-embed-text-v2-moe"
    device: str | None = None
    embedding_dim: int | None = None


class ModelSettings(BaseSettings):
    """Aggregates model configuration settings.

    Centralized configuration for all model-related settings.
    """
    embedder: LocalModelSettings | BaseLLMSettings = LocalModelSettings(
        role=ModelRoleType.embedder,
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384
    )
    orchestrator: LocalModelSettings | BaseLLMSettings = OllamaSettings(
        role=ModelRoleType.orchestrator,
        model_name="ministral-3:3b-cloud",
        tokenizer_model_name="mistralai/Ministral-3-3B-Instruct-2512",
        temperature=0.,
        top_k=10,
        top_p=1.,
        num_predict=2048,
        prompt_version={"graph_index": "v2", "routing": "v1"}
    )
    retriever: LocalModelSettings | BaseLLMSettings = OllamaSettings(
        role=ModelRoleType.retriever,
        model_name="rnj-1:8b-cloud",  # "qwen2.5:14b",
        temperature=0.0,
        num_ctx=8192,
        top_k=20,
        top_p=0.3,
        num_predict=512,
    )
    researcher: dict = {
        "_with_tools": OllamaSettings(
            role=ModelRoleType.researcher,
            model_name="rnj-1:8b-cloud",
            temperature=0.0,
            num_ctx=4096,
            top_k=20,
            top_p=0.3,
            num_predict=1024,
        ),
        "_structured": OllamaSettings(
            role=ModelRoleType.researcher,
            model_name="gemma3:4b-cloud",
            temperature=0.0,
            num_ctx=4096,
            top_k=50,
            top_p=1.,
            num_predict=2048,
        ),
        "prompt_version": "v1"
    }
    analyst: LocalModelSettings | BaseLLMSettings = OllamaSettings(
        role=ModelRoleType.analyst,
        model_name="ministral-3:8b-cloud",
        temperature=0.15,
        num_ctx=32768,
        top_k=40,
        top_p=0.9,
        num_predict=2048,
        reasoning=True,
    )
    mentor: LocalModelSettings | BaseLLMSettings = OllamaSettings(
        role=ModelRoleType.mentor,
        model_name="gemini-3-flash-preview:cloud",
        temperature=0.7,
        num_ctx=8192,
        top_k=40,
        top_p=0.8,
        num_predict=2048,
        reasoning=True,
    )
    reranker: CohereSettings | BaseLLMSettings = CohereSettings(
        role=ModelRoleType.reranker,
        model_name="rerank-multilingual-v3.0",
        top_n=10
    )
