import os

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.configs.constants import ENV_PATH
from backend.configs.enums import ModelRoleType, LLMProviderType


class BaseLLMSettings(BaseSettings):
    """Base settings for language model configurations.

    Provides common configuration options for all language model services.
    """
    model_name: str
    role: ModelRoleType
    provider: LLMProviderType = LLMProviderType.ollama
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
    top_p: float = 0.5
    max_tokens: int = 128
    base_url: str = ""


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


class ResearcherModelSettings(BaseSettings):
    """Configuration settings for the Researcher agent.

    Holds two OllamaSettings variants used in the two-phase research process:
    one for tool-calling and one for structured output synthesis.
    """
    # vLLM self-hosted alternative: Qwen3.5-9B (provider=vllm, base_url=VLLM_ROUTING_URL)
    with_tools: BaseLLMSettings = OllamaSettings(
        role=ModelRoleType.researcher,
        model_name="qwen3.5:cloud",
        temperature=0.0,
        num_ctx=8192,
        top_k=20,
        top_p=0.3,
        num_predict=1024,
    )
    # vLLM self-hosted alternative: Qwen3.5-9B (same instance as with_tools)
    structured: BaseLLMSettings = OllamaSettings(
        role=ModelRoleType.researcher,
        model_name="qwen3.5:cloud",
        temperature=0.0,
        num_ctx=8192,
        top_k=50,
        top_p=1.,
        num_predict=2048,
    )
    prompt_version: str = "v2"


class ModelSettings(BaseSettings):
    """Aggregates model configuration settings.

    Centralized configuration for all model-related settings.
    """
    embedder: LocalModelSettings | BaseLLMSettings = LocalModelSettings(
        role=ModelRoleType.embedder,
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384
    )

    # vLLM self-hosted alternative: Qwen/Qwen3.5-9B on port 8001, provider=LLMProviderType.vllm, base_url="http://<VLLM_HOST>:8001/v1"
    orchestrator: LocalModelSettings | BaseLLMSettings = OllamaSettings(
        role=ModelRoleType.orchestrator,
        model_name="qwen3.5:cloud",
        tokenizer_model_name="Qwen/Qwen3.5-9B",
        temperature=0.,
        top_k=10,
        top_p=1.,
        num_predict=2048,
        prompt_version={"graph_index": "v2", "routing": "v2"},
    )
    # vLLM self-hosted alternative: Qwen/Qwen3.5-9B (same instance as orchestrator)
    retriever: LocalModelSettings | BaseLLMSettings = OllamaSettings(
        role=ModelRoleType.retriever,
        model_name="qwen3.5:cloud",
        temperature=0.0,
        num_ctx=8192,
        top_k=20,
        top_p=0.3,
        num_predict=512,
        prompt_version="v2"
    )
    researcher: ResearcherModelSettings = ResearcherModelSettings()

    # vLLM self-hosted alternative: Qwen/Qwen3.5-27B (Q4) on port 8002, provider=LLMProviderType.vllm, base_url="http://<VLLM_HOST>:8002/v1"
    analyst: LocalModelSettings | BaseLLMSettings = OllamaSettings(
        role=ModelRoleType.analyst,
        model_name="qwen3.5:cloud",
        temperature=0.15,
        num_ctx=32768,
        top_k=40,
        top_p=0.9,
        num_predict=4096,
        reasoning=True,
        prompt_version="v2"
    )
    # vLLM self-hosted alternative: Qwen/Qwen3.5-27B (same instance as analyst)
    mentor: LocalModelSettings | BaseLLMSettings = OllamaSettings(
        role=ModelRoleType.mentor,
        model_name="qwen3.5:cloud",
        temperature=0.7,
        num_ctx=8192,
        top_k=40,
        top_p=0.8,
        num_predict=2048,
        reasoning=True,
        prompt_version="v2"
    )

    reranker: CohereSettings | BaseLLMSettings = CohereSettings(
        role=ModelRoleType.reranker,
        model_name="rerank-multilingual-v3.0",
        top_n=10
    )


def _ollama_models() -> ModelSettings:
    """Ollama Cloud pipeline (default)."""
    return ModelSettings()


def _vllm_models() -> ModelSettings:
    """vLLM Cloud Run GPU pipeline."""
    routing_url = os.environ["VLLM_ROUTING_URL"]
    generation_url = os.environ.get("VLLM_GENERATION_URL", routing_url)
    vllm_model_path = os.environ.get("VLLM_MODEL_PATH")
    return ModelSettings(
        orchestrator=OpenAISettings(
            role=ModelRoleType.orchestrator,
            provider=LLMProviderType.vllm,
            model_name=vllm_model_path,
            base_url=routing_url,
            temperature=0.,
            max_tokens=2048,
            prompt_version={"graph_index": "v2", "routing": "v2"},
        ),
        retriever=OpenAISettings(
            role=ModelRoleType.retriever,
            provider=LLMProviderType.vllm,
            model_name=vllm_model_path,
            base_url=routing_url,
            temperature=0.,
            max_tokens=512,
            prompt_version="v2",
        ),
        researcher=ResearcherModelSettings(
            with_tools=OpenAISettings(
                role=ModelRoleType.researcher,
                provider=LLMProviderType.vllm,
                model_name=vllm_model_path,
                base_url=routing_url,
                temperature=0.,
                max_tokens=1024,
            ),
            structured=OpenAISettings(
                role=ModelRoleType.researcher,
                provider=LLMProviderType.vllm,
                model_name=vllm_model_path,
                base_url=routing_url,
                temperature=0.,
                max_tokens=2048,
            ),
        ),
        analyst=OpenAISettings(
            role=ModelRoleType.analyst,
            provider=LLMProviderType.vllm,
            model_name=vllm_model_path,
            base_url=generation_url,
            temperature=0.15,
            max_tokens=4096,
            prompt_version="v2",
        ),
        mentor=OpenAISettings(
            role=ModelRoleType.mentor,
            provider=LLMProviderType.vllm,
            model_name=vllm_model_path,
            base_url=generation_url,
            temperature=0.7,
            max_tokens=2048,
            prompt_version="v2",
        ),
    )


_PIPELINES = {"ollama": _ollama_models, "vllm": _vllm_models}


def get_model_settings() -> ModelSettings:
    """Returns ModelSettings for the active LLM pipeline.

    Reads the LLM_PIPELINE environment variable (default: 'ollama').
    Supported values: 'ollama', 'vllm'.
    """
    pipeline = os.environ.get("LLM_PIPELINE", "ollama")
    if pipeline not in _PIPELINES:
        raise ValueError(f"Unknown LLM_PIPELINE '{pipeline}'. Must be one of: {list(_PIPELINES)}")
    return _PIPELINES[pipeline]()
