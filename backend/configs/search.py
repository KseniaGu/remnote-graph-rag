from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.configs.constants import ENV_PATH


class TavilySettings(BaseSettings):
    """Tavily search engine settings configuration."""
    api_key: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="TAVILY_",
    )


class KnowledgeGraphSearchSettings(BaseSettings):
    """Knowledge graph search settings configuration."""
    retriever_params: dict = {
        "VectorContextRetriever": {
            "include_text": True, "similarity_top_k": 5, "similarity_score": None, "depth": 2,
            "include_properties": True
        }
    }
    visualizer_retriever_params: dict = {
        "VectorContextRetriever": {
            "include_text": False, "similarity_top_k": 5, "similarity_score": None, "depth": 4,
            "include_properties": False
        },
        # "VectorIndexRetriever": {"similarity_top_k": 10,}
    }
