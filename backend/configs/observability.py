import os

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.configs.constants import ENV_PATH


class LangSmithSettings(BaseSettings):
    """Reads LangSmith configuration from the .env file."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    langsmith_tracing: str = "false"
    langsmith_api_key: SecretStr | str = ""
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_project: str = ""

    def configure(self) -> None:
        """Pushes LangSmith settings into os.environ so the SDK can pick them up."""
        if self.langsmith_tracing.lower() != "true" or not self.langsmith_api_key:
            return

        _set_if_absent("LANGSMITH_TRACING", self.langsmith_tracing)
        _set_if_absent("LANGSMITH_API_KEY", self.langsmith_api_key)
        _set_if_absent("LANGSMITH_ENDPOINT", self.langsmith_endpoint)
        if self.langsmith_project:
            _set_if_absent("LANGSMITH_PROJECT", self.langsmith_project)


def _set_if_absent(key: str, value: str) -> None:
    """Sets an environment variable only if it is not already present."""
    if key not in os.environ:
        os.environ[key] = value
