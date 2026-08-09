from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.configs.constants import ENV_PATH


class ChatLimitsSettings(BaseSettings):
    """Configuration for anonymous chat admission and provider budgets."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="CHAT_LIMITS_",
    )

    shared_quotas_enabled: bool = False
    message_max_chars: int = 4000
    visitor_turns_per_day: int = 50
    visitor_cooldown_seconds: float = 10.0
    global_active_workflows: int = 2
    global_queue_wait_seconds: float = 5.0
    global_queue_poll_seconds: float = 0.25
    workflow_timeout_seconds: float = 180.0
    workflow_lease_seconds: float = 240.0

    max_logical_llm_calls_per_turn: int = 5
    max_llm_attempts_per_turn: int = 6
    max_retries_per_turn: int = 1
    ollama_attempts_per_day: int = 600
    ollama_warning_attempts: int = 450
    ollama_critical_warning_attempts: int = 540
    ollama_request_timeout_seconds: float = 120.0

    tavily_attempts_per_day: int = 30
    tavily_attempts_per_month: int = 500
    tavily_searches_per_turn: int = 1
    tavily_topic_max_chars: int = 256
    tavily_max_results: int = 5
    tavily_result_max_chars: int = 3000
    tavily_context_max_chars: int = 20_000
    tavily_timeout_seconds: float = 60.0

    retrieval_context_max_chars: int = 20_000
    usage_event_ttl_days: int = 35
    state_collection_name: str = "chat_limit_state"
    usage_collection_name: str = "chat_usage_events"

    @model_validator(mode="after")
    def validate_limits(self) -> "ChatLimitsSettings":
        """Validates relationships between independently configurable limits."""
        positive_values = {
            "message_max_chars": self.message_max_chars,
            "visitor_turns_per_day": self.visitor_turns_per_day,
            "visitor_cooldown_seconds": self.visitor_cooldown_seconds,
            "global_active_workflows": self.global_active_workflows,
            "global_queue_poll_seconds": self.global_queue_poll_seconds,
            "workflow_timeout_seconds": self.workflow_timeout_seconds,
            "workflow_lease_seconds": self.workflow_lease_seconds,
            "max_logical_llm_calls_per_turn": self.max_logical_llm_calls_per_turn,
            "max_llm_attempts_per_turn": self.max_llm_attempts_per_turn,
            "ollama_attempts_per_day": self.ollama_attempts_per_day,
            "tavily_attempts_per_day": self.tavily_attempts_per_day,
            "tavily_attempts_per_month": self.tavily_attempts_per_month,
            "tavily_searches_per_turn": self.tavily_searches_per_turn,
            "tavily_topic_max_chars": self.tavily_topic_max_chars,
            "tavily_max_results": self.tavily_max_results,
            "tavily_result_max_chars": self.tavily_result_max_chars,
            "tavily_context_max_chars": self.tavily_context_max_chars,
            "retrieval_context_max_chars": self.retrieval_context_max_chars,
            "usage_event_ttl_days": self.usage_event_ttl_days,
        }
        invalid = [name for name, value in positive_values.items() if value <= 0]
        if invalid:
            raise ValueError(f"chat limits must be positive: {', '.join(invalid)}")
        if self.global_queue_wait_seconds < 0:
            raise ValueError("global_queue_wait_seconds cannot be negative")
        if self.max_retries_per_turn < 0:
            raise ValueError("max_retries_per_turn cannot be negative")
        if self.workflow_lease_seconds <= self.workflow_timeout_seconds:
            raise ValueError(
                "workflow_lease_seconds must exceed workflow_timeout_seconds"
            )
        if self.max_llm_attempts_per_turn < self.max_logical_llm_calls_per_turn:
            raise ValueError(
                "max_llm_attempts_per_turn cannot be below the logical-call limit"
            )
        if not (
            0
            < self.ollama_warning_attempts
            < self.ollama_critical_warning_attempts
            < self.ollama_attempts_per_day
        ):
            raise ValueError(
                "Ollama warning thresholds must be ordered below the daily limit"
            )
        return self
