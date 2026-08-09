"""Typed, content-free errors that are safe to surface in chat."""

from backend.configs.messages import (
    ERROR_AI_REQUEST_REJECTED,
    ERROR_AI_RESPONSE_INVALID,
    ERROR_AI_SERVICE_CAPACITY,
    ERROR_AI_SERVICE_CONFIGURATION,
    ERROR_AI_SERVICE_UNAVAILABLE,
    ERROR_KNOWLEDGE_BASE_UNAVAILABLE,
    ERROR_WEB_SEARCH_CAPACITY,
    ERROR_WEB_SEARCH_UNAVAILABLE,
    ERROR_WORKFLOW_INITIALIZATION,
)


class UserFacingChatError(RuntimeError):
    """Base exception for a chat failure with a sanitized public message."""

    user_message = ""
    reason = "chat_error"

    def __init__(self, user_message: str | None = None) -> None:
        super().__init__(user_message or self.user_message)
        self.user_message = user_message or self.user_message


class WorkflowInitializationUnavailable(UserFacingChatError):
    user_message = ERROR_WORKFLOW_INITIALIZATION
    reason = "workflow_initialization_unavailable"


class AIServiceUnavailable(UserFacingChatError):
    user_message = ERROR_AI_SERVICE_UNAVAILABLE
    reason = "ai_service_unavailable"


class AIServiceCapacity(UserFacingChatError):
    user_message = ERROR_AI_SERVICE_CAPACITY
    reason = "ai_service_capacity"


class AIServiceConfiguration(UserFacingChatError):
    user_message = ERROR_AI_SERVICE_CONFIGURATION
    reason = "ai_service_configuration"


class AIRequestRejected(UserFacingChatError):
    user_message = ERROR_AI_REQUEST_REJECTED
    reason = "ai_request_rejected"


class AIResponseInvalid(UserFacingChatError):
    user_message = ERROR_AI_RESPONSE_INVALID
    reason = "ai_response_invalid"


class KnowledgeBaseUnavailable(UserFacingChatError):
    user_message = ERROR_KNOWLEDGE_BASE_UNAVAILABLE
    reason = "knowledge_base_unavailable"


class WebSearchUnavailable(UserFacingChatError):
    user_message = ERROR_WEB_SEARCH_UNAVAILABLE
    reason = "web_search_unavailable"


class WebSearchCapacity(UserFacingChatError):
    user_message = ERROR_WEB_SEARCH_CAPACITY
    reason = "web_search_capacity"
