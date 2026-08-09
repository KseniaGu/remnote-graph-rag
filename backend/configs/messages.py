"""User-facing message strings emitted by the backend workflow.

All text that surfaces to the user via WorkflowEvent responses or errors is
defined here. Update this file to change any system-generated response text.
"""

# ---------------------------------------------------------------------------
# Fallback responses (no content produced by analyst / mentor)
# ---------------------------------------------------------------------------
FALLBACK_ALL_SOURCES_EXHAUSTED = (
    "I couldn't find information about this topic in your knowledge graph or on the web. "
    "Try rephrasing your question or adding notes on this topic first."
)

FALLBACK_VISUALIZATION_FAILED = (
    "I wasn't able to create a visualization for this topic. "
    "The knowledge graph may not contain enough data about it. "
    "Try adding notes on this topic first."
)

FALLBACK_NO_RESULTS = (
    "I couldn't find any information about this topic in your knowledge graph. "
    "Try adding notes on this topic first, or ask me to research it on the web."
)

FALLBACK_DEFAULT = (
    "I don't have enough information to respond to that right now. "
    "Try rephrasing your question or asking me to search the web for more details."
)

# ---------------------------------------------------------------------------
# Workflow error messages
# ---------------------------------------------------------------------------
ERROR_RECURSION_LIMIT = (
    "The workflow exceeded the maximum number of steps. Please try a simpler query."
)

ERROR_MESSAGE_TOO_LONG = (
    "Your message is too long. Please keep it under 4,000 characters."
)
ERROR_CHAT_COOLDOWN = "Please wait a few seconds before sending another message."
ERROR_VISITOR_DAILY_LIMIT = (
    "You have reached the demo's daily message limit. Please try again after 00:00 UTC."
)
ERROR_VISITOR_ACTIVE_TURN = (
    "Another message is already being processed for this browser."
)
ERROR_GLOBAL_CHAT_BUSY = (
    "The demo is currently busy. Please wait a moment and try again."
)
ERROR_OLLAMA_DAILY_LIMIT = "The demo's daily AI usage limit has been reached. Please try again after 00:00 UTC."
ERROR_TAVILY_DAILY_LIMIT = (
    "The demo's web-research limit has been reached. Please try again later."
)
ERROR_WORKFLOW_BUDGET = (
    "This request required too many processing steps. Please try a simpler question."
)
ERROR_WORKFLOW_TIMEOUT = (
    "This request took too long to complete. Please try a narrower question."
)
ERROR_QUOTA_STORAGE_UNAVAILABLE = (
    "The demo is temporarily unavailable because usage limits cannot be verified. "
    "Please try again shortly."
)
ERROR_WORKFLOW_FAILED = (
    "I encountered an error while processing your request. Please try again."
)

# External runtime dependencies. These messages deliberately describe only the
# user-actionable boundary; detailed provider and infrastructure errors remain
# in server logs.
ERROR_WORKFLOW_INITIALIZATION = (
    "The assistant is temporarily unavailable because a required service or model "
    "could not be loaded. Please try again later."
)
ERROR_AI_SERVICE_UNAVAILABLE = (
    "The AI service did not respond. Please try again in a moment."
)
ERROR_AI_SERVICE_CAPACITY = (
    "The AI service is temporarily at capacity. Please try again later."
)
ERROR_AI_SERVICE_CONFIGURATION = (
    "The AI service is temporarily unavailable because of a service configuration "
    "problem. Please try again later."
)
ERROR_AI_REQUEST_REJECTED = (
    "The AI service could not process this request. Please rephrase it and try again."
)
ERROR_AI_RESPONSE_INVALID = (
    "The AI service returned a response that could not be processed. Please try again."
)
ERROR_KNOWLEDGE_BASE_UNAVAILABLE = (
    "I couldn't access the knowledge base right now. Please try again shortly."
)
ERROR_WEB_SEARCH_UNAVAILABLE = (
    "Web search is temporarily unavailable. Please try again shortly or continue "
    "without web research."
)
ERROR_WEB_SEARCH_CAPACITY = (
    "Web search is temporarily at capacity. Please try again later or continue "
    "without web research."
)
