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
