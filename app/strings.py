"""Frontend UI text constants.

All user-visible strings rendered in the application interface are defined here.
Update this file to change any label, placeholder, header, or tooltip text.
"""

# ---------------------------------------------------------------------------
# App identity
# ---------------------------------------------------------------------------
APP_NAME = "AI Practice"
APP_PAGE_TITLE = "AI Practice | Graph RAG"
APP_TAGLINE = "Study from your personal knowledge graph"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
SIDEBAR_PIPELINE_HEADER = "Agent Pipeline"
SIDEBAR_BTN_CLEAR_CHAT = "Clear Session"
SIDEBAR_BTN_TOGGLE_GRAPH = "Toggle Graph"

# ---------------------------------------------------------------------------
# Workspaces
# ---------------------------------------------------------------------------
WORKSPACE_STUDY_LABEL = "Study Session"
WORKSPACE_STUDY_SUBTITLE = "Ask, retrieve, synthesize, and practice from your notes"
WORKSPACE_GRAPH_LABEL = "Knowledge Graph"
WORKSPACE_GRAPH_SUBTITLE = "Explore relationships discovered from your study session"

# ---------------------------------------------------------------------------
# Agent tooltip descriptions (keyed by agent name as used in the workflow)
# ---------------------------------------------------------------------------
AGENT_DESCRIPTIONS: dict[str, str] = {
    "orchestrator": "Routes your question to the right agent and decides the workflow path",
    "retriever": "Searches your personal knowledge base for relevant notes and concepts",
    "researcher": "Searches the web to supplement or verify knowledge base information",
    "analyst": "Synthesizes retrieved findings into a structured, detailed response",
    "mentor": "Guides you through Socratic interview practice using your knowledge base",
    "visualizer": "Creates an interactive knowledge graph visualization from retrieved data",
}

# ---------------------------------------------------------------------------
# Chat area
# ---------------------------------------------------------------------------
CHAT_EMPTY_HEADING = "Start from a concept"
CHAT_EMPTY_SUBTEXT = "Ask a question, request interview practice, or map relationships from your notes."
CHAT_INPUT_PLACEHOLDER = "Ask your knowledge base..."
CHAT_PROCESSING_LABEL = "Processing"

# ---------------------------------------------------------------------------
# Quick action buttons: list of (icon_name, button_label, prefilled_action)
# ---------------------------------------------------------------------------
QUICK_ACTIONS: list[tuple[str, str, str]] = [
    ("graduation-cap", "Practice a concept", "Quiz me on Transformer architecture"),
    ("search", "Search my notes", "What information do I have about attention mechanisms?"),
    ("book-open-text", "Research a topic", "Research the latest developments in fine-tuning LLMs"),
    ("network", "Map relationships", "Visualize my knowledge about backpropagation algorithm"),
]


def _quick_action_prompt(label: str) -> str:
    """Returns the prefilled prompt for a configured quick-action label."""
    for _, quick_action_label, prompt in QUICK_ACTIONS:
        if quick_action_label == label:
            return prompt
    raise ValueError(f"Unknown quick action label: {label}")


# Cache policy for quick actions. Prompt values are derived from QUICK_ACTIONS;
# aliases let future buttons or wording variants share one cache key.
QUICK_ACTION_CACHE_POLICIES: list[dict[str, object]] = [
    {
        "prompt": _quick_action_prompt("Practice a concept"),
        "enabled": True,
        "ttl_seconds": 14 * 24 * 60 * 60,
        "cache_responses": True,
        "cache_visual_artifacts": False,
        "aliases": [],
    },
    {
        "prompt": _quick_action_prompt("Search my notes"),
        "enabled": True,
        "ttl_seconds": 14 * 24 * 60 * 60,
        "cache_responses": True,
        "cache_visual_artifacts": False,
        "aliases": [],
    },
    {
        "prompt": _quick_action_prompt("Research a topic"),
        "enabled": True,
        "ttl_seconds": 14 * 24 * 60 * 60,
        "cache_responses": True,
        "cache_visual_artifacts": False,
        "aliases": [],
    },
    {
        "prompt": _quick_action_prompt("Map relationships"),
        "enabled": True,
        "ttl_seconds": 14 * 24 * 60 * 60,
        "cache_responses": True,
        "cache_visual_artifacts": True,
        "aliases": [],
    },
]

# ---------------------------------------------------------------------------
# Visualization panel
# ---------------------------------------------------------------------------
VIZ_PANEL_TITLE = "Knowledge Graph"
GRAPH_UPDATED_NOTICE = "Knowledge graph updated from this session"
GRAPH_UPDATED_ACTION = "View graph"
GRAPH_UNAVAILABLE_HEADING = "Knowledge graph unavailable"
GRAPH_LOADING_HEADING = "Preparing knowledge graph"
GRAPH_LOADING_SUBTEXT = "Searching graph data and arranging the visualization."
GRAPH_EMPTY_HEADING = "No knowledge graph yet"
GRAPH_EMPTY_SUBTEXT = "Ask to map relationships from your notes, then the graph will appear here."

# ---------------------------------------------------------------------------
# Context debug panel
# ---------------------------------------------------------------------------
CONTEXT_PANEL_TITLE = "Context"
