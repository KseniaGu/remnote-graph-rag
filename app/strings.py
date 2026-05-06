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
    ("book-open-text", "Research a topic", "Research the latest developments in LLM fine-tuning"),
    ("network", "Map relationships", "Visualize my knowledge about backpropagation algorithm"),
]

# Cache policy for quick actions. The prompt field must match the third item in
# QUICK_ACTIONS; aliases let future buttons or wording variants share one cache key.
QUICK_ACTION_CACHE_POLICIES: list[dict[str, object]] = [
    {
        "prompt": "Quiz me on Transformer architecture",
        "enabled": True,
        "ttl_seconds": 7 * 24 * 60 * 60,
        "cache_responses": True,
        "cache_visual_artifacts": False,
        "aliases": [],
    },
    {
        "prompt": "What information do I have about attention mechanisms?",
        "enabled": True,
        "ttl_seconds": 7 * 24 * 60 * 60,
        "cache_responses": True,
        "cache_visual_artifacts": False,
        "aliases": [],
    },
    {
        "prompt": "Research the latest developments in LLM fine-tuning",
        "enabled": True,
        "ttl_seconds": 2 * 24 * 60 * 60,
        "cache_responses": True,
        "cache_visual_artifacts": False,
        "aliases": [],
    },
    {
        "prompt": "Visualize my knowledge about neural networks",
        "enabled": True,
        "ttl_seconds": 7 * 24 * 60 * 60,
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
