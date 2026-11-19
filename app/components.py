import reflex as rx

from app.practice_components import MermaidDiagram
from app.state import (
    CHAT_MESSAGE_MAX_CHARS,
    AppState,
    LiveAgent,
    RecentMapItem,
    SessionHistoryItem,
    StudySource,
    StudyStage,
    StudyTurn,
)
from app.strings import (
    APP_NAME,
    CHAT_EMPTY_HEADING,
    CHAT_EMPTY_HEADING_SECOND_LINE,
    CHAT_EMPTY_SUBTEXT,
    CHAT_FOLLOW_UP_PLACEHOLDER,
    CHAT_INPUT_PLACEHOLDER,
    CHAT_PROCESSING_LABEL,
    CHAT_PROCESSING_PLACEHOLDER,
    CONTEXT_PANEL_TITLE,
    GRAPH_EMPTY_HEADING,
    GRAPH_EMPTY_SUBTEXT,
    GRAPH_LOADING_HEADING,
    GRAPH_LOADING_SUBTEXT,
    GRAPH_UNAVAILABLE_HEADING,
    QUICK_ACTIONS,
    STUDY_INTENTS,
)
from app.styles import (
    CARD_STYLE,
    COLORS,
)
from app.ui.knowledge_graph import KnowledgeGraph, SourceList
from backend.workflows.presentation import Message


def surface(*children, style: dict | None = None, **props) -> rx.Component:
    """Reusable framed surface for existing panel-style UI."""
    return rx.box(
        *children,
        style={
            **CARD_STYLE,
            **(style or {}),
        },
        **props,
    )


def section_header(title: str, icon_name: str | None = None, **props) -> rx.Component:
    """Compact icon + label header used across panels."""
    text_props = {
        "font_size": props.pop("font_size", "0.875rem"),
        "font_weight": props.pop("font_weight", "600"),
        "color": props.pop("color", COLORS["text_primary"]),
    }
    text_transform = props.pop("text_transform", None)
    letter_spacing = props.pop("letter_spacing", None)
    if text_transform is not None:
        text_props["text_transform"] = text_transform
    if letter_spacing is not None:
        text_props["letter_spacing"] = letter_spacing

    children = []
    if icon_name is not None:
        children.append(rx.icon(icon_name, size=18, color=COLORS["accent_blue"]))
    children.append(rx.text(title, **text_props))

    return rx.hstack(
        *children,
        spacing="2",
        align="center",
        **props,
    )


def workflow_status_label():
    """Human-readable label for the active workflow stage."""
    return rx.cond(
        AppState.active_agent == "retriever",
        "Searching your notes",
        rx.cond(
            AppState.active_agent == "researcher",
            "Researching the web",
            rx.cond(
                AppState.active_agent == "analyst",
                "Drafting a technical answer",
                rx.cond(
                    AppState.active_agent == "mentor",
                    "Preparing interview guidance",
                    rx.cond(
                        AppState.active_agent == "visualizer",
                        "Rendering the knowledge graph",
                        "Preparing response",
                    ),
                ),
            ),
        ),
    )


def state_notice(
    icon_name: str,
    title,
    detail,
    tone: str = "",
    animated: bool = False,
) -> rx.Component:
    """Compact empty/loading/error state block."""
    icon_props = {"class_name": "animate-spin"} if animated else {}
    return rx.vstack(
        rx.icon(icon_name, size=28, **icon_props),
        rx.el.h2(title, class_name="state-notice-title"),
        rx.text(detail, class_name="state-notice-detail"),
        class_name="state-notice",
        data_tone=tone,
        spacing="2",
        align="center",
        width="100%",
    )


def submission_error_notice() -> rx.Component:
    """Inline failed-submission state near the composer."""
    return rx.cond(
        AppState.error_message != "",
        surface(
            rx.hstack(
                rx.icon("circle-alert", size=16, color=COLORS["accent_red"]),
                rx.vstack(
                    rx.text("Submission failed", class_name="submission-error-title"),
                    rx.text(
                        AppState.error_message, class_name="submission-error-detail"
                    ),
                    spacing="0",
                    align="start",
                    min_width="0",
                    flex="1",
                    role="alert",
                ),
                rx.icon_button(
                    rx.icon("x", size=14),
                    on_click=AppState.clear_error,
                    size="1",
                    variant="ghost",
                    class_name="submission-error-close",
                    aria_label="Dismiss error",
                ),
                width="100%",
                spacing="3",
                align="start",
            ),
            class_name="submission-error",
            style={
                "border": f"1px solid {COLORS['accent_red']}50",
                "padding": "0.75rem 0.85rem",
                "width": "100%",
            },
        ),
        rx.fragment(),
    )


def quick_action_button(icon_name: str, label: str, action: str) -> rx.Component:
    """Quick action suggestion button."""
    return rx.el.li(
        rx.el.button(
            rx.el.span(rx.icon(icon_name, size=14), label, class_name="prompt-intent"),
            rx.el.span(action),
            rx.icon("arrow-right", size=14),
            type="button",
            disabled=AppState.is_processing,
            data_draft=action,
        )
    )


def practice_brand() -> rx.Component:
    return rx.el.button(
        rx.html("""
<svg class="brand-mark" viewBox="0 0 24 24" aria-hidden="true">
  <path d="m7 8 4 7m6-7-4 7"/><circle cx="6" cy="6" r="3"/>
  <path d="m18 3 3 3-3 3-3-3Z"/><rect class="hub" x="8.5" y="15" width="7" height="7" rx="1"/>
</svg>
"""),
        rx.el.span(APP_NAME),
        type="button",
        on_click=AppState.open_chat,
        aria_label="AI Practice study workspace",
        class_name="practice-brand",
    )


def practice_theme_button(class_name: str = "") -> rx.Component:
    return rx.el.button(
        rx.el.span(rx.icon("sun", size=16), class_name="theme-sun"),
        rx.el.span(rx.icon("moon", size=16), class_name="theme-moon"),
        type="button",
        aria_label="Switch to light theme",
        data_practice_action="theme",
        class_name=f"practice-icon theme-toggle {class_name}".strip(),
    )


def empty_session_intro() -> rx.Component:
    """Render the approved Welcome heading and illustrative connection."""
    return rx.el.div(
        rx.el.div(
            practice_brand(),
            rx.el.h1(
                CHAT_EMPTY_HEADING + " ", rx.el.br(), CHAT_EMPTY_HEADING_SECOND_LINE
            ),
            rx.el.p(CHAT_EMPTY_SUBTEXT),
            class_name="welcome-copy",
        ),
        rx.html(
            """
<svg viewBox="0 0 440 290" aria-hidden="true">
  <path class="diagram-arc" d="M40 100Q170 0 350 110M60 230Q240 300 380 80"/>
  <path pathLength="100" class="diagram-line" d="M80 65C150 65 150 155 220 155S290 235 365 235"/>
  <circle class="diagram-halo" cx="220" cy="155" r="62"/>
  <circle class="diagram-halo outer" cx="220" cy="155" r="98"/>
  <circle class="diagram-dot" cx="80" cy="65" r="5"/>
  <rect class="diagram-center" x="211" y="146" width="18" height="18" rx="3"/>
  <rect class="diagram-dot" x="360" y="230" width="10" height="10" rx="1"/>
</svg>
<span class="diagram-label diagram-gqa">Grouped-query attention<small>shrinks</small></span>
<span class="diagram-label diagram-kv">KV cache</span>
<span class="diagram-label diagram-paged"><small>allocated in blocks by</small>PagedAttention</span>
""",
            class_name="welcome-diagram",
            role="img",
            aria_label="Illustration of grouped-query attention, KV cache and PagedAttention",
        ),
        class_name="welcome-intro",
    )


def starter_prompt_row() -> rx.Component:
    """Starter prompts attached to the empty-state composer."""
    return rx.el.ul(
        *[
            quick_action_button(icon, intent, action)
            for (icon, _, action), intent in zip(
                QUICK_ACTIONS, ("Practice", "Explain", "Research", "Map"), strict=True
            )
        ],
        class_name="sample-prompts",
        aria_label="Example questions",
    )


def workspace_tabs() -> rx.Component:
    """Top-level workspace navigation."""
    return rx.el.nav(
        rx.el.button(
            "Study",
            on_click=AppState.open_chat,
            aria_current=rx.cond(AppState.active_view == "chat", "page", "false"),
            data_workspace="study",
        ),
        rx.el.button(
            "Graph",
            on_click=AppState.open_graph,
            aria_current=rx.cond(AppState.active_view == "graph", "page", "false"),
            data_workspace="graph",
        ),
        aria_label="Workspace",
        class_name="practice-tabs",
    )


def sidebar_empty_item(label: str) -> rx.Component:
    """Quiet empty state for tab-scoped sidebar history sections."""
    return rx.box(
        rx.text(label, class_name="sidebar-history-meta"),
        class_name="sidebar-history-item",
        style={
            "color": COLORS["text_muted"],
            "min_height": "52px",
        },
    )


def sidebar_session_item(item: SessionHistoryItem) -> rx.Component:
    """Clickable current-tab session history item."""
    return rx.button(
        rx.hstack(
            rx.icon("message-square", size=14, class_name="history-item-icon"),
            rx.vstack(
                rx.text(item.title, class_name="sidebar-history-title"),
                rx.text(item.timestamp, class_name="sidebar-history-meta"),
                spacing="1",
                align="start",
                min_width="0",
            ),
            align="start",
            width="100%",
            min_width="0",
            class_name="sidebar-history-row",
        ),
        on_click=lambda: AppState.select_session_history_item(item.turn_id),
        data_practice_action="history-link",
        data_history_target=item.message_dom_id,
        class_name="sidebar-history-item",
    )


def sidebar_map_item(item: RecentMapItem) -> rx.Component:
    """Clickable current-tab recent map item."""
    return rx.button(
        rx.hstack(
            rx.icon("network", size=14, class_name="history-item-icon"),
            rx.vstack(
                rx.text(item.title, class_name="sidebar-history-title"),
                rx.text(item.timestamp, class_name="sidebar-history-meta"),
                spacing="1",
                align="start",
                min_width="0",
            ),
            align="start",
            width="100%",
            min_width="0",
            class_name="sidebar-history-row",
        ),
        on_click=lambda: AppState.select_recent_map(item.artifact_id),
        data_practice_action="history-link",
        class_name=rx.cond(
            AppState.selected_recent_map_id == item.artifact_id,
            "sidebar-history-item sidebar-history-item-active",
            "sidebar-history-item",
        ),
    )


def sidebar_session_section() -> rx.Component:
    """Current browser-tab session history panel."""
    return rx.vstack(
        rx.cond(
            AppState.has_session_history,
            rx.vstack(
                rx.foreach(AppState.session_history, sidebar_session_item),
                class_name="sidebar-history-list",
                spacing="0",
                width="100%",
            ),
            sidebar_empty_item("No session activity yet"),
        ),
        class_name="sidebar-section",
        spacing="3",
        width="100%",
    )


def sidebar_maps_section() -> rx.Component:
    """Current browser-tab recent maps panel."""
    return rx.vstack(
        rx.el.h3("Maps in this session", class_name="sidebar-section-title"),
        rx.cond(
            AppState.has_recent_maps,
            rx.vstack(
                rx.foreach(AppState.recent_maps, sidebar_map_item),
                class_name="sidebar-history-list",
                spacing="0",
                width="100%",
            ),
            sidebar_empty_item("No maps yet"),
        ),
        class_name="sidebar-section",
        spacing="3",
        width="100%",
    )


def sidebar_history_panels() -> rx.Component:
    """Tab-scoped sidebar history panels."""
    return rx.el.div(
        sidebar_session_section(),
        rx.cond(AppState.has_recent_maps, sidebar_maps_section()),
        class_name="history-items",
    )


def response_article(message: Message, turn: StudyTurn, last) -> rx.Component:
    """Render one finalized response and its own source action."""
    return rx.el.article(
        reading_header(message.agent, message.timestamp),
        reading_markdown(message.content),
        rx.cond(last, rx.foreach(turn.maps, map_result)),
        rx.el.div(
            rx.el.button(
                rx.icon("copy", size=14),
                "Copy answer",
                type="button",
                data_practice_action="copy-answer",
            ),
            rx.el.button(
                rx.icon("panel-right", size=14),
                "Sources & concepts",
                type="button",
                on_click=AppState.select_response_sources(message.response_id),
                aria_pressed="false",
                data_practice_action="open-sources",
                data_source_selected=rx.cond(
                    AppState.study_sources.key == message.response_id, "true", "false"
                ),
            ),
            rx.cond(last & (turn.outcome != "running"), execution_trigger(turn)),
            class_name="answer-actions",
        ),
        rx.cond(last & (turn.outcome != "running"), execution_details(turn)),
        class_name="assistant-message",
        id=message.dom_id,
        key=message.dom_id,
        data_response_id=message.response_id,
    )


def execution_stage(stage: StudyStage, index) -> rx.Component:
    return rx.el.li(
        rx.el.span(index + 1, class_name="stage-number", aria_hidden="true"),
        rx.el.div(
            rx.el.strong(stage.agent),
            rx.el.span(
                rx.match(
                    stage.outcome,
                    ("running", "Running"),
                    ("completed", "Completed"),
                    ("failed", "Failed"),
                    ("interrupted", "Interrupted"),
                    ("cancelled", "Cancelled"),
                    "Incomplete",
                ),
                class_name="stage-status",
            ),
            class_name="stage-label",
        ),
        rx.cond(
            stage.duration != "",
            rx.el.span(stage.duration, class_name="stage-duration"),
        ),
        data_outcome=stage.outcome,
    )


def execution_trigger(turn: StudyTurn) -> rx.Component:
    return rx.el.button(
        "Execution details",
        rx.icon("chevron-right", size=14, class_name="disclosure-chevron"),
        type="button",
        aria_expanded="false",
        aria_controls="execution-" + turn.prompt.dom_id,
        data_practice_action="execution",
        class_name="execution-trigger",
    )


def execution_details(turn: StudyTurn) -> rx.Component:
    outcome = rx.match(
        turn.outcome,
        ("answer", "Answered"),
        ("graph", "Map ready"),
        ("out_of_scope", "Outside study scope"),
        ("no_results", "No results"),
        ("empty", "No response"),
        ("error", "Failed"),
        ("interrupted", "Interrupted"),
        ("cancelled", "Cancelled"),
        ("incomplete", "Incomplete"),
        "Unavailable",
    )
    return rx.el.div(
        rx.el.div(
            rx.el.span(outcome, class_name="execution-badge"),
            rx.cond(
                turn.origin == "cache",
                rx.el.span("Cached", class_name="execution-origin"),
            ),
            class_name="execution-status",
        ),
        rx.el.div(
            rx.el.h3("Observed agent invocations"),
            rx.cond(
                turn.stages.length() > 0,
                rx.el.ol(
                    rx.foreach(turn.stages, execution_stage),
                    class_name="agent-stages",
                    role="list",
                ),
                rx.el.p(
                    rx.match(
                        turn.origin,
                        ("cache", "Replayed from cache. No agents ran for this turn."),
                        ("workflow", "No agent invocations were recorded."),
                        "Execution details are unavailable for this turn.",
                    ),
                    class_name="panel-note",
                ),
            ),
            class_name="execution-body",
        ),
        id="execution-" + turn.prompt.dom_id,
        hidden=True,
        role="region",
        aria_label="Execution details",
        data_outcome=turn.outcome,
        class_name="execution",
    )


def map_result(item: RecentMapItem) -> rx.Component:
    link = rx.el.button(
        rx.icon("network", size=15, aria_hidden=True),
        rx.el.span(item.title),
        rx.icon("arrow-up-right", size=14, aria_hidden=True),
        aria_label="Open map: " + item.title,
        type="button",
        on_click=AppState.select_recent_map(item.artifact_id),
    )
    return rx.el.div(
        rx.cond(
            item.retrieval_text != "",
            rx.tooltip(
                link,
                content=item.retrieval_text,
                class_name="map-request-tooltip",
                align="start",
                collision_padding=12,
                hide_when_detached=True,
            ),
            link,
        ),
        rx.cond(item.timestamp != "", rx.el.time(item.timestamp)),
        class_name="map-result",
        key=item.artifact_id,
    )


def study_turn(turn: StudyTurn) -> rx.Component:
    prompt = turn.prompt
    return rx.el.div(
        rx.el.div(
            rx.el.span("You: ", class_name="sr-only"),
            reading_markdown(prompt.content, class_name="user-content"),
            class_name="user-message",
            id=prompt.dom_id,
            tab_index=-1,
        ),
        rx.foreach(
            turn.responses,
            lambda message, index: response_article(
                message, turn, index == turn.responses.length() - 1
            ),
        ),
        rx.cond(
            turn.responses.length() == 0,
            rx.fragment(
                rx.foreach(turn.maps, map_result),
                rx.cond(
                    turn.outcome != "running",
                    rx.fragment(
                        rx.el.div(execution_trigger(turn), class_name="answer-actions"),
                        execution_details(turn),
                    ),
                ),
            ),
        ),
        rx.cond(
            turn.outcome == "running",
            streaming_bubble(turn.origin),
        ),
        key=prompt.dom_id,
        class_name="study-turn",
    )


def reading_header(agent, detail) -> rx.Component:
    return rx.el.div(
        rx.el.span(
            rx.icon("book-open-text", size=15),
            rx.match(
                agent,
                ("analyst", "Explanation"),
                ("mentor", "Practice guidance"),
                ("researcher", "Research"),
                ("retriever", "Retrieved context"),
                "Response",
            ),
            class_name="answer-kind",
        ),
        rx.el.span(detail),
        class_name="answer-meta",
    )


def reading_code(text, language="python", **props) -> rx.Component:
    return rx.cond(
        language == "mermaid",
        MermaidDiagram.create(source=text),
        rx.el.div(
            rx.el.div(
                rx.el.span(rx.cond(language != "", language, "Code")),
                rx.el.button(
                    rx.icon("copy", size=13),
                    "Copy",
                    type="button",
                    aria_label="Copy code",
                    data_practice_action="copy-code",
                ),
                class_name="code-heading",
            ),
            rx.code_block(
                text,
                language=language,
                theme=rx.code_block.themes.one_dark,
                wrap_long_lines=False,
                **props,
            ),
            class_name="code-block",
        ),
    )


def reading_markdown(
    content, *, class_name: str = "markdown-content answer-body"
) -> rx.Component:
    return rx.markdown(
        content,
        class_name=class_name,
        component_map={
            "h1": rx.el.h1,
            "h2": rx.el.h2,
            "h3": rx.el.h3,
            "h4": rx.el.h4,
            "p": rx.el.p,
            "code": rx.el.code,
            "pre": reading_code,
            "table": lambda text: rx.el.div(
                rx.el.table(text),
                class_name="table-scroll",
                tab_index=0,
                role="region",
                aria_label="Table",
            ),
        },
    )


def streaming_bubble(origin) -> rx.Component:
    """Live streaming bubble that shows tokens as they arrive from analyst/mentor."""
    return rx.cond(
        AppState.is_streaming & (AppState.streaming_agent != "visualizer"),
        rx.el.article(
            reading_header(
                AppState.streaming_agent,
                rx.cond(origin == "cache", "Replaying…", "Writing…"),
            ),
            reading_markdown(AppState.streaming_content),
            class_name="assistant-message streaming-message",
            data_response_id=AppState.streaming_response_id,
        ),
        rx.fragment(),
    )


def chat_messages() -> rx.Component:
    """Chat messages container."""
    return rx.cond(
        AppState.has_messages | AppState.is_streaming | AppState.is_processing,
        rx.box(
            rx.el.div(
                rx.foreach(AppState.study_turns, study_turn),
                class_name="thread-inner",
            ),
            id="chat-messages",
            flex="1",
            overflow_y="auto",
            width="100%",
            tab_index=0,
            role="region",
            aria_label="Conversation",
            class_name="practice-thread",
        ),
        rx.fragment(),
    )


def live_agent(agent: LiveAgent) -> rx.Component:
    return rx.el.li(
        rx.el.span(
            rx.match(
                agent.outcome,
                ("completed", rx.icon("check", size=11)),
                ("failed", rx.icon("x", size=11)),
                ("interrupted", rx.icon("pause", size=11)),
                ("cancelled", rx.icon("minus", size=11)),
                ("incomplete", rx.icon("circle-alert", size=11)),
                rx.fragment(),
            ),
            class_name="agent-marker",
            aria_hidden=True,
        ),
        rx.el.span(agent.name),
        rx.el.span(
            rx.match(
                agent.outcome,
                ("running", "Active"),
                ("completed", "Completed"),
                ("failed", "Failed"),
                ("interrupted", "Interrupted"),
                ("cancelled", "Cancelled"),
                ("incomplete", "Incomplete"),
                "Inactive",
            ),
            class_name="sr-only",
        ),
        data_agent=agent.name.lower(),
        data_outcome=agent.outcome,
    )


def processing_indicator() -> rx.Component:
    """Show one fixed set of agent slots for the current request."""
    return rx.cond(
        AppState.is_processing,
        rx.el.div(
            rx.el.div(
                rx.icon(
                    "loader-circle",
                    size=14,
                    class_name="processing-spinner",
                    aria_hidden=True,
                ),
                rx.el.strong(CHAT_PROCESSING_LABEL),
                rx.el.span(
                    rx.cond(
                        AppState.processing_from_cache,
                        "Replaying a cached response",
                        workflow_status_label(),
                    ),
                    class_name="processing-detail",
                ),
                class_name="processing-status",
                role="status",
            ),
            rx.el.ul(
                rx.foreach(AppState.live_agent_progress, live_agent),
                class_name="live-agents",
                aria_label="Agent progress",
                role="list",
            ),
            class_name="composer-status-line",
        ),
    )


def composer_inner() -> rx.Component:
    """Aligned composer module for status, errors, input, and starter prompts."""
    label = rx.cond(
        AppState.has_messages, CHAT_FOLLOW_UP_PLACEHOLDER, CHAT_INPUT_PLACEHOLDER
    )
    return rx.form(
        rx.el.label(label, html_for="chat-input", class_name="sr-only"),
        rx.el.textarea(
            id="chat-input",
            name="message",
            placeholder=rx.cond(
                AppState.is_processing, CHAT_PROCESSING_PLACEHOLDER, label + "…"
            ),
            read_only=AppState.is_processing,
            max_length=CHAT_MESSAGE_MAX_CHARS,
            rows=1,
            required=True,
        ),
        rx.el.div(
            rx.cond(
                AppState.has_messages,
                rx.el.div(
                    *[
                        rx.el.button(
                            text,
                            type="button",
                            data_draft=draft,
                            disabled=AppState.is_processing,
                        )
                        for text, draft in STUDY_INTENTS
                    ],
                    class_name="intent-controls",
                    role="group",
                    aria_label="Start a question",
                ),
            ),
            rx.el.button(
                "Ask",
                rx.icon("arrow-up", size=16),
                id="chat-send-btn",
                type="submit",
                disabled=AppState.is_processing,
                class_name="practice-primary",
            ),
            class_name="composer-bottom",
        ),
        on_submit=AppState.handle_form_submit,
        reset_on_submit=True,
        class_name="practice-composer",
    )


def chat_input() -> rx.Component:
    """Chat input area."""
    return rx.vstack(
        rx.box(
            submission_error_notice(),
            processing_indicator(),
            composer_inner(),
            rx.cond(~AppState.has_messages, starter_prompt_row()),
            class_name="composer-content",
        ),
        spacing="3",
        width="100%",
        class_name="composer-dock",
    )


def graph_view() -> rx.Component:
    """The graph host stays mounted while its workspace is hidden."""
    active = AppState.active_view == "graph"
    return rx.el.section(
        rx.cond(
            AppState.has_visualization,
            KnowledgeGraph.create(
                graph=AppState.selected_graph,
                active=active,
                maps=AppState.recent_maps,
                on_select_map=AppState.select_recent_map,
                on_prepare=AppState.prepare_graph_explanation,
                can_clear_session=True,
            ),
            rx.cond(
                AppState.error_message != "",
                state_notice(
                    "circle-alert",
                    GRAPH_UNAVAILABLE_HEADING,
                    AppState.error_message,
                    "error",
                ),
                rx.cond(
                    AppState.is_processing
                    & (
                        (AppState.active_agent == "retriever")
                        | (AppState.active_agent == "visualizer")
                    ),
                    state_notice(
                        "loader-circle",
                        GRAPH_LOADING_HEADING,
                        GRAPH_LOADING_SUBTEXT,
                        "loading",
                        animated=True,
                    ),
                    state_notice(
                        "network",
                        GRAPH_EMPTY_HEADING,
                        GRAPH_EMPTY_SUBTEXT,
                    ),
                ),
            ),
        ),
        hidden=~active,
        class_name="practice-graph-workspace",
        aria_label="Graph workspace",
    )


def chat_view() -> rx.Component:
    """Dedicated chat workspace."""
    return rx.el.section(
        chat_messages(),
        rx.el.div(chat_input(), class_name="study-dock"),
        class_name="practice-conversation",
        aria_label="Study session",
    )


def context_panel() -> rx.Component:
    """Panel showing current context (for debugging)."""
    return rx.cond(
        AppState.show_context_panel,
        surface(
            rx.vstack(
                rx.hstack(
                    section_header(
                        CONTEXT_PANEL_TITLE,
                        font_size="0.75rem",
                        font_weight="600",
                        color=COLORS["text_secondary"],
                        text_transform="uppercase",
                    ),
                    rx.icon_button(
                        rx.icon("x", size=14),
                        on_click=AppState.toggle_context_panel,
                        size="1",
                        variant="ghost",
                    ),
                    justify="between",
                    width="100%",
                ),
                rx.scroll_area(
                    rx.code(
                        AppState.current_context,
                        style={
                            "font_size": "0.75rem",
                            "white_space": "pre-wrap",
                            "word_break": "break-word",
                        },
                    ),
                    height="200px",
                ),
                spacing="2",
            ),
            style={
                "margin": "1rem 1.5rem",
            },
        ),
        rx.fragment(),
    )


def clear_session_button() -> rx.Component:
    return rx.el.button(
        rx.icon("trash-2", size=16),
        rx.el.span("Clear session"),
        type="button",
        aria_label="Clear session",
        aria_haspopup="dialog",
        title="Clear session",
        data_practice_action="clear",
        class_name="clear-session",
    )


def practice_icon(name: str, label: str, action: str, **props) -> rx.Component:
    return rx.el.button(
        rx.icon(name, size=16),
        type="button",
        aria_label=label,
        title=label,
        class_name="practice-icon",
        data_practice_action=action,
        **props,
    )


def history_content(*, drawer: bool = False) -> rx.Component:
    return rx.el.div(
        rx.el.div(
            rx.el.h2("Session history"),
            practice_icon(
                "x" if drawer else "panel-left",
                "Close current-session history"
                if drawer
                else "Toggle current-session history",
                "close" if drawer else "collapse-history",
                **(
                    {"auto_focus": True}
                    if drawer
                    else {"aria_expanded": "true", "aria_controls": "study-history"}
                ),
            ),
            class_name="panel-heading",
        ),
        sidebar_history_panels(),
        rx.el.div(clear_session_button(), class_name="history-foot"),
        class_name="history-content",
    )


def source_row(source: StudySource, *, web: bool = False) -> rx.Component:
    source_icon = rx.icon("globe" if web else "file-text", size=14)
    source_info = rx.el.span(
        rx.fragment(
            rx.el.span(source.title),
            rx.cond(source.path != "", rx.el.small(source.path)),
        )
        if web
        else rx.cond(
            source.is_untitled_note,
            rx.cond(
                source.summary != "",
                rx.el.span(source.summary, class_name="source-preview"),
                rx.el.span("Summary unavailable"),
            ),
            rx.el.span(source.title),
        ),
        class_name="source-label",
    )
    disclosure = rx.el.details(
        rx.el.summary(
            source_icon,
            source_info,
            rx.el.span("Summary", class_name="source-summary-label")
            if not web
            else rx.fragment(),
            rx.icon("chevron-right", size=14, class_name="disclosure-chevron"),
        ),
        rx.el.div(
            rx.cond(source.excerpt != "", rx.el.p(source.excerpt))
            if web
            else rx.el.p(
                rx.cond(source.summary != "", source.summary, "Summary unavailable.")
            ),
            rx.cond(
                source.url != "",
                rx.el.a(
                    "Open source",
                    rx.icon("external-link", size=13),
                    aria_label=rx.cond(
                        source.is_untitled_note, "Open source", "Open " + source.title
                    ),
                    href=source.url,
                    target="_blank",
                    rel="noopener noreferrer",
                ),
            ),
            class_name="source-body",
        ),
        class_name="source-note" if not web else "",
    )
    return rx.el.li(
        rx.cond(
            (source.excerpt != "") | (source.url != "")
            if web
            else ~source.is_untitled_note | (source.summary != "") | (source.url != ""),
            disclosure,
            rx.el.div(source_icon, source_info, class_name="source-static"),
        )
    )


def sources_content(*, drawer: bool = False) -> rx.Component:
    sources = AppState.study_sources
    return rx.el.div(
        rx.el.div(
            rx.el.div(
                rx.el.h2("Sources & concepts", tab_index=-1),
                rx.cond(sources.title != "", rx.el.p(sources.title)),
            ),
            practice_icon(
                "x" if drawer else "panel-right",
                "Close sources and concepts",
                "close" if drawer else "collapse-sources",
                **(
                    {"auto_focus": True}
                    if drawer
                    else {"aria_expanded": "true", "aria_controls": "study-sources"}
                ),
            ),
            class_name="panel-heading sources-heading",
        ),
        rx.el.p(
            "Source details are pending.",
            class_name="panel-note source-selection-pending",
            role="status",
        ),
        rx.el.div(
            rx.cond(
                (sources.status != "ready") & (sources.status != "graph"),
                rx.el.p(
                    rx.match(
                        sources.status,
                        ("pending", "Source details are pending."),
                        ("empty", "No source details to display for this response."),
                        (
                            "relationship_notes_unavailable",
                            "The Retriever supplied relationships, but their supporting note excerpts are unavailable.",
                        ),
                        (
                            "unavailable",
                            "Source details are unavailable for this response.",
                        ),
                        (
                            "no_response",
                            "This turn has no response with source details.",
                        ),
                        "Open sources below an answer to see its context.",
                    ),
                    class_name="panel-note",
                    role="status",
                ),
                rx.cond(
                    (sources.status == "graph")
                    | ((sources.notes.length() > 0) & (sources.web.length() == 0)),
                    rx.el.p(
                        rx.cond(
                            sources.status == "graph",
                            "Source details for this graph concept.",
                            "Sources supplied as context for this response.",
                        ),
                        class_name="panel-note",
                    ),
                ),
            ),
            rx.cond(
                (sources.notes.length() + sources.relationship_notes.length()) > 0,
                SourceList.create(
                    rx.foreach(sources.notes, lambda source: source_row(source)),
                    rx.foreach(
                        sources.relationship_notes,
                        lambda source: source_row(source),
                    ),
                    title=rx.cond(
                        sources.status == "graph",
                        "Knowledge graph sources",
                        "Notes used as context",
                    ),
                ),
            ),
            rx.cond(
                sources.web.length() > 0,
                SourceList.create(
                    rx.foreach(
                        sources.web, lambda source: source_row(source, web=True)
                    ),
                    title=rx.cond(
                        sources.status == "graph",
                        "Knowledge graph sources",
                        "Web sources",
                    ),
                ),
            ),
            key=sources.key,
            class_name="sources-scroll",
        ),
        class_name="sources-content",
        data_sources_key=sources.key,
    )


def practice_dialogs() -> rx.Component:
    return rx.fragment(
        rx.el.dialog(
            history_content(drawer=True),
            id="history-dialog",
            aria_label="Current-session history",
            class_name="practice-drawer history-drawer",
        ),
        rx.el.dialog(
            sources_content(drawer=True),
            id="sources-dialog",
            aria_label="Sources and concepts",
            class_name="practice-drawer sources-drawer",
        ),
        rx.el.dialog(
            rx.el.h2("Clear this session?", id="clear-title"),
            rx.el.p("This removes the current conversation and its knowledge graphs."),
            rx.el.div(
                rx.el.button(
                    "Keep studying",
                    type="button",
                    class_name="practice-secondary",
                    auto_focus=True,
                    data_practice_action="close",
                ),
                rx.el.button(
                    "Clear session",
                    type="button",
                    class_name="practice-primary",
                    on_click=AppState.clear_chat,
                    data_practice_action="confirm-clear",
                ),
                class_name="confirmation-actions",
            ),
            id="clear-dialog",
            aria_labelledby="clear-title",
            class_name="practice-confirm",
        ),
    )


def sidebar() -> rx.Component:
    return rx.el.aside(
        history_content(),
        id="study-history",
        aria_label="Current-session history",
        class_name="practice-history",
    )


def shell_header() -> rx.Component:
    """Render workspace navigation at every width."""
    return rx.el.header(
        rx.el.div(
            practice_icon(
                "menu",
                "Open current-session history",
                "history",
                aria_haspopup="dialog",
                aria_controls="history-dialog",
            ),
            practice_brand(),
            class_name="practice-header-brand",
        ),
        workspace_tabs(),
        rx.el.div(
            practice_icon(
                "panel-right",
                "Open sources and concepts",
                "open-sources-panel",
                aria_expanded="false",
            ),
            practice_theme_button(),
            class_name="practice-header-actions",
        ),
        class_name="practice-header",
    )


def main_content() -> rx.Component:
    """Main content area."""
    welcome = (AppState.active_view == "chat") & ~AppState.has_messages
    study = (AppState.active_view == "chat") & AppState.has_messages
    return rx.fragment(
        rx.el.a("Skip to content", href="#main", class_name="practice-skip"),
        rx.cond(welcome, practice_theme_button("welcome-theme"), shell_header()),
        rx.el.main(
            rx.cond(study, sidebar()),
            rx.cond(
                welcome,
                rx.el.section(
                    rx.el.div(
                        empty_session_intro(),
                        rx.el.div(
                            chat_input(),
                            class_name="welcome-entry",
                        ),
                        class_name="landing-layout",
                    ),
                    class_name="welcome",
                    aria_label="Welcome",
                ),
                rx.cond(AppState.active_view == "chat", chat_view()),
            ),
            graph_view(),
            rx.cond(
                study,
                rx.el.aside(
                    sources_content(),
                    id="study-sources",
                    aria_label="Sources and concepts",
                    class_name="practice-sources",
                ),
            ),
            id="main",
            tab_index=-1,
            class_name="practice-main",
        ),
        practice_dialogs(),
        rx.el.div(
            id="practice-toast", role="status", class_name="practice-toast", hidden=True
        ),
    )
