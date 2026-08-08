import reflex as rx

from app.state import AppState, Message, RecentMapItem, SessionHistoryItem
from app.strings import (
    APP_NAME,
    APP_TAGLINE,
    CHAT_EMPTY_HEADING,
    CHAT_EMPTY_SUBTEXT,
    CHAT_INPUT_PLACEHOLDER,
    CHAT_PROCESSING_LABEL,
    CONTEXT_PANEL_TITLE,
    GRAPH_EMPTY_HEADING,
    GRAPH_EMPTY_SUBTEXT,
    GRAPH_LOADING_HEADING,
    GRAPH_LOADING_SUBTEXT,
    GRAPH_UNAVAILABLE_HEADING,
    GRAPH_UPDATED_ACTION,
    GRAPH_UPDATED_NOTICE,
    QUICK_ACTIONS,
    SIDEBAR_BTN_CLEAR_CHAT,
    VIZ_PANEL_TITLE,
    WORKSPACE_GRAPH_LABEL,
    WORKSPACE_GRAPH_SUBTITLE,
    WORKSPACE_STUDY_LABEL,
    WORKSPACE_STUDY_SUBTITLE,
)
from app.styles import (
    AGENT_COLORS,
    ASSISTANT_ARTIFACT_STYLE,
    BUTTON_PRIMARY_STYLE,
    BUTTON_SECONDARY_STYLE,
    CARD_STYLE,
    COLORS,
    INPUT_STYLE,
    MESSAGE_BUBBLE_USER,
)
from backend.configs.constants import LOGO_URL


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


def icon_text_button(
    icon_name: str,
    label: str,
    on_click,
    style: dict,
    icon_color: str | None = None,
    icon_size: int = 16,
    text_size: str = "0.875rem",
) -> rx.Component:
    """Button primitive for icon + text actions."""
    return rx.button(
        rx.hstack(
            rx.icon(icon_name, size=icon_size, color=icon_color),
            rx.text(label, font_size=text_size),
            spacing="2",
            align="center",
        ),
        on_click=on_click,
        style=style,
    )


def agent_label(name, color: str, animated: bool = False) -> rx.Component:
    """Small artifact label used inside assistant study cards."""
    indicator_style = {
        "width": "5px",
        "height": "5px",
        "borderRadius": "50%",
        "background": color,
    }
    if animated:
        indicator_style["animation"] = "pulse 1.5s ease-in-out infinite"

    return rx.hstack(
        rx.box(style=indicator_style),
        rx.text(
            name,
            font_size="0.72rem",
            font_weight="750",
            color=color,
            letter_spacing="0",
            text_transform="uppercase",
        ),
        spacing="2",
        align="center",
        class_name="artifact-label",
    )


def artifact_header(
    label, timestamp, color: str, animated: bool = False
) -> rx.Component:
    """Metadata row for assistant study artifacts."""
    return rx.hstack(
        agent_label(label, color, animated=animated),
        rx.spacer(),
        rx.text(
            timestamp,
            class_name="artifact-timestamp",
        ),
        class_name="artifact-header",
        align="center",
        width="100%",
    )


def message_card(*children, style, class_name: str | None = None) -> rx.Component:
    """Reusable message card wrapper for persisted and streaming messages."""
    return rx.box(
        rx.vstack(
            *children,
            align="start",
            spacing="0",
        ),
        class_name=class_name,
        style=style,
    )


ANALYST_MESSAGE_STYLE = {
    **ASSISTANT_ARTIFACT_STYLE,
    "width": "100%",
    "maxWidth": "100%",
}


def assistant_label_color(agent_name):
    """Return the display color for an assistant message label."""
    return rx.cond(
        agent_name == "analyst",
        COLORS["accent_blue"],
        COLORS["accent_green"],
    )


def assistant_artifact_label(agent_name):
    """Return the study artifact label for an assistant agent."""
    return rx.cond(
        agent_name == "analyst",
        "Synthesis",
        rx.cond(
            agent_name == "mentor",
            "Practice Guidance",
            rx.cond(
                (agent_name == "researcher") | (agent_name == "retriever"),
                "Retrieved Context",
                agent_name,
            ),
        ),
    )


def message_style(message: Message):
    """Choose the card style for a persisted chat message."""
    return rx.cond(
        (message.role != "user") & (message.agent == "analyst"),
        ANALYST_MESSAGE_STYLE,
        rx.cond(
            message.role == "user",
            MESSAGE_BUBBLE_USER,
            ASSISTANT_ARTIFACT_STYLE,
        ),
    )


def message_class_name(message: Message):
    """Choose the message card class for agent-specific rendering."""
    return rx.cond(
        (message.role != "user") & (message.agent == "analyst"),
        "message-card analyst-message",
        rx.cond(
            (message.role != "user") & (message.agent == "mentor"),
            "message-card mentor-message",
            "message-card",
        ),
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
    color: str = COLORS["accent_blue"],
    animated: bool = False,
) -> rx.Component:
    """Compact empty/loading/error state block."""
    icon_props = {"class_name": "animate-spin"} if animated else {}
    return rx.vstack(
        rx.icon(icon_name, size=28, color=color, **icon_props),
        rx.text(title, class_name="state-notice-title"),
        rx.text(detail, class_name="state-notice-detail"),
        class_name="state-notice",
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
                ),
                rx.icon_button(
                    rx.icon("x", size=14),
                    on_click=AppState.clear_error,
                    size="1",
                    variant="ghost",
                    class_name="submission-error-close",
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


def logo() -> rx.Component:
    """Application logo."""
    return rx.hstack(
        rx.image(
            src=LOGO_URL,
            width="32px",
            height="32px",
            style={
                "filter": "brightness(0) saturate(100%) invert(52%) sepia(85%) saturate(1352%) hue-rotate(131deg) brightness(90%) contrast(92%)",
                "opacity": "0.95",
            },
        ),
        rx.text(
            APP_NAME,
            font_size="1.45rem",
            font_weight="800",
            color=COLORS["text_primary"],
            font_family="var(--font-display)",
            letter_spacing="0",
        ),
        spacing="3",
        align="center",
        min_height="44px",
    )


def agent_activity_item(status: dict) -> rx.Component:
    """Compact workflow step marker for the inline agent activity row."""
    agent_color = rx.cond(
        status["name"] == "orchestrator",
        AGENT_COLORS["orchestrator"],
        rx.cond(
            status["name"] == "retriever",
            AGENT_COLORS["retriever"],
            rx.cond(
                status["name"] == "researcher",
                AGENT_COLORS["researcher"],
                rx.cond(
                    status["name"] == "analyst",
                    AGENT_COLORS["analyst"],
                    rx.cond(
                        status["name"] == "mentor",
                        AGENT_COLORS["mentor"],
                        AGENT_COLORS["visualizer"],
                    ),
                ),
            ),
        ),
    )
    visible_color = rx.cond(
        status["is_active"] | status["was_used"],
        agent_color,
        COLORS["text_muted"],
    )
    return rx.tooltip(
        rx.hstack(
            rx.box(
                class_name=rx.cond(status["is_active"], "animate-pulse", ""),
                style={
                    "width": "6px",
                    "height": "6px",
                    "border_radius": "50%",
                    "background": rx.cond(
                        status["is_active"] | status["was_used"],
                        agent_color,
                        COLORS["text_muted"],
                    ),
                },
            ),
            rx.text(
                status["name"],
                font_size="0.78rem",
                font_weight=rx.cond(
                    status["is_active"],
                    "700",
                    rx.cond(status["was_used"], "650", "500"),
                ),
                text_transform="capitalize",
            ),
            spacing="1",
            align="center",
            class_name="agent-activity-row",
            style={
                "color": visible_color,
                "min_width": "0",
            },
        ),
        content=status["description"],
        delay_duration=300,
    )


def agent_activity_strip() -> rx.Component:
    """Inline overview of workflow agents."""
    return rx.hstack(
        rx.foreach(AppState.agent_status_list, agent_activity_item),
        class_name="agent-activity-list",
        spacing="3",
        wrap="wrap",
    )


def quick_action_button(icon_name: str, label: str, action: str) -> rx.Component:
    """Quick action suggestion button."""
    return icon_text_button(
        icon_name,
        label,
        lambda: AppState.set_input(action),
        icon_color=COLORS["text_secondary"],
        icon_size=16,
        text_size="0.9rem",
        style={
            **BUTTON_SECONDARY_STYLE,
            "min_width": "0",
            "padding": "0.75rem 1rem",
            "_hover": {
                "background": "rgba(22, 51, 59, 0.96)",
                "color": COLORS["text_primary"],
                "border_color": "rgba(167, 186, 196, 0.34)",
            },
        },
    )


def empty_session_intro() -> rx.Component:
    """Focused first-viewport introduction shown before the session starts."""
    return rx.vstack(
        rx.icon("message-square-text", size=40, color=COLORS["accent_green"]),
        rx.text(
            CHAT_EMPTY_HEADING,
            class_name="empty-session-heading",
            text_align="center",
        ),
        rx.text(
            CHAT_EMPTY_SUBTEXT,
            class_name="empty-session-subtext",
            font_size="1rem",
            color=COLORS["text_secondary"],
            text_align="center",
            font_family="var(--font-body)",
        ),
        spacing="3",
        align="center",
        width="100%",
        class_name="empty-session-intro",
    )


def starter_prompt_row() -> rx.Component:
    """Starter prompts attached to the empty-state composer."""
    return rx.hstack(
        *[
            quick_action_button(icon, label, action)
            for icon, label, action in QUICK_ACTIONS
        ],
        class_name="starter-prompts",
        spacing="2",
        wrap="wrap",
        justify="center",
        width="100%",
    )


def workspace_tab(icon_name: str, label: str, view: str, on_click) -> rx.Component:
    """Primary workspace navigation item."""
    is_active = AppState.active_view == view
    return rx.button(
        rx.hstack(
            rx.icon(icon_name, size=20),
            rx.text(label, font_size="0.98rem", font_weight="700"),
            spacing="3",
            align="center",
        ),
        on_click=on_click,
        class_name=rx.cond(
            is_active, "workspace-tab workspace-tab-active", "workspace-tab"
        ),
    )


def workspace_tabs() -> rx.Component:
    """Top-level workspace navigation."""
    return rx.vstack(
        workspace_tab(
            "message-square-text", WORKSPACE_STUDY_LABEL, "chat", AppState.open_chat
        ),
        workspace_tab("network", WORKSPACE_GRAPH_LABEL, "graph", AppState.open_graph),
        spacing="2",
        width="100%",
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
            rx.vstack(
                rx.text(item.title, class_name="sidebar-history-title"),
                rx.text(item.meta, class_name="sidebar-history-meta"),
                spacing="2",
                align="start",
                min_width="0",
            ),
            rx.spacer(),
            rx.cond(
                AppState.selected_session_history_id == item.id,
                rx.box(class_name="sidebar-history-dot"),
                rx.fragment(),
            ),
            align="start",
            width="100%",
            min_width="0",
            class_name="sidebar-history-row",
        ),
        on_click=lambda: AppState.select_session_history_item(item.id),
        class_name=rx.cond(
            AppState.selected_session_history_id == item.id,
            "sidebar-history-item sidebar-history-item-active",
            "sidebar-history-item",
        ),
    )


def sidebar_map_item(item: RecentMapItem) -> rx.Component:
    """Clickable current-tab recent map item."""
    return rx.button(
        rx.hstack(
            rx.vstack(
                rx.text(item.title, class_name="sidebar-history-title"),
                rx.text(item.meta, class_name="sidebar-history-meta"),
                spacing="2",
                align="start",
                min_width="0",
            ),
            rx.spacer(),
            rx.cond(
                AppState.selected_recent_map_id == item.id,
                rx.box(class_name="sidebar-history-dot"),
                rx.fragment(),
            ),
            align="start",
            width="100%",
            min_width="0",
            class_name="sidebar-history-row",
        ),
        on_click=lambda: AppState.select_recent_map(item.id),
        class_name=rx.cond(
            AppState.selected_recent_map_id == item.id,
            "sidebar-history-item sidebar-history-item-active",
            "sidebar-history-item",
        ),
    )


def sidebar_session_section() -> rx.Component:
    """Current browser-tab session history panel."""
    return rx.vstack(
        rx.text("Session History", class_name="sidebar-section-title"),
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
        rx.text("Recent Maps", class_name="sidebar-section-title"),
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
    return rx.cond(
        AppState.active_view == "graph",
        sidebar_maps_section(),
        sidebar_session_section(),
    )


def workspace_title():
    """Title for the active workspace."""
    return rx.cond(
        AppState.active_view == "graph",
        WORKSPACE_GRAPH_LABEL,
        WORKSPACE_STUDY_LABEL,
    )


def workspace_subtitle():
    """Subtitle for the active workspace."""
    return rx.cond(
        AppState.active_view == "graph",
        WORKSPACE_GRAPH_SUBTITLE,
        WORKSPACE_STUDY_SUBTITLE,
    )


def workspace_header() -> rx.Component:
    """Header for the active workspace."""
    return rx.hstack(
        rx.vstack(
            rx.text(
                workspace_title(),
                font_size="1.45rem",
                font_weight="800",
                color=COLORS["text_primary"],
                font_family="var(--font-display)",
                letter_spacing="0",
            ),
            rx.text(
                workspace_subtitle(),
                font_size="0.9rem",
                font_weight="400",
                color=COLORS["text_secondary"],
                font_family="var(--font-body)",
            ),
            spacing="1",
            align="start",
        ),
        rx.spacer(),
        padding="1rem 1.5rem",
        width="100%",
        border_bottom=f"1px solid {COLORS['border']}",
        background="rgba(11, 26, 32, 0.96)",
        class_name="desktop-main-header",
    )


def message_bubble(message: Message) -> rx.Component:
    """Chat message bubble - designed for use with rx.foreach."""
    return rx.cond(
        (message.role != "user") & (message.agent == "visualizer"),
        rx.fragment(),
        rx.box(
            message_card(
                rx.cond(
                    (message.role != "user") & (message.agent != ""),
                    artifact_header(
                        assistant_artifact_label(message.agent),
                        message.timestamp,
                        assistant_label_color(message.agent),
                    ),
                    rx.fragment(),
                ),
                rx.markdown(
                    message.content,
                    class_name="markdown-content",
                ),
                rx.text(
                    message.timestamp,
                    font_size="0.65rem",
                    color=COLORS["text_muted"],
                    align_self="flex-end",
                    margin_top="0.25rem",
                    display=rx.cond(
                        (message.role == "user") | (message.agent == ""),
                        "block",
                        "none",
                    ),
                ),
                class_name=message_class_name(message),
                style=message_style(message),
            ),
            width="100%",
            display="flex",
            justify_content=rx.cond(
                message.role == "user",
                "flex-end",
                "flex-start",
            ),
            id=message.dom_id,
            class_name="message-row",
        ),
    )


def streaming_bubble() -> rx.Component:
    """Live streaming bubble that shows tokens as they arrive from analyst/mentor."""
    return rx.cond(
        AppState.is_streaming,
        rx.cond(
            AppState.streaming_agent == "visualizer",
            rx.fragment(),
            rx.box(
                message_card(
                    artifact_header(
                        assistant_artifact_label(AppState.streaming_agent),
                        "",
                        COLORS["accent_blue"],
                        animated=True,
                    ),
                    rx.text(
                        AppState.streaming_content + " ▍",
                        white_space="pre-wrap",
                        overflow_wrap="anywhere",
                        class_name="streaming-content",
                    ),
                    class_name=rx.cond(
                        AppState.streaming_agent == "analyst",
                        "message-card analyst-message",
                        "message-card mentor-message",
                    ),
                    style=rx.cond(
                        AppState.streaming_agent == "analyst",
                        ANALYST_MESSAGE_STYLE,
                        ASSISTANT_ARTIFACT_STYLE,
                    ),
                ),
                width="100%",
                display="flex",
                justify_content="flex-start",
            ),
        ),
        rx.fragment(),
    )


def chat_messages() -> rx.Component:
    """Chat messages container."""
    return rx.cond(
        AppState.has_messages | AppState.is_streaming | AppState.is_processing,
        rx.box(
            rx.box(
                rx.foreach(AppState.messages, message_bubble),
                streaming_bubble(),
                class_name="study-rail study-message-rail",
            ),
            id="chat-messages",
            flex="1",
            overflow_y="auto",
            width="100%",
            class_name="study-message-scroll",
        ),
        rx.fragment(),
    )


def processing_indicator() -> rx.Component:
    """Transparent inline status shown above the composer while processing."""
    return rx.cond(
        AppState.is_processing,
        rx.hstack(
            rx.hstack(
                rx.icon(
                    "loader-circle",
                    size=16,
                    class_name="animate-spin",
                    color=COLORS["accent_blue"],
                ),
                rx.text(
                    CHAT_PROCESSING_LABEL,
                    font_size="0.875rem",
                    font_weight="700",
                    color=COLORS["text_secondary"],
                ),
                rx.text(
                    workflow_status_label(),
                    font_size="0.82rem",
                    color=COLORS["text_secondary"],
                    class_name="workflow-status-detail",
                ),
                class_name="processing-status-group",
                spacing="2",
                align="center",
                min_width="0",
            ),
            agent_activity_strip(),
            class_name="status-card composer-status-line",
            spacing="4",
            align="center",
            justify="between",
            width="100%",
        ),
        rx.fragment(),
    )


def bottom_status_area() -> rx.Component:
    """Predictable notices and workflow status region above the composer."""
    return rx.cond(
        AppState.is_processing
        | (AppState.show_graph_updated_notice & ~AppState.is_processing),
        rx.box(
            rx.vstack(
                graph_updated_notice(),
                processing_indicator(),
                class_name="study-rail bottom-status-stack",
                spacing="3",
                width="100%",
            ),
            class_name="chat-status-rail",
            width="100%",
        ),
        rx.fragment(),
    )


def graph_updated_notice() -> rx.Component:
    """Persistent inline notice shown after a graph update completes."""
    return rx.cond(
        AppState.show_graph_updated_notice & ~AppState.is_processing,
        rx.box(
            rx.hstack(
                rx.icon("network", size=16, color=COLORS["text_secondary"]),
                rx.text(
                    GRAPH_UPDATED_NOTICE,
                    font_size="0.875rem",
                    font_weight="600",
                    color=COLORS["text_secondary"],
                ),
                rx.button(
                    GRAPH_UPDATED_ACTION,
                    on_click=AppState.open_graph,
                    class_name="graph-updated-action",
                ),
                spacing="2",
                align="center",
                width="100%",
                min_width="0",
            ),
            class_name="graph-updated-notice status-card",
            width="100%",
        ),
        rx.fragment(),
    )


def composer_inner() -> rx.Component:
    """Aligned composer module for status, errors, input, and starter prompts."""
    return rx.vstack(
        submission_error_notice(),
        rx.form(
            rx.hstack(
                rx.box(
                    rx.text_area(
                        id="chat-input",
                        name="message",
                        placeholder=CHAT_INPUT_PLACEHOLDER,
                        value=AppState.current_input,
                        on_change=AppState.set_input,
                        disabled=AppState.is_processing,
                        rows="1",
                        style={
                            **INPUT_STYLE,
                            "background": "transparent",
                            "border": "0",
                            "box_shadow": "none",
                            "width": "100%",
                            "min_height": "48px",
                            "padding": "0.75rem 1rem",
                            "resize": "none",
                            "max_height": "120px",
                            "overflow_y": "auto",
                        },
                    ),
                    class_name="composer-textarea-shell",
                    flex="1",
                    min_width="0",
                ),
                rx.button(
                    rx.cond(
                        AppState.is_processing,
                        rx.icon("loader-circle", size=20, class_name="animate-spin"),
                        rx.icon("send", size=20),
                    ),
                    id="chat-send-btn",
                    type="submit",
                    disabled=AppState.is_processing | (AppState.current_input == ""),
                    style=BUTTON_PRIMARY_STYLE,
                ),
                spacing="3",
                width="100%",
                align="center",
            ),
            on_submit=AppState.handle_form_submit,
            reset_on_submit=False,
            width="100%",
        ),
        rx.cond(
            ~AppState.has_messages,
            starter_prompt_row(),
            rx.fragment(),
        ),
        class_name="composer-inner",
        spacing="3",
        width="100%",
    )


def chat_input() -> rx.Component:
    """Chat input area."""
    return rx.vstack(
        rx.script("""
(function() {
    function initChatInput() {
        var ta = document.getElementById('chat-input');
        if (!ta || ta._chatInited) return;
        ta._chatInited = true;

        ta.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });

        function clearComposerDom() {
            ta.value = '';
            ta.style.height = '';
        }

        ta.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
                e.preventDefault();
                e.stopPropagation();
                var form = ta.closest('form');
                var btn = document.getElementById('chat-send-btn');
                if (form && btn && !btn.disabled) {
                    form.requestSubmit();
                    clearComposerDom();
                }
            }
        }, true);

        // Reset textarea height whenever the form is submitted (Enter key or send button)
        var form = ta.closest('form');
        if (form) {
            form.addEventListener('submit', function() {
                window.setTimeout(clearComposerDom, 0);
            });
        }
    }

    function initAutoScroll() {
        var container = document.getElementById('chat-messages');
        if (!container || container._autoScrollInited) return;
        container._autoScrollInited = true;
        new MutationObserver(function() {
            if (Date.now() < (window.__chatHistoryScrollSuppressUntil || 0)) return;
            container.scrollTop = container.scrollHeight;
        }).observe(container, { childList: true, subtree: true });
    }

    function initHistoryScroll() {
        var command = document.getElementById('chat-scroll-command');
        if (!command || command._historyScrollInited) return;
        command._historyScrollInited = true;

        function runHistoryScroll() {
            var raw = (command.textContent || '').trim();
            if (!raw || raw === '0:') return;
            if (window.__lastChatHistoryScrollCommand === raw) return;
            window.__lastChatHistoryScrollCommand = raw;

            var separator = raw.indexOf(':');
            var targetId = separator >= 0 ? raw.slice(separator + 1) : raw;
            if (!targetId) return;

            window.setTimeout(function() {
                var target = document.getElementById(targetId);
                if (!target) return;
                window.__chatHistoryScrollSuppressUntil = Date.now() + 1200;
                target.scrollIntoView({ behavior: 'smooth', block: 'center' });
                target.classList.add('message-scroll-highlight');
                window.setTimeout(function() {
                    target.classList.remove('message-scroll-highlight');
                }, 1400);
            }, 80);
        }

        new MutationObserver(runHistoryScroll).observe(command, {
            childList: true,
            characterData: true,
            subtree: true
        });
        runHistoryScroll();
    }

    // Run now and watch for DOM changes (Reflex hydration)
    initChatInput();
    initAutoScroll();
    initHistoryScroll();
    new MutationObserver(function() {
        initChatInput();
        initAutoScroll();
        initHistoryScroll();
    }).observe(document.body, { childList: true, subtree: true });
})();
"""),
        rx.box(
            AppState.scroll_request_nonce.to_string()
            + ":"
            + AppState.scroll_target_message_dom_id,
            id="chat-scroll-command",
            display="none",
        ),
        rx.cond(
            ~AppState.has_messages,
            empty_session_intro(),
            rx.fragment(),
        ),
        rx.box(
            composer_inner(),
            class_name="study-rail",
        ),
        spacing="3",
        width="100%",
        background=rx.cond(~AppState.has_messages, "transparent", COLORS["bg_card"]),
        border_top=rx.cond(
            ~AppState.has_messages, "none", f"1px solid {COLORS['border']}"
        ),
        class_name=rx.cond(
            ~AppState.has_messages, "empty-session-composer", "chat-composer"
        ),
    )


def graph_panel_body() -> rx.Component:
    """Graph body with visualization, loading, empty, and error states."""
    return rx.box(
        rx.cond(
            AppState.has_visualization,
            rx.plotly(
                data=AppState.plotly_figure,
                config={
                    "displayModeBar": "hover",
                    "scrollZoom": True,
                    "responsive": True,
                },
                class_name="graph-plot",
            ),
            rx.cond(
                AppState.error_message != "",
                state_notice(
                    "circle-alert",
                    GRAPH_UNAVAILABLE_HEADING,
                    AppState.error_message,
                    COLORS["accent_red"],
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
                        COLORS["accent_blue"],
                        animated=True,
                    ),
                    state_notice(
                        "network",
                        GRAPH_EMPTY_HEADING,
                        GRAPH_EMPTY_SUBTEXT,
                        COLORS["text_muted"],
                    ),
                ),
            ),
        ),
        class_name="graph-canvas",
        width="100%",
    )


def visualization_panel() -> rx.Component:
    """Panel for displaying knowledge graph visualization."""
    return rx.cond(
        AppState.show_visualization,
        surface(
            rx.vstack(
                rx.hstack(
                    section_header(VIZ_PANEL_TITLE, "network"),
                    rx.hstack(
                        rx.cond(
                            AppState.plot_count > 1,
                            rx.hstack(
                                rx.icon_button(
                                    rx.icon("chevron-left", size=14),
                                    on_click=AppState.prev_plot,
                                    size="1",
                                    variant="ghost",
                                    color_scheme="gray",
                                ),
                                rx.text(
                                    AppState.current_plot_label,
                                    font_size="0.75rem",
                                    color=COLORS["text_secondary"],
                                ),
                                rx.icon_button(
                                    rx.icon("chevron-right", size=14),
                                    on_click=AppState.next_plot,
                                    size="1",
                                    variant="ghost",
                                    color_scheme="gray",
                                ),
                                spacing="1",
                                align="center",
                            ),
                            rx.fragment(),
                        ),
                        rx.icon_button(
                            rx.icon("x", size=14),
                            on_click=AppState.toggle_visualization,
                            size="1",
                            variant="ghost",
                            color_scheme="gray",
                        ),
                        spacing="1",
                        align="center",
                    ),
                    justify="between",
                    width="100%",
                ),
                graph_panel_body(),
                spacing="3",
                width="100%",
            ),
            style={
                "border_radius": "12px",
                "padding": "1.5rem",
                "margin": "1rem 1.5rem 1rem 1.5rem",
                "width": "calc(100% - 3rem)",
            },
        ),
        rx.fragment(),
    )


def graph_view() -> rx.Component:
    """Dedicated graph workspace."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                section_header(VIZ_PANEL_TITLE, "network"),
                rx.hstack(
                    rx.cond(
                        AppState.plot_count > 1,
                        rx.hstack(
                            rx.icon_button(
                                rx.icon("chevron-left", size=14),
                                on_click=AppState.prev_plot,
                                size="1",
                                variant="ghost",
                                color_scheme="gray",
                            ),
                            rx.text(
                                AppState.current_plot_label,
                                font_size="0.75rem",
                                color=COLORS["text_secondary"],
                            ),
                            rx.icon_button(
                                rx.icon("chevron-right", size=14),
                                on_click=AppState.next_plot,
                                size="1",
                                variant="ghost",
                                color_scheme="gray",
                            ),
                            spacing="1",
                            align="center",
                        ),
                        rx.fragment(),
                    ),
                    spacing="1",
                    align="center",
                ),
                class_name="graph-toolbar",
                justify="between",
                width="100%",
            ),
            graph_panel_body(),
            class_name="graph-workspace-panel",
            spacing="0",
            width="100%",
        ),
        class_name="workspace-view graph-view",
    )


def chat_view() -> rx.Component:
    """Dedicated chat workspace."""
    return rx.vstack(
        chat_messages(),
        bottom_status_area(),
        chat_input(),
        class_name="workspace-view chat-view",
        spacing="0",
        width="100%",
    )


def workspace_body() -> rx.Component:
    """Renders the selected primary workspace."""
    return rx.cond(
        AppState.active_view == "graph",
        graph_view(),
        chat_view(),
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


def sidebar_actions() -> rx.Component:
    """Shared session actions used by desktop and mobile shells."""
    return rx.vstack(
        icon_text_button(
            "trash-2",
            SIDEBAR_BTN_CLEAR_CHAT,
            AppState.clear_chat,
            style={
                **BUTTON_SECONDARY_STYLE,
                "width": "100%",
                "justify_content": "center",
            },
        ),
        spacing="2",
        width="100%",
    )


def sidebar() -> rx.Component:
    """Application sidebar."""
    return rx.box(
        rx.vstack(
            logo(),
            rx.box(
                workspace_tabs(),
                sidebar_history_panels(),
                class_name="sidebar-scroll",
                display="flex",
                flex_direction="column",
                gap="1.25rem",
                width="100%",
            ),
            rx.spacer(),
            sidebar_actions(),
            spacing="5",
            align="start",
            height="100%",
        ),
        class_name="desktop-sidebar",
        style={
            "width": "292px",
            "background": COLORS["bg_card"],
            "border_right": f"1px solid {COLORS['border']}",
            "height": "100vh",
            "position": "fixed",
            "left": "0",
            "top": "0",
            "padding": "1.5rem 1.125rem",
            "display": "flex",
            "flex_direction": "column",
        },
    )


def mobile_shell_header() -> rx.Component:
    """Mobile app shell header and collapsible controls."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                logo(),
                rx.text(
                    APP_TAGLINE,
                    font_size="0.85rem",
                    font_weight="550",
                    color=COLORS["accent_blue_lightest"],
                    font_family="var(--font-mono)",
                    text_align="right",
                ),
                justify="between",
                align="center",
                width="100%",
                gap="1rem",
            ),
            rx.el.details(
                rx.el.summary(
                    section_header(
                        "Workspace",
                        "activity",
                        font_size="0.875rem",
                        font_weight="650",
                    ),
                    class_name="mobile-shell-summary",
                ),
                rx.vstack(
                    workspace_tabs(),
                    sidebar_history_panels(),
                    sidebar_actions(),
                    spacing="4",
                    width="100%",
                    padding_top="0.75rem",
                ),
                class_name="mobile-shell-details",
            ),
            spacing="3",
            width="100%",
        ),
        class_name="mobile-shell-header",
    )


def main_content() -> rx.Component:
    """Main content area."""
    return rx.box(
        rx.vstack(
            mobile_shell_header(),
            workspace_header(),
            workspace_body(),
            spacing="0",
            height="100vh",
            width="100%",
        ),
        class_name="app-main-content",
        style={
            "margin_left": "292px",
            "min_height": "100vh",
            "background": COLORS["bg_dark"],
            "display": "flex",
            "flex_direction": "column",
        },
    )
