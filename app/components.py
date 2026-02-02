import reflex as rx

from app.state import AppState, Message
from app.styles import (
    COLORS, MESSAGE_BUBBLE_USER, MESSAGE_BUBBLE_AGENT,
    BUTTON_PRIMARY_STYLE, BUTTON_SECONDARY_STYLE, INPUT_STYLE,
    BADGE_STYLE, STATUS_INDICATOR_STYLE, CARD_STYLE
)


def logo() -> rx.Component:
    """Application logo."""
    return rx.hstack(
        rx.image(
            src="/book_2.png",
            width="28px",
            height="28px",
            style={
                "filter": "brightness(0) saturate(100%) invert(52%) sepia(85%) saturate(1352%) hue-rotate(131deg) brightness(95%) contrast(101%)",
            },
        ),
        rx.text(
            "AI Practice",
            font_size="1.25rem",
            font_weight="700",
            background=f"linear-gradient(135deg, {COLORS['accent_green']}, {COLORS['accent_blue']})",
            background_clip="text",
            style={"-webkit-background-clip": "text", "-webkit-text-fill-color": "transparent"},
        ),
        spacing="3",
        align="center",
    )


def agent_badge_item(status: dict) -> rx.Component:
    """Badge showing agent status - designed for use with rx.foreach."""
    return rx.hstack(
        rx.cond(
            status["is_active"],
            rx.box(
                class_name="animate-pulse",
                style={
                    **STATUS_INDICATOR_STYLE,
                    "background": COLORS["accent_blue"],
                },
            ),
            rx.box(
                style={
                    **STATUS_INDICATOR_STYLE,
                    "background": rx.cond(
                        status["was_used"],
                        COLORS["accent_green"],
                        COLORS["text_muted"],
                    ),
                },
            ),
        ),
        rx.text(
            status["name"],
            font_size="0.75rem",
            font_weight="500",
            text_transform="capitalize",
        ),
        style={
            **BADGE_STYLE,
            "background": rx.cond(
                status["is_active"],
                f"{COLORS['accent_blue']}33",
                rx.cond(
                    status["was_used"],
                    f"{COLORS['accent_green']}15",
                    "transparent",
                ),
            ),
            "border": rx.cond(
                status["is_active"],
                f"1px solid {COLORS['accent_blue']}",
                rx.cond(
                    status["was_used"],
                    f"1px solid {COLORS['accent_green']}50",
                    f"1px solid {COLORS['border']}",
                ),
            ),
            "color": rx.cond(
                status["is_active"],
                COLORS["accent_blue"],
                rx.cond(
                    status["was_used"],
                    COLORS["accent_green"],
                    COLORS["text_muted"],
                ),
            ),
        },
        spacing="1",
    )


def agent_status_panel() -> rx.Component:
    """Panel showing all agent statuses."""
    return rx.vstack(
        rx.text(
            "Agent Pipeline",
            font_size="0.75rem",
            font_weight="600",
            color=COLORS["text_secondary"],
            text_transform="uppercase",
            letter_spacing="0.05em",
        ),
        rx.foreach(
            AppState.agent_status_list,
            agent_badge_item,
        ),
        align="start",
        spacing="2",
        width="100%",
    )


def quick_action_button(icon_name: str, label: str, action: str) -> rx.Component:
    """Quick action suggestion button."""
    return rx.button(
        rx.hstack(
            rx.icon(icon_name, size=14, color=COLORS["accent_blue"]),
            rx.text(label, font_size="0.8rem"),
            spacing="2",
            align="center",
        ),
        on_click=lambda: AppState.set_input(action),
        style={
            "background": COLORS["bg_card"],
            "color": COLORS["text_secondary"],
            "border": f"1px solid {COLORS['border']}",
            "border_radius": "20px",
            "padding": "0.5rem 1rem",
            "cursor": "pointer",
            "_hover": {
                "background": COLORS["bg_card_hover"],
                "color": COLORS["text_primary"],
                "border_color": COLORS["accent_blue"],
            },
        },
    )


def message_bubble(message: Message) -> rx.Component:
    """Chat message bubble - designed for use with rx.foreach."""
    return rx.box(
        rx.box(
            rx.vstack(
                rx.cond(
                    (message.role != "user") & (message.agent != ""),
                    rx.hstack(
                        rx.box(
                            style={
                                "width": "6px",
                                "height": "6px",
                                "borderRadius": "50%",
                                "background": COLORS["accent_green"],
                            },
                        ),
                        rx.text(
                            message.agent,
                            font_size="0.65rem",
                            font_weight="600",
                            color=COLORS["accent_green"],
                            letter_spacing="0.05em",
                            text_transform="uppercase",
                        ),
                        spacing="2",
                        align="center",
                        margin_bottom="0.25rem",
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
                ),
                align="start",
                spacing="0",
            ),
            style=rx.cond(
                message.role == "user",
                MESSAGE_BUBBLE_USER,
                MESSAGE_BUBBLE_AGENT,
            ),
        ),
        width="100%",
        display="flex",
        justify_content=rx.cond(
            message.role == "user",
            "flex-end",
            "flex-start",
        ),
    )


def chat_messages() -> rx.Component:
    """Chat messages container."""
    return rx.cond(
        AppState.has_messages,
        rx.box(
            rx.foreach(AppState.messages, message_bubble),
            flex="1",
            overflow_y="auto",
            padding="1.5rem",
            display="flex",
            flex_direction="column",
            gap="1rem",
            width="100%",
        ),
        rx.box(
            rx.vstack(
                rx.icon("message-square-text", size=48, color=COLORS["text_muted"]),
                rx.text(
                    "Start your study session",
                    font_size="1rem",
                    color=COLORS["text_secondary"],
                ),
                rx.text(
                    "Ask questions, practice concepts, or explore your knowledge graph",
                    font_size="0.875rem",
                    color=COLORS["text_muted"],
                    text_align="center",
                ),
                spacing="3",
                align="center",
            ),
            flex="1",
            display="flex",
            align_items="center",
            justify_content="center",
            width="100%",
        ),
    )


def processing_indicator() -> rx.Component:
    """Indicator shown while processing."""
    return rx.cond(
        AppState.is_processing,
        rx.hstack(
            rx.hstack(
                rx.icon("loader-2", size=16, class_name="animate-spin", color=COLORS["accent_blue"]),
                rx.text(
                    rx.cond(
                        AppState.active_agent != "",
                        rx.text.span(AppState.active_agent, text_transform="capitalize"),
                        "Processing",
                    ),
                    font_size="0.875rem",
                    color=COLORS["text_secondary"],
                ),
                spacing="2",
                align="center",
            ),
            style={
                "padding": "0.5rem 1rem",
                "background": COLORS["bg_card"],
                "border_radius": "20px",
                "border": f"1px solid {COLORS['border']}",
            },
        ),
        rx.fragment(),
    )


def chat_input() -> rx.Component:
    """Chat input area."""
    return rx.vstack(
        processing_indicator(),
        rx.hstack(
            rx.input(
                placeholder="Ask a question or request practice...",
                value=AppState.current_input,
                on_change=AppState.set_input,
                disabled=AppState.is_processing,
                style={
                    **INPUT_STYLE,
                    "flex": "1",
                },
            ),
            rx.button(
                rx.cond(
                    AppState.is_processing,
                    rx.icon("loader-2", size=20, class_name="animate-spin"),
                    rx.icon("send", size=20),
                ),
                on_click=AppState.send_message,
                disabled=AppState.is_processing | (AppState.current_input == ""),
                style=BUTTON_PRIMARY_STYLE,
            ),
            spacing="3",
            width="100%",
        ),
        rx.cond(
            ~AppState.has_messages,
            rx.hstack(
                quick_action_button("graduation-cap", "Quiz me on Transformers", "Quiz me on Transformer architecture"),
                quick_action_button("search", "Search my notes",
                                    "What information do I have about attention mechanisms?"),
                quick_action_button("globe", "Research a topic", "Research the latest developments in LLM fine-tuning"),
                quick_action_button("network", "Visualize concepts", "Visualize my knowledge about neural networks"),
                spacing="2",
                wrap="wrap",
                justify="center",
            ),
            rx.fragment(),
        ),
        spacing="3",
        width="100%",
        padding="1.5rem",
        background=COLORS["bg_card"],
        border_top=f"1px solid {COLORS['border']}",
    )


def error_toast() -> rx.Component:
    """Error message toast."""
    return rx.cond(
        AppState.error_message != "",
        rx.box(
            rx.hstack(
                rx.icon("alert-circle", size=18, color=COLORS["accent_red"]),
                rx.text(AppState.error_message, font_size="0.875rem"),
                rx.icon_button(
                    rx.icon("x", size=14),
                    on_click=AppState.clear_error,
                    size="1",
                    variant="ghost",
                ),
                spacing="3",
                align="center",
                justify="between",
                width="100%",
            ),
            style={
                "position": "fixed",
                "bottom": "100px",
                "left": "50%",
                "transform": "translateX(-50%)",
                "background": COLORS["bg_card"],
                "border": f"1px solid {COLORS['accent_red']}50",
                "border_radius": "8px",
                "padding": "0.75rem 1rem",
                "color": COLORS["text_primary"],
                "z_index": "1000",
                "max_width": "90%",
            },
        ),
        rx.fragment(),
    )


def visualization_panel() -> rx.Component:
    """Panel for displaying knowledge graph visualization."""
    return rx.cond(
        AppState.show_visualization & AppState.has_visualization,
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.hstack(
                        rx.icon("network", size=18, color=COLORS["accent_blue"]),
                        rx.text(
                            "Knowledge Graph",
                            font_size="0.875rem",
                            font_weight="600",
                        ),
                        spacing="2",
                        align="center",
                    ),
                    rx.hstack(
                        rx.icon_button(
                            rx.icon("minus", size=14),
                            size="1",
                            variant="ghost",
                            color_scheme="gray",
                        ),
                        rx.icon_button(
                            rx.icon("plus", size=14),
                            size="1",
                            variant="ghost",
                            color_scheme="gray",
                        ),
                        rx.icon_button(
                            rx.icon("x", size=14),
                            on_click=AppState.toggle_visualization,
                            size="1",
                            variant="ghost",
                            color_scheme="gray",
                        ),
                        spacing="1",
                    ),
                    justify="between",
                    width="100%",
                ),
                rx.box(
                    rx.plotly(
                        data=AppState.plotly_figure,
                        use_resize_handler=True,
                        config={"responsive": True},
                    ),
                    width="100%",
                    height="450px",
                ),
                spacing="3",
                width="100%",
            ),
            style={
                "background": COLORS["bg_card"],
                "border_radius": "12px",
                "border": f"1px solid {COLORS['border']}",
                "padding": "1.5rem",
                "margin": "1rem 1.5rem 1rem 1.5rem",
                "width": "calc(100% - 3rem)",
            },
        ),
        rx.fragment(),
    )


def context_panel() -> rx.Component:
    """Panel showing current context (for debugging)."""
    return rx.cond(
        AppState.show_context_panel,
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.text(
                        "Context",
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
                **CARD_STYLE,
                "margin": "1rem 1.5rem",
            },
        ),
        rx.fragment(),
    )


def sidebar() -> rx.Component:
    """Application sidebar."""
    return rx.box(
        rx.vstack(
            logo(),
            rx.divider(margin_y="1.5rem", border_color=COLORS["border"]),
            agent_status_panel(),
            rx.spacer(),
            rx.vstack(
                rx.button(
                    rx.hstack(
                        rx.icon("trash-2", size=16),
                        rx.text("Clear Chat", font_size="0.875rem"),
                        spacing="2",
                    ),
                    on_click=AppState.clear_chat,
                    style={
                        **BUTTON_SECONDARY_STYLE,
                        "width": "100%",
                        "justify_content": "center",
                    },
                ),
                rx.button(
                    rx.hstack(
                        rx.icon("code", size=16),
                        rx.text("Show Context", font_size="0.875rem"),
                        spacing="2",
                    ),
                    on_click=AppState.toggle_context_panel,
                    style={
                        **BUTTON_SECONDARY_STYLE,
                        "width": "100%",
                        "justify_content": "center",
                    },
                ),
                spacing="2",
                width="100%",
            ),
            spacing="0",
            align="start",
            height="100%",
        ),
        style={
            "width": "280px",
            "background": COLORS["bg_card"],
            "border_right": f"1px solid {COLORS['border']}",
            "height": "100vh",
            "position": "fixed",
            "left": "0",
            "top": "0",
            "padding": "1.5rem",
            "display": "flex",
            "flex_direction": "column",
        },
    )


def main_content() -> rx.Component:
    """Main content area."""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text(
                    "Study AI using your personal knowledge base",
                    font_size="1.125rem",
                    font_weight="600",
                    color=COLORS["text_primary"],
                ),
                rx.spacer(),
                rx.cond(
                    AppState.has_visualization,
                    rx.button(
                        rx.hstack(
                            rx.icon("network", size=16),
                            rx.text("Toggle Graph", font_size="0.875rem"),
                            spacing="2",
                        ),
                        on_click=AppState.toggle_visualization,
                        style=BUTTON_SECONDARY_STYLE,
                    ),
                    rx.fragment(),
                ),
                padding="1rem 1.5rem",
                width="100%",
                border_bottom=f"1px solid {COLORS['border']}",
                background=COLORS["bg_card"],
            ),
            visualization_panel(),
            context_panel(),
            chat_messages(),
            chat_input(),
            spacing="0",
            height="100vh",
            width="100%",
        ),
        style={
            "margin_left": "280px",
            "min_height": "100vh",
            "background": COLORS["bg_dark"],
            "display": "flex",
            "flex_direction": "column",
        },
    )
