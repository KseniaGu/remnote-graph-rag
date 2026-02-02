"""Styling constants for the Interview Prep application."""

# Color scheme - WhatsApp-style dark theme
COLORS = {
    "bg_dark": "#0b141a",
    "bg_card": "#1f2c34",
    "bg_card_hover": "#2a3942",
    "bg_input": "#2a3942",
    "border": "#2a3942",
    "border_focus": "#00a884",
    "text_primary": "#e9edef",
    "text_secondary": "#8696a0",
    "text_muted": "#667781",
    "accent_blue": "#53bdeb",
    "accent_purple": "#a855f7",
    "accent_green": "#00a884",
    "accent_orange": "#f97316",
    "accent_red": "#ef4444",
    "user_bubble": "#005c4b",
    "agent_bubble": "#202c33",
}

# Agent colors for visual distinction
AGENT_COLORS = {
    "orchestrator": "#f97316",
    "retriever": "#22c55e",
    "researcher": "#a855f7",
    "analyst": "#0ea5e9",
    "mentor": "#ec4899",
    "visualizer": "#14b8a6",
}

# Common styles
CONTAINER_STYLE = {
    "width": "100%",
    "max_width": "1400px",
    "margin": "0 auto",
    "padding": "0 1rem",
}

CARD_STYLE = {
    "background": COLORS["bg_card"],
    "border_radius": "12px",
    "border": f"1px solid {COLORS['border']}",
    "padding": "1.5rem",
}

INPUT_STYLE = {
    "background": COLORS["bg_input"],
    "border": f"1px solid {COLORS['border']}",
    "border_radius": "8px",
    "color": COLORS["text_primary"],
    "padding": "0.75rem 1rem",
    "width": "100%",
    "height": "auto",
    "min_height": "44px",
    "line_height": "1.5",
    "font_size": "1rem",
    "_focus": {
        "border_color": COLORS["border_focus"],
        "outline": "none",
        "box_shadow": f"0 0 0 2px {COLORS['border_focus']}33",
    },
    "_placeholder": {
        "color": COLORS["text_muted"],
    },
}

BUTTON_PRIMARY_STYLE = {
    "background": COLORS["accent_green"],
    "color": "white",
    "border": "none",
    "border_radius": "50%",
    "width": "44px",
    "height": "44px",
    "min_width": "44px",
    "padding": "0",
    "display": "flex",
    "align_items": "center",
    "justify_content": "center",
    "cursor": "pointer",
    "transition": "all 0.2s ease",
    "_hover": {
        "opacity": "0.9",
    },
    "_disabled": {
        "opacity": "0.5",
        "cursor": "not-allowed",
    },
}

BUTTON_SECONDARY_STYLE = {
    "background": "transparent",
    "color": COLORS["text_secondary"],
    "border": f"1px solid {COLORS['border']}",
    "border_radius": "8px",
    "padding": "0.5rem 1rem",
    "cursor": "pointer",
    "transition": "all 0.2s ease",
    "_hover": {
        "background": COLORS["bg_card_hover"],
        "color": COLORS["text_primary"],
    },
}

MESSAGE_BUBBLE_USER = {
    "background": COLORS["user_bubble"],
    "color": COLORS["text_primary"],
    "borderRadius": "18px 18px 4px 18px",
    "padding": "0.625rem 0.875rem",
    "maxWidth": "65%",
    "marginLeft": "auto",
    "wordWrap": "break-word",
    "boxShadow": "0 1px 0.5px rgba(11, 20, 26, 0.13)",
}

MESSAGE_BUBBLE_AGENT = {
    "background": COLORS["agent_bubble"],
    "color": COLORS["text_primary"],
    "borderRadius": "18px 18px 18px 4px",
    "padding": "0.625rem 0.875rem",
    "maxWidth": "65%",
    "marginRight": "auto",
    "wordWrap": "break-word",
    "boxShadow": "0 1px 0.5px rgba(11, 20, 26, 0.13)",
}

SIDEBAR_STYLE = {
    "width": "280px",
    "background": COLORS["bg_card"],
    "border_right": f"1px solid {COLORS['border']}",
    "height": "100vh",
    "position": "fixed",
    "left": "0",
    "top": "0",
    "padding": "1.5rem",
    "overflow_y": "auto",
}

MAIN_CONTENT_STYLE = {
    "margin_left": "280px",
    "min_height": "100vh",
    "background": COLORS["bg_dark"],
    "display": "flex",
    "flex_direction": "column",
}

CHAT_CONTAINER_STYLE = {
    "flex": "1",
    "overflow_y": "auto",
    "padding": "1.5rem",
    "display": "flex",
    "flex_direction": "column",
    "gap": "1rem",
}

VISUALIZATION_PANEL_STYLE = {
    "background": COLORS["bg_card"],
    "border_radius": "12px",
    "border": f"1px solid {COLORS['border']}",
    "padding": "1rem",
    "height": "400px",
    "margin_bottom": "1rem",
}

STATUS_INDICATOR_STYLE = {
    "width": "8px",
    "height": "8px",
    "border_radius": "50%",
    "display": "inline-block",
    "margin_right": "0.5rem",
}

BADGE_STYLE = {
    "display": "inline-flex",
    "align_items": "center",
    "padding": "0.25rem 0.75rem",
    "border_radius": "9999px",
    "font_size": "0.75rem",
    "font_weight": "500",
}

GLOBAL_STYLES = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f172a;
    color: #f8fafc;
    line-height: 1.6;
}

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #0b141a;
}

::-webkit-scrollbar-thumb {
    background: #374045;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #4a5568;
}

.markdown-content h1,
.markdown-content h2,
.markdown-content h3 {
    margin-top: 0.5rem;
    margin-bottom: 0.25rem;
    font-weight: 600;
}

.markdown-content code {
    background: #0b141a;
    padding: 0.125rem 0.375rem;
    border-radius: 4px;
    font-size: 0.85rem;
}

.markdown-content pre {
    background: #0b141a;
    padding: 0.75rem;
    border-radius: 6px;
    overflow-x: auto;
    margin: 0.5rem 0;
}

.markdown-content ul,
.markdown-content ol {
    padding-left: 1.25rem;
    margin: 0.25rem 0;
}

.markdown-content table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.5rem 0;
}

.markdown-content th,
.markdown-content td {
    border: 1px solid #2a3942;
    padding: 0.5rem;
    text-align: left;
}

.markdown-content th {
    background: #1f2c34;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.animate-pulse {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.animate-spin {
    animation: spin 1s linear infinite;
}

.js-plotly-plot, .plotly, .plot-container {
    width: 100% !important;
}

.js-plotly-plot .plotly .main-svg {
    width: 100% !important;
}
"""
