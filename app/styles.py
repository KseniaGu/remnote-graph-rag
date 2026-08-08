"""Styling constants for the AI Practice application."""

# Color scheme - compact dark study workbench
COLORS = {
    "bg_dark": "#061216",
    "bg_card": "#0b1a20",
    "bg_card_hover": "#102a31",
    "bg_input": "#10242b",
    "border": "#1d3942",
    "border_focus": "#19d7ca",
    "text_primary": "#f3fbff",
    "text_secondary": "#a7bac4",
    "text_muted": "#6f8791",
    "accent_blue": "#75b8f2",
    "accent_blue_light": "#b8dff8",
    "accent_blue_lightest": "#e2f5ff",
    "accent_purple": "#8d6af4",
    "accent_green": "#19d7ca",
    "accent_orange": "#f6a336",
    "accent_red": "#ff756f",
    "user_bubble": "#087466",
    "agent_bubble": "#10242b",
}

# Agent colors for the compact activity line, tuned to the dark teal workbench
AGENT_COLORS = {
    "orchestrator": "#8fd3ff",
    "retriever": "#39d9c0",
    "researcher": "#75b8f2",
    "analyst": "#5fd2ee",
    "mentor": "#a8c7ff",
    "visualizer": "#19d7ca",
}

# Common styles
CONTROL_TRANSITION = "background 180ms ease, border-color 180ms ease, color 180ms ease, box-shadow 180ms ease"
FOCUS_RING = "0 0 0 2px rgba(25, 215, 202, 0.28)"
SURFACE_SHADOW = (
    "inset 0 1px 0 rgba(243, 251, 255, 0.035), 0 18px 54px rgba(0, 0, 0, 0.22)"
)

CONTAINER_STYLE = {
    "width": "100%",
    "max_width": "1400px",
    "margin": "0 auto",
    "padding": "0 1rem",
}

CARD_STYLE = {
    "background": COLORS["bg_card"],
    "border_radius": "var(--radius-surface)",
    "border": "var(--border-default)",
    "padding": "var(--artifact-padding)",
}

INPUT_STYLE = {
    "background": COLORS["bg_input"],
    "border": "var(--border-default)",
    "border_radius": "var(--radius-control)",
    "color": COLORS["text_primary"],
    "padding": "0.875rem 1rem",
    "width": "100%",
    "height": "auto",
    "min_height": "50px",
    "line_height": "1.5",
    "font_size": "1rem",
    "transition": CONTROL_TRANSITION,
    "_focus": {
        "border_color": COLORS["accent_blue"],
        "outline": "none",
        "box_shadow": FOCUS_RING,
    },
    "_placeholder": {
        "color": COLORS["text_muted"],
    },
}

BUTTON_PRIMARY_STYLE = {
    "background": COLORS["accent_green"],
    "color": "#031315",
    "border": "none",
    "border_radius": "var(--radius-round)",
    "width": "48px",
    "height": "48px",
    "min_width": "48px",
    "padding": "0",
    "display": "flex",
    "align_items": "center",
    "justify_content": "center",
    "cursor": "pointer",
    "transition": CONTROL_TRANSITION,
    "_hover": {
        "background": "#31eee1",
        "box_shadow": "0 0 0 4px rgba(25, 215, 202, 0.12)",
    },
    "_disabled": {
        "opacity": "0.5",
        "cursor": "not-allowed",
    },
}

BUTTON_SECONDARY_STYLE = {
    "background": "rgba(16, 36, 43, 0.54)",
    "color": COLORS["text_secondary"],
    "border": "var(--border-default)",
    "border_radius": "var(--radius-control)",
    "padding": "var(--control-padding-standard)",
    "cursor": "pointer",
    "min_height": "40px",
    "transition": CONTROL_TRANSITION,
    "_hover": {
        "background": "rgba(22, 51, 59, 0.96)",
        "border_color": "rgba(167, 186, 196, 0.36)",
        "color": COLORS["text_primary"],
    },
}

MESSAGE_BUBBLE_USER = {
    "background": COLORS["user_bubble"],
    "color": COLORS["text_primary"],
    "borderRadius": "var(--radius-surface) var(--radius-surface) 4px var(--radius-surface)",
    "padding": "0.875rem 1rem",
    "maxWidth": "65%",
    "marginLeft": "auto",
    "wordWrap": "break-word",
    "boxShadow": SURFACE_SHADOW,
}

ASSISTANT_ARTIFACT_STYLE = {
    "background": "linear-gradient(180deg, rgba(16, 36, 43, 0.98), rgba(12, 28, 34, 0.98))",
    "color": COLORS["text_primary"],
    "border": "var(--border-default)",
    "borderLeft": "var(--border-artifact)",
    "borderRadius": "var(--radius-surface)",
    "padding": "var(--artifact-padding)",
    "width": "100%",
    "maxWidth": "100%",
    "marginRight": "auto",
    "wordWrap": "break-word",
    "boxShadow": SURFACE_SHADOW,
}

SIDEBAR_STYLE = {
    "width": "292px",
    "background": COLORS["bg_card"],
    "border_right": f"1px solid {COLORS['border']}",
    "height": "100vh",
    "position": "fixed",
    "left": "0",
    "top": "0",
    "padding": "1.5rem 1.125rem",
    "overflow_y": "auto",
}

MAIN_CONTENT_STYLE = {
    "margin_left": "292px",
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
    "border_radius": "var(--radius-surface)",
    "border": "var(--border-default)",
    "padding": "var(--artifact-padding)",
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
    "border_radius": "var(--radius-round)",
    "font_size": "0.75rem",
    "font_weight": "500",
}

GLOBAL_STYLES = """
:root {
    --font-display: 'IBM Plex Sans', sans-serif;
    --font-body: 'Inter', sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;
    --color-bg-base: #061216;
    --color-bg-page: #07161b;
    --color-surface: #0b1a20;
    --color-surface-raised: #10242b;
    --color-surface-hover: #102a31;
    --color-control: rgba(16, 36, 43, 0.54);
    --color-control-hover: rgba(22, 51, 59, 0.96);
    --color-input: #10242b;
    --color-artifact: #10242b;
    --color-border: #1d3942;
    --color-text-primary: #f3fbff;
    --color-text-secondary: #a7bac4;
    --color-text-muted: #6f8791;
    --color-accent: #19d7ca;
    --color-accent-soft: #75b8f2;
    --color-danger: #ef4444;
    --color-status-active: #75b8f2;
    --color-status-complete: #19d7ca;
    --color-status-idle: #6f8791;
    --radius-control: 8px;
    --radius-surface: 8px;
    --radius-round: 9999px;
    --border-default: 1px solid rgba(167, 186, 196, 0.18);
    --border-hover: 1px solid rgba(167, 186, 196, 0.34);
    --border-active: 1px solid rgba(25, 215, 202, 0.72);
    --border-artifact: 2px solid rgba(25, 215, 202, 0.58);
    --control-padding-compact: 0.55rem 0.75rem;
    --control-padding-standard: 0.7rem 0.9rem;
    --surface-header-padding: 0.875rem 1.25rem;
    --artifact-padding: 1.15rem 1.25rem;
    --control-transition: background 180ms ease, border-color 180ms ease, color 180ms ease, box-shadow 180ms ease;
    --study-rail-width: 940px;
    --workspace-gutter: 1.5rem;
    --focus-ring: 0 0 0 2px rgba(25, 215, 202, 0.28);
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: var(--font-body);
    background: var(--color-bg-page);
    color: var(--color-text-primary);
    line-height: 1.55;
}

*:focus-visible {
    outline: none;
    box-shadow: var(--focus-ring);
}

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--color-bg-base);
}

::-webkit-scrollbar-thumb {
    background: #1d3942;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #2b515b;
}

.markdown-content {
    color: var(--color-text-primary);
}

.markdown-content h1,
.markdown-content h2 {
    font-family: var(--font-display);
    font-size: 1.22rem;
    font-weight: 800;
    line-height: 1.25;
    margin-top: 1rem;
    margin-bottom: 0.35rem;
}

.markdown-content h3 {
    font-size: 1rem;
    font-weight: 750;
    line-height: 1.35;
    margin-top: 0.85rem;
    margin-bottom: 0.25rem;
}

.markdown-content p {
    margin: 0.45rem 0;
}

.markdown-content strong {
    color: var(--color-text-primary);
    font-weight: 650;
}

.markdown-content code {
    background: rgba(6, 18, 22, 0.82);
    font-family: var(--font-mono);
    padding: 0.125rem 0.375rem;
    border-radius: 4px;
    font-size: 0.85rem;
}

.markdown-content pre {
    background: rgba(6, 18, 22, 0.82);
    padding: 0.75rem;
    border-radius: 6px;
    overflow-x: auto;
    margin: 0.5rem 0;
}

.markdown-content ul,
.markdown-content ol {
    padding-left: 1.25rem;
    margin: 0.35rem 0 0.5rem;
}

.markdown-content table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.65rem 0 0.8rem;
}

.markdown-content th,
.markdown-content td {
    border: 1px solid rgba(167, 186, 196, 0.16);
    padding: 0.5rem;
    text-align: left;
}

.markdown-content th {
    background: rgba(6, 18, 22, 0.42);
    color: var(--color-text-secondary);
    font-weight: 650;
}

.analyst-message .markdown-content {
    max-width: none;
    width: 100%;
}

.analyst-message .markdown-content table {
    display: block;
    overflow-x: auto;
}

.artifact-header {
    border-bottom: var(--border-default);
    margin: -0.15rem -0.05rem 0.85rem;
    min-height: 1.35rem;
    padding: 0 0.05rem 0.6rem;
}

.artifact-label {
    min-width: 0;
    text-transform: uppercase;
}

.artifact-timestamp {
    color: var(--color-text-muted);
    font-family: var(--font-mono);
    font-size: 0.7rem;
    white-space: nowrap;
}

.state-notice {
    min-height: 220px;
    justify-content: center;
    padding: 2.25rem 1rem;
    text-align: center;
}

.state-notice svg {
    height: 40px !important;
    width: 40px !important;
}

.state-notice-title {
    color: var(--color-text-primary);
    font-size: 1.22rem;
    font-weight: 800;
}

.state-notice-detail {
    color: var(--color-text-secondary);
    font-size: 0.98rem;
    max-width: 34rem;
}

.submission-error {
    background: rgba(32, 13, 17, 0.72) !important;
    box-shadow: none !important;
    color: var(--color-text-primary);
    margin-bottom: 0.15rem;
}

.submission-error svg {
    flex: 0 0 auto;
    margin-top: 0.15rem;
}

.submission-error-title {
    color: var(--color-text-primary);
    font-size: 0.875rem;
    font-weight: 700;
    line-height: 1.25;
}

.submission-error-detail {
    color: var(--color-text-secondary);
    font-size: 0.8rem;
    line-height: 1.35;
    overflow-wrap: anywhere;
}

.submission-error-close {
    align-self: flex-start;
    color: var(--color-text-secondary);
    flex: 0 0 auto;
}

.workspace-view {
    flex: 1;
    min-height: 0;
    width: 100%;
}

.chat-view {
    display: flex;
}

.study-rail {
    margin-left: auto;
    margin-right: auto;
    max-width: var(--study-rail-width);
    width: min(100%, var(--study-rail-width));
}

.study-message-scroll {
    padding: 1.35rem var(--workspace-gutter) 1rem;
}

.study-message-rail {
    display: flex;
    flex-direction: column;
    gap: 1.05rem;
    min-height: 100%;
}

.message-row {
    scroll-margin-block: 5rem;
}

.message-scroll-highlight {
    animation: message-target-flash 1.4s ease-out;
}

@keyframes message-target-flash {
    0% {
        filter: brightness(1.28);
    }
    20% {
        filter: brightness(1.18);
    }
    100% {
        filter: brightness(1);
    }
}

.graph-view {
    display: flex;
    min-height: 0;
    overflow: hidden;
    padding: 1.25rem var(--workspace-gutter) 1.5rem;
}

.composer-inner {
    width: 100%;
}

.composer-textarea-shell {
    background: var(--color-input);
    border: var(--border-default);
    border-radius: var(--radius-control);
    box-shadow: inset 0 1px 0 rgba(243, 251, 255, 0.03);
    min-height: 50px;
    transition: var(--control-transition);
}

.composer-textarea-shell:focus-within {
    border-color: var(--color-accent-soft);
    box-shadow: var(--focus-ring);
}

.chat-composer {
    border-top: var(--border-default);
    padding: 0.9rem var(--workspace-gutter) 1.2rem;
    box-shadow: 0 -18px 44px rgba(0, 0, 0, 0.18);
}

.chat-composer textarea,
.empty-session-composer textarea {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

.chat-composer textarea:focus-visible,
.empty-session-composer textarea:focus-visible {
    border-color: transparent !important;
    box-shadow: none !important;
    outline: none !important;
}

.graph-updated-notice {
    color: var(--color-text-secondary);
    display: flex;
    justify-content: flex-start;
}

.chat-status-rail {
    background: transparent;
    padding: 0 var(--workspace-gutter) 0.9rem;
}

.bottom-status-stack {
    align-items: flex-start;
}

.composer-status-line {
    align-items: center;
    background: transparent;
    color: var(--color-text-secondary);
    min-height: 1.35rem;
    overflow: visible;
    width: 100%;
}

.composer-status-divider {
    display: none;
}

.status-card {
    background: transparent;
    max-width: 100%;
    width: 100%;
}

.agent-activity-list {
    align-items: center;
    border-left: 1px solid rgba(167, 186, 196, 0.24);
    justify-content: flex-end;
    margin-left: auto;
    min-width: 0;
    padding-left: 0.95rem;
    row-gap: 0.35rem;
}

.processing-status-group {
    flex-shrink: 0;
}

.workflow-status-detail {
    min-width: 0;
}

.agent-activity-row {
    min-height: 1.25rem;
}

.agent-activity-row span {
    white-space: nowrap;
}

.graph-updated-action {
    background: transparent;
    border: 0;
    color: var(--color-accent-soft);
    cursor: pointer;
    font-size: 0.8rem;
    font-weight: 700;
    padding: 0;
    white-space: nowrap;
    transition: var(--control-transition);
}

.graph-updated-action:hover {
    color: var(--color-text-primary);
}

.graph-workspace-panel {
    background: linear-gradient(180deg, rgba(11, 26, 32, 0.98), rgba(7, 22, 27, 0.98));
    border: var(--border-default);
    border-radius: var(--radius-surface);
    box-shadow: inset 0 1px 0 rgba(243, 251, 255, 0.04), 0 24px 70px rgba(0, 0, 0, 0.24);
    display: flex;
    flex-direction: column;
    flex: 1;
    min-height: 0;
    overflow: hidden;
    width: 100%;
}

.graph-toolbar {
    background: rgba(11, 26, 32, 0.92);
    border-bottom: var(--border-default);
    flex: 0 0 auto;
    min-height: 56px;
    padding: var(--surface-header-padding);
}

.graph-toolbar button {
    border-radius: var(--radius-control);
    color: var(--color-text-secondary);
    transition: var(--control-transition);
}

.graph-toolbar button:hover {
    background: var(--color-surface-hover);
    color: var(--color-text-primary);
}

.graph-canvas {
    background:
        radial-gradient(circle at 50% 46%, rgba(25, 215, 202, 0.09), transparent 28rem),
        linear-gradient(180deg, #08171c 0%, #061216 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    flex: 1;
    min-height: 0;
    overflow: hidden;
}

.graph-plot,
.graph-plot > div,
.graph-plot .js-plotly-plot,
.graph-plot .plot-container,
.graph-plot .svg-container {
    height: 100% !important;
    width: 100% !important;
}

.graph-plot {
    align-self: stretch;
    flex: 1;
}

.workspace-tab {
    align-items: center;
    background: var(--color-control);
    border: var(--border-default);
    border-radius: var(--radius-control);
    color: var(--color-text-secondary);
    cursor: pointer;
    display: flex;
    justify-content: flex-start;
    min-height: 56px;
    padding: 0.875rem 1rem;
    transition: var(--control-transition);
    width: 100%;
}

.workspace-tab:hover {
    background: var(--color-control-hover) !important;
    border: var(--border-hover) !important;
    color: var(--color-text-primary) !important;
}

.workspace-tab-active {
    background: linear-gradient(90deg, rgba(25, 215, 202, 0.18), rgba(25, 215, 202, 0.07));
    border: var(--border-active);
    color: var(--color-accent);
    box-shadow: inset 3px 0 0 var(--color-accent), 0 12px 34px rgba(25, 215, 202, 0.08);
}

.workspace-tab-active:hover {
    background: linear-gradient(90deg, rgba(25, 215, 202, 0.22), rgba(25, 215, 202, 0.09)) !important;
    border: var(--border-active) !important;
    color: var(--color-accent) !important;
}

.sidebar-scroll {
    min-height: 0;
    overflow-y: auto;
    padding-right: 0;
    width: 100%;
}

.sidebar-section {
    border-top: var(--border-default);
    padding-top: 1.05rem;
    width: 100%;
}

.sidebar-section-title {
    color: var(--color-text-secondary);
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0;
    text-transform: uppercase;
}

.sidebar-history-list {
    border: var(--border-default);
    border-radius: var(--radius-surface);
    overflow: hidden;
    width: 100%;
}

.sidebar-history-item {
    appearance: none;
    background: rgba(16, 36, 43, 0.46);
    border-bottom: var(--border-default);
    border-left: 0;
    border-right: 0;
    border-top: 0;
    color: inherit;
    cursor: pointer;
    display: block;
    font: inherit;
    height: auto;
    min-height: 88px;
    min-width: 0;
    overflow: hidden;
    padding: 0.85rem 0.9rem;
    position: relative;
    text-align: left;
    transition: var(--control-transition);
    width: 100%;
}

.sidebar-history-item:hover {
    background: rgba(22, 51, 59, 0.72);
}

.sidebar-history-item:last-child {
    border-bottom: 0;
}

.sidebar-history-item-active {
    background: linear-gradient(90deg, rgba(25, 215, 202, 0.16), rgba(16, 36, 43, 0.56) 100%);
    box-shadow: inset 3px 0 0 var(--color-accent);
}

.sidebar-history-row {
    align-items: flex-start !important;
}

.sidebar-history-title {
    color: var(--color-text-primary);
    display: -webkit-box;
    font-size: 0.875rem;
    font-weight: 650;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
    line-height: 1.35;
    min-width: 0;
    overflow: hidden;
    overflow-wrap: anywhere;
    white-space: normal;
}

.sidebar-history-meta {
    color: var(--color-text-secondary);
    display: block;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    line-height: 1.2;
    min-height: 0.9rem;
    white-space: nowrap;
}

.sidebar-history-dot {
    background: var(--color-accent);
    border-radius: 9999px;
    box-shadow: 0 0 12px rgba(25, 215, 202, 0.5);
    align-self: center;
    flex: 0 0 auto;
    height: 8px;
    width: 8px;
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

.mermaid-wrapper {
    background: var(--color-bg-base);
    border-radius: 6px;
    padding: 0.75rem;
    margin: 0.5rem 0;
    overflow-x: auto;
}

.mermaid-wrapper svg {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
}

.mermaid-wrapper .label,
.mermaid-wrapper text {
    font-family: var(--font-body) !important;
}

.empty-session-composer {
    align-items: center;
    background:
        radial-gradient(circle at 50% 34%, rgba(25, 215, 202, 0.13), transparent 34rem),
        linear-gradient(180deg, rgba(16, 36, 43, 0.26), rgba(6, 18, 22, 0));
    flex: 1;
    justify-content: flex-start;
    min-height: 0;
    padding: clamp(8rem, 24vh, 13.25rem) var(--workspace-gutter) var(--workspace-gutter);
}

.empty-session-intro {
    max-width: 760px;
}

.empty-session-heading {
    color: var(--color-text-primary);
    font-family: var(--font-display);
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0;
    line-height: 1.22;
    text-wrap: balance;
}

.empty-session-subtext {
    max-width: min(100%, 760px);
    white-space: nowrap;
}

.empty-session-composer .composer-inner {
    align-self: center;
    max-width: 920px;
    width: 100%;
}

.empty-session-composer form,
.starter-prompts {
    width: 100%;
}

.starter-prompts {
    gap: 0.75rem;
    margin-top: 0.35rem;
}

.mobile-shell-header {
    display: none;
}

.mobile-shell-details {
    width: 100%;
}

.mobile-shell-summary {
    align-items: center;
    border: 1px solid var(--color-border);
    border-radius: 8px;
    color: var(--color-text-secondary);
    cursor: pointer;
    display: flex;
    list-style: none;
    padding: 0.625rem 0.75rem;
}

.mobile-shell-summary::-webkit-details-marker {
    display: none;
}

@media (max-width: 768px) {
    :root {
        --workspace-gutter: 1rem;
    }

    .desktop-sidebar,
    .desktop-main-header {
        display: none !important;
    }

    .mobile-shell-header {
        background: var(--color-surface);
        border-bottom: 1px solid var(--color-border);
        display: flex;
        padding: 1rem;
        width: 100%;
    }

    .app-main-content {
        margin-left: 0 !important;
    }

    .app-main-content > div {
        min-height: 100dvh;
    }

    .graph-view {
        padding: var(--workspace-gutter);
    }

    .empty-session-composer {
        justify-content: flex-start;
        padding-top: 3.25rem;
    }

    .empty-session-heading {
        font-size: 1.75rem;
    }

    .empty-session-subtext {
        max-width: 34rem;
        white-space: normal;
    }

    .starter-prompts {
        align-items: stretch;
        flex-direction: column !important;
    }

    .starter-prompts button {
        justify-content: center;
        width: 100%;
    }

    .sidebar-section {
        padding-top: 0.9rem;
    }

    .sidebar-history-item {
        min-height: 62px;
    }

    .composer-status-line {
        align-items: flex-start !important;
        flex-direction: column !important;
    }

    .agent-activity-list {
        border-left: 0;
        justify-content: flex-start;
        margin-left: 0;
        padding-left: 0;
    }

    .processing-status-group {
        flex-wrap: wrap;
    }

    .composer-status-divider {
        display: none;
    }
}
"""
