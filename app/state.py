import asyncio
import re
import uuid
from datetime import datetime
from typing import Any

import plotly.graph_objects as go
import reflex as rx
from pydantic import BaseModel

from app.strings import AGENT_DESCRIPTIONS, QUICK_ACTION_CACHE_POLICIES, QUICK_ACTIONS
from backend.configs.chat_limits import ChatLimitsSettings
from backend.configs.constants import (
    CACHED_AGENT_REPLAY_LATENCY,
    CACHED_TOKENS_REPLAY_LATENCY,
    DEFAULT_RECENT_MESSAGE_LIMIT,
    RECURSION_LIMIT,
)
from backend.configs.enums import WorkflowEventType
from backend.configs.messages import ERROR_MESSAGE_TOO_LONG
from backend.utils.cache import get_quick_action_cache, normalize_quick_action_prompt
from backend.utils.chat_errors import UserFacingChatError
from backend.utils.chat_limits import (
    ChatAdmission,
    WorkflowTimeoutExceeded,
    get_chat_limit_service,
)
from backend.utils.helpers import logger
from backend.utils.session_memory import get_session_memory_store
from backend.workflows.learner_reflex import get_workflow

MAX_SIDEBAR_HISTORY_ITEMS = 10
SIDEBAR_TITLE_MAX_CHARS = 48


CHAT_LIMITS_SETTINGS = ChatLimitsSettings()
CHAT_MESSAGE_MAX_CHARS = CHAT_LIMITS_SETTINGS.message_max_chars
_MATH_SPAN_RE = re.compile(r"\$\$.*?\$\$|\$.*?\$", flags=re.DOTALL)
_TABLE_ROW_BOUNDARY_RE = re.compile(r"(?<!\\)\|\s+(?<!\\)\|")
_TABLE_CELL_BOUNDARY_RE = re.compile(r"(?<!\\)\|")
_TABLE_SEPARATOR_CELL_RE = re.compile(r":?-{3,}:?")


def _demote_display_math_to_inline(text: str) -> str:
    return re.sub(
        r"\$\$(.+?)\$\$",
        lambda match: f"${match.group(1).strip()}$",
        text,
        flags=re.DOTALL,
    )


def _collapse_table_display_math(text: str) -> str:
    """Keeps Markdown table rows single-line when an agent puts display math in a cell."""
    lines = text.splitlines()
    collapsed: list[str] = []
    idx = 0

    while idx < len(lines):
        line = lines[idx]
        if line.lstrip().startswith("|") and line.count("$$") % 2 == 1:
            row_parts = [line.strip()]
            idx += 1
            while idx < len(lines):
                row_parts.append(lines[idx].strip())
                if lines[idx].count("$$") % 2 == 1:
                    idx += 1
                    break
                idx += 1

            row = " ".join(part for part in row_parts if part)
            collapsed.append(_demote_display_math_to_inline(row))
            continue

        if line.lstrip().startswith("|") and "$$" in line:
            line = _demote_display_math_to_inline(line)
        collapsed.append(line)
        idx += 1

    return "\n".join(collapsed)


def _replace_math_pipes(text: str) -> str:
    """Avoids Markdown table splits from raw conditional bars inside math spans."""

    def replace_in_span(match: re.Match[str]) -> str:
        return re.sub(r"(?<!\\)\|", r"\\mid ", match.group(0))

    return _MATH_SPAN_RE.sub(replace_in_span, text)


def _escape_math_hashes(text: str) -> str:
    """Escapes literal hashes that KaTeX treats as macro parameter characters."""

    def replace_in_span(match: re.Match[str]) -> str:
        return re.sub(r"(?<!\\)#", r"\\#", match.group(0))

    return _MATH_SPAN_RE.sub(replace_in_span, text)


def _markdown_table_cells(row: str) -> list[str]:
    stripped = row.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in _TABLE_CELL_BOUNDARY_RE.split(stripped[1:-1])]


def _repair_concatenated_markdown_tables(text: str) -> str:
    """Splits table rows that an LLM concatenated onto one physical line."""
    repaired_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            repaired_lines.append(line)
            continue

        segments = _TABLE_ROW_BOUNDARY_RE.split(stripped)
        candidate_rows: list[str] = []
        for segment in segments:
            row = segment.strip()
            if not row.startswith("|"):
                row = f"| {row}"
            if not row.endswith("|"):
                row = f"{row} |"
            candidate_rows.append(row)

        candidate_cells = [_markdown_table_cells(row) for row in candidate_rows]
        column_count = len(candidate_cells[0]) if candidate_cells else 0
        has_separator = len(candidate_cells) > 1 and all(
            _TABLE_SEPARATOR_CELL_RE.fullmatch(cell) for cell in candidate_cells[1]
        )
        has_consistent_columns = column_count >= 2 and all(
            len(cells) == column_count for cells in candidate_cells
        )

        if has_separator and has_consistent_columns:
            repaired_lines.extend(candidate_rows)
        else:
            repaired_lines.append(line)

    return "\n".join(repaired_lines)


def _normalize_math_delimiters(text: str) -> str:
    """Normalizes LLM Markdown/LaTeX into forms supported by the chat renderer."""
    text = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text, flags=re.DOTALL)
    text = _collapse_table_display_math(text)
    text = _replace_math_pipes(text)
    text = _repair_concatenated_markdown_tables(text)
    return _escape_math_hashes(text)


def _sidebar_title(text: str, max_chars: int = SIDEBAR_TITLE_MAX_CHARS) -> str:
    """Builds a compact, single-line sidebar title."""
    title = re.sub(r"\s+", " ", text.strip())
    if len(title) <= max_chars:
        return title
    if max_chars <= 3:
        return title[:max_chars]
    return f"{title[: max_chars - 3].rstrip()}..."


def _extract_map_title(
    artifact: dict[str, Any], fallback_prompt: str, position: int
) -> str:
    """Extracts the best Recent Maps title from a Plotly artifact."""
    raw_title = (
        artifact.get("layout", {}).get("title", "")
        if isinstance(artifact, dict)
        else ""
    )
    if isinstance(raw_title, dict):
        raw_title = raw_title.get("text", "")
    if isinstance(raw_title, str) and raw_title.strip():
        return _sidebar_title(raw_title)
    if fallback_prompt.strip():
        return _sidebar_title(fallback_prompt)
    return f"Knowledge Map {position}"


def _build_quick_action_policy_map() -> dict[str, dict[str, Any]]:
    policy_map: dict[str, dict[str, Any]] = {}
    quick_action_prompts = {action for _, _, action in QUICK_ACTIONS}
    for raw_policy in QUICK_ACTION_CACHE_POLICIES:
        policy = dict(raw_policy)
        prompt = str(policy.get("prompt", ""))
        if (
            not prompt
            or prompt not in quick_action_prompts
            or not bool(policy.get("enabled", True))
        ):
            continue
        raw_aliases = policy.get("aliases", [])
        aliases = raw_aliases if isinstance(raw_aliases, (list, tuple)) else []
        candidates = [prompt, *aliases]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                policy_map[normalize_quick_action_prompt(candidate)] = policy
    return policy_map


QUICK_ACTION_POLICY_BY_PROMPT = _build_quick_action_policy_map()


def _get_quick_action_cache_policy(prompt: str) -> dict[str, Any] | None:
    return QUICK_ACTION_POLICY_BY_PROMPT.get(normalize_quick_action_prompt(prompt))


def _canonical_visitor_id(value: str) -> str:
    """Returns a canonical anonymous UUID, or an empty string when invalid."""
    try:
        return str(uuid.UUID(value.strip()))
    except (AttributeError, ValueError):
        return ""


async def _events_with_timeout(events, timeout_seconds: float):
    """Yields workflow events under one wall-clock timeout."""
    try:
        async with asyncio.timeout(timeout_seconds):
            async for event in events:
                yield event
    finally:
        close = getattr(events, "aclose", None)
        if close is not None:
            await close()


def _submitted_message_from_form(form_data: dict) -> str:
    """Returns the exact submitted composer value, normalized for processing."""
    return str(form_data.get("message", "")).strip()


def _model_payload(model: BaseModel) -> dict[str, Any]:
    return model.model_dump()


def _message_to_memory_payload(message: "Message") -> dict[str, str]:
    return {
        "role": message.role,
        "content": message.content,
        "agent": message.agent,
        "timestamp": message.timestamp,
        "dom_id": message.dom_id,
    }


def _message_from_memory_payload(message: dict[str, Any]) -> "Message":
    payload = dict(message)
    if payload.get("role") != "user":
        payload["content"] = _normalize_math_delimiters(str(payload.get("content", "")))
    return Message(**payload)


def _cached_response_chunks(content: str, chunk_size: int = 24) -> list[str]:
    """Splits cached content into small chunks so cache hits feel like live streaming."""
    if not content:
        return []
    return [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]


async def _persist_session_snapshot(
    session_id: str,
    messages: list[dict[str, str]],
    session_history: list[dict[str, Any]] | None = None,
    recent_maps: list[dict[str, Any]] | None = None,
    visual_artifacts: list[dict[str, Any]] | None = None,
) -> None:
    if not session_id:
        return
    try:
        await asyncio.to_thread(
            lambda: get_session_memory_store().replace_messages(
                session_id,
                messages,
                session_history=session_history,
                recent_maps=recent_maps,
                visual_artifacts=visual_artifacts,
            )
        )
    except Exception as e:
        logger.warning(f"Persisting session snapshot failed: {e}")
        pass


class Message(BaseModel):
    """A chat message."""

    content: str
    role: str  # "user" or "assistant"
    agent: str = ""  # Agent name for assistant messages
    timestamp: str = ""
    dom_id: str = ""


class SessionHistoryItem(BaseModel):
    """Sidebar metadata for one user turn in the current browser tab session."""

    id: str
    title: str
    meta: str
    message_index: int
    timestamp: str
    message_dom_id: str = ""


class RecentMapItem(BaseModel):
    """Sidebar metadata for one generated graph in the current browser tab session."""

    id: str
    title: str
    meta: str
    artifact_index: int
    timestamp: str


def _message_dom_id(position: int) -> str:
    """Builds a stable fallback DOM id for restored messages."""
    return f"message-{position}"


def _restore_ui_navigation_targets(
    messages: list[Message],
    session_history: list[SessionHistoryItem],
) -> tuple[list[Message], list[SessionHistoryItem]]:
    """Backfills message DOM ids and legacy history targets after session hydration."""
    restored_messages = [
        message
        if message.dom_id
        else message.model_copy(update={"dom_id": _message_dom_id(index)})
        for index, message in enumerate(messages)
    ]
    restored_history = []
    for item in session_history:
        message_dom_id = item.message_dom_id
        if not message_dom_id and 0 <= item.message_index < len(restored_messages):
            message_dom_id = restored_messages[item.message_index].dom_id
        restored_history.append(
            item.model_copy(update={"message_dom_id": message_dom_id})
        )
    return restored_messages, restored_history


def _session_history_selection_state(
    item_id: str,
    session_history: list[SessionHistoryItem],
    current_nonce: int,
) -> dict[str, str | int] | None:
    """Returns state updates for navigating to a current-session history item."""
    for item in session_history:
        if item.id != item_id:
            continue
        target_dom_id = item.message_dom_id or _message_dom_id(item.message_index)
        return {
            "selected_session_history_id": item.id,
            "scroll_target_message_dom_id": target_dom_id,
            "scroll_request_nonce": current_nonce + 1,
        }
    return None


def _recent_map_selection_state(
    item_id: str,
    item: RecentMapItem,
    visual_artifact_count: int,
) -> dict[str, str | int] | None:
    """Returns state updates for navigating to a valid recent map item."""
    if item.id != item_id or not 0 <= item.artifact_index < visual_artifact_count:
        return None
    return {
        "selected_recent_map_id": item.id,
        "selected_plot_index": item.artifact_index,
    }


def _merge_recent_map_items(
    visual_artifacts: list[dict[str, Any]],
    recent_maps: list[RecentMapItem],
    new_artifacts: list[dict[str, Any]],
    fallback_prompt: str,
    timestamp: str | None = None,
) -> tuple[list[dict[str, Any]], list[RecentMapItem], int, str]:
    """Appends graph artifacts while keeping recent-map indexes aligned after trimming."""
    combined_artifacts = visual_artifacts + new_artifacts
    dropped_count = max(0, len(combined_artifacts) - MAX_SIDEBAR_HISTORY_ITEMS)
    kept_artifacts = combined_artifacts[dropped_count:]

    adjusted_maps = [
        item.model_copy(update={"artifact_index": item.artifact_index - dropped_count})
        for item in recent_maps
        if item.artifact_index >= dropped_count
    ]

    map_timestamp = timestamp or datetime.now().strftime("%H:%M")
    start_index = len(visual_artifacts)
    new_items = []
    for offset, artifact in enumerate(new_artifacts):
        artifact_index = start_index + offset - dropped_count
        if artifact_index < 0:
            continue
        new_items.append(
            RecentMapItem(
                id=str(uuid.uuid4()),
                title=_extract_map_title(artifact, fallback_prompt, artifact_index + 1),
                meta=map_timestamp,
                artifact_index=artifact_index,
                timestamp=map_timestamp,
            )
        )

    merged_maps = (adjusted_maps + new_items)[-MAX_SIDEBAR_HISTORY_ITEMS:]
    merged_maps = [
        item for item in merged_maps if 0 <= item.artifact_index < len(kept_artifacts)
    ]
    selected_plot_index = len(kept_artifacts) - 1 if kept_artifacts else 0
    selected_recent_map_id = new_items[-1].id if new_items else ""
    return kept_artifacts, merged_maps, selected_plot_index, selected_recent_map_id


class AgentStatus(BaseModel):
    """Status of an agent in the workflow."""

    name: str
    is_active: bool = False
    last_action: str = ""


class AppState(rx.State):
    """The main application state."""

    # Chat state
    messages: list[Message] = []
    current_input: str = ""
    is_processing: bool = False

    # Agent status tracking
    active_agent: str = ""
    agent_history: list[str] = []

    # Visualization state
    visual_artifacts: list[dict[str, Any]] = []
    selected_plot_index: int = 0
    show_visualization: bool = False
    show_graph_updated_notice: bool = False

    # Context state (for debugging/display)
    current_context: str = ""
    show_context_panel: bool = False

    # Workspace state
    active_view: str = "chat"  # chat, graph

    # Session state
    session_started: bool = False
    session_id: str = rx.SessionStorage("", name="ai_practice_session_id")
    session_history: list[SessionHistoryItem] = []
    recent_maps: list[RecentMapItem] = []
    selected_session_history_id: str = ""
    visitor_id: str = rx.LocalStorage("", name="ai_practice_visitor_id", sync=True)
    selected_recent_map_id: str = ""
    scroll_target_message_dom_id: str = ""
    scroll_request_nonce: int = 0
    error_message: str = ""

    # Streaming state
    streaming_content: str = ""
    streaming_agent: str = ""

    @rx.var
    def is_streaming(self) -> bool:
        """True while a token stream is in progress."""
        return self.streaming_content != ""

    @rx.var
    def has_messages(self) -> bool:
        """Checks if there are any messages."""
        return len(self.messages) > 0

    @rx.var
    def has_visualization(self) -> bool:
        """Checks if there are any visualizations to display."""
        return len(self.visual_artifacts) > 0

    @rx.var
    def has_session_history(self) -> bool:
        """Checks if there are current-tab session history items."""
        return len(self.session_history) > 0

    @rx.var
    def has_recent_maps(self) -> bool:
        """Checks if there are current-tab map history items."""
        return len(self.recent_maps) > 0

    @rx.var
    def plot_count(self) -> int:
        """Total number of plots generated in this session."""
        return len(self.visual_artifacts)

    @rx.var
    def current_plot_label(self) -> str:
        """Human-readable label for the current plot position."""
        if not self.visual_artifacts:
            return ""
        return f"{self.selected_plot_index + 1} / {len(self.visual_artifacts)}"

    @rx.var(cache=True)
    def plotly_figure(self) -> go.Figure:
        """Convert the currently selected visual artifact to a Plotly Figure.

        Cached so it only recalculates when visual_artifacts or selected_plot_index
        changes, preventing Plotly re-renders on unrelated state updates.
        """
        if self.visual_artifacts and 0 <= self.selected_plot_index < len(
            self.visual_artifacts
        ):
            fig = go.Figure(self.visual_artifacts[self.selected_plot_index])
            fig.update_layout(
                autosize=True,
                margin={"l": 24, "r": 24, "t": 42, "b": 24},
            )
            return fig
        return go.Figure()

    @rx.var
    def agent_status_list(self) -> list[dict]:
        """Get list of agent statuses for display."""
        agents = [
            "orchestrator",
            "retriever",
            "researcher",
            "analyst",
            "mentor",
            "visualizer",
        ]
        return [
            {
                "name": agent,
                "is_active": agent == self.active_agent,
                "was_used": agent in self.agent_history,
                "description": AGENT_DESCRIPTIONS.get(agent, ""),
            }
            for agent in agents
        ]

    def _append_session_history_item(
        self,
        user_message: str,
        meta: str,
        message_index: int,
        message_dom_id: str,
    ) -> None:
        item = SessionHistoryItem(
            id=str(uuid.uuid4()),
            title=_sidebar_title(user_message),
            meta=meta,
            message_index=message_index,
            message_dom_id=message_dom_id,
            timestamp=meta,
        )
        self.session_history = (self.session_history + [item])[
            -MAX_SIDEBAR_HISTORY_ITEMS:
        ]
        self.selected_session_history_id = item.id

    def _append_recent_map_items(
        self, artifacts: list[dict[str, Any]], fallback_prompt: str
    ) -> None:
        if not artifacts:
            return

        visual_artifacts, recent_maps, selected_plot_index, selected_recent_map_id = (
            _merge_recent_map_items(
                self.visual_artifacts,
                self.recent_maps,
                artifacts,
                fallback_prompt,
            )
        )
        self.visual_artifacts = visual_artifacts
        self.recent_maps = recent_maps
        self.selected_plot_index = selected_plot_index
        self.selected_recent_map_id = (
            selected_recent_map_id or self.selected_recent_map_id
        )

    async def _persist_current_session_snapshot(self, session_id: str) -> None:
        async with self:
            messages = [_message_to_memory_payload(msg) for msg in self.messages]
            session_history = [_model_payload(item) for item in self.session_history]
            recent_maps = [_model_payload(item) for item in self.recent_maps]
            visual_artifacts = self.visual_artifacts[-MAX_SIDEBAR_HISTORY_ITEMS:]

        await _persist_session_snapshot(
            session_id,
            messages,
            session_history=session_history,
            recent_maps=recent_maps,
            visual_artifacts=visual_artifacts,
        )

    @rx.event(background=True)
    async def initialize_session(self):
        """Hydrates the current browser-tab session from best-effort Mongo memory."""
        async with self:
            canonical_visitor_id = _canonical_visitor_id(self.visitor_id)
            if not canonical_visitor_id:
                canonical_visitor_id = str(uuid.uuid4())
            self.visitor_id = canonical_visitor_id
            canonical_session_id = _canonical_visitor_id(self.session_id)
            self.session_id = canonical_session_id or str(uuid.uuid4())
            if not canonical_session_id:
                return
            session_id = self.session_id
            should_hydrate = (
                not self.messages and not self.session_history and not self.recent_maps
            )

        if not should_hydrate:
            return

        try:
            doc = await asyncio.to_thread(
                lambda: get_session_memory_store().get(session_id)
            )
        except Exception as e:
            logger.warning(f"Loading session snapshot failed: {e}")
            return
        if not doc:
            return

        try:
            messages = [
                _message_from_memory_payload(message)
                for message in doc.get("messages", [])
                if isinstance(message, dict)
            ]
            session_history = [
                SessionHistoryItem(**item)
                for item in doc.get("session_history", [])
                if isinstance(item, dict)
            ][-MAX_SIDEBAR_HISTORY_ITEMS:]
            visual_artifacts = doc.get("visual_artifacts", [])
            visual_artifacts = (
                visual_artifacts if isinstance(visual_artifacts, list) else []
            )
            visual_artifacts = visual_artifacts[-MAX_SIDEBAR_HISTORY_ITEMS:]
            recent_maps = [
                RecentMapItem(**item)
                for item in doc.get("recent_maps", [])
                if isinstance(item, dict)
            ][-MAX_SIDEBAR_HISTORY_ITEMS:]
        except Exception as e:
            logger.warning(f"Hydrating session snapshot failed: {e}")
            return

        async with self:
            if (
                self.session_id != session_id
                or self.messages
                or self.session_history
                or self.recent_maps
            ):
                return
            messages, session_history = _restore_ui_navigation_targets(
                messages, session_history
            )
            self.messages = messages
            self.session_history = session_history
            self.visual_artifacts = visual_artifacts
            self.recent_maps = [
                item
                for item in recent_maps
                if 0 <= item.artifact_index < len(self.visual_artifacts)
            ]
            self.selected_session_history_id = (
                self.session_history[-1].id if self.session_history else ""
            )
            self.selected_recent_map_id = (
                self.recent_maps[-1].id if self.recent_maps else ""
            )
            self.selected_plot_index = (
                self.recent_maps[-1].artifact_index if self.recent_maps else 0
            )
            self.show_visualization = bool(self.visual_artifacts)

    def set_input(self, value: str):
        """Updates the current input value."""
        if self.is_processing:
            return
        self.current_input = value

    def handle_form_submit(self, form_data: dict):
        """Handles chat form submission using the current DOM value to avoid WebSocket sync race conditions."""
        value = _submitted_message_from_form(form_data)
        if not value or self.is_processing:
            return
        self.current_input = ""
        self.is_processing = True
        return AppState.process_submitted_message(value, True)

    def toggle_context_panel(self):
        """Toggles the context panel visibility."""
        self.show_context_panel = not self.show_context_panel

    def set_active_view(self, view: str):
        """Sets the active workspace view."""
        if view in {"chat", "graph"}:
            self.active_view = view

    def open_chat(self):
        """Shows the chat workspace."""
        self.active_view = "chat"

    def open_graph(self):
        """Shows the graph workspace."""
        self.active_view = "graph"
        self.show_visualization = True
        self.show_graph_updated_notice = False

    def toggle_visualization(self):
        """Toggles between chat and graph workspaces."""
        if self.active_view == "graph":
            self.active_view = "chat"
        else:
            self.active_view = "graph"
            self.show_visualization = True

    def next_plot(self):
        """Navigates to the next plot in the history."""
        if self.selected_plot_index < len(self.visual_artifacts) - 1:
            self.selected_plot_index += 1

    def prev_plot(self):
        """Navigates to the previous plot in the history."""
        if self.selected_plot_index > 0:
            self.selected_plot_index -= 1

    def select_session_history_item(self, item_id: str):
        """Selects a current-tab session history item."""
        selection = _session_history_selection_state(
            item_id,
            self.session_history,
            self.scroll_request_nonce,
        )
        if selection is None:
            return
        self.selected_session_history_id = str(selection["selected_session_history_id"])
        self.scroll_target_message_dom_id = str(
            selection["scroll_target_message_dom_id"]
        )
        self.scroll_request_nonce = int(selection["scroll_request_nonce"])
        self.active_view = "chat"

    def select_recent_map(self, item_id: str):
        """Selects a generated map from the current browser-tab map history."""
        for item in self.recent_maps:
            selection = _recent_map_selection_state(
                item_id,
                item,
                len(self.visual_artifacts),
            )
            if selection is None:
                continue
            self.selected_recent_map_id = str(selection["selected_recent_map_id"])
            self.selected_plot_index = int(selection["selected_plot_index"])
            self.active_view = "graph"
            self.show_visualization = True
            self.show_graph_updated_notice = False
            return

    @rx.event(background=True)
    async def clear_chat(self):
        """Clears the chat history."""
        async with self:
            old_session_id = self.session_id
            self.messages = []
            self.agent_history = []
            self.session_id = str(uuid.uuid4())
            # self.current_context = "" # It's hidden now
            self.visual_artifacts = []
            self.session_history = []
            self.recent_maps = []
            self.selected_session_history_id = ""
            self.selected_recent_map_id = ""
            self.scroll_target_message_dom_id = ""
            self.scroll_request_nonce = 0
            self.selected_plot_index = 0
            self.show_visualization = False
            self.show_graph_updated_notice = False
            self.active_view = "chat"
            self.error_message = ""

        if old_session_id:
            try:
                await asyncio.to_thread(
                    lambda: get_session_memory_store().delete(old_session_id)
                )
            except Exception as e:
                logger.warning(f"Deleting session snapshot failed: {e}")

    def clear_error(self):
        """Clears the error message."""
        self.error_message = ""

    async def _replay_cached_agent_history(self, agent_history: list[str]) -> None:
        """Replays cached agent activity into the sidebar without running the workflow."""
        for agent_name in agent_history:
            if not agent_name:
                continue
            async with self:
                self.active_agent = agent_name
                if agent_name not in self.agent_history:
                    self.agent_history = self.agent_history + [agent_name]
            await asyncio.sleep(CACHED_AGENT_REPLAY_LATENCY)
        async with self:
            self.active_agent = ""

    async def _stream_cached_response(self, response: dict[str, str]) -> None:
        """Streams a cached assistant response through the same UI state as live tokens."""
        agent_name = response.get("agent", "cache")
        content = response.get("content", "")
        rendered_content = _normalize_math_delimiters(content)

        async with self:
            self.active_agent = agent_name if agent_name != "system" else ""
            self.streaming_agent = agent_name
            self.streaming_content = ""
            if (
                agent_name
                and agent_name != "system"
                and agent_name not in self.agent_history
            ):
                self.agent_history = self.agent_history + [agent_name]

        for chunk in _cached_response_chunks(rendered_content):
            async with self:
                self.streaming_content = self.streaming_content + chunk
            await asyncio.sleep(CACHED_TOKENS_REPLAY_LATENCY)

        async with self:
            self.active_agent = ""
            self.streaming_content = ""
            self.streaming_agent = ""
            self.messages = self.messages + [
                Message(
                    content=rendered_content,
                    role="assistant",
                    agent=agent_name,
                    timestamp=datetime.now().strftime("%H:%M"),
                    dom_id=f"message-{uuid.uuid4().hex}",
                )
            ]

    @rx.event(background=True)
    async def send_message(self):
        """Processes the current composer value for non-form callers."""
        async with self:
            if not self.current_input.strip() or self.is_processing:
                return
            user_message = self.current_input.strip()
            self.current_input = ""
            self.is_processing = True
        await self._process_message(user_message)

    @rx.event(background=True)
    async def process_submitted_message(
        self, user_message: str, submit_reserved: bool = False
    ):
        """Processes a submitted form value without depending on composer state timing."""
        user_message = user_message.strip()
        if not user_message:
            return
        async with self:
            if self.is_processing and not submit_reserved:
                return
            self.current_input = ""
            self.is_processing = True
        await self._process_message(user_message)

    async def _process_message(self, user_message: str):
        """Sends a captured user message through the workflow."""
        if len(user_message) > CHAT_MESSAGE_MAX_CHARS:
            async with self:
                self.error_message = ERROR_MESSAGE_TOO_LONG
                self.is_processing = False
            return

        async with self:
            existing_message_count = len(self.messages)
        quick_action_policy = _get_quick_action_cache_policy(user_message)
        should_cache_quick_action = (
            existing_message_count == 0 and quick_action_policy is not None
        )
        cache_prompt = (
            str(quick_action_policy.get("prompt", user_message))
            if quick_action_policy
            else user_message
        )
        graph_updated_this_run = False
        limit_service = get_chat_limit_service()
        admission: ChatAdmission | None = None
        turn_status = "success"
        cached = None

        async with self:
            canonical_session_id = _canonical_visitor_id(self.session_id)
            if not canonical_session_id:
                canonical_session_id = str(uuid.uuid4())
            self.session_id = canonical_session_id
            session_id = self.session_id
            visitor_id = _canonical_visitor_id(self.visitor_id)
            if not visitor_id:
                visitor_id = str(uuid.uuid4())
                self.visitor_id = visitor_id

        if should_cache_quick_action:
            try:
                cached = await asyncio.to_thread(
                    lambda: get_quick_action_cache().get(cache_prompt)
                )
            except Exception as exc:
                logger.warning(
                    "Quick-action cache lookup failed",
                    error_type=type(exc).__name__,
                )

        try:
            if cached:
                admission = await limit_service.admit_cached_turn(
                    visitor_id, session_id
                )
            else:
                admission = await limit_service.admit_expensive_turn(
                    visitor_id, session_id
                )
        except UserFacingChatError as exc:
            async with self:
                self.error_message = exc.user_message
                self.is_processing = False
            logger.info("Chat turn rejected", reason=exc.reason)
            return

        async with self:
            self.is_processing = True
            self.error_message = ""
            self.agent_history = []
            self.show_graph_updated_notice = False
            timestamp = datetime.now().strftime("%H:%M")
            message_index = len(self.messages)
            message_dom_id = f"message-{uuid.uuid4().hex}"

            # Add user message
            self.messages = self.messages + [
                Message(
                    content=user_message,
                    role="user",
                    timestamp=timestamp,
                    dom_id=message_dom_id,
                )
            ]
            self._append_session_history_item(
                user_message, timestamp, message_index, message_dom_id
            )

        try:
            if should_cache_quick_action:
                if cached:
                    cached_responses = cached.get("responses", [])
                    cached_visual_artifacts = cached.get("visual_artifacts", [])
                    cached_agent_history = cached.get("agent_history", [])
                    if not cached_agent_history:
                        cached_agent_history = [
                            response.get("agent", "")
                            for response in cached_responses
                            if response.get("agent")
                        ]
                        if cached_visual_artifacts:
                            cached_agent_history.append("visualizer")
                    await self._replay_cached_agent_history(cached_agent_history)
                    async with self:
                        if cached_visual_artifacts:
                            self._append_recent_map_items(
                                cached_visual_artifacts, user_message
                            )
                            self.show_visualization = True
                            graph_updated_this_run = True
                    for response in cached_responses:
                        await self._stream_cached_response(response)
                    await self._persist_current_session_snapshot(session_id)
                    return

            workflow = get_workflow()

            # Prepare message history
            recent_messages = self.messages[:-1][-DEFAULT_RECENT_MESSAGE_LIMIT:]
            message_history = [
                {"role": msg.role, "content": msg.content} for msg in recent_messages
            ]
            cache_responses: list[dict[str, str]] = []
            cache_visual_artifacts: list[dict[str, Any]] = []
            cache_context = ""
            cache_error = False
            try:
                session_summary = await asyncio.to_thread(
                    lambda: get_session_memory_store().get_prompt_summary(
                        session_id, user_message
                    )
                )
            except Exception as e:
                logger.warning(f"Loading session summary failed: {e}")
                session_summary = ""

            # Stream through workflow with per-token updates
            async for event in _events_with_timeout(
                workflow.stream_with_tokens(
                    user_message=user_message,
                    message_history=message_history,
                    recursion_limit=RECURSION_LIMIT,
                    session_id=session_id,
                    session_summary=session_summary,
                    turn_context=admission.context,
                ),
                limit_service.settings.workflow_timeout_seconds,
            ):
                async with self:
                    if event.type == WorkflowEventType.AGENT_START:
                        self.active_agent = event.data["agent"]
                        admission.context.agents.add(event.data["agent"])
                        if event.data["agent"] not in self.agent_history:
                            self.agent_history = self.agent_history + [
                                event.data["agent"]
                            ]

                    elif event.type == WorkflowEventType.AGENT_END:
                        self.active_agent = ""

                    elif event.type == WorkflowEventType.TOKEN:
                        agent_name = event.data.get("agent", "")
                        self.streaming_agent = agent_name
                        self.streaming_content = (
                            self.streaming_content + event.data["chunk"]
                        )

                    # Context is hidden for now
                    # elif event.type == WorkflowEventType.CONTEXT_UPDATE:
                    #     raw = (event.data["context"]
                    #            .replace("[RESEARCH_COMPLETE]", "")
                    #            .replace("Visual artifact generated", "")
                    #            .replace("Visualization failed", "")
                    #            .strip())
                    #     try:
                    #         import json as _json
                    #         parsed = _json.loads(raw)
                    #         self.current_context = _json.dumps(parsed, ensure_ascii=False, indent=2)
                    #     except Exception:
                    #         self.current_context = raw

                    elif event.type == WorkflowEventType.VISUALIZATION:
                        artifacts = event.data["artifacts"]
                        cache_visual_artifacts.extend(artifacts)
                        self._append_recent_map_items(artifacts, user_message)
                        self.show_visualization = True
                        graph_updated_this_run = True

                    elif event.type == WorkflowEventType.RESPONSE:
                        agent_name = event.data.get("agent", "")
                        content = event.data["content"]
                        if agent_name != "system":
                            cache_responses.append(
                                {"content": content, "agent": agent_name}
                            )
                        self.streaming_content = ""
                        self.streaming_agent = ""
                        self.messages = self.messages + [
                            Message(
                                content=_normalize_math_delimiters(content),
                                role="assistant",
                                agent=agent_name,
                                timestamp=datetime.now().strftime("%H:%M"),
                                dom_id=f"message-{uuid.uuid4().hex}",
                            )
                        ]

                    elif event.type == WorkflowEventType.ERROR:
                        cache_error = True
                        turn_status = event.data.get("reason", "workflow_error")
                        self.error_message = event.data.get(
                            "message", "Unknown error occurred"
                        )

                    elif event.type == WorkflowEventType.CONTEXT_UPDATE:
                        cache_context = event.data.get("context", "")

            if should_cache_quick_action and not cache_error:
                async with self:
                    cache_agent_history = list(self.agent_history)
                ttl_seconds = (
                    quick_action_policy.get("ttl_seconds")
                    if quick_action_policy
                    else None
                )
                try:
                    ttl_seconds = int(ttl_seconds) if ttl_seconds is not None else None
                except (TypeError, ValueError):
                    ttl_seconds = None
                responses_to_cache = (
                    cache_responses
                    if bool(quick_action_policy.get("cache_responses", True))
                    else []
                )
                artifacts_to_cache = (
                    cache_visual_artifacts
                    if bool(quick_action_policy.get("cache_visual_artifacts", True))
                    else []
                )
                try:
                    await asyncio.to_thread(
                        lambda: get_quick_action_cache().set(
                            cache_prompt,
                            responses_to_cache,
                            artifacts_to_cache,
                            cache_context,
                            agent_history=cache_agent_history,
                            ttl_seconds=ttl_seconds,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Quick-action cache persistence failed: {e}")
                    pass

            await self._persist_current_session_snapshot(session_id)

        except TimeoutError:
            turn_status = WorkflowTimeoutExceeded.reason
            async with self:
                self.error_message = WorkflowTimeoutExceeded.user_message
            await self._persist_current_session_snapshot(session_id)
        except UserFacingChatError as exc:
            turn_status = exc.reason
            async with self:
                self.error_message = exc.user_message
            await self._persist_current_session_snapshot(session_id)
        except Exception as exc:
            turn_status = "workflow_error"
            logger.error(
                "Chat request processing failed",
                error_type=type(exc).__name__,
            )
            async with self:
                self.error_message = "I encountered an error while processing your request. Please try again."
            await self._persist_current_session_snapshot(session_id)
        finally:
            if admission is not None:
                await limit_service.finish_turn(admission, turn_status)
            async with self:
                self.is_processing = False
                self.active_agent = ""
                self.streaming_content = ""
                self.streaming_agent = ""
                if graph_updated_this_run and self.active_view == "chat":
                    self.show_graph_updated_notice = True


class SidebarState(rx.State):
    """State for sidebar interactions."""

    selected_mode: str = "interview"  # interview, research, review

    def set_mode(self, mode: str):
        """Set the current mode."""
        self.selected_mode = mode


class VisualizationState(rx.State):
    """State for visualization controls."""

    zoom_level: float = 1.0
    show_labels: bool = True

    def zoom_in(self):
        """Zoom in on the visualization."""
        self.zoom_level = min(self.zoom_level + 0.2, 3.0)

    def zoom_out(self):
        """Zoom out on the visualization."""
        self.zoom_level = max(self.zoom_level - 0.2, 0.5)

    def reset_zoom(self):
        """Reset zoom level."""
        self.zoom_level = 1.0

    def toggle_labels(self):
        """Toggle label visibility."""
        self.show_labels = not self.show_labels
