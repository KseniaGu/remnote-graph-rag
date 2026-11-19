import asyncio
import re
import uuid
from collections.abc import Callable
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from threading import Event
from time import monotonic
from typing import Any

import reflex as rx
from pydantic import BaseModel

from app.strings import AGENT_DESCRIPTIONS, QUICK_ACTION_CACHE_POLICIES, QUICK_ACTIONS
from backend.configs.chat_limits import ChatLimitsSettings
from backend.configs.constants import (
    CACHED_TOKENS_REPLAY_LATENCY,
    DEFAULT_RECENT_MESSAGE_LIMIT,
    RECURSION_LIMIT,
)
from backend.configs.enums import WorkflowEventType
from backend.configs.messages import ERROR_MESSAGE_TOO_LONG, ERROR_WORKFLOW_FAILED
from backend.utils.async_tasks import complete_on_cancel
from backend.utils.cache import get_quick_action_cache, normalize_quick_action_prompt
from backend.utils.chat_errors import UserFacingChatError
from backend.utils.chat_limits import (
    ChatAdmission,
    VisitorActiveTurnExceeded,
    WorkflowTimeoutExceeded,
    get_chat_limit_service,
)
from backend.utils.helpers import logger
from backend.utils.session_memory import get_session_memory_store
from backend.workflows.learner_reflex import get_workflow, resolve_response_evidence
from backend.workflows.presentation import (
    MAX_RECENT_MAPS,
    CachedTurn,
    ExecutionStage,
    GraphArtifact,
    GraphNode,
    GraphPayload,
    Message,
    ResponseSource,
    SessionSnapshot,
    StageOutcome,
    TurnExecution,
    TurnIdentity,
    TurnOutcome,
    validated_web_url,
)

MAX_SESSION_HISTORY_ITEMS = 10
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


@asynccontextmanager
async def _events_with_timeout(events, timeout_seconds: float):
    """Keep the wall-clock timeout and stream cleanup in the consuming task."""
    async with asyncio.timeout(timeout_seconds), aclosing(events):
        yield events


def _submitted_message_from_form(form_data: dict) -> str:
    """Returns the exact submitted composer value, normalized for processing."""
    return str(form_data.get("message", "")).strip()


def _cached_response_chunks(content: str, chunk_size: int = 24) -> list[str]:
    """Splits cached content into small chunks so cache hits feel like live streaming."""
    if not content:
        return []
    return [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]


@dataclass
class _ActiveRequest:
    session_id: str
    nonce: str
    task: asyncio.Task
    invalidated: Event = field(default_factory=Event)

    def cancel(self) -> None:
        self.invalidated.set()
        if not self.task.cancelling():
            self.task.cancel()


_active_requests: dict[str, _ActiveRequest] = {}


class SessionHistoryItem(BaseModel):
    """Sidebar link to the user message that opened a turn."""

    turn_id: str
    title: str
    timestamp: str
    message_dom_id: str


class RecentMapItem(BaseModel):
    """Sidebar link to a retained graph artifact."""

    artifact_id: str
    title: str
    timestamp: str
    retrieval_text: str = ""


@dataclass
class LiveAgent:
    name: str
    outcome: str


@dataclass
class StudyStage:
    agent: str
    outcome: StageOutcome
    duration: str


@dataclass
class StudyTurn:
    prompt: Message
    responses: list[Message] = field(default_factory=list)
    maps: list[RecentMapItem] = field(default_factory=list)
    outcome: str = ""
    origin: str = ""
    stages: list[StudyStage] = field(default_factory=list)


@dataclass
class StudySource:
    title: str
    path: str
    excerpt: str
    url: str
    summary: str = ""
    is_untitled_note: bool = False


@dataclass
class StudySources:
    key: str = ""
    title: str = ""
    status: str = ""
    notes: list[StudySource] = field(default_factory=list)
    relationship_notes: list[StudySource] = field(default_factory=list)
    web: list[StudySource] = field(default_factory=list)


def _study_source(source: ResponseSource, *, note_number: int = 1) -> StudySource:
    web = source.kind == "web"
    path = (
        " / ".join(part.strip() for part in source.path if part.strip()) if web else ""
    )
    excerpt = (source.excerpt or "").strip() if web else ""
    title = (source.title or "").strip()
    return StudySource(
        title or ("Untitled web source" if web else f"Note {note_number}"),
        path,
        excerpt,
        validated_web_url(source.url) or "",
        (source.summary or "").strip(),
        is_untitled_note=not web and not title,
    )


def _with_note_previews(message: Message) -> Message:
    evidence = resolve_response_evidence(message.evidence)
    if evidence is message.evidence:
        return message
    return message.model_copy(update={"evidence": evidence})


def _stage_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    return "<0.01 s" if 0 < seconds < 0.01 else f"{seconds:.2f} s"


class AppState(rx.State):
    """The main application state."""

    # Chat state
    messages: list[Message] = []
    current_input: str = ""
    is_processing: bool = False

    # Agent status tracking
    active_agent: str = ""
    agent_history: list[str] = []
    executions: list[TurnExecution] = []
    _processing_turn_id: str = ""

    # Visualization state
    _visual_artifacts: list[GraphArtifact] = []
    graph_draft: str = ""
    graph_draft_nonce: int = 0
    graph_source_nonce: int = 0
    _selected_graph_source: tuple[str, str, str] | None = None

    # Context state (for debugging/display)
    current_context: str = ""
    show_context_panel: bool = False

    # Workspace state
    active_view: str = "chat"  # chat, graph

    # Session state
    session_started: bool = False
    session_id: str = rx.SessionStorage("", name="ai_practice_session_id")
    selected_session_history_id: str = ""
    selected_source_response_id: str = ""
    visitor_id: str = rx.LocalStorage("", name="ai_practice_visitor_id", sync=True)
    selected_recent_map_id: str = ""
    scroll_target_message_dom_id: str = ""
    scroll_request_nonce: int = 0
    error_message: str = ""
    _bound_session_id: str = ""
    _hydrated_session_id: str = ""
    _request_nonce: str = ""

    # Streaming state
    streaming_content: str = ""
    streaming_agent: str = ""
    streaming_response_id: str = ""

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
        return len(self._visual_artifacts) > 0

    @rx.var
    def session_history(self) -> list[SessionHistoryItem]:
        """Builds recent turn links from the committed prompt records."""
        return [
            SessionHistoryItem(
                turn_id=message.turn_id,
                title=_sidebar_title(message.content),
                timestamp=message.timestamp,
                message_dom_id=message.dom_id,
            )
            for message in self.messages
            if message.role == "user"
        ][-MAX_SESSION_HISTORY_ITEMS:]

    @rx.var
    def recent_maps(self) -> list[RecentMapItem]:
        """Builds map links from the retained artifacts in display order."""
        return [
            RecentMapItem(
                artifact_id=artifact.artifact_id,
                title=artifact.title.strip() or "Untitled map",
                timestamp=artifact.timestamp,
            )
            for artifact in self._visual_artifacts
        ]

    @rx.var
    def has_session_history(self) -> bool:
        """Checks if there are current-tab session history items."""
        return len(self.session_history) > 0

    @rx.var
    def has_recent_maps(self) -> bool:
        """Checks if there are current-tab map history items."""
        return len(self.recent_maps) > 0

    @rx.var
    def selected_map_index(self) -> int:
        return next(
            (
                index
                for index, graph in enumerate(self._visual_artifacts)
                if graph.artifact_id == self.selected_recent_map_id
            ),
            -1,
        )

    @rx.var
    def selected_graph(self) -> GraphArtifact | None:
        if self.selected_map_index >= 0:
            return self._visual_artifacts[self.selected_map_index]
        return None

    @property
    def _processing_execution(self) -> TurnExecution | None:
        if self.is_processing and self.executions:
            report = self.executions[-1]
            if report.turn_id == self._processing_turn_id:
                return report
        return None

    @rx.var
    def processing_from_cache(self) -> bool:
        report = self._processing_execution
        return report is not None and report.origin == "cache"

    @rx.var
    def live_agent_progress(self) -> list[LiveAgent]:
        """Summarize the observed invocations without changing the execution log."""
        report = self._processing_execution
        outcomes: dict[str, StageOutcome] = {}
        running: set[str] = set()
        if report is not None:
            for stage in report.stages:
                outcomes[stage.agent] = stage.outcome
                if stage.outcome == "running":
                    running.add(stage.agent)
        return [
            LiveAgent(
                name.capitalize(),
                "running" if name in running else outcomes.get(name, "inactive"),
            )
            for name in AGENT_DESCRIPTIONS
        ]

    @rx.var
    def study_turns(self) -> list[StudyTurn]:
        """Group visible responses, retained maps and observed execution by turn."""
        turns: dict[tuple[str, str], StudyTurn] = {}
        for message in self.messages:
            key = (message.session_id, message.turn_id)
            if message.role == "user":
                turns[key] = StudyTurn(prompt=message)
            elif message.agent != "visualizer" and key in turns:
                turns[key].responses.append(message)
        for report in self.executions:
            turn = turns.get((report.session_id, report.turn_id))
            if turn is not None:
                turn.outcome = report.outcome
                turn.origin = report.origin
                turn.stages = [
                    StudyStage(
                        agent=stage.agent.capitalize(),
                        outcome=stage.outcome,
                        duration=_stage_duration(stage.duration_seconds),
                    )
                    for stage in report.stages
                ]
        for artifact in self._visual_artifacts:
            turn = turns.get((artifact.session_id, artifact.turn_id))
            if turn is not None:
                turn.maps.append(
                    RecentMapItem(
                        artifact_id=artifact.artifact_id,
                        title=artifact.title.strip() or "Untitled map",
                        timestamp=artifact.timestamp,
                        retrieval_text=" & ".join(artifact.retrieval_queries),
                    )
                )
        return list(turns.values())

    @rx.var
    def study_sources(self) -> StudySources:
        """Resolve an explicit response or the selected turn's latest response."""
        if self._selected_graph_source:
            artifact_id, concept_id, source_id = self._selected_graph_source
            graph = next(
                (
                    item
                    for item in self._visual_artifacts
                    if item.artifact_id == artifact_id
                    and item.session_id == self.session_id
                ),
                None,
            )
            if graph:
                concept = next(node for node in graph.nodes if node.id == concept_id)
                source = next(
                    item for item in graph.sources if item.source_id == source_id
                )
                result = StudySources(
                    key=artifact_id,
                    title=f"{concept.label} · {graph.title}",
                    status="graph",
                )
                (result.notes if source.kind == "note" else result.web).append(
                    _study_source(source)
                )
                return result
        turns = self.study_turns
        if not turns:
            return StudySources()
        turn = next(
            (
                turn
                for turn in turns
                if turn.prompt.turn_id == self.selected_session_history_id
            ),
            turns[-1],
        )
        response = next(
            (
                response
                for response in turn.responses
                if response.response_id == self.selected_source_response_id
            ),
            None,
        )
        if response is None:
            if turn.outcome == "running" and (
                not turn.responses
                or (
                    turn.prompt.dom_id == turns[-1].prompt.dom_id
                    and self.streaming_response_id
                )
            ):
                return StudySources(
                    key=turn.prompt.turn_id,
                    title=_sidebar_title(turn.prompt.content),
                    status="pending",
                )
            if turn.responses:
                response = turn.responses[-1]
            else:
                return StudySources(
                    key=turn.prompt.turn_id,
                    title=_sidebar_title(turn.prompt.content),
                    status="no_response",
                )

        title = _sidebar_title(turn.prompt.content)
        if len(turn.responses) > 1:
            position = turn.responses.index(response) + 1
            title = f"{title} · Response {position} of {len(turn.responses)}"
        evidence = response.evidence
        result = StudySources(
            key=response.response_id or "",
            title=title,
            status="unavailable" if evidence is None else "ready",
        )
        if evidence is not None:
            for source in sorted(
                evidence.sources, key=lambda row: row.supports_relationship
            ):
                if source.kind == "web":
                    result.web.append(_study_source(source))
                elif (source.excerpt or "").strip() or (
                    source.supports_relationship and source.summary
                ):
                    notes = (
                        result.relationship_notes
                        if source.supports_relationship
                        else result.notes
                    )
                    note_number = len(result.notes) + len(result.relationship_notes) + 1
                    notes.append(_study_source(source, note_number=note_number))
            if not result.notes and not result.relationship_notes and not result.web:
                result.status = (
                    "relationship_notes_unavailable"
                    if evidence.has_relationship_context
                    or any(source.kind == "note" for source in evidence.sources)
                    else "empty"
                )
        return result

    @rx.event
    def select_response_sources(self, response_id: str):
        """Pin sources to a visible assistant response in the current session."""
        for message in self.messages:
            if (
                message.session_id == self.session_id
                and message.role == "assistant"
                and message.agent != "visualizer"
                and message.response_id == response_id
            ):
                self.selected_session_history_id = message.turn_id
                self.selected_source_response_id = response_id
                self._selected_graph_source = None
                return

    def _append_maps(self, artifacts: list[GraphArtifact]) -> None:
        if artifacts:
            self._visual_artifacts = [*self._visual_artifacts, *artifacts][
                -MAX_RECENT_MAPS:
            ]
            if self.selected_map_index < 0:
                self.selected_recent_map_id = artifacts[-1].artifact_id

    def _owns_request(self, request: _ActiveRequest) -> bool:
        return (
            self.session_id == request.session_id
            and self._request_nonce == request.nonce
            and self.is_processing
            and not request.invalidated.is_set()
        )

    async def _write_for_request(
        self, request: _ActiveRequest, write: Callable
    ) -> None:
        """Skip queued writes after invalidation and drain writes already in progress."""

        def guarded_write():
            if not request.invalidated.is_set():
                write()

        await complete_on_cancel(asyncio.to_thread(guarded_write))

    async def _persist_current_session_snapshot(self, request: _ActiveRequest) -> None:
        try:
            async with self:
                if (
                    not self._owns_request(request)
                    or self._hydrated_session_id != request.session_id
                ):
                    return
                snapshot = SessionSnapshot(
                    session_id=request.session_id,
                    messages=[
                        message.model_copy(deep=True) for message in self.messages
                    ],
                    artifacts=[
                        artifact.model_copy() for artifact in self._visual_artifacts
                    ],
                    executions=[
                        report.model_copy(deep=True) for report in self.executions
                    ],
                )
            await self._write_for_request(
                request, lambda: get_session_memory_store().replace_snapshot(snapshot)
            )
        except Exception as exc:
            logger.warning(
                "Persisting session snapshot failed", error_type=type(exc).__name__
            )

    async def _load_session_snapshot(
        self, session_id: str
    ) -> tuple[bool, SessionSnapshot | None]:
        def load():
            snapshot = get_session_memory_store().get(
                session_id, require_available=True
            )
            if snapshot is not None and snapshot.session_id == session_id:
                snapshot = snapshot.model_copy(
                    update={
                        "messages": [
                            _with_note_previews(message)
                            for message in snapshot.messages
                        ]
                    }
                )
            return snapshot

        try:
            snapshot = await asyncio.to_thread(load)
        except Exception as exc:
            logger.warning(
                "Loading session snapshot failed", error_type=type(exc).__name__
            )
            return False, None
        return True, snapshot

    def _restore_snapshot(
        self, session_id: str, snapshot: SessionSnapshot | None
    ) -> None:
        self._hydrated_session_id = session_id
        if snapshot is None or snapshot.session_id != session_id:
            return
        self.messages = snapshot.messages
        self.executions = snapshot.executions
        user_turns = [
            message.turn_id for message in snapshot.messages if message.role == "user"
        ]
        self.selected_session_history_id = user_turns[-1] if user_turns else ""
        self.selected_source_response_id = ""
        self._append_maps(snapshot.artifacts)

    @rx.event(background=True)
    async def initialize_session(self):
        """Hydrates the current browser-tab session from best-effort Mongo memory."""
        async with self:
            canonical_visitor_id = _canonical_visitor_id(self.visitor_id)
            if not canonical_visitor_id:
                canonical_visitor_id = str(uuid.uuid4())
            self.visitor_id = canonical_visitor_id
            if not self._bind_session():
                return
            session_id = self.session_id
            nonce = self._request_nonce
            if (
                self.messages
                or self.is_processing
                or self._hydrated_session_id == session_id
            ):
                return

        # A fresh Reflex client may reconnect to a conversation still finishing.
        active = _active_requests.get(session_id)
        if active:
            await asyncio.gather(asyncio.shield(active.task), return_exceptions=True)
        available, snapshot = await self._load_session_snapshot(session_id)
        if not available:
            return

        async with self:
            if (
                self.session_id != session_id
                or self._request_nonce != nonce
                or self.is_processing
                or self.messages
                or self._hydrated_session_id == session_id
            ):
                return
            self._restore_snapshot(session_id, snapshot)

    def set_input(self, value: str):
        """Updates the current input value."""
        if self.is_processing:
            return
        self.current_input = value

    @rx.event(background=True)
    async def handle_form_submit(self, form_data: dict):
        """Handles chat form submission using the current DOM value to avoid WebSocket sync race conditions."""
        await self._process_message(_submitted_message_from_form(form_data))

    def toggle_context_panel(self):
        """Toggles the context panel visibility."""
        self.show_context_panel = not self.show_context_panel

    def open_chat(self):
        """Shows the Study workspace."""
        self.active_view = "chat"

    def open_graph(self):
        """Shows the graph workspace."""
        self.active_view = "graph"

    def select_session_history_item(self, item_id: str):
        """Selects a current-tab session history item."""
        for item in self.session_history:
            if item.turn_id == item_id:
                self.selected_session_history_id = item.turn_id
                self.selected_source_response_id = ""
                self._selected_graph_source = None
                self.scroll_target_message_dom_id = item.message_dom_id
                self.scroll_request_nonce += 1
                self.active_view = "chat"
                return

    def select_recent_map(self, item_id: str):
        """Opens a retained map owned by the current conversation."""
        if any(
            artifact.artifact_id == item_id and artifact.session_id == self.session_id
            for artifact in self._visual_artifacts
        ):
            self.selected_recent_map_id = item_id
            self.open_graph()

    def _active_graph_concept(
        self, artifact_id: str, concept_id: str
    ) -> GraphNode | None:
        graph = self.selected_graph
        if (
            self.active_view == "graph"
            and graph is not None
            and graph.session_id == self.session_id
            and graph.artifact_id == artifact_id
        ):
            return next((node for node in graph.nodes if node.id == concept_id), None)
        return None

    @rx.event
    def prepare_graph_explanation(self, artifact_id: str, concept_id: str):
        """Prepare an editable Study draft for a concept in the active map."""
        concept = self._active_graph_concept(artifact_id, concept_id)
        if concept:
            suffix = "." if not concept.label.endswith((".", "?", "!")) else ""
            self.graph_draft = f"Explain {concept.label}{suffix}"[
                :CHAT_MESSAGE_MAX_CHARS
            ]
            self.graph_draft_nonce += 1
            self.open_chat()

    @rx.event
    def show_graph_source(self, artifact_id: str, concept_id: str, source_id: str):
        concept = self._active_graph_concept(artifact_id, concept_id)
        if (
            concept
            and source_id in concept.source_ids
            and any(
                source.source_id == source_id for source in self.selected_graph.sources
            )
        ):
            self._selected_graph_source = (artifact_id, concept_id, source_id)
            self.graph_source_nonce += 1
            self.open_chat()

    def _reset_stream(self) -> None:
        self.streaming_content = ""
        self.streaming_agent = ""
        self.streaming_response_id = ""

    def _reset_conversation(self) -> None:
        self.messages = []
        self.current_input = ""
        self.agent_history = []
        self.active_agent = ""
        self.executions = []
        self._processing_turn_id = ""
        self._visual_artifacts = []
        self.current_context = ""
        self.selected_session_history_id = ""
        self.selected_source_response_id = ""
        self.selected_recent_map_id = ""
        self.scroll_target_message_dom_id = ""
        self.scroll_request_nonce = 0
        self.graph_draft = ""
        self._selected_graph_source = None
        self.active_view = "chat"
        self.error_message = ""
        self.is_processing = False
        self._hydrated_session_id = ""
        self._request_nonce = ""
        self._reset_stream()

    @rx.event(background=True)
    async def clear_chat(self):
        """Rotates the conversation before draining writes and retiring the old snapshot."""
        async with self:
            old_session_id = self.session_id
            request = _active_requests.get(old_session_id)
            if request:
                request.cancel()
            self._reset_conversation()
            self.session_id = str(uuid.uuid4())
            self._bound_session_id = self._hydrated_session_id = self.session_id

        if old_session_id:
            try:
                await complete_on_cancel(self._retire_session(old_session_id, request))
            except Exception as exc:
                logger.warning(
                    "Deleting session snapshot failed", error_type=type(exc).__name__
                )

    async def _retire_session(
        self, session_id: str, request: _ActiveRequest | None
    ) -> None:
        if request:
            await asyncio.gather(request.task, return_exceptions=True)
        await asyncio.to_thread(lambda: get_session_memory_store().delete(session_id))

    def clear_error(self):
        """Clears the error message."""
        self.error_message = ""

    async def _stream_cached_response(
        self, request: _ActiveRequest, response: Message
    ) -> None:
        """Streams a remapped cached response through the live token UI."""
        for chunk in _cached_response_chunks(response.content):
            async with self:
                if not self._owns_request(request):
                    return
                self.streaming_response_id = response.response_id or ""
                self.streaming_agent = response.agent
                self.streaming_content += chunk
            await asyncio.sleep(CACHED_TOKENS_REPLAY_LATENCY)

        response = await asyncio.to_thread(_with_note_previews, response)
        async with self:
            if self._owns_request(request):
                self._reset_stream()
                self.messages = [*self.messages, response]

    def _bind_session(self) -> bool:
        canonical_session_id = _canonical_visitor_id(self.session_id)
        session_id = canonical_session_id or str(uuid.uuid4())
        if self._bound_session_id and self._bound_session_id != session_id:
            old_request = _active_requests.get(self._bound_session_id)
            if old_request and old_request.nonce == self._request_nonce:
                old_request.cancel()
            self._reset_conversation()
        self.session_id = self._bound_session_id = session_id
        if not canonical_session_id:
            self._hydrated_session_id = session_id
        return bool(canonical_session_id)

    @rx.event(background=True)
    async def send_message(self):
        """Processes the current composer value for non-form callers."""
        await self._process_message(None)

    async def _process_message(self, user_message: str | None):
        """Own admission, mutation, persistence and cleanup for one request."""
        request: _ActiveRequest | None = None
        admission: ChatAdmission | None = None
        execution: TurnExecution | None = None
        execution_started_at = monotonic()
        outcome: TurnOutcome = "incomplete"
        status = "workflow_incomplete"
        limit_service = None
        try:
            async with self:
                if user_message is None:
                    user_message = self.current_input.strip()
                if not user_message:
                    return
                self._bind_session()
                if self.is_processing:
                    return
                session_id = self.session_id
                if session_id in _active_requests:
                    self.error_message = VisitorActiveTurnExceeded.user_message
                    return
                self.current_input = ""
                if len(user_message) > CHAT_MESSAGE_MAX_CHARS:
                    self.error_message = ERROR_MESSAGE_TOO_LONG
                    return
                task = asyncio.current_task()
                assert task is not None
                self.is_processing = True
                self._processing_turn_id = ""
                self._request_nonce = str(uuid.uuid4())
                request = _ActiveRequest(session_id, self._request_nonce, task)
                _active_requests[session_id] = request
                visitor_id = _canonical_visitor_id(self.visitor_id) or str(uuid.uuid4())
                self.visitor_id = visitor_id
                needs_hydration = (
                    not self.messages and self._hydrated_session_id != session_id
                )
            if needs_hydration:
                available, snapshot = await self._load_session_snapshot(session_id)
            async with self:
                if not self._owns_request(request):
                    return
                if needs_hydration and available:
                    self._restore_snapshot(session_id, snapshot)
                known_empty = (
                    self._hydrated_session_id == session_id and not self.messages
                )
                message_history = [
                    {"role": msg.role, "content": msg.content}
                    for msg in self.messages[-DEFAULT_RECENT_MESSAGE_LIMIT:]
                ]
            policy = (
                _get_quick_action_cache_policy(user_message) if known_empty else None
            )
            limit_service = get_chat_limit_service()
            cached = None
            if policy:
                cache_prompt = str(policy.get("prompt", user_message))
                try:
                    cached = await asyncio.to_thread(
                        lambda: get_quick_action_cache().get(cache_prompt)
                    )
                except Exception as exc:
                    logger.warning(
                        "Quick-action cache lookup failed",
                        error_type=type(exc).__name__,
                    )
            async with self:
                if not self._owns_request(request):
                    return
            admit = (
                limit_service.admit_cached_turn
                if cached
                else limit_service.admit_expensive_turn
            )
            admission = await admit(visitor_id, session_id)
            turn = TurnIdentity(
                session_id=session_id, turn_id=admission.context.turn_id
            )
            identity = turn.model_dump()

            async with self:
                if not self._owns_request(request):
                    status = "cancelled"
                    return
                timestamp = datetime.now().strftime("%H:%M")
                self.error_message = ""
                self.agent_history = []
                self.messages = [
                    *self.messages,
                    Message(
                        **identity,
                        content=user_message,
                        role="user",
                        agent="",
                        timestamp=timestamp,
                        dom_id=f"message-{uuid.uuid4().hex}",
                    ),
                ]
                self.selected_session_history_id = turn.turn_id
                self.selected_source_response_id = ""
                self._selected_graph_source = None
                execution = TurnExecution(
                    **identity, origin="cache" if cached else "workflow"
                )
                self._processing_turn_id = turn.turn_id
                self.executions = [*self.executions, execution]

            if cached:
                replay = cached.replay(turn, timestamp)
                async with self:
                    if not self._owns_request(request):
                        return
                    self.agent_history = replay.agent_history
                    self._append_maps(replay.artifacts)
                for response in replay.responses:
                    await self._stream_cached_response(request, response)
                outcome = "answer" if replay.responses else "graph"
                status = "success"
                return

            try:
                session_summary = await asyncio.to_thread(
                    lambda: get_session_memory_store().get_prompt_summary(
                        session_id, user_message
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Loading session summary failed", error_type=type(exc).__name__
                )
                session_summary = ""
            async with self:
                if not self._owns_request(request):
                    return
            execution_started_at = monotonic()
            events = get_workflow().stream_with_tokens(
                user_message=user_message,
                message_history=message_history,
                recursion_limit=RECURSION_LIMIT,
                session_id=session_id,
                session_summary=session_summary,
                turn_context=admission.context,
                execution_started_at=execution_started_at,
            )
            artifacts: list[GraphArtifact] = []
            outcome, status = await self._consume_workflow(
                request,
                admission,
                events,
                user_message,
                artifacts,
                limit_service.settings.workflow_timeout_seconds,
            )
            await self._resolve_turn_note_previews(request, turn.turn_id)
            if policy and outcome in ("answer", "graph"):
                await self._cache_completed_turn(
                    request, turn, policy, cache_prompt, artifacts
                )

        except asyncio.CancelledError:
            status = outcome = "cancelled"
            raise
        except TimeoutError:
            status = WorkflowTimeoutExceeded.reason
            outcome = "interrupted"
            await self._set_request_error(request, WorkflowTimeoutExceeded.user_message)
        except UserFacingChatError as exc:
            status, outcome = exc.reason, "error"
            await self._set_request_error(request, exc.user_message)
            if admission is None:
                logger.info("Chat turn rejected", reason=exc.reason)
        except Exception as exc:
            status, outcome = "workflow_error", "error"
            logger.error(
                "Chat request processing failed", error_type=type(exc).__name__
            )
            if request:
                await self._set_request_error(request, ERROR_WORKFLOW_FAILED)
        finally:
            if request is not None:
                try:
                    if execution is not None:
                        async with self:
                            if self._owns_request(request):
                                self._finish_execution(
                                    outcome, monotonic() - execution_started_at
                                )
                        await self._persist_current_session_snapshot(request)
                finally:
                    try:
                        if admission is not None:
                            await complete_on_cancel(
                                limit_service.finish_turn(admission, status)
                            )
                    finally:
                        try:
                            async with self:
                                if _active_requests.get(session_id) is request:
                                    del _active_requests[session_id]
                                if self._owns_request(request):
                                    self.is_processing = False
                                    self._processing_turn_id = ""
                                    self.active_agent = ""
                                    self._reset_stream()
                        finally:
                            if _active_requests.get(session_id) is request:
                                del _active_requests[session_id]

    async def _set_request_error(self, request: _ActiveRequest, message: str) -> None:
        async with self:
            if self._owns_request(request):
                self.error_message = message

    async def _resolve_turn_note_previews(
        self, request: _ActiveRequest, turn_id: str
    ) -> None:
        async with self:
            if not self._owns_request(request):
                return
            responses = [
                message
                for message in self.messages
                if message.turn_id == turn_id and message.evidence is not None
            ]
        if not responses:
            return
        resolved = await asyncio.to_thread(
            lambda: {
                message.response_id: _with_note_previews(message)
                for message in responses
            }
        )
        async with self:
            if self._owns_request(request):
                self.messages = [
                    resolved.get(message.response_id, message)
                    for message in self.messages
                ]

    def _finish_execution(self, outcome: TurnOutcome, end_offset: float) -> None:
        report = self.executions[-1]
        unfinished: StageOutcome = {
            "error": "failed",
            "interrupted": "interrupted",
            "cancelled": "cancelled",
        }.get(outcome, "incomplete")
        stages = [
            stage.model_copy(update={"end_offset": end_offset, "outcome": unfinished})
            if stage.outcome == "running"
            else stage
            for stage in report.stages
        ]
        self.executions = [
            *self.executions[:-1],
            report.model_copy(update={"outcome": outcome, "stages": stages}),
        ]

    def _record_stage(self, stage: ExecutionStage) -> None:
        report = self.executions[-1]
        if stage.session_id != report.session_id or stage.turn_id != report.turn_id:
            return
        stages = list(report.stages)
        if stage.sequence == len(stages) and stage.outcome == "running":
            stages.append(stage)
        elif (
            stage.sequence < len(stages)
            and stages[stage.sequence].invocation_id == stage.invocation_id
            and stages[stage.sequence].outcome == "running"
            and stage.outcome != "running"
        ):
            stages[stage.sequence] = stage
        else:
            return
        self.executions = [
            *self.executions[:-1],
            report.model_copy(update={"stages": stages}),
        ]

    async def _consume_workflow(
        self,
        request: _ActiveRequest,
        admission: ChatAdmission,
        events,
        user_message: str,
        artifacts: list[GraphArtifact],
        timeout_seconds: float,
    ) -> tuple[TurnOutcome, str]:
        identity = {
            "session_id": request.session_id,
            "turn_id": admission.context.turn_id,
        }
        pending: dict[str, tuple[str, str]] = {}
        finalized: set[str] = set()
        async with _events_with_timeout(events, timeout_seconds) as stream:
            async for event in stream:
                async with self:
                    if not self._owns_request(request):
                        return "cancelled", "cancelled"
                    if event.type in (
                        WorkflowEventType.AGENT_START,
                        WorkflowEventType.AGENT_END,
                    ):
                        agent = event.data["agent"]
                        stage = event.data.get("stage")
                        if stage is not None:
                            self._record_stage(stage)
                        if event.type == WorkflowEventType.AGENT_START:
                            self.active_agent = agent
                            admission.context.agents.add(agent)
                            if agent not in self.agent_history:
                                self.agent_history = [*self.agent_history, agent]
                        else:
                            self.active_agent = next(
                                (
                                    item.agent
                                    for item in reversed(self.executions[-1].stages)
                                    if item.outcome == "running"
                                ),
                                "",
                            )
                            if stage is not None:
                                retained = set(event.data.get("response_ids", ()))
                                if (
                                    pending.get(self.streaming_response_id, ("", ""))[0]
                                    == stage.invocation_id
                                    and self.streaming_response_id not in retained
                                ):
                                    self._reset_stream()
                                pending = {
                                    key: item
                                    for key, item in pending.items()
                                    if item[0] != stage.invocation_id or key in retained
                                }

                    elif event.type == WorkflowEventType.TOKEN:
                        response_id = event.data["response_id"]
                        if response_id not in finalized:
                            _, content = pending.get(response_id, ("", ""))
                            content += event.data["chunk"]
                            pending[response_id] = (
                                event.data.get("invocation_id", ""),
                                content,
                            )
                            self.streaming_response_id = response_id
                            self.streaming_agent = event.data["agent"]
                            self.streaming_content = content

                    elif event.type == WorkflowEventType.VISUALIZATION:
                        retained_ids = {
                            artifact.artifact_id
                            for artifact in [*self._visual_artifacts, *artifacts]
                        }
                        new_artifacts = []
                        timestamp = datetime.now().strftime("%H:%M")
                        for raw in event.data["artifacts"]:
                            try:
                                payload = GraphPayload.model_validate(raw)
                                graph = GraphArtifact(
                                    **payload.model_dump(),
                                    **identity,
                                    timestamp=timestamp,
                                )
                            except (TypeError, ValueError):
                                logger.warning("Ignoring an invalid graph artifact")
                                continue
                            if graph.artifact_id not in retained_ids:
                                retained_ids.add(graph.artifact_id)
                                new_artifacts.append(graph)
                        if not new_artifacts and not artifacts:
                            self.error_message = (
                                "The knowledge map could not be displayed."
                            )
                        artifacts.extend(new_artifacts)
                        self._append_maps(new_artifacts)

                    elif event.type == WorkflowEventType.RESPONSE:
                        response_id = event.data.get("response_id") or str(uuid.uuid4())
                        if response_id in finalized:
                            continue
                        finalized.add(response_id)
                        pending.pop(response_id, None)
                        if self.streaming_response_id == response_id:
                            self._reset_stream()
                        response = Message(
                            **identity,
                            response_id=response_id,
                            invocation_id=event.data.get("invocation_id"),
                            content=_normalize_math_delimiters(event.data["content"]),
                            evidence=event.data.get("evidence"),
                            role="assistant",
                            agent=event.data.get("agent", ""),
                            timestamp=datetime.now().strftime("%H:%M"),
                            dom_id=f"message-{uuid.uuid4().hex}",
                        )
                        self.messages = [*self.messages, response]

                    elif event.type == WorkflowEventType.ERROR:
                        self.error_message = event.data.get(
                            "message", ERROR_WORKFLOW_FAILED
                        )
                        return event.data.get("outcome", "error"), event.data.get(
                            "reason", "workflow_error"
                        )

                    elif event.type == WorkflowEventType.COMPLETE:
                        outcome = event.data["outcome"]
                        if outcome == "graph" and not artifacts:
                            self.error_message = (
                                "The knowledge map could not be displayed."
                            )
                            return "error", "workflow_error"
                        return (
                            outcome,
                            "workflow_error" if outcome == "error" else "success",
                        )

        await self._set_request_error(request, ERROR_WORKFLOW_FAILED)
        return "incomplete", "workflow_incomplete"

    async def _cache_completed_turn(
        self,
        request: _ActiveRequest,
        turn: TurnIdentity,
        policy: dict,
        cache_prompt: str,
        artifacts: list[GraphArtifact],
    ) -> None:
        async with self:
            if not self._owns_request(request):
                return
            responses = (
                [
                    msg.model_copy(deep=True, update={"invocation_id": None})
                    for msg in self.messages
                    if msg.turn_id == turn.turn_id
                    and msg.role == "assistant"
                    and msg.agent != "system"
                ]
                if policy.get("cache_responses", True)
                else []
            )
            cache_artifacts = (
                artifacts if policy.get("cache_visual_artifacts", True) else []
            )
            if not responses and not cache_artifacts:
                return
            cached = CachedTurn(
                **turn.model_dump(),
                responses=responses,
                artifacts=cache_artifacts,
                agent_history=list(self.agent_history),
            )
        try:
            ttl = policy.get("ttl_seconds")
            ttl_seconds = int(ttl) if ttl is not None else None
        except (TypeError, ValueError):
            ttl_seconds = None
        try:
            await self._write_for_request(
                request,
                lambda: get_quick_action_cache().set(
                    cache_prompt, cached, ttl_seconds=ttl_seconds
                ),
            )
        except Exception as exc:
            logger.warning(
                "Quick-action cache persistence failed", error_type=type(exc).__name__
            )


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
