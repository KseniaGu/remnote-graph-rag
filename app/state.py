import asyncio
import re
import uuid
from datetime import datetime
from typing import Any

import plotly.graph_objects as go
import reflex as rx
from pydantic import BaseModel

from app.strings import AGENT_DESCRIPTIONS, QUICK_ACTIONS, QUICK_ACTION_CACHE_POLICIES
from backend.configs.constants import RECURSION_LIMIT, CACHED_AGENT_REPLAY_LATENCY, CACHED_TOKENS_REPLAY_LATENCY, \
    DEFAULT_RECENT_MESSAGE_LIMIT
from backend.configs.enums import WorkflowEventType
from backend.utils.cache import get_quick_action_cache, normalize_quick_action_prompt
from backend.utils.helpers import logger
from backend.utils.session_memory import get_session_memory_store
from backend.workflows.learner_reflex import get_workflow


def _normalize_math_delimiters(text: str) -> str:
    """Converts LaTeX \\[...\\] and \\(...\\) delimiters to $$...$$ and $...$ so remark-math can render them."""
    text = re.sub(r'\\\[(.+?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.+?)\\\)', r'$\1$', text, flags=re.DOTALL)
    return text


def _build_quick_action_policy_map() -> dict[str, dict[str, Any]]:
    policy_map: dict[str, dict[str, Any]] = {}
    quick_action_prompts = {action for _, _, action in QUICK_ACTIONS}
    for raw_policy in QUICK_ACTION_CACHE_POLICIES:
        policy = dict(raw_policy)
        prompt = str(policy.get("prompt", ""))
        if not prompt or prompt not in quick_action_prompts or not bool(policy.get("enabled", True)):
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


def _message_to_memory_payload(message: "Message") -> dict[str, str]:
    return {
        "role": message.role,
        "content": message.content,
        "agent": message.agent,
        "timestamp": message.timestamp,
    }


def _cached_response_chunks(content: str, chunk_size: int = 24) -> list[str]:
    """Splits cached content into small chunks so cache hits feel like live streaming."""
    if not content:
        return []
    return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]


async def _persist_session_messages(session_id: str, messages: list[dict[str, str]]) -> None:
    if not session_id:
        return
    try:
        await asyncio.to_thread(
            lambda: get_session_memory_store().replace_messages(session_id, messages)
        )
    except Exception as e:
        logger.warning(f"Persisting session messages failed: {e}")
        pass


class Message(BaseModel):
    """A chat message."""
    content: str
    role: str  # "user" or "assistant"
    agent: str = ""  # Agent name for assistant messages
    timestamp: str = ""


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

    # Context state (for debugging/display)
    current_context: str = ""
    show_context_panel: bool = False

    # Session state
    session_started: bool = False
    session_id: str = ""
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
        if self.visual_artifacts and 0 <= self.selected_plot_index < len(self.visual_artifacts):
            fig = go.Figure(self.visual_artifacts[self.selected_plot_index])
            fig.update_layout(height=410)
            return fig
        return go.Figure()

    @rx.var
    def agent_status_list(self) -> list[dict]:
        """Get list of agent statuses for display."""
        agents = ["orchestrator", "retriever", "researcher", "analyst", "mentor", "visualizer"]
        return [
            {
                "name": agent,
                "is_active": agent == self.active_agent,
                "was_used": agent in self.agent_history,
                "description": AGENT_DESCRIPTIONS.get(agent, ""),
            }
            for agent in agents
        ]

    def set_input(self, value: str):
        """Updates the current input value."""
        self.current_input = value

    def handle_form_submit(self, form_data: dict):
        """Handles chat form submission using the current DOM value to avoid WebSocket sync race conditions."""
        value = form_data.get("message", "").strip()
        if not value or self.is_processing:
            return
        self.current_input = value
        return AppState.send_message

    def toggle_context_panel(self):
        """Toggles the context panel visibility."""
        self.show_context_panel = not self.show_context_panel

    def toggle_visualization(self):
        """Toggles visualization panel."""
        self.show_visualization = not self.show_visualization

    def next_plot(self):
        """Navigates to the next plot in the history."""
        if self.selected_plot_index < len(self.visual_artifacts) - 1:
            self.selected_plot_index += 1

    def prev_plot(self):
        """Navigates to the previous plot in the history."""
        if self.selected_plot_index > 0:
            self.selected_plot_index -= 1

    def clear_chat(self):
        """Clears the chat history."""
        self.messages = []
        self.agent_history = []
        self.session_id = ""
        # self.current_context = "" # It's hidden now
        self.visual_artifacts = []
        self.selected_plot_index = 0
        self.show_visualization = False
        self.error_message = ""

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
            if agent_name and agent_name != "system" and agent_name not in self.agent_history:
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
                    timestamp=datetime.now().strftime("%H:%M")
                )
            ]

    @rx.event(background=True)
    async def send_message(self):
        """Sends a message and process the response."""
        if not self.current_input.strip() or self.is_processing:
            return

        user_message = self.current_input.strip()
        quick_action_policy = _get_quick_action_cache_policy(user_message)
        should_cache_quick_action = len(self.messages) == 0 and quick_action_policy is not None
        cache_prompt = str(quick_action_policy.get("prompt", user_message)) if quick_action_policy else user_message

        async with self:
            self.current_input = ""
            if not self.session_id:
                self.session_id = str(uuid.uuid4())
            session_id = self.session_id

        async with self:
            self.is_processing = True
            self.error_message = ""
            self.agent_history = []

            # Add user message
            self.messages = self.messages + [
                Message(
                    content=user_message,
                    role="user",
                    timestamp=datetime.now().strftime("%H:%M")
                )
            ]

        try:
            if should_cache_quick_action:
                try:
                    cached = await asyncio.to_thread(lambda: get_quick_action_cache().get(cache_prompt))
                except Exception as e:
                    logger.warning(f"Quick-action cache lookup failed: {e}")
                    cached = None
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
                            self.visual_artifacts = self.visual_artifacts + cached_visual_artifacts
                            self.selected_plot_index = len(self.visual_artifacts) - 1
                            self.show_visualization = True
                    for response in cached_responses:
                        await self._stream_cached_response(response)
                    async with self:
                        messages_for_memory = [_message_to_memory_payload(msg) for msg in self.messages]
                    await _persist_session_messages(session_id, messages_for_memory)
                    return

            workflow = get_workflow()

            # Prepare message history
            recent_messages = self.messages[:-1][-DEFAULT_RECENT_MESSAGE_LIMIT:]
            message_history = [
                {"role": msg.role, "content": msg.content}
                for msg in recent_messages
            ]
            cache_responses: list[dict[str, str]] = []
            cache_visual_artifacts: list[dict[str, Any]] = []
            cache_context = ""
            cache_error = False
            try:
                session_summary = await asyncio.to_thread(
                    lambda: get_session_memory_store().get_prompt_summary(session_id, user_message)
                )
            except Exception as e:
                logger.warning(f"Loading session summary failed: {e}")
                session_summary = ""

            # Stream through workflow with per-token updates
            async for event in workflow.stream_with_tokens(
                    user_message=user_message,
                    message_history=message_history,
                    recursion_limit=RECURSION_LIMIT,
                    session_id=session_id,
                    session_summary=session_summary,
            ):
                async with self:
                    if event.type == WorkflowEventType.AGENT_START:
                        self.active_agent = event.data["agent"]
                        if event.data["agent"] not in self.agent_history:
                            self.agent_history = self.agent_history + [event.data["agent"]]

                    elif event.type == WorkflowEventType.AGENT_END:
                        self.active_agent = ""

                    elif event.type == WorkflowEventType.TOKEN:
                        agent_name = event.data.get("agent", "")
                        self.streaming_agent = agent_name
                        self.streaming_content = self.streaming_content + event.data["chunk"]

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
                        self.visual_artifacts = self.visual_artifacts + artifacts
                        self.selected_plot_index = len(self.visual_artifacts) - 1
                        self.show_visualization = True

                    elif event.type == WorkflowEventType.RESPONSE:
                        agent_name = event.data.get("agent", "")
                        content = event.data["content"]
                        if agent_name != "system":
                            cache_responses.append({"content": content, "agent": agent_name})
                        self.streaming_content = ""
                        self.streaming_agent = ""
                        self.messages = self.messages + [
                            Message(
                                content=_normalize_math_delimiters(content),
                                role="assistant",
                                agent=agent_name,
                                timestamp=datetime.now().strftime("%H:%M")
                            )
                        ]

                    elif event.type == WorkflowEventType.ERROR:
                        cache_error = True
                        self.error_message = event.data.get("message", "Unknown error occurred")

                    elif event.type == WorkflowEventType.CONTEXT_UPDATE:
                        cache_context = event.data.get("context", "")

            if should_cache_quick_action and not cache_error:
                async with self:
                    cache_agent_history = list(self.agent_history)
                ttl_seconds = quick_action_policy.get("ttl_seconds") if quick_action_policy else None
                try:
                    ttl_seconds = int(ttl_seconds) if ttl_seconds is not None else None
                except (TypeError, ValueError):
                    ttl_seconds = None
                responses_to_cache = (
                    cache_responses if bool(quick_action_policy.get("cache_responses", True)) else []
                )
                artifacts_to_cache = (
                    cache_visual_artifacts if bool(quick_action_policy.get("cache_visual_artifacts", True)) else []
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

            async with self:
                messages_for_memory = [_message_to_memory_payload(msg) for msg in self.messages]
            await _persist_session_messages(session_id, messages_for_memory)

        except Exception as e:
            async with self:
                self.error_message = f"Error processing request: {str(e)}"
                self.messages = self.messages + [
                    Message(
                        content="I encountered an error while processing your request. Please try again.",
                        role="assistant",
                        agent="system",
                        timestamp=datetime.now().strftime("%H:%M")
                    )
                ]
                messages_for_memory = [_message_to_memory_payload(msg) for msg in self.messages]
            await _persist_session_messages(session_id, messages_for_memory)
        finally:
            async with self:
                self.is_processing = False
                self.active_agent = ""
                self.streaming_content = ""
                self.streaming_agent = ""


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
