import re
from datetime import datetime
from typing import Any

import plotly.graph_objects as go
import reflex as rx
from pydantic import BaseModel

from app.strings import AGENT_DESCRIPTIONS
from backend.configs.enums import WorkflowEventType


def _normalize_math_delimiters(text: str) -> str:
    """Converts LaTeX \\[...\\] and \\(...\\) delimiters to $$...$$ and $...$
    so remark-math can render them."""
    text = re.sub(r'\\\[(.+?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.+?)\\\)', r'$\1$', text, flags=re.DOTALL)
    return text


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
    error_message: str = ""

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
        self.current_context = ""
        self.visual_artifacts = []
        self.selected_plot_index = 0
        self.show_visualization = False
        self.error_message = ""

    def clear_error(self):
        """Clears the error message."""
        self.error_message = ""

    @rx.event(background=True)
    async def send_message(self):
        """Sends a message and process the response."""
        if not self.current_input.strip() or self.is_processing:
            return

        user_message = self.current_input.strip()

        async with self:
            self.current_input = ""
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
            # Get workflow instance
            from backend.workflows.learner_reflex import get_workflow

            workflow = get_workflow()

            # Prepare message history
            message_history = [
                {"role": msg.role, "content": msg.content}
                for msg in self.messages[:-1]  # Exclude the message we just added
            ]

            # Stream through workflow with status updates
            async for event in workflow.stream_with_status(
                    user_message=user_message,
                    message_history=message_history,
                    recursion_limit=25
            ):
                async with self:
                    if event.type == WorkflowEventType.AGENT_START:
                        self.active_agent = event.data["agent"]
                        if event.data["agent"] not in self.agent_history:
                            self.agent_history = self.agent_history + [event.data["agent"]]

                    elif event.type == WorkflowEventType.AGENT_END:
                        self.active_agent = ""

                    elif event.type == WorkflowEventType.CONTEXT_UPDATE:
                        raw = (event.data["context"]
                               .replace("[RESEARCH_COMPLETE]", "")
                               .replace("Visual artifact generated", "")
                               .replace("Visualization failed", "")
                               .strip())
                        try:
                            import json as _json
                            parsed = _json.loads(raw)
                            self.current_context = _json.dumps(parsed, ensure_ascii=False, indent=2)
                        except Exception:
                            self.current_context = raw

                    elif event.type == WorkflowEventType.VISUALIZATION:
                        self.visual_artifacts = self.visual_artifacts + event.data["artifacts"]
                        self.selected_plot_index = len(self.visual_artifacts) - 1
                        self.show_visualization = True

                    elif event.type == WorkflowEventType.RESPONSE:
                        agent_name = event.data.get("agent", "")
                        self.messages = self.messages + [
                            Message(
                                content=_normalize_math_delimiters(event.data["content"]),
                                role="assistant",
                                agent=agent_name,
                                timestamp=datetime.now().strftime("%H:%M")
                            )
                        ]

                    elif event.type == WorkflowEventType.ERROR:
                        self.error_message = event.data.get("message", "Unknown error occurred")

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
        finally:
            async with self:
                self.is_processing = False
                self.active_agent = ""


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
