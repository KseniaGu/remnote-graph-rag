from datetime import datetime
from typing import Any, Optional
import reflex as rx
from pydantic import BaseModel
import plotly.graph_objects as go

from backend.configs.enums import WorkflowEventType


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
    visual_artifact: Optional[dict[str, Any]] = None
    show_visualization: bool = False

    # Context state (for debugging/display)
    current_context: str = ""
    show_context_panel: bool = False

    # Session state
    session_started: bool = False
    error_message: str = ""

    @rx.var
    def has_messages(self) -> bool:
        """Check if there are any messages."""
        return len(self.messages) > 0

    @rx.var
    def has_visualization(self) -> bool:
        """Check if there's a visualization to display."""
        return self.visual_artifact is not None

    @rx.var
    def plotly_figure(self) -> go.Figure:
        """Convert visual_artifact dict to a Plotly Figure for rendering."""
        if self.visual_artifact is not None:
            fig = go.Figure(self.visual_artifact)
            fig.update_layout(
                autosize=True,
                width=None,
                height=None,
            )
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
            }
            for agent in agents
        ]

    def set_input(self, value: str):
        """Update the current input value."""
        self.current_input = value

    def toggle_context_panel(self):
        """Toggle the context panel visibility."""
        self.show_context_panel = not self.show_context_panel

    def toggle_visualization(self):
        """Toggle visualization panel."""
        self.show_visualization = not self.show_visualization

    def clear_chat(self):
        """Clear the chat history."""
        self.messages = []
        self.agent_history = []
        self.current_context = ""
        self.visual_artifact = None
        self.show_visualization = False
        self.error_message = ""

    def clear_error(self):
        """Clear the error message."""
        self.error_message = ""

    @rx.event(background=True)
    async def send_message(self):
        """Send a message and process the response."""
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
                        self.current_context = event.data["context"]

                    elif event.type == WorkflowEventType.VISUALIZATION:
                        self.visual_artifact = event.data["artifact"]
                        self.show_visualization = True

                    elif event.type == WorkflowEventType.RESPONSE:
                        agent_name = event.data.get("agent", "")
                        self.messages = self.messages + [
                            Message(
                                content=event.data["content"],
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
