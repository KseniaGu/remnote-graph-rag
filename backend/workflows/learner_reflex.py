import threading
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from backend.configs.constants import RECURSION_LIMIT, VISUALIZATION_EMPTY_CONTEXT
from backend.configs.enums import WorkflowEventType
from backend.configs.messages import (
    ERROR_RECURSION_LIMIT,
    ERROR_WORKFLOW_FAILED,
    FALLBACK_ALL_SOURCES_EXHAUSTED,
    FALLBACK_DEFAULT,
    FALLBACK_NO_RESULTS,
    FALLBACK_OUT_OF_SCOPE,
    FALLBACK_VISUALIZATION_FAILED,
)
from backend.configs.models import get_model_settings
from backend.configs.observability import LangSmithSettings
from backend.configs.paths import PathSettings
from backend.configs.search import KnowledgeGraphSearchSettings, TavilySettings
from backend.configs.storage import StorageSettings
from backend.utils.chat_errors import (
    UserFacingChatError,
    WorkflowInitializationUnavailable,
)
from backend.utils.chat_limits import (
    ChatTurnContext,
    reset_current_chat_turn,
    set_current_chat_turn,
)
from backend.utils.helpers import logger
from backend.workflows.learner import LearnerWorkflow


@dataclass
class WorkflowEvent:
    """Event emitted during workflow execution."""

    type: WorkflowEventType
    data: dict


class ReflexLearnerWorkflow:
    """Reflex-compatible wrapper for LearnerWorkflow with streaming support."""

    _init_lock = threading.Lock()

    def __init__(self):
        """Initializes the lightweight wrapper; heavy workflow setup is lazy."""
        self._workflow = None
        self._graph = None

    def _ensure_initialized(self):
        """Ensures the workflow is initialized, thread-safe."""
        if self._graph is not None:
            return

        with ReflexLearnerWorkflow._init_lock:
            if self._graph is not None:
                return
            try:
                LangSmithSettings().configure()
                tavily_settings = TavilySettings()
                models_settings = get_model_settings()
                path_settings = PathSettings()
                kg_search_settings = KnowledgeGraphSearchSettings()
                storage_settings = StorageSettings()
                self._workflow = LearnerWorkflow(
                    models_settings,
                    path_settings,
                    storage_settings,
                    tavily_settings,
                    kg_search_settings,
                )
                self._graph = self._workflow.run()
                logger.info("ReflexLearnerWorkflow initialized successfully")

            except Exception as exc:
                logger.exception(
                    "Failed to initialize ReflexLearnerWorkflow",
                    error_type=type(exc).__name__,
                )
                raise WorkflowInitializationUnavailable() from exc

    @staticmethod
    def _get_fallback_message(final_state: dict) -> str:
        """Returns a user-facing fallback selected from explicit workflow state."""
        if final_state.get("request_scope") == "out_of_scope":
            return FALLBACK_OUT_OF_SCOPE
        if final_state.get("sources_exhausted"):
            return FALLBACK_ALL_SOURCES_EXHAUSTED

        context = str(final_state.get("context", ""))
        if VISUALIZATION_EMPTY_CONTEXT in context or "Visualization failed" in context:
            return FALLBACK_VISUALIZATION_FAILED
        if '"no_results": true' in context:
            return FALLBACK_NO_RESULTS
        return FALLBACK_DEFAULT

    @staticmethod
    def _convert_messages(messages: list[dict]) -> list[BaseMessage]:
        """Converts message dicts to LangChain messages."""
        langchain_messages = []
        for msg in messages:
            if msg.get("role") == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            else:
                langchain_messages.append(AIMessage(content=msg["content"]))
        return langchain_messages

    async def process_message(
        self,
        user_message: str,
        message_history: list[dict],
        recursion_limit: int = RECURSION_LIMIT,
        session_id: str = "",
        session_summary: str = "",
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """
        Processes a user message through the workflow and yield events.

        Args:
            user_message: The user's message
            message_history: Previous messages in the conversation
            recursion_limit: Maximum recursion depth for the graph
            session_id: Session ID.
            session_summary: Session's conversation history summary.

        Yields:
            WorkflowEvent: Events during workflow execution
        """
        self._ensure_initialized()

        # Convert messages to LangChain format
        messages = self._convert_messages(message_history)
        messages.append(HumanMessage(content=user_message))

        thread_id = str(uuid.uuid4())
        configurable = {"thread_id": thread_id}
        if session_id:
            configurable["checkpoint_ns"] = session_id
        config = {"recursion_limit": recursion_limit, "configurable": configurable}
        initial_state = {
            "messages": messages,
            "session_summary": session_summary,
            "request_scope": "unclassified",
            "retrieval_status": "not_run",
            "retriever_empty": False,
            "sources_exhausted": False,
        }

        try:
            # Track which agents are being called
            # current_agent = None

            result = await self._graph.ainvoke(initial_state, config=config)

            if result:
                # Extract and yield context update
                context = result.get("context", "")
                if context:
                    yield WorkflowEvent(
                        type=WorkflowEventType.CONTEXT_UPDATE, data={"context": context}
                    )

                # Extract and yield visualization
                visual_artifacts = result.get("visual_artifacts", [])
                response_emitted = bool(visual_artifacts)
                if visual_artifacts:
                    yield WorkflowEvent(
                        type=WorkflowEventType.VISUALIZATION,
                        data={"artifacts": visual_artifacts},
                    )

                # Extract and yield agent responses
                result_messages = result.get("messages", [])
                for msg in result_messages:
                    if (
                        hasattr(msg, "content")
                        and hasattr(msg, "type")
                        and msg.type == "ai"
                    ):
                        agent_name = (
                            msg.additional_kwargs.get("agent", "").strip("[]").lower()
                        )
                        if agent_name in ("analyst", "mentor") and msg.content:
                            response_emitted = True
                            yield WorkflowEvent(
                                type=WorkflowEventType.RESPONSE,
                                data={"content": msg.content, "agent": agent_name},
                            )

                if not response_emitted:
                    context = result.get("context", "")
                    fallback = self._get_fallback_message(result)
                    yield WorkflowEvent(
                        type=WorkflowEventType.RESPONSE,
                        data={"content": fallback, "agent": "system"},
                    )

                yield WorkflowEvent(
                    type=WorkflowEventType.COMPLETE,
                    data={"next_step": result.get("next_step", "__end__")},
                )

        except UserFacingChatError as exc:
            logger.warning("Workflow stopped", reason=exc.reason)
            yield WorkflowEvent(
                type=WorkflowEventType.ERROR,
                data={"message": exc.user_message, "reason": exc.reason},
            )

        except GraphRecursionError as e:
            logger.error(f"Workflow exceeded recursion limit: {e}")
            yield WorkflowEvent(
                type=WorkflowEventType.ERROR, data={"message": ERROR_RECURSION_LIMIT}
            )

        except Exception as exc:
            logger.error("Workflow error", error_type=type(exc).__name__)
            yield WorkflowEvent(
                type=WorkflowEventType.ERROR,
                data={"message": ERROR_WORKFLOW_FAILED},
            )

    async def stream_with_status(
        self,
        user_message: str,
        message_history: list[dict],
        recursion_limit: int = RECURSION_LIMIT,
        session_id: str = "",
        session_summary: str = "",
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Streams workflow execution with detailed status updates.

        This method uses the graph's stream mode to provide real-time
        updates about which agent is currently active.
        """
        self._ensure_initialized()

        messages = self._convert_messages(message_history)
        messages.append(HumanMessage(content=user_message))

        thread_id = str(uuid.uuid4())
        configurable = {"thread_id": thread_id}
        if session_id:
            configurable["checkpoint_ns"] = session_id
        config = {"recursion_limit": recursion_limit, "configurable": configurable}
        initial_state = {
            "messages": messages,
            "session_summary": session_summary,
            "request_scope": "unclassified",
            "retrieval_status": "not_run",
            "retriever_empty": False,
            "sources_exhausted": False,
        }

        try:
            final_state = {}

            async for event in self._graph.astream(initial_state, config=config):
                # Each event is a dict with node name as key
                for node_name, node_output in event.items():
                    yield WorkflowEvent(
                        type=WorkflowEventType.AGENT_START, data={"agent": node_name}
                    )

                    # Update final state with node output
                    if isinstance(node_output, dict):
                        final_state.update(node_output)

                    yield WorkflowEvent(
                        type=WorkflowEventType.AGENT_END, data={"agent": node_name}
                    )

            # Yield final results
            if final_state:
                context = final_state.get("context", "")
                if context:
                    yield WorkflowEvent(
                        type=WorkflowEventType.CONTEXT_UPDATE, data={"context": context}
                    )

                visual_artifacts = final_state.get("visual_artifacts", [])
                response_emitted = bool(visual_artifacts)
                if visual_artifacts:
                    yield WorkflowEvent(
                        type=WorkflowEventType.VISUALIZATION,
                        data={"artifacts": visual_artifacts},
                    )

                result_messages = final_state.get("messages", [])
                if isinstance(result_messages, list):
                    for msg in result_messages:
                        if (
                            hasattr(msg, "content")
                            and hasattr(msg, "type")
                            and msg.type == "ai"
                        ):
                            agent_name = (
                                msg.additional_kwargs.get("agent", "")
                                .strip("[]")
                                .lower()
                            )
                            if agent_name in ("analyst", "mentor") and msg.content:
                                response_emitted = True
                                yield WorkflowEvent(
                                    type=WorkflowEventType.RESPONSE,
                                    data={"content": msg.content, "agent": agent_name},
                                )

                if not response_emitted:
                    context = final_state.get("context", "")
                    fallback = self._get_fallback_message(final_state)
                    yield WorkflowEvent(
                        type=WorkflowEventType.RESPONSE,
                        data={"content": fallback, "agent": "system"},
                    )

            yield WorkflowEvent(
                type=WorkflowEventType.COMPLETE, data={"status": "success"}
            )

        except UserFacingChatError as exc:
            logger.warning("Workflow stream stopped", reason=exc.reason)
            yield WorkflowEvent(
                type=WorkflowEventType.ERROR,
                data={"message": exc.user_message, "reason": exc.reason},
            )

        except GraphRecursionError as e:
            logger.error(f"Workflow exceeded recursion limit: {e}")
            yield WorkflowEvent(
                type=WorkflowEventType.ERROR, data={"message": ERROR_RECURSION_LIMIT}
            )

        except Exception as exc:
            logger.error("Workflow stream error", error_type=type(exc).__name__)
            yield WorkflowEvent(
                type=WorkflowEventType.ERROR,
                data={"message": ERROR_WORKFLOW_FAILED},
            )

    async def stream_with_tokens(
        self,
        user_message: str,
        message_history: list[dict],
        recursion_limit: int = RECURSION_LIMIT,
        session_id: str = "",
        session_summary: str = "",
        turn_context: ChatTurnContext | None = None,
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Streams workflow execution with per-token updates for analyst/mentor responses.

        Uses LangGraph's astream_events(version="v2") to intercept token chunks from
        the LLM layer inside analyst and mentor nodes, yielding TOKEN events in real time.
        All other event types (AGENT_START/END, CONTEXT_UPDATE, VISUALIZATION, RESPONSE)
        are preserved with the same semantics as stream_with_status().
        """
        self._ensure_initialized()

        messages = self._convert_messages(message_history)
        messages.append(HumanMessage(content=user_message))

        thread_id = str(uuid.uuid4())
        configurable = {"thread_id": thread_id}
        if session_id:
            configurable["checkpoint_ns"] = session_id
        config = {"recursion_limit": recursion_limit, "configurable": configurable}
        initial_state = {
            "messages": messages,
            "session_summary": session_summary,
            "request_scope": "unclassified",
            "retrieval_status": "not_run",
            "retriever_empty": False,
            "sources_exhausted": False,
        }

        _AGENT_NODES = {
            "orchestrator",
            "retriever",
            "researcher",
            "analyst",
            "mentor",
            "visualizer",
        }
        _STREAMING_NODES = {"analyst", "mentor"}

        accumulated: dict[str, str] = {}
        final_state: dict = {}

        turn_token = set_current_chat_turn(turn_context)
        try:
            async for event in self._graph.astream_events(
                initial_state, config=config, version="v2"
            ):
                event_type = event.get("event", "")
                name = event.get("name", "")
                metadata = event.get("metadata", {})
                node = metadata.get("langgraph_node", "")

                if (
                    event_type == "on_chain_start"
                    and name in _AGENT_NODES
                    and name == node
                ):
                    yield WorkflowEvent(
                        type=WorkflowEventType.AGENT_START, data={"agent": node}
                    )

                elif (
                    event_type == "on_chain_end"
                    and name in _AGENT_NODES
                    and name == node
                ):
                    yield WorkflowEvent(
                        type=WorkflowEventType.AGENT_END, data={"agent": node}
                    )

                elif event_type == "on_chat_model_stream" and node in _STREAMING_NODES:
                    chunk = event.get("data", {}).get("chunk")
                    if chunk is not None:
                        content = chunk.content if hasattr(chunk, "content") else ""
                        if content:
                            accumulated[node] = accumulated.get(node, "") + content
                            yield WorkflowEvent(
                                type=WorkflowEventType.TOKEN,
                                data={"chunk": content, "agent": node},
                            )

                elif event_type == "on_chain_end" and name == "LangGraph":
                    output = event.get("data", {}).get("output")
                    if isinstance(output, dict):
                        final_state = output

            # Yield final results from captured state
            context = final_state.get("context", "")
            if context:
                yield WorkflowEvent(
                    type=WorkflowEventType.CONTEXT_UPDATE, data={"context": context}
                )

            visual_artifacts = final_state.get("visual_artifacts", [])
            response_emitted = bool(visual_artifacts)
            if visual_artifacts:
                yield WorkflowEvent(
                    type=WorkflowEventType.VISUALIZATION,
                    data={"artifacts": visual_artifacts},
                )

            if accumulated:
                for agent_name, content in accumulated.items():
                    if content:
                        response_emitted = True
                        yield WorkflowEvent(
                            type=WorkflowEventType.RESPONSE,
                            data={"content": content, "agent": agent_name},
                        )
            else:
                result_messages = final_state.get("messages", [])
                if isinstance(result_messages, list):
                    for msg in result_messages:
                        if (
                            hasattr(msg, "content")
                            and hasattr(msg, "type")
                            and msg.type == "ai"
                        ):
                            agent_name = (
                                msg.additional_kwargs.get("agent", "")
                                .strip("[]")
                                .lower()
                            )
                            if agent_name in _STREAMING_NODES and msg.content:
                                response_emitted = True
                                yield WorkflowEvent(
                                    type=WorkflowEventType.RESPONSE,
                                    data={"content": msg.content, "agent": agent_name},
                                )

            if not response_emitted:
                fallback = self._get_fallback_message(final_state)
                yield WorkflowEvent(
                    type=WorkflowEventType.RESPONSE,
                    data={"content": fallback, "agent": "system"},
                )

            yield WorkflowEvent(
                type=WorkflowEventType.COMPLETE, data={"status": "success"}
            )

        except UserFacingChatError as exc:
            logger.warning("Workflow token stream stopped", reason=exc.reason)
            yield WorkflowEvent(
                type=WorkflowEventType.ERROR,
                data={"message": exc.user_message, "reason": exc.reason},
            )

        except GraphRecursionError as e:
            logger.error(f"Workflow exceeded recursion limit: {e}")
            yield WorkflowEvent(
                type=WorkflowEventType.ERROR, data={"message": ERROR_RECURSION_LIMIT}
            )

        except Exception as exc:
            logger.error("Workflow token stream error", error_type=type(exc).__name__)
            yield WorkflowEvent(
                type=WorkflowEventType.ERROR,
                data={"message": ERROR_WORKFLOW_FAILED},
            )
        finally:
            reset_current_chat_turn(turn_token)


@lru_cache(maxsize=1)
def get_workflow() -> ReflexLearnerWorkflow:
    """Returns the process-local workflow wrapper."""
    return ReflexLearnerWorkflow()
