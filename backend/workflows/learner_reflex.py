import threading
import uuid
from collections.abc import AsyncGenerator, Iterator
from contextlib import aclosing
from dataclasses import dataclass
from time import monotonic

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.errors import GraphInterrupt, GraphRecursionError
from pydantic import ValidationError

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
from backend.knowledge_graph.storage import open_document_store
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
from backend.workflows.agents.retrieval_evidence import resolve_note_chunks
from backend.workflows.learner import LearnerWorkflow
from backend.workflows.presentation import (
    ExecutionStage,
    ResponseEvidence,
    StageOutcome,
    TurnOutcome,
)

AGENT_NODES = {
    "orchestrator",
    "retriever",
    "researcher",
    "analyst",
    "mentor",
    "visualizer",
}
STREAMING_NODES = {"analyst", "mentor"}


@dataclass
class WorkflowEvent:
    """Event emitted during workflow execution."""

    type: WorkflowEventType
    data: dict


@dataclass
class _StreamedResponse:
    response_id: str
    message_id: str | None = None


def _stage_event(kind: WorkflowEventType, stage: ExecutionStage) -> WorkflowEvent:
    return WorkflowEvent(kind, {"agent": stage.agent, "stage": stage})


def _turn_outcome(state: dict, *, answer: bool, graph: bool) -> TurnOutcome:
    if answer:
        return "answer"
    if graph:
        return "graph"
    if state.get("request_scope") == "out_of_scope":
        return "out_of_scope"
    context = str(state.get("context", ""))
    if (
        state.get("sources_exhausted")
        or state.get("retrieval_status") == "no_results"
        or VISUALIZATION_EMPTY_CONTEXT in context
        or '"no_results": true' in context
    ):
        return "no_results"
    if "Visualization failed" in context:
        return "error"
    return "empty"


class ReflexLearnerWorkflow:
    """Reflex-compatible wrapper for LearnerWorkflow with streaming support."""

    _init_lock = threading.Lock()

    def __init__(self):
        """Initializes the lightweight wrapper; heavy workflow setup is lazy."""
        self._workflow = None
        self._graph = None
        self._document_store = None
        self._document_store_lock = threading.Lock()

    def _get_document_store(self):
        with self._document_store_lock:
            if self._document_store is None:
                self._document_store = open_document_store(
                    StorageSettings().document_storage
                )
            return self._document_store

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
                    document_store_loader=self._get_document_store,
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

    @staticmethod
    def _current_turn_messages(messages: list) -> list:
        """Select messages after the latest user input, or a node's message delta."""
        for index in range(len(messages) - 1, -1, -1):
            if getattr(messages[index], "type", None) == "human":
                return messages[index + 1 :]
        return messages

    @staticmethod
    def _response_event(content, agent: str, message=None) -> WorkflowEvent:
        data = {"content": content, "agent": agent}
        evidence = getattr(message, "additional_kwargs", {}).get("response_evidence")
        if evidence is not None:
            try:
                data["evidence"] = ResponseEvidence.model_validate(evidence)
            except ValidationError:
                logger.warning("Invalid response evidence")
        return WorkflowEvent(type=WorkflowEventType.RESPONSE, data=data)

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
            "routing_validated": False,
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
                for msg in self._current_turn_messages(result_messages):
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
                            yield self._response_event(msg.content, agent_name, msg)

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
            "routing_validated": False,
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
                    for msg in self._current_turn_messages(result_messages):
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
                                yield self._response_event(msg.content, agent_name, msg)

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
        execution_started_at: float | None = None,
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Streams observed node boundaries and reconciles model tokens with node outputs."""
        messages = self._convert_messages(message_history)
        messages.append(HumanMessage(content=user_message))
        configurable = {"thread_id": str(uuid.uuid4())}
        if session_id:
            configurable["checkpoint_ns"] = session_id
        config = {"recursion_limit": recursion_limit, "configurable": configurable}
        initial_state = {
            "messages": messages,
            "session_summary": session_summary,
            "request_scope": "unclassified",
            "routing_validated": False,
            "retrieval_status": "not_run",
            "retriever_empty": False,
            "sources_exhausted": False,
        }

        identity = {
            "session_id": turn_context.session_id
            if turn_context
            else str(uuid.uuid4()),
            "turn_id": turn_context.turn_id if turn_context else str(uuid.uuid4()),
        }
        started_at = (
            monotonic() if execution_started_at is None else execution_started_at
        )
        active: dict[str, ExecutionStage] = {}
        seen_runs: set[str] = set()
        streams: dict[str, dict[str, _StreamedResponse]] = {}
        emitted_ids: set[str] = set()
        emitted_messages: list[BaseMessage] = []
        root_id = None
        final_state = None
        interrupted = False
        incomplete = False
        failed = False
        node_failed = False

        def close_stage(stage: ExecutionStage, outcome: StageOutcome) -> WorkflowEvent:
            return _stage_event(
                WorkflowEventType.AGENT_END,
                stage.model_copy(
                    update={"end_offset": monotonic() - started_at, "outcome": outcome}
                ),
            )

        def responses(
            output: dict,
            stage: ExecutionStage | None = None,
        ) -> Iterator[WorkflowEvent]:
            raw_messages = output.get("messages")
            if not isinstance(raw_messages, list):
                return
            invocation_message_ids: set[str] = set()
            for message in self._current_turn_messages(raw_messages):
                if (
                    not isinstance(message, AIMessage)
                    or not isinstance(message.content, str)
                    or not message.content
                ):
                    continue
                agent = message.additional_kwargs.get("agent")
                agent = agent.strip("[]").lower() if isinstance(agent, str) else ""
                if agent not in STREAMING_NODES or (stage and agent != stage.agent):
                    continue
                known_ids = invocation_message_ids if stage else emitted_ids
                if (message.id and message.id in known_ids) or (
                    stage is None
                    and any(earlier is message for earlier in emitted_messages)
                ):
                    continue
                if message.id:
                    invocation_message_ids.add(message.id)
                    emitted_ids.add(message.id)
                emitted_messages.append(message)
                result = self._response_event(message.content, agent, message)
                if stage:
                    candidates = [
                        stream.response_id
                        for stream in streams.get(stage.invocation_id, {}).values()
                        if message.id and stream.message_id == message.id
                    ]
                    result.data.update(
                        invocation_id=stage.invocation_id,
                        response_id=candidates[0]
                        if len(candidates) == 1
                        else str(uuid.uuid4()),
                    )
                yield result

        error = None
        terminal_outcome: TurnOutcome = "error"
        stage_outcome: StageOutcome = "failed"
        turn_token = set_current_chat_turn(turn_context)
        try:
            self._ensure_initialized()
            events = self._graph.astream_events(
                initial_state, config=config, version="v2"
            )
            async with aclosing(events):
                async for event in events:
                    if not isinstance(event, dict):
                        continue
                    event_type = event.get("event")
                    name = event.get("name")
                    run_id = event.get("run_id")
                    parents = event.get("parent_ids", [])
                    metadata = event.get("metadata", {})
                    data = event.get("data", {})
                    if (
                        not isinstance(parents, list)
                        or not isinstance(metadata, dict)
                        or not isinstance(data, dict)
                    ):
                        continue
                    node = metadata.get("langgraph_node")
                    if not parents and name == "LangGraph":
                        if (
                            root_id is None
                            and event_type == "on_chain_start"
                            and isinstance(run_id, str)
                            and run_id
                        ):
                            root_id = run_id
                        elif root_id is not None and run_id == root_id:
                            if event_type == "on_chain_stream":
                                chunk = data.get("chunk")
                                if isinstance(chunk, dict) and "__interrupt__" in chunk:
                                    interrupted = True
                            elif event_type == "on_chain_end":
                                output = data.get("output")
                                if isinstance(output, dict) and output:
                                    final_state = output
                            elif event_type == "on_chain_error":
                                failed = True
                        continue
                    if not isinstance(run_id, str) or not run_id:
                        continue
                    if (
                        isinstance(name, str)
                        and root_id is not None
                        and name in AGENT_NODES
                        and name == node
                        and parents == [root_id]
                    ):
                        if event_type == "on_chain_start" and run_id not in seen_runs:
                            stage = ExecutionStage(
                                **identity,
                                invocation_id=run_id,
                                sequence=len(seen_runs),
                                agent=name,
                                start_offset=monotonic() - started_at,
                            )
                            seen_runs.add(run_id)
                            active[run_id] = stage
                            yield _stage_event(WorkflowEventType.AGENT_START, stage)
                        elif (
                            event_type in ("on_chain_end", "on_chain_error")
                            and run_id in active
                            and active[run_id].agent == node
                        ):
                            stage = active.pop(run_id)
                            output = data.get("output")
                            final_responses = []
                            if event_type == "on_chain_error":
                                boundary = close_stage(stage, "failed")
                                failed = node_failed = True
                            elif isinstance(output, dict):
                                boundary = close_stage(stage, "completed")
                                if node in STREAMING_NODES:
                                    final_responses = list(responses(output, stage))
                                    boundary.data["response_ids"] = [
                                        result.data["response_id"]
                                        for result in final_responses
                                    ]
                            else:
                                boundary = close_stage(stage, "incomplete")
                                incomplete = True
                            streams.pop(run_id, None)
                            yield boundary
                            for result in final_responses:
                                yield result
                        continue
                    if event_type not in ("on_chat_model_stream", "on_chat_model_end"):
                        continue
                    if not isinstance(node, str) or node not in STREAMING_NODES:
                        continue
                    stage = next(
                        (
                            active[parent]
                            for parent in reversed(parents)
                            if isinstance(parent, str) and parent in active
                        ),
                        None,
                    )
                    if stage is None or stage.agent != node:
                        continue
                    invocation_streams = streams.setdefault(stage.invocation_id, {})
                    if event_type == "on_chat_model_end":
                        message_id = getattr(data.get("output"), "id", None)
                        if (
                            run_id in invocation_streams
                            and isinstance(message_id, str)
                            and message_id
                        ):
                            invocation_streams[run_id].message_id = message_id
                        continue
                    chunk = data.get("chunk")
                    content = getattr(chunk, "content", None)
                    if not isinstance(content, str) or not content:
                        continue
                    stream = invocation_streams.get(run_id)
                    if stream is None:
                        stream = _StreamedResponse(
                            str(uuid.uuid4()), getattr(chunk, "id", None)
                        )
                        invocation_streams[run_id] = stream
                    yield WorkflowEvent(
                        WorkflowEventType.TOKEN,
                        {
                            "chunk": content,
                            "agent": node,
                            "invocation_id": stage.invocation_id,
                            "response_id": stream.response_id,
                        },
                    )

            if interrupted or (final_state and "__interrupt__" in final_state):
                terminal_outcome = stage_outcome = "interrupted"
            elif not failed and (final_state is None or active or incomplete):
                terminal_outcome = stage_outcome = "incomplete"
            elif not failed:
                context = final_state.get("context", "")
                if context:
                    yield WorkflowEvent(
                        WorkflowEventType.CONTEXT_UPDATE, {"context": context}
                    )
                artifacts = final_state.get("visual_artifacts", [])
                if artifacts:
                    yield WorkflowEvent(
                        WorkflowEventType.VISUALIZATION, {"artifacts": artifacts}
                    )
                for result in responses(final_state):
                    yield result
                outcome = _turn_outcome(
                    final_state, answer=bool(emitted_messages), graph=bool(artifacts)
                )
                if not emitted_messages and not artifacts:
                    yield self._response_event(
                        self._get_fallback_message(final_state), "system"
                    )
                yield WorkflowEvent(
                    WorkflowEventType.COMPLETE,
                    {
                        "status": "error" if outcome == "error" else "success",
                        "outcome": outcome,
                    },
                )
                return

        except UserFacingChatError as exc:
            logger.warning("Workflow token stream stopped", reason=exc.reason)
            error = {"message": exc.user_message, "reason": exc.reason}
        except GraphInterrupt:
            terminal_outcome = stage_outcome = "interrupted"
        except GraphRecursionError:
            logger.warning("Workflow exceeded recursion limit")
            error = {"message": ERROR_RECURSION_LIMIT, "reason": "recursion_limit"}
        except Exception as exc:
            logger.error("Workflow token stream error", error_type=type(exc).__name__)
        finally:
            reset_current_chat_turn(turn_token)

        if stage_outcome == "failed" and (len(active) > 1 or node_failed):
            stage_outcome = "incomplete"
        for stage in active.values():
            yield close_stage(stage, stage_outcome)
        yield WorkflowEvent(
            WorkflowEventType.ERROR,
            {
                "message": ERROR_WORKFLOW_FAILED,
                "reason": f"workflow_{terminal_outcome}",
                **(error or {}),
                "outcome": terminal_outcome,
            },
        )


_workflow = ReflexLearnerWorkflow()


def get_workflow() -> ReflexLearnerWorkflow:
    """Returns the process-local workflow wrapper."""
    return _workflow


def resolve_response_evidence(
    evidence: ResponseEvidence | None,
) -> ResponseEvidence | None:
    """Resolve relationship references and backfill summaries for saved passages."""
    if evidence is None:
        return None
    previewed_notes = [
        source
        for source in evidence.sources
        if source.kind == "note"
        and ((source.excerpt or "").strip() or source.supports_relationship)
    ]
    summarized_ids = {
        source.chunk_id for source in previewed_notes if (source.summary or "").strip()
    }
    references = [
        source.chunk_id
        for source in evidence.sources
        if source.kind == "note"
        and source.chunk_id
        and source.chunk_id not in summarized_ids
    ]
    if not references:
        return evidence
    try:
        store = get_workflow()._get_document_store()
        details = {
            source.chunk_id: source
            for source in resolve_note_chunks(
                lambda chunk_id: store.get_document(chunk_id, raise_error=False),
                references,
            )
            if source.excerpt or source.summary
        }
    except Exception as exc:
        logger.warning(
            "Response note previews unavailable", error_type=type(exc).__name__
        )
        return evidence
    previewed_ids = {source.chunk_id for source in previewed_notes}
    sources = []
    for source in evidence.sources:
        note = details.get(source.chunk_id) if source.kind == "note" else None
        if note is not None:
            if source.chunk_id not in previewed_ids:
                continue
            if (
                (source.excerpt or "").strip() or source.supports_relationship
            ) and note.summary:
                source = source.model_copy(update={"summary": note.summary})
        sources.append(source)
    sources.extend(
        note.model_copy(update={"supports_relationship": True})
        for chunk_id, note in details.items()
        if chunk_id not in previewed_ids
    )
    return evidence.model_copy(update={"sources": sources})
