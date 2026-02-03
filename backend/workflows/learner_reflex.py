import asyncio
from dataclasses import dataclass
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.errors import GraphRecursionError

from backend.configs.enums import WorkflowEventType
from backend.configs.models import ModelSettings
from backend.configs.paths import PathSettings
from backend.configs.search import TavilySettings, KnowledgeGraphSearchSettings
from backend.configs.storage import StorageSettings
from backend.utils.helpers import logger
from backend.workflows.learner import LearnerWorkflow


@dataclass
class WorkflowEvent:
    """Event emitted during workflow execution."""
    type: WorkflowEventType
    data: dict


class ReflexLearnerWorkflow:
    """Reflex-compatible wrapper for LearnerWorkflow with streaming support."""

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern to ensure only one workflow instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the workflow (only runs once due to singleton)."""
        if ReflexLearnerWorkflow._initialized:
            return

        self._workflow = None
        self._graph = None
        ReflexLearnerWorkflow._initialized = True

    def _ensure_initialized(self):
        """Ensure the workflow is initialized."""
        if self._graph is not None:
            return

        try:
            tavily_settings = TavilySettings()
            models_settings = ModelSettings()
            path_settings = PathSettings()
            kg_search_settings = KnowledgeGraphSearchSettings()

            storage_settings = StorageSettings()

            self._workflow = LearnerWorkflow(
                models_settings,
                path_settings,
                storage_settings,
                tavily_settings,
                kg_search_settings
            )
            self._graph = self._workflow.run()
            logger.info("ReflexLearnerWorkflow initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ReflexLearnerWorkflow: {e}")
            raise RuntimeError(f"Workflow initialization failed: {str(e)}")

    def _convert_messages(self, messages: list[dict]) -> list[BaseMessage]:
        """Convert message dicts to LangChain messages."""
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
            recursion_limit: int = 25
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """
        Process a user message through the workflow and yield events.
        
        Args:
            user_message: The user's message
            message_history: Previous messages in the conversation
            recursion_limit: Maximum recursion depth for the graph
            
        Yields:
            WorkflowEvent: Events during workflow execution
        """
        self._ensure_initialized()

        # Convert messages to LangChain format
        messages = self._convert_messages(message_history)
        messages.append(HumanMessage(content=user_message))

        config = {"recursion_limit": recursion_limit}
        initial_state = {"messages": messages}

        try:
            # Track which agents are being called
            # current_agent = None

            if hasattr(self._graph, "ainvoke"):
                result = await self._graph.ainvoke(initial_state, config=config)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._graph.invoke(initial_state, config=config)
                )

            if result:
                # Extract and yield context update
                context = result.get("context", "")
                if context:
                    yield WorkflowEvent(
                        type=WorkflowEventType.CONTEXT_UPDATE,
                        data={"context": context}
                    )

                # Extract and yield visualization
                visual_artifact = result.get("visual_artifact")
                if visual_artifact:
                    yield WorkflowEvent(
                        type=WorkflowEventType.VISUALIZATION,
                        data={"artifact": visual_artifact}
                    )

                # Extract and yield agent responses
                result_messages = result.get("messages", [])
                for msg in result_messages:
                    if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == 'ai':
                        agent_name = msg.additional_kwargs.get('agent', '').strip('[]').lower()
                        yield WorkflowEvent(
                            type=WorkflowEventType.RESPONSE,
                            data={
                                "content": msg.content,
                                "agent": agent_name
                            }
                        )

                yield WorkflowEvent(
                    type=WorkflowEventType.COMPLETE,
                    data={"next_step": result.get("next_step", "__end__")}
                )

        except GraphRecursionError as e:
            logger.error(f"Workflow exceeded recursion limit: {e}")
            yield WorkflowEvent(
                type=WorkflowEventType.ERROR,
                data={"message": "The workflow exceeded the maximum number of steps. Please try a simpler query."}
            )

        except Exception as e:
            logger.error(f"Workflow error: {e}")
            yield WorkflowEvent(
                type=WorkflowEventType.ERROR,
                data={"message": str(e)}
            )

    async def stream_with_status(
            self,
            user_message: str,
            message_history: list[dict],
            recursion_limit: int = 25
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """
        Stream workflow execution with detailed status updates.
        
        This method uses the graph's stream mode to provide real-time
        updates about which agent is currently active.
        """
        self._ensure_initialized()

        messages = self._convert_messages(message_history)
        messages.append(HumanMessage(content=user_message))

        config = {"recursion_limit": recursion_limit}
        initial_state = {"messages": messages}

        try:
            final_state = {}

            if hasattr(self._graph, "astream"):
                async for event in self._graph.astream(initial_state, config=config):
                    # Each event is a dict with node name as key
                    for node_name, node_output in event.items():
                        yield WorkflowEvent(
                            type=WorkflowEventType.AGENT_START,
                            data={"agent": node_name}
                        )

                        # Update final state with node output
                        if isinstance(node_output, dict):
                            final_state.update(node_output)

                        yield WorkflowEvent(
                            type=WorkflowEventType.AGENT_END,
                            data={"agent": node_name}
                        )
            else:
                loop = asyncio.get_running_loop()

                # Use stream to get node-by-node updates
                def run_stream():
                    results = []
                    for event in self._graph.stream(initial_state, config=config):
                        results.append(event)
                    return results

                events = await loop.run_in_executor(None, run_stream)

                for event in events:
                    # Each event is a dict with node name as key
                    for node_name, node_output in event.items():
                        yield WorkflowEvent(
                            type=WorkflowEventType.AGENT_START,
                            data={"agent": node_name}
                        )

                        # Update final state with node output
                        if isinstance(node_output, dict):
                            final_state.update(node_output)

                        yield WorkflowEvent(
                            type=WorkflowEventType.AGENT_END,
                            data={"agent": node_name}
                        )

            # Yield final results
            if final_state:
                context = final_state.get("context", "")
                if context:
                    yield WorkflowEvent(
                        type=WorkflowEventType.CONTEXT_UPDATE,
                        data={"context": context}
                    )

                visual_artifact = final_state.get("visual_artifact")
                if visual_artifact:
                    yield WorkflowEvent(
                        type=WorkflowEventType.VISUALIZATION,
                        data={"artifact": visual_artifact}
                    )

                result_messages = final_state.get("messages", [])
                if isinstance(result_messages, list):
                    for msg in result_messages:
                        if hasattr(msg, 'content') and hasattr(msg, 'type') and msg.type == 'ai':
                            agent_name = msg.additional_kwargs.get('agent', '').strip('[]').lower()
                            yield WorkflowEvent(
                                type=WorkflowEventType.RESPONSE,
                                data={
                                    "content": msg.content,
                                    "agent": agent_name
                                }
                            )

            yield WorkflowEvent(
                type=WorkflowEventType.COMPLETE,
                data={"status": "success"}
            )

        except GraphRecursionError as e:
            logger.error(f"Workflow exceeded recursion limit: {e}")
            yield WorkflowEvent(
                type=WorkflowEventType.ERROR,
                data={"message": "The workflow exceeded the maximum number of steps. Please try a simpler query."}
            )

        except Exception as e:
            logger.error(f"Workflow stream error: {e}")
            yield WorkflowEvent(
                type=WorkflowEventType.ERROR,
                data={"message": str(e)}
            )


# Global workflow instance
_workflow_instance = None


def get_workflow() -> ReflexLearnerWorkflow:
    """Get the singleton workflow instance."""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = ReflexLearnerWorkflow()
    return _workflow_instance
