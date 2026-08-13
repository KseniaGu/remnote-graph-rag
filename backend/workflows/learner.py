import asyncio
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pymongo import MongoClient
from tavily import TavilyClient

from backend.configs.constants import (
    DEFAULT_RECENT_MESSAGE_LIMIT,
    RETRIEVAL_BELOW_THRESHOLD,
    RETRIEVAL_NO_RESULTS,
    RETRIEVAL_TOPIC_MISMATCH,
    TITLE_MAX_LENGTH,
    VISUALIZATION_EMPTY_CONTEXT,
)
from backend.configs.enums import ModelRoleType, PromptType
from backend.configs.models import ModelSettings
from backend.configs.paths import PathSettings
from backend.configs.search import KnowledgeGraphSearchSettings, TavilySettings
from backend.configs.storage import StorageSettings
from backend.knowledge_graph.indexer import KnowledgeGraphIndexer
from backend.knowledge_graph.storage import KnowledgeGraphStorage
from backend.utils.chat_errors import (
    AIRequestRejected,
    AIResponseInvalid,
    AIServiceCapacity,
    AIServiceConfiguration,
    AIServiceUnavailable,
    UserFacingChatError,
)
from backend.utils.chat_limits import (
    get_chat_limit_service,
    get_current_chat_turn,
    is_transient_provider_error,
    provider_status_code,
)
from backend.utils.helpers import add_trace_metadata
from backend.utils.prompt_engine import PromptEngine
from backend.utils.session_memory import compress_text
from backend.workflows.agents.analyst_retrieval import AnalystRetrievalPipeline
from backend.workflows.agents.factory import AgentsFactory
from backend.workflows.agents.schemas import *
from backend.workflows.agents.tools import *
from backend.workflows.agents.visualizer_retrieval import VisualizerRetrievalPipeline

logger = get_logger(WORKFLOW_LOGGING)


class LearnerWorkflow:
    """Multi-agent workflow for technical interview preparation and learning.

    Orchestrates multiple specialized agents (Orchestrator, Retriever, Researcher, Analyst,
    Mentor, Visualizer) to provide comprehensive learning support through knowledge base
    retrieval, web research, technical analysis, and interactive mentoring.
    """

    def __init__(
        self,
        models_settings: ModelSettings,
        path_settings: PathSettings,
        storage_settings: StorageSettings,
        tavily_settings: TavilySettings,
        kg_search_settings: KnowledgeGraphSearchSettings,
    ):
        """Initializes the learner workflow with all required settings.

        Args:
            models_settings: Configuration for all LLM models and agents.
            path_settings: File system paths for data and prompts.
            storage_settings: Storage backend configuration.
            tavily_settings: Web search API configuration.
            kg_search_settings: Knowledge graph search parameters.
        """
        self.models_settings = models_settings
        self.path_settings = path_settings
        self.storage_settings = storage_settings
        self.chat_limit_service = get_chat_limit_service()
        self.chat_limits_settings = self.chat_limit_service.settings

        self.workflow = StateGraph(State)

        self.prompt_engine = PromptEngine(self.path_settings.prompts_dir)
        self.search_engine = TavilyClient(
            api_key=tavily_settings.api_key.get_secret_value()
        )

        # Initialize knowledge graph, agents, and workflow nodes
        self._init_graph(kg_search_settings, storage_settings)
        self._init_agents()
        self._init_nodes()

    def _init_graph(
        self,
        kg_search_settings: KnowledgeGraphSearchSettings,
        storage_settings: StorageSettings,
    ):
        """Initializes the knowledge graph indexer from prepared runtime storage.

        Runtime storage is load-only. Index construction, embedding backfill, and
        local-to-remote migration must be performed by the dedicated offline scripts.

        Args:
            kg_search_settings: Knowledge graph search configuration.
            storage_settings: Storage backend settings.
        """

        runtime_storage_backends = (
            storage_settings.document_storage,
            storage_settings.index_storage,
            storage_settings.vector_storage,
            storage_settings.property_graph_storage,
        )
        if any(
            getattr(backend, "init_from_local", False)
            for backend in runtime_storage_backends
        ) or getattr(storage_settings.vector_storage, "overwrite_index", False):
            raise RuntimeError(
                "Runtime retrieval storage must be load-only; disable "
                "init_from_local and vector-store overwrite settings."
            )

        def _create_embedder():
            return HuggingFaceEmbedding(
                self.models_settings.embedder.model_path,
                trust_remote_code=True,
                device=self.models_settings.embedder.device,
                embed_batch_size=10,
                local_files_only=True,
            )

        def _create_kg_storage():
            return KnowledgeGraphStorage(
                self.path_settings,
                storage_settings,
                embedding_dim=self.models_settings.embedder.embedding_dim,
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            embedder_future = executor.submit(_create_embedder)
            kg_storage_future = executor.submit(_create_kg_storage)
            embedder = embedder_future.result()
            kg_storage = kg_storage_future.result()
        logger.info("Knowledge graph storage initialized")

        self.knowledge_graph_indexer = KnowledgeGraphIndexer(
            kg_storage.storage_context,
            self.path_settings,
            kg_storage.storage_settings.document_storage.storage_type,
            kg_search_settings,
            embedder,
        )
        logger.info("Knowledge graph indexer initialized")

        # Runtime serving must never construct or repair retrieval storage.
        try:
            self.knowledge_graph_indexer.load_index()
        except ValueError as exc:
            raise RuntimeError(
                "Runtime retrieval storage is missing or invalid; build or migrate "
                "prepared storage before starting the application."
            ) from exc

    def _init_agents(self):
        """Initializes all agent models and tools.

        Sets up:
        - Tools: knowledge base search, web research, graph visualization
        - Prompts: loads and configures prompts for each agent role
        - Models: initializes LLM instances with role-specific configurations

        Special handling for:
        - Researcher: creates two variants (with_tools, structured output)
        - Orchestrator: configured with structured output for routing decisions
        - Retriever: bound with KB search and visualization tools
        """
        # Initialize tools
        base_kb_search_tool = self._build_kb_search_tool()
        limit_service = (
            getattr(self, "chat_limit_service", None) or get_chat_limit_service()
        )
        web_search_tool = deep_web_research(self.search_engine, limit_service)
        visualizer_kb_search_tool = self._build_visualizer_tool()
        self.tools = {
            tool.name: tool
            for tool in (
                base_kb_search_tool,
                web_search_tool,
                visualizer_kb_search_tool,
            )
        }

        # Load prompts and initialize models for each agent role
        self.prompts = {}
        model_agents_iter = ModelRoleType.get_all_members(is_agent=True)

        for next_step_type in model_agents_iter:
            role_type = getattr(ModelRoleType, next_step_type)
            model_settings = getattr(self.models_settings, role_type.name)

            prompt_version = model_settings.prompt_version
            model_function = ""
            if role_type == ModelRoleType.orchestrator:
                prompt_version, model_function = prompt_version["routing"], "routing"

            # Load and store prompts for this agent
            prompt, system_prompt = self.prompt_engine.render(
                PromptType.learner_workflow,
                role_type,
                prompt_version,
                model_function=model_function,
            )
            self.prompts[role_type] = (prompt, system_prompt)

            # Initialize model agents with role-specific configurations
            if role_type == ModelRoleType.researcher:
                self.researcher_with_tools = AgentsFactory.get_llm_by_role(
                    model_settings.with_tools
                )
                self.researcher_structured = AgentsFactory.get_llm_by_role(
                    model_settings.structured
                )
                self.researcher_with_tools = self.researcher_with_tools.bind_tools(
                    [web_search_tool]
                )
                self.researcher_structured = (
                    self.researcher_structured.with_structured_output(ResearchResult)
                )
                continue
            else:
                model = AgentsFactory.get_llm_by_role(model_settings)

                # Configure role-specific model capabilities
                if role_type == ModelRoleType.orchestrator:
                    model = model.with_structured_output(RoutingDecision)
                elif role_type == ModelRoleType.retriever:
                    model = model.bind_tools(
                        [base_kb_search_tool, visualizer_kb_search_tool]
                    )

            setattr(self, role_type.name, model)

    def _build_kb_search_tool(self):
        mode = self.knowledge_graph_indexer.kg_search_settings.analyst_retrieval_mode
        if mode == "optimized":
            reranker_settings = getattr(
                getattr(self, "models_settings", None), "reranker", None
            )
            if (
                self.knowledge_graph_indexer.kg_search_settings.analyst_reranker_mode
                == "ollama_llm_rerank"
                and getattr(
                    getattr(self, "chat_limits_settings", None),
                    "shared_quotas_enabled",
                    False,
                )
            ):
                raise ValueError(
                    "ollama_llm_rerank is not supported while shared chat quotas are enabled"
                )
            analyst_retrieval_pipeline = AnalystRetrievalPipeline(
                self.knowledge_graph_indexer,
                reranker_settings=reranker_settings,
            )
            return search_knowledge_base(analyst_pipeline=analyst_retrieval_pipeline)
        if mode == "legacy_vector_context":
            legacy_retriever = self.knowledge_graph_indexer.get_retriever(
                self.knowledge_graph_indexer.kg_search_settings.retriever_params
            )
            return search_knowledge_base(retriever=legacy_retriever)
        raise ValueError(f"Unsupported analyst retrieval mode: {mode}")

    def _build_visualizer_tool(self):
        mode = self.knowledge_graph_indexer.kg_search_settings.visualizer_retrieval_mode
        if mode == "optimized":
            visualizer_retrieval_pipeline = VisualizerRetrievalPipeline(
                self.knowledge_graph_indexer
            )
            return get_subgraphs_to_visualize(
                visualizer_pipeline=visualizer_retrieval_pipeline
            )
        if mode == "legacy_vector_context":
            legacy_retriever = self.knowledge_graph_indexer.get_retriever(
                self.knowledge_graph_indexer.kg_search_settings.visualizer_retriever_params
            )
            return get_subgraphs_to_visualize(retriever=legacy_retriever)
        raise ValueError(f"Unsupported visualizer retrieval mode: {mode}")

    def _init_nodes(self):
        """Initializes workflow graph nodes and edges.

        Creates the LangGraph workflow structure:
        - Adds all agent nodes (mentor, analyst, orchestrator, researcher, retriever, visualizer)
        - Defines edges: START -> orchestrator -> conditional routing -> agents -> orchestrator
        - Orchestrator acts as central router, directing flow based on conversation state
        """
        # Add all agent nodes to the workflow graph
        self.workflow.add_node("mentor", self.mentor_node)
        self.workflow.add_node("analyst", self.analyst_node)
        self.workflow.add_node("orchestrator", self.orchestrator_node)
        self.workflow.add_node("researcher", self.researcher_node)
        self.workflow.add_node("retriever", self.retriever_node)
        self.workflow.add_node("visualizer", self.visualizer_node)

        self.workflow.add_edge(START, "orchestrator")

        self.workflow.add_conditional_edges(
            "orchestrator",
            self.get_next_step,
            {
                ModelRoleType.analyst.name: "analyst",
                ModelRoleType.mentor.name: "mentor",
                ModelRoleType.researcher.name: "researcher",
                ModelRoleType.retriever.name: "retriever",
                "visualizer": "visualizer",
                "__end__": END,
            },
        )

        self.workflow.add_edge("retriever", "orchestrator")
        self.workflow.add_edge("researcher", "orchestrator")
        self.workflow.add_edge("visualizer", "orchestrator")
        self.workflow.add_edge("mentor", "orchestrator")
        self.workflow.add_edge("analyst", "orchestrator")

    def get_checkpointer(self):
        """Builds a MongoDBSaver checkpointer."""
        try:
            mongo_cfg = self.storage_settings.checkpoint_storage
            client = MongoClient(mongo_cfg.uri.get_secret_value())
            saver = MongoDBSaver(client, db_name=mongo_cfg.db_name)
            logger.info("LangGraph MongoDB checkpointer initialized")
            return saver
        except Exception as e:
            logger.warning(
                f"MongoDB checkpointer unavailable, falling back to MemorySaver: {e}"
            )
        return MemorySaver()

    @staticmethod
    def get_next_step(state: State) -> str:
        """Extracts the next routing step from workflow state.

        Args:
            state: Current workflow state containing next_step decision.

        Returns:
            Name of the next agent node to execute.
        """
        return state.next_step

    @staticmethod
    @traceable(run_type="chain", name="format_conversation_history")
    def format_conversation_history(
        messages: list, max_message_chars: int = 1200
    ) -> str:
        """Formats conversation messages into a readable history string.

        Args:
            messages: List of conversation messages.
            max_message_chars: Maximum content length per message before deterministic compression.

        Returns:
            Formatted conversation history with agent labels and content.
        """
        text = ""
        for message in messages:
            text += f"<message role={message.additional_kwargs.get('agent', message.type)}>\n"
            text += "<content>\n"
            text += f"{compress_text(str(message.content), max_message_chars)}\n"
            text += "</content>\n"
            text += "</message>"

        return text

    @staticmethod
    def _with_session_summary(conversation_history: str, session_summary: str) -> str:
        if not session_summary:
            return conversation_history
        return (
            "<older_conversation_summary>\n"
            f"{session_summary}\n"
            "</older_conversation_summary>\n\n"
            f"{conversation_history}"
        )

    @traceable(run_type="chain", name="create_messages_to_pass")
    def create_messages_to_pass(
        self,
        role_type: ModelRoleType,
        state: State,
        prompt_template_arguments: tuple,
        conversation_history_limit: int = DEFAULT_RECENT_MESSAGE_LIMIT,
    ) -> list[tuple[str, str]]:
        """Creates formatted message list for agent invocation.

        Builds prompt messages by:
        1. Extracting required data from state based on template arguments
        2. Formatting prompt template with extracted data
        3. Returning system instruction + formatted user prompt

        Args:
            role_type: Type of agent receiving the messages.
            state: Current workflow state.
            prompt_template_arguments: Tuple of required template argument names.
            conversation_history_limit: The maximum number of last messages passed to the agent.

        Returns:
            List of (role, content) tuples for model invocation.
        """
        prompt_kwargs = {}

        # Extract user's latest input if needed
        if "input_text" in prompt_template_arguments:
            user_messages = [
                message for message in state.messages if message.type == "human"
            ]
            prompt_kwargs["input_text"] = user_messages[-1].content

        # Add context (KB results, research findings, etc.)
        if "context" in prompt_template_arguments:
            prompt_kwargs["context"] = state.context
            # Notify if visualization was generated
            if (
                "visual_artifact" in prompt_template_arguments
                and state.visual_artifacts
            ):
                prompt_kwargs["context"] += "\nVisual artifact generated"

        if "conversation_history" in prompt_template_arguments:
            conversation_history = self.format_conversation_history(
                state.messages[-conversation_history_limit:]
            )
            prompt_kwargs["conversation_history"] = self._with_session_summary(
                conversation_history,
                state.session_summary,
            )

        prompt, system_prompt = self.prompts[role_type]
        prompt = prompt.format(**prompt_kwargs)

        return [("system", system_prompt["system_instruction"]), ("human", prompt)]

    async def call_model(
        self, messages_to_pass: list, role_type: ModelRoleType, **kwargs
    ) -> AIMessage | None:
        """Invokes one chat agent with bounded, status-aware retry behavior."""
        if role_type == ModelRoleType.researcher:
            model_type = kwargs.get("model_type")
            model = getattr(self, role_type.name + model_type)
            run_name = f"{role_type.name}{model_type}"
            provider_settings = getattr(
                self.models_settings.researcher, model_type.lstrip("_")
            )
        else:
            model = getattr(self, role_type.name)
            run_name = role_type.name
            provider_settings = getattr(self.models_settings, role_type.name)

        context = get_current_chat_turn()
        self.chat_limit_service.begin_llm_call(context, role_type.name)
        attempt_number = 0
        while True:
            attempt_number += 1
            await self.chat_limit_service.reserve_llm_attempt(
                context,
                getattr(provider_settings, "provider", "ollama"),
                role_type.name,
            )
            try:
                response = await model.ainvoke(
                    messages_to_pass, config={"run_name": run_name}
                )
                self.chat_limit_service.record_model_usage(context, response)
                return response
            except UserFacingChatError:
                raise
            except Exception as exc:
                should_retry = (
                    attempt_number == 1
                    and is_transient_provider_error(exc)
                    and self.chat_limit_service.claim_retry(context)
                )
                if should_retry:
                    logger.warning(
                        "Retrying transient chat-model failure",
                        role=role_type.name,
                        error_type=type(exc).__name__,
                    )
                    await asyncio.sleep(1.0 + random.uniform(0.0, 0.25))
                    continue

                err_type = self._classify_error(exc)
                add_trace_metadata("error_type", err_type)
                add_trace_metadata("error_role", role_type.name)
                logger.error(
                    "Chat-model invocation failed",
                    role=role_type.name,
                    error_type=err_type,
                )
                raise self._user_facing_model_error(exc, err_type) from exc

    @staticmethod
    def _classify_error(exc: BaseException) -> str:
        """Bucket an exception into a coarse `error_type` tag for observability."""
        name = type(exc).__name__
        msg = str(exc)
        if isinstance(exc, UnicodeEncodeError) or "surrogates not allowed" in msg:
            return "unicode_encode_error"
        if (
            name in ("OutputParserException", "ValidationError")
            or "validation error" in msg.lower()
        ):
            return "structured_output_parse_error"
        if (
            name in ("TimeoutError", "ReadTimeout", "WriteTimeout", "ConnectTimeout")
            or "timeout" in msg.lower()
        ):
            return "timeout"
        if "connection" in msg.lower() or name in ("ConnectError", "ConnectionError"):
            return "connection_error"
        return f"unexpected:{name}"

    @staticmethod
    def _user_facing_model_error(
        exc: BaseException, error_type: str
    ) -> UserFacingChatError:
        """Maps provider failures to stable public error categories."""
        status = provider_status_code(exc)
        if status == 429:
            return AIServiceCapacity()
        if status in {401, 403, 404}:
            return AIServiceConfiguration()
        if status in {400, 422}:
            return AIRequestRejected()
        if error_type == "structured_output_parse_error":
            return AIResponseInvalid()
        if error_type == "unicode_encode_error":
            return AIRequestRejected()
        return AIServiceUnavailable()

    @traceable(run_type="chain", name="call_tools")
    async def call_tools(self, response: Any) -> dict[str, Any]:
        """Executes at most the first recognized tool call."""
        tool_calls = list(getattr(response, "tool_calls", []) or [])
        recognized = [call for call in tool_calls if call.get("name") in self.tools]
        if len(tool_calls) > 1:
            logger.warning(
                "Discarding excess model tool calls",
                returned_count=len(tool_calls),
                executed_count=min(1, len(recognized)),
            )
        if not recognized:
            return {}

        tool_call = recognized[0]
        tool_name = tool_call["name"]
        try:
            logger.info("Executing tool", tool_name=tool_name)
            result = await self.tools[tool_name].ainvoke(tool_call.get("args", {}))
        except UserFacingChatError:
            raise
        except Exception as exc:
            logger.error(
                "Workflow tool failed",
                tool_name=tool_name,
                error_type=type(exc).__name__,
            )
            return {}
        return {tool_name: result} if result is not None else {}

    @staticmethod
    def _deterministic_route(state: State) -> str | None:
        """Deterministic routing decisions.

        Returns the next step name if the context + last message uniquely determine it, otherwise None.
        """
        ctx = state.context or ""

        # Priority 0: terminal signals set earlier in this turn.
        if state.request_scope == "out_of_scope" or state.sources_exhausted:
            return "__end__"
        if "[RESEARCH_COMPLETE]" in ctx:
            return ModelRoleType.analyst.name
        if state.retriever_empty:
            return ModelRoleType.researcher.name
        if (
            "Visual artifact generated" in ctx
            or "Visualization failed" in ctx
            or VISUALIZATION_EMPTY_CONTEXT in ctx
        ):
            return "__end__"

        # Priority 1: if an agent (analyst/mentor) has already produced a response this turn,
        # its message is the last one in state.messages. Workers add additional_kwargs["agent"]
        # to their response; presence of that tag means the turn is complete.
        if state.messages:
            last = state.messages[-1]
            agent_tag = (
                last.additional_kwargs.get("agent", "")
                if hasattr(last, "additional_kwargs")
                else ""
            )
            if agent_tag in ("[ANALYST]", "[MENTOR]"):
                return "__end__"

        # Priority 2: visualization data has been gathered, route to visualizer.
        if "get_subgraphs_to_visualize" in ctx:
            return "visualizer"

        # Priority 3: KB results are in context, route to analyst for synthesis.
        if "search_knowledge_base" in ctx:
            return ModelRoleType.analyst.name

        # Priority 4: web research findings are in context (covered by RESEARCH_COMPLETE above,
        # but handle the case where the sentinel got stripped).
        if "## Web Research Findings" in ctx:
            return ModelRoleType.analyst.name

        # Context is empty or unrecognized; an LLM classifier is needed for intent routing.
        return None

    async def orchestrator_node(self, state: State) -> dict[str, str]:
        """Orchestrator agent: central router that decides next workflow step.

        Fast path: when the context uniquely determines the next step (sentinels, KB results,
        agent completion), route deterministically without any LLM call. This eliminates ~80%
        of orchestrator LLM traffic based on log analysis.

        Slow path: when intent is ambiguous (empty context, fresh human question), call the
        Orchestrator LLM with the structured-output schema for intent classification.

        Args:
            state: Current workflow state.

        Returns:
            Dictionary with next_step routing decision.
        """
        ctx = state.context or ""
        add_trace_metadata("context", ctx)

        deterministic = self._deterministic_route(state)
        if deterministic is not None:
            logger.info(f"[ORCHESTRATOR] Routing to: {deterministic} (deterministic)")
            return {"next_step": deterministic}

        messages_to_pass = self.create_messages_to_pass(
            ModelRoleType.orchestrator,
            state,
            ("conversation_history", "context", "visual_artifact"),
        )
        response = await self.call_model(messages_to_pass, ModelRoleType.orchestrator)

        if response:
            request_scope = response.request_scope
            next_step = response.next_step
            if request_scope == "out_of_scope":
                next_step = "__end__"

            # Safety: don't route to visualizer without graph data (first attempt only).
            if next_step == "visualizer" and "get_subgraphs_to_visualize" not in ctx:
                logger.warning(
                    "[ORCHESTRATOR] No graph data for visualizer. Routing to retriever."
                )
                next_step = ModelRoleType.retriever.name

            add_trace_metadata("request_scope", request_scope)
            logger.info(
                "[ORCHESTRATOR] Routing decision",
                next_step=next_step,
                request_scope=request_scope,
                reasoning=response.reasoning,
            )
            return {"next_step": next_step, "request_scope": request_scope}

        logger.warning(
            "[ORCHESTRATOR] LLM failed on ambiguous input; defaulting to retriever."
        )
        add_trace_metadata("request_scope", "ambiguous")
        return {
            "next_step": ModelRoleType.retriever.name,
            "request_scope": "ambiguous",
        }

    async def retriever_node(self, state: State) -> dict[str, str]:
        """Retriever agent: fetches information from knowledge base or prepares visualization data.

        Analyzes user request to determine intent:
        - Information search: calls search_knowledge_base tool to retrieve relevant facts
        - Visualization: calls get_subgraphs_to_visualize to gather graph structure data

        The retrieved data is stored in context for downstream agents (Analyst, Visualizer).

        Args:
            state: Current workflow state.

        Returns:
            Dictionary with updated context containing tool results.
        """
        model_name = ModelRoleType.retriever.name.upper()
        add_trace_metadata("context", state.context or "")
        messages_to_pass = self.create_messages_to_pass(
            ModelRoleType.retriever, state, ("input_text", "conversation_history")
        )

        response = await self.call_model(messages_to_pass, ModelRoleType.retriever)
        if response and response.tool_calls:
            tool_results = await self.call_tools(response)
            retrieval_status = self._assess_retrieval_results(tool_results)
        else:
            tool_results = {}
            retrieval_status = "no_results"

        add_trace_metadata("retrieval_status", retrieval_status)
        logger.info(
            f"[{model_name}] Retrieval assessed",
            retrieval_status=retrieval_status,
        )
        if retrieval_status != "adequate":
            return {
                "context": "",
                "retriever_empty": True,
                "retrieval_status": retrieval_status,
            }
        return {
            "context": json.dumps(tool_results, ensure_ascii=False),
            "retriever_empty": False,
            "retrieval_status": retrieval_status,
        }

    # Minimum score a retrieved item must have for us to treat the retrieval as useful.
    _KB_MIN_USEFUL_SCORE = 0.30

    @classmethod
    def _assess_retrieval_results(cls, tool_results: dict[str, Any] | None) -> str:
        """Classifies retrieval output without conflating distinct failure modes."""
        if not tool_results:
            return "no_results"

        for tool_name, value in tool_results.items():
            if tool_name == "get_subgraphs_to_visualize":
                return "adequate"
            if tool_name != "search_knowledge_base" or not isinstance(value, str):
                continue

            result = value.strip()
            if not result or result == RETRIEVAL_NO_RESULTS:
                return "no_results"
            if result == RETRIEVAL_BELOW_THRESHOLD:
                return "below_threshold"
            if result == RETRIEVAL_TOPIC_MISMATCH:
                return "topic_mismatch"
            if "[RELATION]" not in result and "[SOURCE]" not in result:
                return "no_results"

            scores = re.findall(r"\(Score:\s*([\d.]+)", result)
            if scores and not any(
                float(score) >= cls._KB_MIN_USEFUL_SCORE for score in scores
            ):
                return "below_threshold"
            return "adequate"

        return "no_results"

    async def researcher_node(self, state: State) -> dict[str, str]:
        """Researcher agent: conducts web research for information not in knowledge base.

        Two-phase process:
            1. researcher_with_tools: analyzes request and calls deep_web_research tool
            2. researcher_structured: synthesizes search results into structured ResearchResult

        Outputs structured findings with:
            - key_findings: synthesized summary of relevant information
            - sources: list of web sources with titles, URLs, and types
            - confidence_level: assessment of source quality (high/medium/low)
            - status: success, partial_match, or no_relevant_info

        Args:
            state: Current workflow state.

        Returns:
            Dictionary with updated context containing formatted research findings.
        """
        model_name = ModelRoleType.researcher.name.upper()
        add_trace_metadata("context", state.context or "")
        messages_to_pass = self.create_messages_to_pass(
            ModelRoleType.researcher, state, ("input_text", "context")
        )

        response = await self.call_model(
            messages_to_pass, ModelRoleType.researcher, model_type="_with_tools"
        )
        logger.debug(f"[{model_name}] Initial response received")

        # Execute tools and synthesize results
        if response and response.tool_calls:
            tool_call_results = await self.call_tools(response)

            if tool_call_results:
                # Format tool results for structured output model
                tool_results_text = "\n\n".join(
                    [
                        f"Tool: {tool_name}\n{result}"
                        for tool_name, result in tool_call_results.items()
                    ]
                )

                # Create clean message history for researcher_structured
                clean_messages = [
                    *messages_to_pass,
                    ("assistant", response.text),
                    (
                        "human",
                        f"Here are the search results:\n\n{tool_results_text}\n\n.",
                    ),
                ]

                # Get structured research output
                research_result = await self.call_model(
                    clean_messages, ModelRoleType.researcher, model_type="_structured"
                )

                if research_result:
                    logger.info(
                        f"[{model_name}] Research status: {research_result.status}, "
                        f"confidence: {research_result.confidence_level}"
                    )
                    add_trace_metadata("research_status", research_result.status)

                    if (
                        research_result.status == "no_relevant_info"
                        or not research_result.key_findings.strip()
                    ):
                        logger.warning(
                            f"[{model_name}] Web evidence did not establish the requested topic; marking sources_exhausted."
                        )
                        return {
                            "context": "",
                            "retriever_empty": False,
                            "sources_exhausted": True,
                        }

                    # Format research findings with sources for analyst. Strip any prior
                    # `retriever_empty`-era context so we don't concatenate stale markers.
                    base_ctx = (
                        state.context
                        if "[RESEARCH_COMPLETE]" not in (state.context or "")
                        else ""
                    )
                    formatted_context = f"{base_ctx}\n\n## Web Research Findings\n\n{research_result.key_findings}".lstrip()
                    if research_result.sources:
                        formatted_context += "\n\n### Sources:\n"
                        for source in research_result.sources:
                            formatted_context += (
                                f"- [{source.get('title', 'Unknown')}]({source.get('url', '')}) "
                                f"({source.get('type', 'web')})\n"
                            )

                    return {
                        "context": formatted_context + "\n[RESEARCH_COMPLETE]",
                        "retriever_empty": False,
                        "sources_exhausted": False,
                    }

        if state.retriever_empty:
            logger.warning(
                f"[{model_name}] No search result generated and retriever was empty. Marking sources_exhausted."
            )
            return {"context": "", "sources_exhausted": True}
        logger.warning(
            f"[{model_name}] No search result generated, returning original context"
        )
        return {"context": state.context}

    async def analyst_node(self, state: State) -> dict[str, list]:
        """Analyst agent: synthesizes retrieved data into professional technical responses.

        Acts as the "Internal Scribe" that combines:
            - Knowledge base facts (from Retriever)
            - Web research findings (from Researcher)

        Into a cohesive, well-formatted response with:
            - Technical precision (LaTeX for formulas, proper ML nomenclature)
            - Structural clarity (Markdown headers, tables, bolding)
            - Source attribution (KB vs. web sources)

        Args:
            state: Current workflow state.

        Returns:
            Dictionary with analyst's response message.
        """
        model_name = ModelRoleType.analyst.name.upper()
        add_trace_metadata("context", state.context or "")
        messages_to_pass = self.create_messages_to_pass(
            ModelRoleType.analyst, state, ("input_text", "context")
        )
        response = await self.call_model(messages_to_pass, ModelRoleType.analyst)

        if response:
            content, stripped = response.content, ""
            if isinstance(content, str):
                stripped = content.strip()
                if stripped.startswith("```"):
                    stripped = (
                        stripped[stripped.index("\n") + 1 :]
                        if "\n" in stripped
                        else stripped[3:]
                    )
                if stripped.endswith("```"):
                    stripped = stripped[: stripped.rfind("```")]
                stripped = stripped.strip()
                if stripped:
                    response.content = stripped
                    response.additional_kwargs["agent"] = f"[{model_name}]"
                    return {"messages": [response], "context": ""}

        logger.warning(
            f"[{model_name}] No meaningful response generated, returning empty messages for fallback handling"
        )
        return {"messages": []}

    async def mentor_node(self, state: State) -> dict[str, list]:
        """Mentor agent: conducts Socratic interview practice and technical questioning.

        Acts as a Senior Technical Interviewer that:
            - Uses retrieved context as "Ground Truth" to fact-check user answers
            - Applies Socratic method: asks follow-up questions to expose logic gaps
            - Provides targeted explanations when user is fundamentally stuck

        Args:
            state: Current workflow state.

        Returns:
            Dictionary with mentor's response message.
        """
        model_name = ModelRoleType.mentor.name.upper()
        add_trace_metadata("context", state.context or "")
        messages_to_pass = self.create_messages_to_pass(
            ModelRoleType.mentor, state, ("conversation_history", "context")
        )
        response = await self.call_model(messages_to_pass, ModelRoleType.mentor)
        if response:
            response.additional_kwargs["agent"] = f"[{model_name}]"
            return {"messages": [response]}

        logger.warning(
            f"[{model_name}] No response generated, returning empty messages for fallback handling"
        )
        return {"messages": []}

    async def visualizer_node(self, state: State) -> dict[str, list]:
        """Visualizer agent: creates interactive graph visualizations from retrieved data.

        Processes graph structure data from get_subgraphs_to_visualize tool and generates an interactive
        Plotly visualization. Appends the new plot to the existing visual_artifacts list.

        Args:
            state: Current workflow state containing graph data in context.

        Returns:
            Dictionary with updated visual_artifacts list.
        """
        add_trace_metadata("context", state.context or "")
        try:
            tool_results = state.context
            if tool_results:
                visualizer_tool_results = json.loads(tool_results).get(
                    "get_subgraphs_to_visualize"
                )
                if visualizer_tool_results:
                    nodes, relation_triplets, queries = visualizer_tool_results
                    if not nodes and not relation_triplets:
                        logger.warning("[VISUALIZER] No focused graph data found.")
                        return {
                            "visual_artifacts": state.visual_artifacts,
                            "context": VISUALIZATION_EMPTY_CONTEXT,
                        }
                    title = " & ".join(queries)
                    title = (
                        (title[:TITLE_MAX_LENGTH] + "…")
                        if len(title) > TITLE_MAX_LENGTH
                        else title
                    )
                    plotly_figure = (
                        self.knowledge_graph_indexer.get_graph_visualization(
                            nodes, relation_triplets, title=title.title()
                        )
                    )
                    return {
                        "visual_artifacts": [plotly_figure.to_dict()],
                        "context": "Visual artifact generated",
                    }
            logger.error("[VISUALIZER] No graph data found in context")
        except Exception as e:
            logger.error(f"[VISUALIZER] Failed to create visualization: {e!s}")

        return {
            "visual_artifacts": state.visual_artifacts,
            "context": "Visualization failed",
        }

    def run(self) -> Any:
        """Compiles and returns the executable workflow graph.

        Returns:
            Compiled LangGraph workflow ready for invocation.
        """
        return self.workflow.compile(checkpointer=self.get_checkpointer())

    def show_graph(self, jupyter_notebook: bool = False) -> None:
        """Visualizes the workflow graph structure.

        Args:
            jupyter_notebook: If True, returns IPython display object for notebooks. If False, opens graph image in
                              system viewer.

        Returns:
            IPython display object if jupyter_notebook=True, None otherwise.
        """
        graph = self.run()
        png_graph = graph.get_graph().draw_mermaid_png()
        if jupyter_notebook:
            try:
                from IPython.display import Image as IPyImage
                from IPython.display import display
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "IPython is required for jupyter_notebook=True."
                ) from e

            return display(IPyImage(png_graph))

        try:
            from PIL import Image as PILImage
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Pillow is required for show_graph().") from e

        img = PILImage.open(BytesIO(png_graph))
        img.show()


if __name__ == "__main__":
    # ...
    """
    from backend.configs.storage import StorageSettings
    from langgraph.errors import GraphRecursionError

    tavily_settings = TavilySettings()
    models_settings = ModelSettings()
    path_settings = PathSettings()
    kg_search_settings = KnowledgeGraphSearchSettings()

    # For local storage configuration. Comment the lines below to use non-local
    storage_settings = StorageSettings()
    # storage_settings.document_storage = LocalStorageSettings(storage_path=path_settings.local_storage_dir)
    # storage_settings.index_storage = LocalStorageSettings(storage_path=path_settings.local_storage_dir)
    # storage_settings.vector_storage = LocalStorageSettings(storage_path=path_settings.local_storage_dir)
    # storage_settings.property_graph_storage = LocalStorageSettings(storage_path=path_settings.local_storage_dir)

    workflow = LearnerWorkflow(models_settings, path_settings, storage_settings, tavily_settings, kg_search_settings)
    graph = workflow.run()
    # workflow.show_graph()
    # messages = ('Visualize my knowledge about neural networks',)
    messages = ("Research the latest developments in LLM fine-tuning",)
    # messages = ("What information do I have about attention mechanism?",)
    # "What information do I have about EAGLE?",
    # "What information do I have about attention mechanism?",
    # "Tell me please more about VAE",
    # "Can you please visualize this information?")

    try:
        for message in messages:
            state = asyncio.run(
                graph.ainvoke({"messages": [message]}, config={"recursion_limit": 25, "configurable": {"thread_id": 1}})
            )
    except GraphRecursionError as e:
        logger.error(f"Workflow exceeded recursion limit: {e}")

    import plotly.graph_objects as go

    fig = go.Figure(state["visual_artifacts"][-1])
    fig.update_layout(
        autosize=True,
        width=None,
        height=None,
    )
    fig.show()
    """
