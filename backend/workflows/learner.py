import json
from io import BytesIO

from IPython.display import Image, display
from PIL import Image
from langgraph.graph import StateGraph, START, END
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from ollama._types import ResponseError
from tavily import TavilyClient

from backend.configs.enums import ModelRoleType, PromptType
from backend.configs.models import ModelSettings
from backend.configs.paths import PathSettings
from backend.configs.search import TavilySettings, KnowledgeGraphSearchSettings
from backend.configs.storage import StorageSettings
from backend.knowledge_graph.indexer import KnowledgeGraphIndexer
from backend.knowledge_graph.storage import KnowledgeGraphStorage
from backend.utils.prompt_engine import PromptEngine
from backend.workflows.agents.factory import AgentsFactory
from backend.workflows.agents.schemas import *
from backend.workflows.agents.tools import *

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
            kg_search_settings: KnowledgeGraphSearchSettings
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

        self.workflow = StateGraph(State)

        self.prompt_engine = PromptEngine(self.path_settings.prompts_dir)
        self.search_engine = TavilyClient(api_key=tavily_settings.api_key.get_secret_value())

        # Initialize knowledge graph, agents, and workflow nodes
        self._init_graph(kg_search_settings, storage_settings)
        self._init_agents()
        self._init_nodes()

    def _init_graph(self, kg_search_settings: KnowledgeGraphSearchSettings, storage_settings: StorageSettings):
        """Initializes the knowledge graph indexer with embeddings and reranking.
        
        Sets up the knowledge graph storage, embedder, reranker, and attempts to load existing index.
        If no index exists, builds a new one using the orchestrator agent.
        
        Args:
            kg_search_settings: Knowledge graph search configuration.
            storage_settings: Storage backend settings.
        """
        embedder = HuggingFaceEmbedding(
            self.models_settings.embedder.model_path,
            trust_remote_code=True,
            device=self.models_settings.embedder.device,
            embed_batch_size=10,
        )
        reranker = CohereRerank(
            api_key=self.models_settings.reranker.api_key.get_secret_value(),
            model=self.models_settings.reranker.model_name,
            top_n=self.models_settings.reranker.top_n,
        )
        kg_storage = KnowledgeGraphStorage(
            self.path_settings,
            storage_settings,
            embedding_dim=self.models_settings.embedder.embedding_dim,
        )
        self.knowledge_graph_indexer = KnowledgeGraphIndexer(
            kg_storage.storage_context,
            self.path_settings,
            kg_storage.storage_settings.document_storage.storage_type,
            kg_search_settings,
            embedder,
            reranker,
        )
        logger.info("Knowledge graph indexer initialized")

        # Attempt to load existing index, build new one if not found
        try:
            self.knowledge_graph_indexer.load_index()
        except ValueError:
            logger.info("No existing index found. Building new knowledge graph index...")
            role_settings = getattr(self.models_settings, ModelRoleType.orchestrator.name)
            prompt_version = role_settings.prompt_version["graph_index"]
            graph_index_prompt, graph_index_system_prompt = self.prompt_engine.render(
                PromptType.learner_workflow, ModelRoleType.orchestrator, prompt_version, "graph_index"
            )
            role_params = role_settings.model_dump(include={"temperature", "top_k", "top_p", "base_url"})
            llm_params = {"model": role_settings.model_name, **role_params}
            self.knowledge_graph_indexer.build_index(
                llm=llm_params,
                graph_index_prompt=graph_index_system_prompt["system_instruction"] + "\n" + graph_index_prompt,
            )
            logger.info("Knowledge graph index built successfully")

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
        base_kb_search_tool = search_knowledge_base(
            self.knowledge_graph_indexer.get_retriever(), self.knowledge_graph_indexer.reranker
        )
        web_search_tool = deep_web_research(self.search_engine)
        visualizer_retriever_params = self.knowledge_graph_indexer.kg_search_settings.visualizer_retriever_params
        visualizer_kb_search_tool = get_sub_graphs_to_visualize(
            self.knowledge_graph_indexer.get_retriever(visualizer_retriever_params)
        )
        self.tools = {tool.name: tool for tool in (base_kb_search_tool, web_search_tool, visualizer_kb_search_tool)}

        # Load prompts and initialize models for each agent role
        self.prompts = {}
        model_agents_iter = ModelRoleType.get_all_members(is_agent=True)

        for next_step_type in model_agents_iter:
            role_type = getattr(ModelRoleType, next_step_type)
            model_settings = getattr(self.models_settings, role_type.name)

            # Handle different prompt version formats
            if role_type == ModelRoleType.researcher:
                prompt_version = model_settings["prompt_version"]
            else:
                prompt_version = model_settings.prompt_version
            model_function = ""
            if role_type == ModelRoleType.orchestrator:
                prompt_version, model_function = prompt_version["routing"], "routing"

            # Load and store prompts for this agent
            prompt, system_prompt = self.prompt_engine.render(
                PromptType.learner_workflow, role_type, prompt_version, model_function=model_function
            )
            self.prompts[role_type] = (prompt, system_prompt)

            # Initialize model agents with role-specific configurations
            if role_type == ModelRoleType.researcher:
                self.researcher_with_tools = AgentsFactory.get_llm_by_role(model_settings["_with_tools"])
                self.researcher_structured = AgentsFactory.get_llm_by_role(model_settings["_structured"])
                self.researcher_with_tools = AgentsFactory.add_retry(
                    self.researcher_with_tools.bind_tools([web_search_tool])
                )
                self.researcher_structured = AgentsFactory.add_retry(
                    self.researcher_structured.with_structured_output(ResearchResult)
                )
            else:
                model = AgentsFactory.get_llm_by_role(model_settings)

                # Configure role-specific model capabilities
                if role_type == ModelRoleType.orchestrator:
                    model = model.with_structured_output(RoutingDecision)
                elif role_type == ModelRoleType.retriever:
                    model = model.bind_tools([base_kb_search_tool, visualizer_kb_search_tool])

                setattr(self, role_type.name, AgentsFactory.add_retry(model))

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
            }
        )

        self.workflow.add_edge("retriever", "orchestrator")
        self.workflow.add_edge("researcher", "orchestrator")
        self.workflow.add_edge("visualizer", "orchestrator")
        self.workflow.add_edge("mentor", "orchestrator")
        self.workflow.add_edge("analyst", "orchestrator")

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
    def format_conversation_history(messages: list) -> str:
        """Formats conversation messages into a readable history string.
        
        Args:
            messages: List of conversation messages.
            
        Returns:
            Formatted conversation history with agent labels and content.
        """
        return "\n".join(f"{message.additional_kwargs.get('agent', message.type)}: "
                         f"{message.content}" for message in messages)

    def create_messages_to_pass(
            self,
            role_type: ModelRoleType,
            state: State,
            prompt_template_arguments: tuple
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
            
        Returns:
            List of (role, content) tuples for model invocation.
        """
        prompt_kwargs = {}

        # Extract user's latest input if needed
        if "input_text" in prompt_template_arguments:
            user_messages = [message for message in state.messages if message.type == 'human']
            prompt_kwargs["input_text"] = user_messages[-1].content

        # Add context (KB results, research findings, etc.)
        if "context" in prompt_template_arguments:
            prompt_kwargs["context"] = state.context
            # Notify if visualization was generated
            if "visual_artifact" in prompt_template_arguments and state.visual_artifact:
                prompt_kwargs["context"] += "\nVisual artifact generated"

        if "conversation_history" in prompt_template_arguments:
            prompt_kwargs["conversation_history"] = self.format_conversation_history(state.messages)

        prompt, system_prompt = self.prompts[role_type]
        prompt = prompt.format(**prompt_kwargs)

        return [("system", system_prompt["system_instruction"]), ("human", prompt)]

    def call_model(self, messages_to_pass: list, role_type: ModelRoleType, **kwargs) -> Optional[Any]:
        """Invokes an agent model with error handling.
        
        Args:
            messages_to_pass: Formatted messages for model input.
            role_type: Type of agent to invoke.
            **kwargs: Additional arguments (e.g., model_type for researcher variants).
            
        Returns:
            Model response or None if invocation failed.
        """
        if role_type == ModelRoleType.researcher:
            model_type = kwargs.get("model_type")
            model = getattr(self, role_type.name + model_type)
        else:
            model = getattr(self, role_type.name)
        try:
            return model.invoke(messages_to_pass)
        except ResponseError as e:
            logger.error(f"[{role_type.name.upper()}] Service error after retries exhausted: {e.status_code}")
        except Exception as e:
            logger.error(f"[{role_type.name.upper()}] Unexpected error: {str(e)}")
        return None

    def call_tools(self, response: Any) -> dict[str, Any]:
        """Executes tool calls from agent response.
        
        Args:
            response: Agent response containing tool_calls.
            
        Returns:
            Dictionary mapping tool names to their results.
        """
        tool_results = {}
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            try:
                logger.info(f"Executing tool: {tool_name}")
                result = self.tools[tool_name].invoke(tool_call["args"])
                tool_results[tool_name] = result
            except Exception as e:
                logger.error(f"{tool_name} failed. Error: {str(e)}")

        return tool_results

    def orchestrator_node(self, state: State) -> dict[str, str]:
        """Orchestrator agent: central router that decides next workflow step.
        
        Analyzes conversation history and current context to determine which specialized agent should handle the request
        next. Routes to:
            - retriever: for KB searches or visualization data gathering
            - researcher: for web research on topics not in KB
            - analyst: to synthesize retrieved data into professional responses
            - mentor: for interactive interview practice and Socratic questioning
            - visualizer: to create graph visualizations from retrieved data
            - __end__: when agent has responded and awaiting user input
        
        Args:
            state: Current workflow state.
            
        Returns:
            Dictionary with next_step routing decision.
        """
        messages_to_pass = self.create_messages_to_pass(
            ModelRoleType.orchestrator, state, ("conversation_history", "context", "visual_artifact")
        )
        response = self.call_model(messages_to_pass, ModelRoleType.orchestrator)
        if response:
            next_step = response.next_step
            if next_step == "visualizer":
                if not state.context or "get_sub_graphs_to_visualize" not in state.context:
                    logger.warning(
                        "[ORCHESTRATOR] Cannot route to visualizer without graph data. "
                        "Routing to retriever first."
                    )
                    next_step = "retriever"

            logger.info(f"[ORCHESTRATOR] Routing to: {next_step}")
            logger.info(f"[ORCHESTRATOR] Reasoning: {response.reasoning}")
            return {"next_step": next_step}
        return {"next_step": ModelRoleType.retriever.name}

    def retriever_node(self, state: State) -> dict[str, str]:
        """Retriever agent: fetches information from knowledge base or prepares visualization data.
        
        Analyzes user request to determine intent:
        - Information search: calls search_knowledge_base tool to retrieve relevant facts
        - Visualization: calls get_sub_graphs_to_visualize to gather graph structure data
        
        The retrieved data is stored in context for downstream agents (Analyst, Visualizer).
        
        Args:
            state: Current workflow state.
            
        Returns:
            Dictionary with updated context containing tool results.
        """
        model_name = ModelRoleType.retriever.name.upper()
        messages_to_pass = self.create_messages_to_pass(
            ModelRoleType.retriever, state, ("input_text", "conversation_history")
        )

        response = self.call_model(messages_to_pass, ModelRoleType.retriever)
        if response and response.tool_calls:
            tool_results = self.call_tools(response)
            return {"context": json.dumps(tool_results)}

        logger.warning(f"[{model_name}] Nothing retrieved, returning original context")
        return {"context": state.context}

    def researcher_node(self, state: State) -> dict[str, str]:
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
        messages_to_pass = self.create_messages_to_pass(
            ModelRoleType.researcher, state, ("input_text", "context")
        )

        response = self.call_model(messages_to_pass, ModelRoleType.researcher, model_type="_with_tools")
        logger.debug(f"[{model_name}] Initial response received")

        # Execute tools and synthesize results
        if response and response.tool_calls:
            tool_call_results = self.call_tools(response)

            if tool_call_results:
                # Format tool results for structured output model
                tool_results_text = "\n\n".join([
                    f"Tool: {tool_name}\n{result}"
                    for tool_name, result in tool_call_results.items()
                ])

                # Create clean message history for researcher_structured
                clean_messages = [
                    *messages_to_pass,
                    ("assistant", response.text),
                    ("human", f"Here are the search results:\n\n{tool_results_text}\n\n.")
                ]

                # Get structured research output
                research_result = self.call_model(clean_messages, ModelRoleType.researcher, model_type="_structured")

                if research_result:
                    logger.info(
                        f"[{model_name}] Research status: {research_result.status}, "
                        f"confidence: {research_result.confidence_level}"
                    )

                    # Format research findings with sources for analyst
                    formatted_context = f"{state.context}\n\n## Web Research Findings\n\n{research_result.key_findings}"
                    if research_result.sources:
                        formatted_context += "\n\n### Sources:\n"
                        for source in research_result.sources:
                            formatted_context += f"- [{source.get('title', 'Unknown')}]({source.get('url', '')}) " \
                                                 f"({source.get('type', 'web')})\n"

                    return {"context": formatted_context}

        logger.warning(f"[{model_name}] No search result generated, returning original context")
        return {"context": state.context}

    def analyst_node(self, state: State) -> dict[str, list]:
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
        messages_to_pass = self.create_messages_to_pass(ModelRoleType.analyst, state, ("input_text", "context"))
        response = self.call_model(messages_to_pass, ModelRoleType.analyst)
        if response:
            response.additional_kwargs["agent"] = f"[{model_name}]"
            return {"messages": [response]}

        logger.warning(f"[{model_name}] No response generated, returning empty list")
        return {"messages": []}

    def mentor_node(self, state: State) -> dict[str, list]:
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
        messages_to_pass = self.create_messages_to_pass(ModelRoleType.mentor, state, ("input_text", "context"))
        response = self.call_model(messages_to_pass, ModelRoleType.mentor)
        if response:
            response.additional_kwargs["agent"] = f"[{model_name}]"
            return {"messages": [response]}

        logger.warning(f"[{model_name}] No response generated, returning empty list")
        return {"messages": []}

    def visualizer_node(self, state: State) -> dict[str, Optional[dict]]:
        """Visualizer agent: creates interactive graph visualizations from retrieved data.
        
        Processes graph structure data from get_sub_graphs_to_visualize tool and generates an interactive
        Plotly visualization.
        
        Args:
            state: Current workflow state containing graph data in context.
            
        Returns:
            Dictionary with visual_artifact (Plotly figure dict) or None if failed.
        """
        try:
            tool_results = state.context
            if tool_results:
                visualizer_tool_results = json.loads(tool_results).get("get_sub_graphs_to_visualize")
                if visualizer_tool_results:
                    plotly_figure = self.knowledge_graph_indexer.get_graph_visualization(*visualizer_tool_results)
                    return {"visual_artifact": plotly_figure.to_dict()}
            logger.error("[VISUALIZER] No graph data found in context")
        except Exception as e:
            logger.error(f"[VISUALIZER] Failed to create visualization: {str(e)}")

        return {"visual_artifact": None}

    def run(self) -> Any:
        """Compiles and returns the executable workflow graph.
        
        Returns:
            Compiled LangGraph workflow ready for invocation.
        """
        return self.workflow.compile()

    def show_graph(self, jupyter_notebook: bool = False) -> Optional[Any]:
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
            return display(Image(graph.get_graph().draw_mermaid_png()))

        img = Image.open(BytesIO(png_graph))
        img.show()


if __name__ == '__main__':
    # ...
    """
    from backend.configs.storage import StorageSettings, LocalStorageSettings
    from llama_index.core.schema import MetadataMode
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

    try:
        state = graph.invoke({"messages": [
            # 'Research the latest developments in LLM fine-tuning'
            'Please visualize my knowledge about attention mechanism'
        ]},
            config={"recursion_limit": 25}
        )
    except GraphRecursionError as e:
        logger.error(f"Workflow exceeded recursion limit: {e}")

    import plotly.graph_objects as go
    fig = go.Figure(state["visual_artifact"])
    fig.update_layout(
        autosize=True,
        width=None,
        height=None,
    )
    fig.show()
    """
