import asyncio
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph

from backend.configs.constants import (
    RETRIEVAL_BELOW_THRESHOLD,
    RETRIEVAL_TOPIC_MISMATCH,
    VISUALIZATION_EMPTY_CONTEXT,
)
from backend.configs.enums import ModelRoleType
from backend.configs.messages import (
    FALLBACK_ALL_SOURCES_EXHAUSTED,
    FALLBACK_OUT_OF_SCOPE,
    FALLBACK_VISUALIZATION_FAILED,
)
from backend.configs.models import ModelSettings
from backend.configs.search import KnowledgeGraphSearchSettings
from backend.workflows.agents.schemas import ResearchResult, RoutingDecision, State
from backend.workflows.learner import LearnerWorkflow
from backend.workflows.learner_reflex import ReflexLearnerWorkflow


class FakeNode:
    def __init__(
        self, text: str, node_id: str = "node_1", metadata: dict | None = None
    ) -> None:
        self.text = text
        self.node_id = node_id
        self.metadata = metadata or {}


class FakeNodeWithScore:
    def __init__(self, node: FakeNode, score: float | None = 0.9) -> None:
        self.node = node
        self.score = score


class FakeRetriever:
    def __init__(self, results_by_query: dict[str, list]) -> None:
        self.results_by_query = results_by_query

    def retrieve(self, query: str) -> list:
        return list(self.results_by_query.get(query, []))


class FakeIndexer:
    def __init__(
        self, settings: KnowledgeGraphSearchSettings, retriever: FakeRetriever
    ) -> None:
        self.kg_search_settings = settings
        self.retriever = retriever
        self.get_retriever_calls: list[dict] = []

    def get_retriever(self, retriever_params: dict) -> FakeRetriever:
        self.get_retriever_calls.append(retriever_params)
        return self.retriever


LAST_FAKE_ANALYST_PIPELINE = None


class FakeAnalystPipeline:
    def __init__(self, indexer, reranker_settings=None) -> None:
        global LAST_FAKE_ANALYST_PIPELINE
        self.indexer = indexer
        self.reranker_settings = reranker_settings
        LAST_FAKE_ANALYST_PIPELINE = self

    def search(self, queries: list[str]) -> str:
        return f"optimized analyst: {queries[0]}"


class FakeVisualizerPipeline:
    def __init__(self, indexer) -> None:
        self.indexer = indexer

    def visualize(
        self, queries: list[str]
    ) -> tuple[list[str], list[tuple[str, str, str]], list[str]]:
        return ["optimized_node"], [("A", "REL", "B")], queries


class FakePromptEngine:
    def render(self, *args, **kwargs):
        return "prompt", {"system_instruction": "system"}


class FakeLLM:
    def bind_tools(self, tools: list) -> "FakeLLM":
        self.bound_tools = tools
        return self

    def with_structured_output(self, schema) -> "FakeLLM":
        self.structured_schema = schema
        return self


def make_workflow(indexer: FakeIndexer):
    workflow = object.__new__(LearnerWorkflow)
    workflow.knowledge_graph_indexer = indexer
    return workflow


def test_retrieval_assessment_accepts_useful_scored_sources() -> None:
    output = (
        "RETRIEVER RESULTS:\n\n"
        "QUERY: How is BERT related to FastText?\n"
        "[SOURCE] [S1] (Score: 0.82; Chunk: chunk_1) BERT and FastText are text models."
    )

    assert (
        LearnerWorkflow._assess_retrieval_results({"search_knowledge_base": output})
        == "adequate"
    )


def test_retrieval_assessment_distinguishes_empty_sentinel() -> None:
    assert (
        LearnerWorkflow._assess_retrieval_results(
            {"search_knowledge_base": "No relevant information found."}
        )
        == "no_results"
    )


def test_retrieval_assessment_preserves_visualizer_result_for_terminal_handling() -> (
    None
):
    assert (
        LearnerWorkflow._assess_retrieval_results(
            {"get_subgraphs_to_visualize": ([], [], ["BERT"])}
        )
        == "adequate"
    )


def test_retrieval_assessment_distinguishes_inadequate_statuses() -> None:
    assert (
        LearnerWorkflow._assess_retrieval_results(
            {"search_knowledge_base": RETRIEVAL_BELOW_THRESHOLD}
        )
        == "below_threshold"
    )
    assert (
        LearnerWorkflow._assess_retrieval_results(
            {"search_knowledge_base": RETRIEVAL_TOPIC_MISMATCH}
        )
        == "topic_mismatch"
    )


def test_inadequate_retrieval_routes_to_researcher_without_reset() -> None:
    workflow = object.__new__(LearnerWorkflow)
    workflow.call_model = AsyncMock()
    state = State(
        messages=[HumanMessage(content="Tell me about RetNet")],
        retriever_empty=True,
        retrieval_status="topic_mismatch",
        request_scope="in_scope",
    )

    result = asyncio.run(workflow.orchestrator_node(state))

    assert result["next_step"] == "researcher"
    workflow.call_model.assert_not_awaited()


def test_out_of_scope_decision_overrides_model_route() -> None:
    workflow = object.__new__(LearnerWorkflow)
    workflow.create_messages_to_pass = Mock(return_value=[])
    workflow.call_model = AsyncMock(
        return_value=RoutingDecision(
            request_scope="out_of_scope",
            next_step="researcher",
            reasoning="Clearly unrelated request.",
        )
    )
    state = State(messages=[HumanMessage(content="Tell me about cats")])

    result = asyncio.run(workflow.orchestrator_node(state))

    assert result == {"next_step": "__end__", "request_scope": "out_of_scope"}
    workflow.call_model.assert_awaited_once()


def test_researcher_no_relevant_info_exhausts_sources_without_analyst_marker() -> None:
    workflow = object.__new__(LearnerWorkflow)
    workflow.create_messages_to_pass = Mock(return_value=[])
    workflow.call_tools = AsyncMock(
        return_value={"deep_web_research": "Only unrelated search results."}
    )
    workflow.call_model = AsyncMock(
        side_effect=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "deep_web_research",
                        "args": {"topic": "EAGLT"},
                        "id": "research-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ResearchResult(
                key_findings="No relevant technical identity was established.",
                sources=[],
                confidence_level="low",
                status="no_relevant_info",
                gap_analysis="Only unrelated names were returned.",
            ),
        ]
    )
    state = State(
        messages=[HumanMessage(content="Tell me about EAGLT")],
        retriever_empty=True,
        retrieval_status="topic_mismatch",
        request_scope="ambiguous",
    )

    result = asyncio.run(workflow.researcher_node(state))

    assert result["sources_exhausted"] is True
    assert result["context"] == ""
    assert "RESEARCH_COMPLETE" not in result["context"]


def test_fallback_selection_uses_explicit_state_priority() -> None:
    visual_context = {"context": VISUALIZATION_EMPTY_CONTEXT}

    assert (
        ReflexLearnerWorkflow._get_fallback_message(
            {
                "request_scope": "out_of_scope",
                "sources_exhausted": True,
                **visual_context,
            }
        )
        == FALLBACK_OUT_OF_SCOPE
    )
    assert (
        ReflexLearnerWorkflow._get_fallback_message(
            {"sources_exhausted": True, **visual_context}
        )
        == FALLBACK_ALL_SOURCES_EXHAUSTED
    )
    assert (
        ReflexLearnerWorkflow._get_fallback_message(visual_context)
        == FALLBACK_VISUALIZATION_FAILED
    )


def make_mocked_runtime_graph(call_model, call_tools):
    workflow = object.__new__(LearnerWorkflow)
    workflow.workflow = StateGraph(State)
    workflow.create_messages_to_pass = Mock(return_value=[])
    workflow.call_model = call_model
    workflow.call_tools = call_tools
    workflow._init_nodes()
    return workflow.workflow.compile()


def test_out_of_scope_graph_trajectory_uses_only_orchestrator() -> None:
    roles = []

    async def call_model(_messages, role_type, **_kwargs):
        roles.append(role_type)
        return RoutingDecision(
            request_scope="out_of_scope",
            next_step="researcher",
            reasoning="Clearly unrelated request.",
        )

    call_tools = AsyncMock()
    graph = make_mocked_runtime_graph(call_model, call_tools)

    result = asyncio.run(
        graph.ainvoke({"messages": [HumanMessage(content="Tell me about cats")]})
    )

    assert roles == [ModelRoleType.orchestrator]
    assert result["next_step"] == "__end__"
    assert result["request_scope"] == "out_of_scope"
    call_tools.assert_not_awaited()


def test_adequate_local_graph_trajectory_reaches_analyst() -> None:
    roles = []

    async def call_model(_messages, role_type, **_kwargs):
        roles.append(role_type)
        if role_type == ModelRoleType.orchestrator:
            return RoutingDecision(
                request_scope="in_scope",
                next_step="retriever",
                reasoning="Technical lookup.",
            )
        if role_type == ModelRoleType.retriever:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge_base",
                        "args": {
                            "queries": ["RetNet"],
                            "required_topics": [["RetNet"]],
                        },
                        "id": "retrieve-1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="RetNet is supported by the retrieved evidence.")

    call_tools = AsyncMock(
        return_value={
            "search_knowledge_base": (
                "RETRIEVER RESULTS:\n\nQUERY: RetNet\n"
                "[SOURCE] (Score: 0.90) RetNet is a sequence model."
            )
        }
    )
    graph = make_mocked_runtime_graph(call_model, call_tools)

    result = asyncio.run(
        graph.ainvoke({"messages": [HumanMessage(content="Tell me about RetNet")]})
    )

    assert roles == [
        ModelRoleType.orchestrator,
        ModelRoleType.retriever,
        ModelRoleType.analyst,
    ]
    assert result["retrieval_status"] == "adequate"
    assert result["messages"][-1].additional_kwargs["agent"] == "[ANALYST]"


def test_topic_mismatch_graph_trajectory_uses_web_and_five_logical_calls() -> None:
    roles = []

    async def call_model(_messages, role_type, **kwargs):
        roles.append(role_type)
        if role_type == ModelRoleType.orchestrator:
            return RoutingDecision(
                request_scope="in_scope",
                next_step="retriever",
                reasoning="Technical lookup.",
            )
        if role_type == ModelRoleType.retriever:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge_base",
                        "args": {"queries": ["RetNet"]},
                        "id": "retrieve-1",
                        "type": "tool_call",
                    }
                ],
            )
        if (
            role_type == ModelRoleType.researcher
            and kwargs.get("model_type") == "_with_tools"
        ):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "deep_web_research",
                        "args": {"topic": "RetNet"},
                        "id": "research-1",
                        "type": "tool_call",
                    }
                ],
            )
        if role_type == ModelRoleType.researcher:
            return ResearchResult(
                key_findings="RetNet is established by web evidence.",
                sources=[
                    {
                        "title": "RetNet paper",
                        "url": "https://example.test",
                        "type": "paper",
                    }
                ],
                confidence_level="high",
                status="success",
                gap_analysis=None,
            )
        return AIMessage(content="RetNet answer grounded in web evidence.")

    async def call_tools(response):
        name = response.tool_calls[0]["name"]
        if name == "search_knowledge_base":
            return {name: RETRIEVAL_TOPIC_MISMATCH}
        return {name: "Authoritative RetNet web result."}

    graph = make_mocked_runtime_graph(call_model, call_tools)
    result = asyncio.run(
        graph.ainvoke({"messages": [HumanMessage(content="Tell me about RetNet")]})
    )

    assert roles == [
        ModelRoleType.orchestrator,
        ModelRoleType.retriever,
        ModelRoleType.researcher,
        ModelRoleType.researcher,
        ModelRoleType.analyst,
    ]
    assert len(roles) == 5
    assert result["retrieval_status"] == "topic_mismatch"
    assert result["sources_exhausted"] is False


def test_web_no_relevant_info_graph_trajectory_never_invokes_analyst() -> None:
    roles = []

    async def call_model(_messages, role_type, **kwargs):
        roles.append(role_type)
        if role_type == ModelRoleType.orchestrator:
            return RoutingDecision(
                request_scope="ambiguous",
                next_step="retriever",
                reasoning="Unknown technical-looking term.",
            )
        if role_type == ModelRoleType.retriever:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge_base",
                        "args": {"queries": ["EAGLT"]},
                        "id": "retrieve-1",
                        "type": "tool_call",
                    }
                ],
            )
        if (
            role_type == ModelRoleType.researcher
            and kwargs.get("model_type") == "_with_tools"
        ):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "deep_web_research",
                        "args": {"topic": "EAGLT"},
                        "id": "research-1",
                        "type": "tool_call",
                    }
                ],
            )
        return ResearchResult(
            key_findings="No exact identity was established.",
            sources=[],
            confidence_level="low",
            status="no_relevant_info",
            gap_analysis="Only unrelated acronyms were returned.",
        )

    async def call_tools(response):
        name = response.tool_calls[0]["name"]
        if name == "search_knowledge_base":
            return {name: RETRIEVAL_TOPIC_MISMATCH}
        return {name: "Unrelated web results."}

    graph = make_mocked_runtime_graph(call_model, call_tools)
    result = asyncio.run(
        graph.ainvoke({"messages": [HumanMessage(content="Tell me about EAGLT")]})
    )

    assert roles == [
        ModelRoleType.orchestrator,
        ModelRoleType.retriever,
        ModelRoleType.researcher,
        ModelRoleType.researcher,
    ]
    assert ModelRoleType.analyst not in roles
    assert result["sources_exhausted"] is True
    assert result["context"] == ""


def test_deterministic_route_ends_after_empty_visualization_signal() -> None:
    state = State(context=VISUALIZATION_EMPTY_CONTEXT)

    assert LearnerWorkflow._deterministic_route(state) == "__end__"


def test_visualizer_node_returns_terminal_empty_signal_for_empty_graph_result() -> None:
    workflow = object.__new__(LearnerWorkflow)
    state = State(context='{"get_subgraphs_to_visualize": [[], [], ["BERT"]]}')

    result = asyncio.run(workflow.visualizer_node(state))

    assert result["visual_artifacts"] == []
    assert result["context"] == VISUALIZATION_EMPTY_CONTEXT


def test_kb_search_tool_can_use_optimized_analyst_pipeline() -> None:
    settings = KnowledgeGraphSearchSettings(analyst_retrieval_mode="optimized")
    indexer = FakeIndexer(settings, FakeRetriever({}))
    workflow = make_workflow(indexer)

    with patch(
        "backend.workflows.learner.AnalystRetrievalPipeline", FakeAnalystPipeline
    ):
        tool = workflow._build_kb_search_tool()

    assert tool.invoke({"queries": ["BERT"]}) == "optimized analyst: BERT"
    assert indexer.get_retriever_calls == []


def test_kb_search_tool_can_use_legacy_vector_context_retriever() -> None:
    settings = KnowledgeGraphSearchSettings(
        analyst_retrieval_mode="legacy_vector_context"
    )
    retriever = FakeRetriever(
        {"BERT": [FakeNodeWithScore(FakeNode("legacy source text"), 0.91)]}
    )
    indexer = FakeIndexer(settings, retriever)
    workflow = make_workflow(indexer)

    tool = workflow._build_kb_search_tool()
    output = tool.invoke({"queries": ["BERT"]})

    assert "legacy source text" in output
    assert indexer.get_retriever_calls == [settings.retriever_params]


def test_visualizer_tool_uses_optimized_visualizer_pipeline_by_default() -> None:
    settings = KnowledgeGraphSearchSettings(visualizer_retrieval_mode="optimized")
    indexer = FakeIndexer(settings, FakeRetriever({}))
    workflow = make_workflow(indexer)

    with patch(
        "backend.workflows.learner.VisualizerRetrievalPipeline", FakeVisualizerPipeline
    ):
        tool = workflow._build_visualizer_tool()

    assert tool.invoke({"queries": ["BERT"]}) == (
        ["optimized_node"],
        [("A", "REL", "B")],
        ["BERT"],
    )
    assert indexer.get_retriever_calls == []


def test_visualizer_tool_can_use_legacy_vector_context_retriever() -> None:
    settings = KnowledgeGraphSearchSettings(
        visualizer_retrieval_mode="legacy_vector_context"
    )
    retriever = FakeRetriever(
        {"graph": [FakeNodeWithScore(FakeNode("A -> REL -> B"), 0.95)]}
    )
    indexer = FakeIndexer(settings, retriever)
    workflow = make_workflow(indexer)

    tool = workflow._build_visualizer_tool()
    nodes, triplets, queries = tool.invoke({"queries": ["graph"]})

    assert nodes == ["A", "B"]
    assert triplets == [("A", "REL", "B")]
    assert queries == ["graph"]
    assert indexer.get_retriever_calls == [settings.visualizer_retriever_params]


def test_init_agents_handles_researcher_composite_model_settings() -> None:
    settings = KnowledgeGraphSearchSettings(
        analyst_retrieval_mode="optimized",
        visualizer_retrieval_mode="optimized",
    )
    indexer = FakeIndexer(settings, FakeRetriever({}))
    workflow = make_workflow(indexer)
    workflow.models_settings = ModelSettings()
    workflow.prompt_engine = FakePromptEngine()
    workflow.search_engine = object()

    with (
        patch(
            "backend.workflows.learner.AgentsFactory.get_llm_by_role",
            side_effect=lambda model_settings: FakeLLM(),
        ),
        patch(
            "backend.workflows.learner.AgentsFactory.add_retry",
            side_effect=lambda runnable, provider=None: runnable,
        ),
        patch(
            "backend.workflows.learner.AnalystRetrievalPipeline", FakeAnalystPipeline
        ),
        patch(
            "backend.workflows.learner.VisualizerRetrievalPipeline",
            FakeVisualizerPipeline,
        ),
    ):
        workflow._init_agents()

    assert isinstance(workflow.researcher_with_tools, FakeLLM)
    assert isinstance(workflow.researcher_structured, FakeLLM)
    assert "search_knowledge_base" in workflow.tools
    assert "get_subgraphs_to_visualize" in workflow.tools


def test_kb_search_tool_passes_model_reranker_settings_to_optimized_analyst_pipeline() -> (
    None
):
    settings = KnowledgeGraphSearchSettings(analyst_retrieval_mode="optimized")
    indexer = FakeIndexer(settings, FakeRetriever({}))
    workflow = make_workflow(indexer)
    workflow.models_settings = ModelSettings()

    with patch(
        "backend.workflows.learner.AnalystRetrievalPipeline", FakeAnalystPipeline
    ):
        workflow._build_kb_search_tool()

    assert LAST_FAKE_ANALYST_PIPELINE is not None
    assert (
        LAST_FAKE_ANALYST_PIPELINE.reranker_settings
        is workflow.models_settings.reranker
    )
