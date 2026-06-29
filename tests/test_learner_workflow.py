import asyncio
from unittest.mock import patch

from backend.configs.constants import VISUALIZATION_EMPTY_CONTEXT
from backend.configs.models import ModelSettings
from backend.configs.search import KnowledgeGraphSearchSettings
from backend.workflows.agents.schemas import State
from backend.workflows.learner import LearnerWorkflow


class FakeNode:
    def __init__(self, text: str, node_id: str = "node_1", metadata: dict | None = None) -> None:
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
    def __init__(self, settings: KnowledgeGraphSearchSettings, retriever: FakeRetriever) -> None:
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

    def visualize(self, queries: list[str]) -> tuple[list[str], list[tuple[str, str, str]], list[str]]:
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


def test_kb_results_empty_accepts_useful_scored_sources() -> None:
    output = (
        "RETRIEVER RESULTS:\n\n"
        "QUERY: How is BERT related to FastText?\n"
        "[SOURCE] [S1] (Score: 0.82; Chunk: chunk_1) BERT and FastText are text models."
    )

    assert LearnerWorkflow._kb_results_empty({"search_knowledge_base": output}) is False


def test_kb_results_empty_rejects_empty_sentinel() -> None:
    assert LearnerWorkflow._kb_results_empty({"search_knowledge_base": "No relevant information found."}) is True


def test_kb_results_empty_preserves_empty_visualizer_result_for_terminal_handling() -> None:
    assert LearnerWorkflow._kb_results_empty({"get_subgraphs_to_visualize": ([], [], ["BERT"])}) is False


def test_deterministic_route_ends_after_empty_visualization_signal() -> None:
    state = State(context=VISUALIZATION_EMPTY_CONTEXT)

    assert LearnerWorkflow._deterministic_route(state) == "__end__"


def test_visualizer_node_returns_terminal_empty_signal_for_empty_graph_result() -> None:
    workflow = object.__new__(LearnerWorkflow)
    state = State(context='{"get_subgraphs_to_visualize": [[], [], ["BERT"]]}')

    result = asyncio.run(workflow.visualizer_node(state))

    assert result["visual_artifacts"] == []
    assert result["context"] == VISUALIZATION_EMPTY_CONTEXT


def test_kb_search_tool_uses_optimized_analyst_pipeline_by_default() -> None:
    settings = KnowledgeGraphSearchSettings(analyst_retrieval_mode="optimized")
    indexer = FakeIndexer(settings, FakeRetriever({}))
    workflow = make_workflow(indexer)

    with patch("backend.workflows.learner.AnalystRetrievalPipeline", FakeAnalystPipeline):
        tool = workflow._build_kb_search_tool()

    assert tool.invoke({"queries": ["BERT"]}) == "optimized analyst: BERT"
    assert indexer.get_retriever_calls == []


def test_kb_search_tool_can_use_legacy_vector_context_retriever() -> None:
    settings = KnowledgeGraphSearchSettings(analyst_retrieval_mode="legacy_vector_context")
    retriever = FakeRetriever({"BERT": [FakeNodeWithScore(FakeNode("legacy source text"), 0.91)]})
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

    with patch("backend.workflows.learner.VisualizerRetrievalPipeline", FakeVisualizerPipeline):
        tool = workflow._build_visualizer_tool()

    assert tool.invoke({"queries": ["BERT"]}) == (["optimized_node"], [("A", "REL", "B")], ["BERT"])
    assert indexer.get_retriever_calls == []


def test_visualizer_tool_can_use_legacy_vector_context_retriever() -> None:
    settings = KnowledgeGraphSearchSettings(visualizer_retrieval_mode="legacy_vector_context")
    retriever = FakeRetriever({"graph": [FakeNodeWithScore(FakeNode("A -> REL -> B"), 0.95)]})
    indexer = FakeIndexer(settings, retriever)
    workflow = make_workflow(indexer)

    tool = workflow._build_visualizer_tool()
    nodes, triplets, queries = tool.invoke({"queries": ["graph"]})

    assert nodes == []
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
        patch("backend.workflows.learner.AgentsFactory.get_llm_by_role", side_effect=lambda model_settings: FakeLLM()),
        patch("backend.workflows.learner.AgentsFactory.add_retry", side_effect=lambda runnable, provider=None: runnable),
        patch("backend.workflows.learner.AnalystRetrievalPipeline", FakeAnalystPipeline),
        patch("backend.workflows.learner.VisualizerRetrievalPipeline", FakeVisualizerPipeline),
    ):
        workflow._init_agents()

    assert isinstance(workflow.researcher_with_tools, FakeLLM)
    assert isinstance(workflow.researcher_structured, FakeLLM)
    assert "search_knowledge_base" in workflow.tools
    assert "get_subgraphs_to_visualize" in workflow.tools


def test_kb_search_tool_passes_model_reranker_settings_to_optimized_analyst_pipeline() -> None:
    settings = KnowledgeGraphSearchSettings(analyst_retrieval_mode="optimized")
    indexer = FakeIndexer(settings, FakeRetriever({}))
    workflow = make_workflow(indexer)
    workflow.models_settings = ModelSettings()

    with patch("backend.workflows.learner.AnalystRetrievalPipeline", FakeAnalystPipeline):
        workflow._build_kb_search_tool()

    assert LAST_FAKE_ANALYST_PIPELINE is not None
    assert LAST_FAKE_ANALYST_PIPELINE.reranker_settings is workflow.models_settings.reranker
