import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph

from backend.configs.enums import WorkflowEventType
from backend.configs.models import ModelRoleType
from backend.workflows.agents.analyst_retrieval import (
    AnalystRetrievalPipeline,
    SourceCandidate,
)
from backend.workflows.agents.retrieval_access import RetrievalStoreAccess
from backend.workflows.agents.retrieval_evidence import NormalizedMetadata
from backend.workflows.agents.schemas import ResearchResult, RoutingDecision, State
from backend.workflows.agents.tools import search_knowledge_base
from backend.workflows.learner import LearnerWorkflow
from backend.workflows.learner_reflex import ReflexLearnerWorkflow
from backend.workflows.presentation import ResponseEvidence


def _hit(chunk_id, text, score=0.9):
    return SimpleNamespace(
        score=score,
        node=SimpleNamespace(id_=chunk_id, text=text, metadata={"note_id": "note-1"}),
    )


def test_source_artifact_tracks_only_admitted_text_and_deduplicates():
    hits = [
        _hit("chunk-filtered", "Unrelated", 0.01),
        _hit("chunk-a", "Attention " * 18, score=0.84),
        _hit("chunk-dup", "Attention " * 18),
        _hit("chunk-b", "Transformer " * 30, score=None),
        _hit("chunk-omitted", "Later evidence"),
    ]
    hits[1].node.metadata["postprocess_chunk_summary"] = (
        "A stored summary of attention."
    )
    tool = search_knowledge_base(
        SimpleNamespace(retrieve=lambda _: hits), max_context_chars=350
    )
    args = {
        "queries": ["Attention"],
        "topic_policy": "named",
        "required_topics": [["Attention"]],
    }
    result = tool.invoke(
        {"type": "tool_call", "id": "kb", "name": tool.name, "args": args}
    )

    assert result.content == tool.invoke(args)
    sources = result.artifact["sources"]
    assert [item["chunk_id"] for item in sources] == ["chunk-a", "chunk-b"]
    assert [item["retrieval_score"] for item in sources] == [0.84, None]
    assert sources[0].get("summary") == "A stored summary of attention."
    assert sources[1].get("summary") is None
    assert "A stored summary of attention." not in result.content
    assert all(item["note_id"] == "note-1" for item in sources)
    assert sources[1]["excerpt"] in result.content
    assert len(sources[1]["excerpt"]) < len(hits[3].node.text)


def test_analyst_keeps_sources_when_final_context_is_cleared():
    workflow = object.__new__(LearnerWorkflow)
    tool = search_knowledge_base(
        SimpleNamespace(retrieve=lambda _: [_hit("chunk-a", "Attention evidence")])
    )
    workflow.tools = {tool.name: tool}
    workflow.create_messages_to_pass = lambda *args: []

    async def call_model(messages, role, **kwargs):
        if role == ModelRoleType.retriever:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool.name,
                        "id": "kb",
                        "args": {
                            "queries": ["Attention"],
                            "topic_policy": "named",
                            "required_topics": [["Attention"]],
                        },
                    }
                ],
            )
        return AIMessage(content="An answer")

    workflow.call_model = call_model

    async def run():
        state = State(messages=[HumanMessage(content="Explain attention")])
        update = await workflow.retriever_node(state)
        state = state.model_copy(update=update)
        result = await workflow.analyst_node(state)
        assert result["context"] == ""
        assert (
            result["messages"][0].additional_kwargs["response_evidence"]["sources"][0][
                "chunk_id"
            ]
            == "chunk-a"
        )

    asyncio.run(run())


def test_optimized_sources_follow_internal_and_outer_budgets():
    pipeline = object.__new__(AnalystRetrievalPipeline)
    pipeline.settings = SimpleNamespace(analyst_context_max_chars=260)
    pipeline.access = RetrievalStoreAccess
    candidates = [
        SourceCandidate(
            node_id=chunk,
            node=None,
            metadata=NormalizedMetadata(),
            text=text,
            base_score=0.9,
            score=0.9,
            rank=index,
            mentioned_concepts=[SimpleNamespace(id="concept_attention")],
        )
        for index, (chunk, text) in enumerate(
            [
                ("chunk_short", "Attention evidence."),
                ("chunk_long", "Transformer " * 100),
                ("chunk_late", "Unsent"),
            ],
            start=1,
        )
    ]
    pipeline._search_one = lambda query: {
        "query": query,
        "sources": candidates,
        "relations": [],
    }
    search = search_knowledge_base(analyst_pipeline=pipeline, max_context_chars=220)
    args = {
        "queries": ["Attention"],
        "topic_policy": "named",
        "required_topics": [["Attention"]],
    }
    result = search.invoke(
        {"type": "tool_call", "id": "kb", "name": search.name, "args": args}
    )
    assert result.content == search.invoke(args)
    assert [source["chunk_id"] for source in result.artifact["sources"]] == [
        "chunk_short",
        "chunk_long",
    ]
    assert [source["retrieval_score"] for source in result.artifact["sources"]] == [
        0.9,
        0.9,
    ]
    assert result.artifact["concept_ids"] == ["concept_attention"]
    assert result.artifact["sources"][1]["excerpt"] in result.content
    assert "Unsent" not in result.content


def test_optimized_source_header_cut_does_not_admit_generated_marker():
    pipeline = object.__new__(AnalystRetrievalPipeline)
    pipeline.access = RetrievalStoreAccess
    source = SourceCandidate(
        node_id="chunk_source",
        node=None,
        metadata=NormalizedMetadata(source="Long display title " * 8),
        text="Actual evidence. " * 40,
        base_score=0.9,
        score=0.9,
        rank=1,
        mentioned_concepts=[SimpleNamespace(id="concept_a")],
    )
    prefix = f"[SOURCE] [S1] (Score: 0.90; Chunk: {source.node_id}; Source: {source.metadata.source}) "
    pipeline.settings = SimpleNamespace(
        analyst_context_max_chars=len("RETRIEVER RESULTS:\n\nQUERY: Topic\n" + prefix)
        + 2
    )
    pipeline._search_one = lambda query: {
        "query": query,
        "sources": [source],
        "relations": [],
    }
    search = search_knowledge_base(analyst_pipeline=pipeline)
    output = search.invoke(
        {
            "type": "tool_call",
            "id": "kb",
            "name": search.name,
            "args": {"queries": ["Topic"], "topic_policy": "broad"},
        }
    )
    assert output.content.endswith("...[truncated]")
    assert "Actual evidence" not in output.content
    assert ResponseEvidence.model_validate(output.artifact) == ResponseEvidence()


def test_malformed_note_metadata_does_not_become_provenance():
    hit = _hit("chunk-a", "Actual evidence")
    hit.node.metadata = {
        "chunk_id": {"corrupt": 1},
        "note_id": {"corrupt": 1},
        "title": ["corrupt"],
    }
    search = search_knowledge_base(SimpleNamespace(retrieve=lambda _: [hit]))
    result = search.invoke(
        {
            "type": "tool_call",
            "id": "kb",
            "name": search.name,
            "args": {"queries": ["Topic"], "topic_policy": "broad"},
        }
    )
    [source] = result.artifact["sources"]
    assert source["chunk_id"] == "chunk-a"
    assert source["source_id"] == "note:chunk-a"
    assert source["note_id"] is None and source["title"] is None


def test_concurrent_workflow_responses_keep_local_web_and_empty_evidence_separate():
    calls = []
    hits = {
        "Attention": [_hit("shell-a", "Attention evidence")],
        "LocalB": [_hit("shell-b", "LocalB evidence")],
        "Mismatch": [_hit("shell-rejected", "Unrelated evidence")],
    }

    def retrieve(query):
        calls.append(query)
        return hits.get(query, [])

    @tool("deep_web_research")
    async def web(topic: str) -> str:
        """Provide a fixed research result for the exercise."""
        calls.append(f"web:{topic}")
        return "Web context"

    workflow = object.__new__(LearnerWorkflow)
    workflow.workflow = StateGraph(State)
    search = search_knowledge_base(SimpleNamespace(retrieve=retrieve))
    workflow.tools = {search.name: search, web.name: web}
    workflow.create_messages_to_pass = lambda role, state, fields: (
        state.messages[0].content,
        state.context,
    )

    async def call_model(messages, role, **kwargs):
        query, context = messages[:2]
        if role == ModelRoleType.orchestrator:
            return RoutingDecision(
                request_scope="in_scope",
                next_step="mentor" if query == "Quiz" else "retriever",
                reasoning="fixture",
            )
        if role == ModelRoleType.retriever:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": search.name,
                        "id": "kb",
                        "args": {
                            "queries": [query],
                            "topic_policy": "named",
                            "required_topics": [[query]],
                        },
                    }
                ],
            )
        if role == ModelRoleType.researcher:
            if kwargs["model_type"] == "_with_tools":
                assert context == ""
                return AIMessage(
                    content="",
                    tool_calls=[
                        {"name": web.name, "id": "web", "args": {"topic": query}}
                    ],
                )
            return ResearchResult(
                key_findings="Grounded web context",
                sources=[
                    {"title": "Reference", "url": "https://example.org/reference"},
                    {"title": "Duplicate URL", "url": "https://example.org/reference"},
                    {"url": "javascript:alert(1)"},
                    {"url": "https://example.org:99999/path"},
                ],
                status="success",
                confidence_level="high",
            )
        if query in {"Empty", "Mismatch"}:
            assert "Web Research Findings" in context and "Unrelated" not in context
        return AIMessage(content=f"Response to {query}")

    workflow.call_model = call_model
    workflow._init_nodes()
    wrapper = ReflexLearnerWorkflow()
    wrapper._graph = workflow.workflow.compile()

    async def run():
        async def collect(query):
            events = [event async for event in wrapper.stream_with_tokens(query, [])]
            assert all(
                event.type
                not in {WorkflowEventType.ERROR, WorkflowEventType.CONTEXT_UPDATE}
                for event in events
            )
            return next(
                event.data
                for event in events
                if event.type == WorkflowEventType.RESPONSE
            )

        return await asyncio.gather(
            *(
                collect(query)
                for query in ("Attention", "LocalB", "Mismatch", "Empty", "Quiz")
            )
        )

    first, second, mismatch, empty, mentor = asyncio.run(run())
    assert first["evidence"].sources[0].chunk_id == "shell-a"
    assert second["evidence"].sources[0].chunk_id == "shell-b"
    for fallback in (mismatch, empty):
        [source] = fallback["evidence"].sources
        assert source.url == "https://example.org/reference"
        assert source.chunk_id is None and source.excerpt is None
    assert mentor["agent"] == "mentor" and mentor["content"] == "Response to Quiz"
    assert mentor["invocation_id"] and mentor["response_id"]
    assert "evidence" not in mentor
    assert sorted(calls) == [
        "Attention",
        "Empty",
        "LocalB",
        "Mismatch",
        "web:Empty",
        "web:Mismatch",
    ]
