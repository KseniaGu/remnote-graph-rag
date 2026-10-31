"""Offline WP4.4 retrieval and contextual web-handoff regressions."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from langchain_core.messages import AIMessage, HumanMessage

from backend.configs.enums import ModelRoleType
from backend.configs.models import ModelSettings
from backend.workflows.agents.schemas import State
from backend.workflows.learner import LearnerWorkflow


def test_researcher_receives_current_followup_and_speculative_decoding_history():
    """The real prompt renderer retains the domain even when local context is empty."""
    version = ModelSettings().researcher.prompt_version
    prompt = yaml.safe_load(
        Path(
            f"backend/llm/prompts/learner_workflow/researcher/{version}.yaml"
        ).read_text()
    )
    workflow = LearnerWorkflow.__new__(LearnerWorkflow)
    workflow.prompts = {ModelRoleType.researcher: (prompt["user"], prompt["system"])}
    workflow.call_model = AsyncMock(return_value=None)
    state = State(
        messages=[
            HumanMessage(content="What other speculative decoding methods exist?"),
            AIMessage(
                content="We discussed speculative decoding alternatives including Medusa."
            ),
            HumanMessage(content="Please tell me more about Medusa"),
        ],
        session_summary="The current topic is speculative decoding for LLM inference.",
        context="",
        retriever_empty=True,
        retrieval_status="no_results",
        request_scope="in_scope",
        routing_validated=True,
    )
    asyncio.run(workflow.researcher_node(state))
    sent = workflow.call_model.await_args.args[0]
    rendered = "\n".join(text for _, text in sent)
    assert "Please tell me more about Medusa" in rendered
    assert "speculative decoding" in rendered


# Synthetic agent proposals test a structural gate, not reviewed answerability labels.
# Partial and identity-only material is intentionally classified as adequate by
# the existing identity-presence check; that does not establish a complete answer.
DEVELOPMENT_CASES = [
    (
        "named_substantive",
        "Medusa speculative decoding",
        "named",
        [["Medusa"]],
        "Medusa uses multiple decoding heads to propose tokens.",
        "adequate",
        "substantive proposal",
    ),
    (
        "adjacent_only",
        "Medusa speculative decoding",
        "named",
        [["Medusa"]],
        "Speculative decoding can use a draft model.",
        "topic_mismatch",
        "wrong identity proposal",
    ),
    (
        "retnet_resnet",
        "RetNet",
        "named",
        [["RetNet"]],
        "ResNet uses residual image blocks.",
        "topic_mismatch",
        "synthetic identity distinction, WP4.1 control",
    ),
    (
        "mla_attention",
        "MLA",
        "named",
        [["MLA", "Multi-Head Latent Attention"]],
        "Ordinary attention uses queries and keys.",
        "topic_mismatch",
        "synthetic identity distinction, WP4.1 control",
    ),
    (
        "partial",
        "Medusa speculative decoding",
        "named",
        [["Medusa"]],
        "Medusa uses several heads.",
        "adequate",
        "agent proposed partial, not complete answer",
    ),
    (
        "identity_only",
        "Medusa speculative decoding",
        "named",
        [["Medusa"]],
        "Medusa is mentioned in a list.",
        "adequate",
        "agent proposed identity-only, not sufficient evidence",
    ),
    (
        "path_only",
        "Medusa speculative decoding",
        "named",
        [["Medusa"]],
        "A document contains no substantive method details.",
        "adequate",
        "agent proposed path-only, not sufficient evidence",
    ),
    (
        "broad",
        "optimizers",
        "broad",
        [],
        "Adam uses adaptive moments.",
        "adequate",
        "synthetic broad positive, WP4.1 control",
    ),
    (
        "empty",
        "Medusa speculative decoding",
        "named",
        [["Medusa"]],
        "",
        "no_results",
        "synthetic empty",
    ),
    (
        "below_threshold",
        "Medusa speculative decoding",
        "named",
        [["Medusa"]],
        "Medusa evidence with low similarity.",
        "below_threshold",
        "synthetic low score",
    ),
]


def _synthetic_kb(case, mode):
    from backend.configs.constants import (
        RETRIEVAL_BELOW_THRESHOLD,
        RETRIEVAL_NO_RESULTS,
    )
    from backend.workflows.agents.tools import search_knowledge_base

    name, query, _, _, evidence, _, _ = case
    if mode == "optimized":

        class Pipeline:
            def search(self, queries):
                if name == "empty":
                    return RETRIEVAL_NO_RESULTS
                if name == "below_threshold":
                    return RETRIEVAL_BELOW_THRESHOLD
                path = "\n[SOURCE PATH] Medusa" if name == "path_only" else ""
                return f"RETRIEVER RESULTS:\n\nQUERY: {queries[0]}\n[SOURCE] (Score: 0.90) {evidence}{path}"

        return search_knowledge_base(analyst_pipeline=Pipeline())

    from test_retrieval_evidence import FakeNode, FakeNodeWithScore, FakeRetriever

    metadata = {"path": ["Medusa"]} if name == "path_only" else {}
    nodes = (
        []
        if name == "empty"
        else [
            FakeNodeWithScore(
                FakeNode(evidence, metadata=metadata),
                0.01 if name == "below_threshold" else 0.9,
            )
        ]
    )
    return search_knowledge_base(FakeRetriever({query: nodes}))


class OptimizedBelowThresholdMisclassified(AssertionError):
    """Known incorrect conversion of a below-threshold sentinel."""


@pytest.mark.parametrize(
    ("case", "mode"),
    [
        pytest.param(
            case,
            mode,
            id=f"{mode}-{case[0]}",
            marks=(
                pytest.mark.xfail(
                    strict=True,
                    raises=OptimizedBelowThresholdMisclassified,
                    reason="Optimized below-threshold sentinel is converted to topic mismatch",
                )
                if case[0] == "below_threshold" and mode == "optimized"
                else ()
            ),
        )
        for case in DEVELOPMENT_CASES
        for mode in ("optimized", "legacy")
    ],
)
def test_development_structural_adequacy_table(case, mode):
    name, query, policy, topics, _, expected, _ = case
    tool = _synthetic_kb(case, mode)
    result = tool.invoke(
        {"queries": [query], "topic_policy": policy, "required_topics": topics}
    )
    actual = LearnerWorkflow._assess_retrieval_results(
        {"search_knowledge_base": result}
    )
    if name == "below_threshold" and mode == "optimized" and actual == "topic_mismatch":
        raise OptimizedBelowThresholdMisclassified((actual, result))
    assert actual == expected, (name, mode, result)


def _compiled_medusa_graph(case, *, invalid_args=False, research_status="success"):
    """Real nodes and tool boundary; all model and web responses are scripted."""
    from langchain_core.tools import tool
    from langgraph.graph import StateGraph

    from backend.configs.models import ModelSettings
    from backend.workflows.agents.schemas import ResearchResult, RoutingDecision

    calls = []
    sent_researcher = []
    web_topics = []
    kb_tool = _synthetic_kb(case, "legacy")

    @tool("deep_web_research")
    async def fake_web(topic: str) -> str:
        """Record a synthetic web topic without network access."""
        web_topics.append(topic)
        return "Synthetic relevant web evidence."

    workflow = LearnerWorkflow.__new__(LearnerWorkflow)
    workflow.workflow = StateGraph(State)
    workflow.tools = {"search_knowledge_base": kb_tool, "deep_web_research": fake_web}
    settings = ModelSettings()
    workflow.prompts = {}
    for role, version in (
        (ModelRoleType.orchestrator, settings.orchestrator.prompt_version["routing"]),
        (ModelRoleType.retriever, settings.retriever.prompt_version),
        (ModelRoleType.researcher, settings.researcher.prompt_version),
        (ModelRoleType.analyst, settings.analyst.prompt_version),
    ):
        folder = (
            "orchestrator/routing" if role == ModelRoleType.orchestrator else role.name
        )
        prompt = yaml.safe_load(
            Path(
                f"backend/llm/prompts/learner_workflow/{folder}/{version}.yaml"
            ).read_text()
        )
        workflow.prompts[role] = (prompt["user"], prompt["system"])

    original_call_tools = workflow.call_tools
    workflow.call_tools = AsyncMock(wraps=original_call_tools)

    async def scripted_model(messages, role, **kwargs):
        calls.append((role.name, kwargs.get("model_type")))
        if role == ModelRoleType.orchestrator:
            return RoutingDecision(
                request_scope="in_scope", next_step="retriever", reasoning="scripted"
            )
        if role == ModelRoleType.retriever:
            args = {"queries": [case[1]], "required_topics": case[3]}
            if not invalid_args:
                args["topic_policy"] = case[2]
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge_base",
                        "args": args,
                        "id": "kb",
                        "type": "tool_call",
                    }
                ],
            )
        if (
            role == ModelRoleType.researcher
            and kwargs.get("model_type") == "_with_tools"
        ):
            sent_researcher.append(messages)
            # This response is prescribed, not evidence a real model follows the prompt.
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "deep_web_research",
                        "args": {"topic": "Medusa speculative decoding"},
                        "id": "web",
                        "type": "tool_call",
                    }
                ],
            )
        if role == ModelRoleType.researcher:
            return ResearchResult(
                key_findings="Synthetic Medusa evidence",
                sources=[],
                confidence_level="low",
                status=research_status,
                gap_analysis=None,
            )
        if role == ModelRoleType.analyst:
            return AIMessage(content="Synthetic local or web response.")
        raise AssertionError(f"unexpected worker {role}")

    workflow.call_model = scripted_model
    workflow._init_nodes()
    graph = workflow.workflow.compile()
    state = {
        "messages": [
            HumanMessage(content="What other speculative decoding methods exist?"),
            AIMessage(
                content="We discussed speculative decoding alternatives including Medusa."
            ),
            HumanMessage(content="Please tell me more about Medusa"),
        ],
        "session_summary": "The current topic is speculative decoding for LLM inference.",
    }
    return workflow, graph, state, calls, sent_researcher, web_topics


@pytest.mark.parametrize("case_name", ["adjacent_only", "empty", "below_threshold"])
def test_scripted_medusa_gaps_preserve_researcher_context_and_web_topic(case_name):
    from unittest.mock import patch

    case = next(item for item in DEVELOPMENT_CASES if item[0] == case_name)
    workflow, graph, state, calls, researcher_messages, web_topics = (
        _compiled_medusa_graph(case)
    )
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        result = asyncio.run(graph.ainvoke(state))
    assert result["retrieval_status"] == case[5]
    assert result["request_scope"] == "in_scope"
    rendered = "\n".join(text for _, text in researcher_messages[0])
    assert "Please tell me more about Medusa" in rendered
    assert "speculative decoding" in rendered
    assert web_topics == ["Medusa speculative decoding"]
    assert [name for name, _ in calls].count("retriever") == 1
    assert [
        call.args[0].tool_calls[0]["name"]
        for call in workflow.call_tools.await_args_list
    ] == ["search_knowledge_base", "deep_web_research"]


def test_scripted_medusa_adequate_local_stays_local():
    from unittest.mock import patch

    case = DEVELOPMENT_CASES[0]
    workflow, graph, state, calls, researcher_messages, web_topics = (
        _compiled_medusa_graph(case)
    )
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        result = asyncio.run(graph.ainvoke(state))
    assert result["retrieval_status"] == "adequate"
    assert not researcher_messages and not web_topics
    assert [name for name, _ in calls] == ["orchestrator", "retriever", "analyst"]
    assert workflow.call_tools.await_count == 1


def test_original_missing_topic_policy_stops_before_tool_or_web():
    from unittest.mock import patch

    from backend.utils.chat_errors import AIResponseInvalid

    case = DEVELOPMENT_CASES[0]
    workflow, graph, state, _, researcher_messages, web_topics = _compiled_medusa_graph(
        case, invalid_args=True
    )
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        with pytest.raises(AIResponseInvalid):
            asyncio.run(graph.ainvoke(state))
    workflow.call_tools.assert_not_awaited()
    assert not researcher_messages and not web_topics


def test_scripted_no_relevant_web_results_exhaust_sources_without_analyst():
    from unittest.mock import patch

    case = next(item for item in DEVELOPMENT_CASES if item[0] == "empty")
    _, graph, state, calls, _, web_topics = _compiled_medusa_graph(
        case, research_status="no_relevant_info"
    )
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        result = asyncio.run(graph.ainvoke(state))
    assert result["sources_exhausted"] is True
    assert "analyst" not in [name for name, _ in calls]
    assert web_topics == ["Medusa speculative decoding"]


def test_researcher_prompt_marks_latest_request_as_authoritative():
    version = ModelSettings().researcher.prompt_version
    prompt = yaml.safe_load(
        Path(
            f"backend/llm/prompts/learner_workflow/researcher/{version}.yaml"
        ).read_text()
    )
    assert (
        "latest user request overrides stale context"
        in prompt["system"]["system_instruction"]
    )
    assert "{conversation_history}" in prompt["user"]
    assert "{input_text}" in prompt["user"]
