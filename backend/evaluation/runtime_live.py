"""Controlled headless invocation of the production learner graph for evaluation."""

from __future__ import annotations

import time
import uuid
from typing import Any

from backend.configs.constants import RECURSION_LIMIT
from backend.configs.search import KnowledgeGraphSearchSettings
from backend.evaluation.runtime_evaluation import (
    FailureObservation,
    GraphObservation,
    ProviderObservation,
    RuntimeRecord,
    RuntimeScenario,
    RuntimeUsage,
    ToolObservation,
    _action_signatures,
    _bounded_evidence_from_tool_output,
    _graph_labels_from_artifacts,
    _merge_bounded_evidence,
    _redact_mapping,
)
from backend.utils.chat_limits import (
    ChatTurnContext,
    reset_current_chat_turn,
    set_current_chat_turn,
)
from backend.workflows.learner_reflex import get_workflow


async def run_controlled_live_scenario(
    scenario: RuntimeScenario,
    *,
    repetition: int = 1,
    recursion_limit: int = RECURSION_LIMIT,
) -> RuntimeRecord:
    """Invokes the same compiled graph used by Reflex with explicit turn context.

    The caller is responsible for the command-level live-call confirmation and
    Tavily guard. This function never loads, rebuilds, or mutates retrieval data.
    """
    wrapper = get_workflow()
    wrapper._ensure_initialized()
    graph = wrapper._graph
    if graph is None:
        raise RuntimeError("learner workflow did not initialize")

    run_id = str(uuid.uuid4())
    checkpoint_ns = (
        f"runtime-eval:{scenario.id}:v{scenario.version}:r{repetition}:{run_id}"
    )
    thread_id = str(uuid.uuid4())
    messages = wrapper._convert_messages(scenario.input.message_history)
    from langchain_core.messages import HumanMessage

    messages.append(HumanMessage(content=scenario.input.user_message))
    initial_state = {
        "messages": messages,
        "session_summary": scenario.input.session_summary,
        "request_scope": "unclassified",
        "retrieval_status": "not_run",
        "retriever_empty": False,
        "sources_exhausted": False,
    }
    config = {
        "recursion_limit": recursion_limit,
        "configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns},
        "metadata": {
            "evaluation_suite": "runtime_graph_rag",
            "evaluation_case_id": scenario.id,
            "evaluation_case_version": scenario.version,
            "evaluation_repetition": repetition,
            "evaluation_run_id": run_id,
        },
        "tags": ["runtime-evaluation", scenario.id],
    }
    turn_context = ChatTurnContext(
        visitor_id=f"runtime-eval-{run_id}",
        session_id=checkpoint_ns,
        turn_id=run_id,
    )

    agent_sequence: list[str] = []
    route_decisions: list[str] = []
    retrieval_statuses: list[str] = []
    tools_by_run_id: dict[str, ToolObservation] = {}
    raw_graph_output: Any = None
    providers: list[ProviderObservation] = []
    failures: list[FailureObservation] = []
    final_state: dict[str, Any] = {}
    trace_status = "success"
    started = time.monotonic()
    turn_token = set_current_chat_turn(turn_context)
    try:
        async for event in graph.astream_events(
            initial_state, config=config, version="v2"
        ):
            event_type = str(event.get("event") or "")
            name = str(event.get("name") or "")
            event_run_id = str(event.get("run_id") or "")
            metadata = (
                event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
            )
            node = str(metadata.get("langgraph_node") or "")
            data = event.get("data") if isinstance(event.get("data"), dict) else {}

            if (
                event_type == "on_chain_start"
                and name == node
                and node
                in {
                    "orchestrator",
                    "retriever",
                    "researcher",
                    "analyst",
                    "mentor",
                    "visualizer",
                }
            ):
                if not agent_sequence or agent_sequence[-1] != node:
                    agent_sequence.append(node)
            elif event_type == "on_chain_end" and name == node:
                output = data.get("output")
                if isinstance(output, dict):
                    if node == "orchestrator" and output.get("next_step"):
                        route_decisions.append(str(output["next_step"]))
                    if node == "retriever" and output.get("retrieval_status"):
                        retrieval_statuses.append(str(output["retrieval_status"]))
            elif event_type == "on_tool_start":
                arguments = (
                    _redact_mapping(data["input"])
                    if isinstance(data.get("input"), dict)
                    else {}
                )
                tools_by_run_id[event_run_id] = ToolObservation(
                    run_id=event_run_id,
                    name=name,
                    node=node,
                    arguments=arguments,
                    status="running",
                )
            elif event_type in {"on_tool_end", "on_tool_error"}:
                observation = tools_by_run_id.get(event_run_id)
                if observation is None:
                    observation = ToolObservation(
                        run_id=event_run_id,
                        name=name,
                        node=node,
                    )
                    tools_by_run_id[event_run_id] = observation
                observation.status = (
                    "success" if event_type == "on_tool_end" else "error"
                )
                output = data.get("output")
                observation.evidence_ids = sorted(_find_evidence_ids(output))
                observation.bounded_evidence = _bounded_evidence_from_tool_output(
                    output,
                    tool_name=name,
                    arguments=observation.arguments,
                )
                if name == "get_subgraphs_to_visualize" and event_type == "on_tool_end":
                    raw_graph_output = output
                if event_type == "on_tool_error":
                    failures.append(
                        FailureObservation(
                            code="tool_error", node=node, run_id=event_run_id
                        )
                    )
            elif event_type == "on_chat_model_end":
                output = data.get("output")
                response_metadata = getattr(output, "response_metadata", {}) or {}
                usage_metadata = getattr(output, "usage_metadata", {}) or {}
                content = getattr(output, "content", "")
                tool_calls = getattr(output, "tool_calls", []) or []
                provider = ProviderObservation(
                    run_id=event_run_id,
                    node=node,
                    provider=str(response_metadata.get("model_provider") or ""),
                    model=str(
                        response_metadata.get("model_name")
                        or response_metadata.get("model")
                        or ""
                    ),
                    status="success",
                    stop_reason=str(response_metadata.get("done_reason") or ""),
                    input_tokens=_int_or_zero(usage_metadata.get("input_tokens")),
                    output_tokens=_int_or_zero(usage_metadata.get("output_tokens")),
                    content_present=bool(str(content or "").strip()),
                    tool_call_count=len(tool_calls),
                    configured_output_token_limit=_int_or_none(
                        metadata.get("ls_max_tokens")
                        or metadata.get("max_tokens")
                        or metadata.get("num_predict")
                    ),
                )
                providers.append(provider)
                if provider.stop_reason == "length":
                    failures.append(
                        FailureObservation(
                            code=(
                                "model_output_truncated"
                                if not provider.content_present
                                and not provider.tool_call_count
                                else "model_output_length_limit"
                            ),
                            node=node,
                            run_id=event_run_id,
                        )
                    )
            elif event_type == "on_chain_end" and name == "LangGraph":
                output = data.get("output")
                if isinstance(output, dict):
                    final_state = output
    except Exception as exc:
        trace_status = "error"
        code = _live_error_code(exc)
        failures.append(FailureObservation(code=code, node="workflow"))
    finally:
        reset_current_chat_turn(turn_token)

    tools = list(tools_by_run_id.values())
    graph_observation = _live_graph_observation(
        raw_graph_output, final_state.get("visual_artifacts")
    )
    final_response = _live_final_response(final_state, len(messages))
    if (
        not final_response
        and trace_status == "success"
        and not graph_observation.artifact_count
    ):
        final_response = wrapper._get_fallback_message(final_state)
    if graph_observation.artifact_count:
        output_modality = "graph"
    elif final_response:
        output_modality = "text"
    elif trace_status == "error":
        output_modality = "error"
    else:
        output_modality = "none"

    retrieval_outcome = _live_retrieval_outcome(retrieval_statuses, providers, tools)
    if retrieval_outcome == "invalid_model_output":
        failures.append(
            FailureObservation(code="invalid_retriever_output", node="retriever")
        )
    elif retrieval_outcome == "no_tool_call":
        failures.append(
            FailureObservation(code="retriever_no_tool_call", node="retriever")
        )
    sources_exhausted = bool(final_state.get("sources_exhausted", False))
    if sources_exhausted:
        failures.append(FailureObservation(code="sources_exhausted"))

    settings = KnowledgeGraphSearchSettings()
    provider_input_tokens = sum(provider.input_tokens for provider in providers)
    provider_output_tokens = sum(provider.output_tokens for provider in providers)
    input_tokens = turn_context.input_tokens or provider_input_tokens
    output_tokens = turn_context.output_tokens or provider_output_tokens
    evidence_ids = sorted(
        {evidence_id for tool in tools for evidence_id in tool.evidence_ids}
        | set(graph_observation.node_ids)
    )
    bounded_evidence = _merge_bounded_evidence(
        item for tool in tools for item in tool.bounded_evidence
    )
    evidence_tools_observed = any(
        tool.name
        in {"search_knowledge_base", "deep_web_research", "get_subgraphs_to_visualize"}
        for tool in tools
    )
    worker_sequence = [name for name in agent_sequence if name != "orchestrator"]
    return RuntimeRecord(
        scenario_id=scenario.id,
        scenario_version=scenario.version,
        run_id=run_id,
        repetition=repetition,
        source_mode="controlled_live",
        trace_status=trace_status,
        input=scenario.input,
        agent_sequence=agent_sequence,
        worker_sequence=worker_sequence,
        route_decisions=route_decisions,
        tools=tools,
        providers=providers,
        retrieval_statuses=retrieval_statuses,
        retrieval_outcome=retrieval_outcome,
        request_scope=str(final_state.get("request_scope") or "unclassified"),
        sources_exhausted=sources_exhausted,
        final_response=final_response,
        output_modality=output_modality,
        graph=graph_observation,
        evidence_ids=evidence_ids,
        bounded_evidence=bounded_evidence,
        evidence_capture_status=(
            "captured"
            if bounded_evidence
            else ("unavailable" if evidence_tools_observed else "not_applicable")
        ),
        action_signatures=_action_signatures(worker_sequence, tools),
        failures=failures,
        terminated=(
            trace_status == "success"
            and (
                final_state.get("next_step") == "__end__"
                or route_decisions[-1:] == ["__end__"]
            )
        ),
        usage=RuntimeUsage(
            latency_seconds=time.monotonic() - started,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=turn_context.total_tokens or input_tokens + output_tokens,
            logical_llm_calls=turn_context.logical_llm_calls,
            provider_attempts=turn_context.llm_provider_attempts,
            retries=turn_context.retries_used,
            tavily_searches=turn_context.tavily_attempts,
        ),
        provenance={
            "source_configuration_status": "verified",
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "evaluation_run_id": run_id,
            "analyst_retrieval_mode": settings.analyst_retrieval_mode,
            "visualizer_retrieval_mode": settings.visualizer_retrieval_mode,
        },
    )


def _live_graph_observation(raw_output: Any, artifacts: Any) -> GraphObservation:
    artifact_count = len(artifacts) if isinstance(artifacts, list) else 0
    artifact_labels, artifact_edges = _graph_labels_from_artifacts(artifacts)
    if not isinstance(raw_output, (list, tuple)) or len(raw_output) < 2:
        return GraphObservation(
            artifact_count=artifact_count,
            node_labels=artifact_labels,
            labeled_edges=artifact_edges,
            capture_status=(
                "captured"
                if artifact_labels and artifact_edges
                else ("unavailable" if artifact_count else "not_applicable")
            ),
        )
    raw_nodes, raw_edges = raw_output[0], raw_output[1]
    node_ids: list[str] = []
    node_labels: dict[str, str] = {}
    if isinstance(raw_nodes, list):
        for node in raw_nodes[:25]:
            if isinstance(node, dict):
                node_id = str(node.get("id") or node.get("node_id") or "")
                label = str(node.get("label") or node.get("name") or "")
            else:
                node_id, label = str(node), ""
            if node_id:
                node_ids.append(node_id)
                if label:
                    node_labels[node_id] = label[:200]
    edges: list[tuple[str, str, str]] = []
    if isinstance(raw_edges, list):
        for edge in raw_edges[:35]:
            if isinstance(edge, dict):
                subject = str(edge.get("source") or edge.get("subject") or "")
                predicate = str(edge.get("predicate") or edge.get("label") or "")
                object_ = str(edge.get("target") or edge.get("object") or "")
            elif isinstance(edge, (list, tuple)) and len(edge) == 3:
                subject, predicate, object_ = map(str, edge)
            else:
                continue
            if subject and predicate and object_:
                edges.append((subject, predicate, object_))
    if len(raw_output) > 3 and isinstance(raw_output[3], list):
        for detail in raw_output[3][:25]:
            if isinstance(detail, dict):
                node_id = str(detail.get("id") or "")
                label = str(detail.get("label") or detail.get("name") or "")
                if node_id and label:
                    node_labels[node_id] = label[:200]
    node_ids = list(dict.fromkeys(node_ids))[:25]
    node_set = set(node_ids)
    dangling = sum(
        left not in node_set or right not in node_set for left, _, right in edges
    )
    labeled_edges = [
        (node_labels[left], predicate, node_labels[right])
        for left, predicate, right in edges
        if left in node_labels and right in node_labels
    ][:35]
    if not node_labels:
        node_labels = artifact_labels
    if not labeled_edges:
        labeled_edges = artifact_edges

    capture_status = (
        "captured"
        if node_labels and labeled_edges
        else (
            "unavailable" if node_ids or edges or artifact_count else "not_applicable"
        )
    )
    return GraphObservation(
        artifact_count=artifact_count,
        node_ids=node_ids,
        edges=edges,
        dangling_edge_count=dangling,
        node_labels=node_labels,
        labeled_edges=labeled_edges,
        capture_status=capture_status,
    )


def _live_final_response(final_state: dict[str, Any], input_message_count: int) -> str:
    messages = final_state.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages[input_message_count:]):
        additional = getattr(message, "additional_kwargs", {}) or {}
        if additional.get("agent") in {"[ANALYST]", "[MENTOR]"}:
            return str(getattr(message, "content", "") or "")
    return ""


def _live_retrieval_outcome(
    statuses: list[str],
    providers: list[ProviderObservation],
    tools: list[ToolObservation],
) -> str:
    if not statuses:
        return "not_run"
    status = statuses[-1]
    if status == "adequate":
        return "adequate"
    if any(
        provider.node == "retriever"
        and provider.stop_reason == "length"
        and not provider.content_present
        and not provider.tool_call_count
        for provider in providers
    ):
        return "invalid_model_output"
    if status == "no_results" and any(
        tool.name in {"search_knowledge_base", "get_subgraphs_to_visualize"}
        for tool in tools
    ):
        return "true_empty"
    return "no_tool_call" if status == "no_results" else status


def _find_evidence_ids(value: Any) -> set[str]:
    from backend.evaluation.runtime_evaluation import _find_evidence_ids as find_ids

    return find_ids(value)


def _live_error_code(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".casefold()
    if "responseinvalid" in text or "outputparser" in text or "validation" in text:
        return "structured_output_parse_error"
    if "timeout" in text:
        return "timeout"
    if "recursion" in text:
        return "recursion_limit"
    if any(
        term in text
        for term in ("redis", "pinecone", "neo4j", "memgraph", "mongodb", "storage")
    ):
        return "storage_error"
    if "tavily" in text or "tool" in text:
        return "tool_error"
    if any(term in text for term in ("ollama", "openai", "anthropic", "provider")):
        return "provider_error"
    return "workflow_error"


def _int_or_none(value: Any) -> int | None:
    try:
        return max(0, int(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
