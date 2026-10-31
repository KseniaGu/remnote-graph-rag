"""Controlled headless invocation of the production learner graph for evaluation."""

from __future__ import annotations

import json
import os
import re
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from backend.configs.constants import RECURSION_LIMIT
from backend.configs.search import KnowledgeGraphSearchSettings
from backend.evaluation.portfolio_budget import (
    EvaluationBudgetExceeded,
    EvaluationBudgetUnavailable,
    current_evaluation_budget,
)
from backend.evaluation.runtime_evaluation import (
    FailureObservation,
    GraphObservation,
    ProviderObservation,
    RuntimeRecord,
    RuntimeScenario,
    ToolObservation,
    _action_signatures,
    _aggregate_event_usage,
    _bounded_evidence_from_tool_output,
    _graph_labels_from_artifacts,
    _merge_bounded_evidence,
    _provider_response_fields,
    _redact_mapping,
)
from backend.utils.chat_limits import (
    ChatTurnContext,
    provider_status_code,
    reset_current_chat_turn,
    set_current_chat_turn,
)
from backend.workflows.learner_reflex import ReflexLearnerWorkflow


def get_workflow() -> ReflexLearnerWorkflow:
    """Creates a session-local wrapper so cached production clients cannot bypass guards."""
    budget = current_evaluation_budget()
    if budget is None:
        raise EvaluationBudgetUnavailable(
            "headless service evaluation requires the guarded runner"
        )
    if not hasattr(budget, "runtime_workflow"):
        budget.runtime_workflow = ReflexLearnerWorkflow()
    return budget.runtime_workflow


def _disable_tavily_transport_retries(wrapper: ReflexLearnerWorkflow) -> None:
    from functools import partial

    from urllib3.util import Retry

    session = wrapper._workflow.search_engine.session
    for adapter in session.adapters.values():
        adapter.max_retries = Retry.from_int(0)
    session.post = partial(session.post, allow_redirects=False)


@contextmanager
def _capture_openrouter_wire_usage(wrapper: ReflexLearnerWorkflow):
    """Observe JSON usage on evaluation clients without reading streaming bodies."""
    hooks = []
    seen = set()
    captured: dict[str, dict[str, Any] | None] = {}

    async def after_response(response: Any) -> None:
        # Streaming must remain streaming; normalized-only metadata stays unknown.
        if "application/json" not in response.headers.get("content-type", "").lower():
            return
        try:
            request_body = json.loads(response.request.content)
            if isinstance(request_body, dict) and request_body.get("stream") is True:
                return
            await response.aread()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("id"), str):
                return
            request_id = payload["id"]
            if not request_id:
                return
            usage = payload.get("usage")
            value = {
                key: usage[key]
                for key in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "input_tokens",
                    "output_tokens",
                )
                if isinstance(usage, dict) and key in usage
            }
            # Duplicate response IDs cannot be matched safely, even with equal data.
            captured[request_id] = value if request_id not in captured else None
        except Exception:
            # Diagnostics must never change request behavior or expose payloads.
            return

    def visit(value: Any) -> None:
        if value is None or id(value) in seen:
            return
        seen.add(id(value))
        sdk = getattr(value, "client", None)
        client = getattr(getattr(sdk, "sdk_configuration", None), "async_client", None)
        if client is not None and hasattr(client, "event_hooks"):
            entries = client.event_hooks["response"]
            if id(entries) not in seen:
                seen.add(id(entries))
                entries.append(after_response)
                hooks.append(entries)
        for attr in ("bound", "steps", "steps__"):
            child = getattr(value, attr, None)
            if isinstance(child, dict):
                for item in child.values():
                    visit(item)
            elif isinstance(child, (list, tuple)):
                for item in child:
                    visit(item)
            elif child is not None:
                visit(child)

    workflow = getattr(wrapper, "_workflow", None)
    for role in (
        "orchestrator",
        "retriever",
        "researcher_with_tools",
        "researcher_structured",
        "analyst",
        "mentor",
        "visualizer",
    ):
        visit(getattr(workflow, role, None))
    try:
        yield captured
    finally:
        for entries in hooks:
            entries.remove(after_response)


def _raise_capacity_stop(exc: BaseException) -> None:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if provider_status_code(current) == 429:
            raise EvaluationBudgetUnavailable(
                "provider HTTP 429 stopped the batch; refresh quota before resuming"
            ) from None
        current = current.__cause__ or current.__context__


def _evaluation_retryable(exc: BaseException) -> bool:
    """Permit only concrete transport errors or explicit 5xx responses."""
    import httpx

    status = provider_status_code(exc)
    if status is not None:
        return 500 <= status <= 599
    return isinstance(
        exc, (TimeoutError, ConnectionError, httpx.TimeoutException, httpx.NetworkError)
    )


@contextmanager
def _evaluation_retry_guard():
    # The production retry token and limits remain in force. Narrow only the
    # classifier used by this isolated evaluation runner, then restore it.
    import backend.workflows.learner as learner

    original = learner.is_transient_provider_error

    def classify(exc: BaseException) -> bool:
        return (
            _evaluation_retryable(exc) if current_evaluation_budget() else original(exc)
        )

    learner.is_transient_provider_error = classify
    try:
        yield
    finally:
        learner.is_transient_provider_error = original


async def run_controlled_live_scenario(
    scenario: RuntimeScenario,
    *,
    repetition: int = 1,
    recursion_limit: int = RECURSION_LIMIT,
    diagnostics_path: Path | None = None,
) -> RuntimeRecord:
    """Invokes the same compiled graph used by Reflex with explicit turn context.

    The caller is responsible for the command-level live-call confirmation and
    Tavily guard. This function never loads, rebuilds, or mutates retrieval data.
    """
    wrapper = get_workflow()
    wrapper._ensure_initialized()
    if current_evaluation_budget() is not None:
        _disable_tavily_transport_retries(wrapper)
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
        "routing_validated": False,
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
    observed_wire_ids: set[str] = set()
    failures: list[FailureObservation] = []
    final_state: dict[str, Any] = {}
    trace_status = "success"
    started = time.monotonic()
    turn_token = set_current_chat_turn(turn_context)
    retry_guard = _evaluation_retry_guard()
    retry_guard.__enter__()
    wire_guard = _capture_openrouter_wire_usage(wrapper)
    wire_usage_by_id = wire_guard.__enter__()
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

            if diagnostics_path is not None:
                _write_live_diagnostic(diagnostics_path, event, turn_context)

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
                    tool_error = data.get("error")
                    if isinstance(
                        tool_error,
                        (EvaluationBudgetExceeded, EvaluationBudgetUnavailable),
                    ):
                        raise tool_error
                    if isinstance(tool_error, BaseException):
                        _raise_capacity_stop(tool_error)
                    failures.append(
                        FailureObservation(
                            code="tool_error", node=node, run_id=event_run_id
                        )
                    )
            elif event_type == "on_chat_model_end":
                output = data.get("output")
                metadata_value = getattr(output, "response_metadata", {})
                response_metadata = (
                    metadata_value if isinstance(metadata_value, dict) else {}
                )
                usage_metadata = getattr(output, "usage_metadata", {}) or {}
                response_id = response_metadata.get("id")
                wire_usage = (
                    wire_usage_by_id.get(response_id)
                    if isinstance(response_id, str)
                    else None
                )
                if isinstance(wire_usage, dict):
                    observed_wire_ids.add(response_id)
                    response_metadata = {**response_metadata, "token_usage": wire_usage}
                configured_adapter = metadata.get("ls_provider")
                observed_adapter = response_metadata.get("model_provider")
                normalized_untrusted = (
                    name == "ChatOpenRouter"
                    or "openrouter" in (configured_adapter, observed_adapter)
                    or not (configured_adapter or name == "ChatOllama")
                )
                content = getattr(output, "content", "")
                tool_calls = getattr(output, "tool_calls", []) or []
                provider = ProviderObservation(
                    # Multiple callbacks for one response share the wire identity.
                    run_id=(
                        f"wire-response:{response_id}"
                        if isinstance(wire_usage, dict)
                        else event_run_id
                    ),
                    node=node,
                    configured_provider=str(metadata.get("ls_provider") or ""),
                    configured_model=str(metadata.get("ls_model_name") or ""),
                    status="success",
                    **_provider_response_fields(
                        response_metadata,
                        usage_metadata,
                        normalized_untrusted=normalized_untrusted,
                        wire_captured=isinstance(wire_usage, dict),
                    ),
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
    except (EvaluationBudgetExceeded, EvaluationBudgetUnavailable) as exc:
        if diagnostics_path is not None:
            _write_live_exception(diagnostics_path, exc, turn_context)
        raise
    except Exception as exc:
        if diagnostics_path is not None:
            _write_live_exception(diagnostics_path, exc, turn_context)
        _raise_capacity_stop(exc)
        if diagnostics_path is not None:
            raise RuntimeError(
                f"controlled live execution stopped: {_diagnostic_text(exc)}"
            ) from exc
        trace_status = "error"
        code = _live_error_code(exc)
        failures.append(FailureObservation(code=code, node="workflow"))
    finally:
        wire_guard.__exit__(None, None, None)
        retry_guard.__exit__(None, None, None)
        reset_current_chat_turn(turn_token)

    # A JSON response may be rejected by SDK validation before a model-end
    # callback exists. Preserve its uniquely identified wire fields separately.
    for response_id, wire_usage in wire_usage_by_id.items():
        if response_id not in observed_wire_ids and isinstance(wire_usage, dict):
            providers.append(
                ProviderObservation(
                    run_id=f"wire-response:{response_id}",
                    status="response_observed",
                    **_provider_response_fields(
                        {"token_usage": wire_usage}, {}, wire_captured=True
                    ),
                )
            )

    tools = list(tools_by_run_id.values())
    graph_observation = _live_graph_observation(
        raw_graph_output, final_state.get("visual_artifacts")
    )

    final_response = _live_final_response(final_state, len(messages))
    response_origin = "graph_message" if final_response else "unavailable"
    if (
        not final_response
        and trace_status == "success"
        and not graph_observation.artifact_count
    ):
        final_response = wrapper._get_fallback_message(final_state)
        response_origin = "application_fallback_reconstruction"
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
    usage = _aggregate_event_usage(
        providers, attempted=turn_context.llm_provider_attempts
    )
    usage.latency_seconds = time.monotonic() - started
    usage.logical_llm_calls = turn_context.logical_llm_calls
    usage.provider_attempts = turn_context.llm_provider_attempts
    usage.retries = turn_context.retries_used
    usage.tavily_searches = turn_context.tavily_attempts
    usage.turn_telemetry = {
        "input_tokens": turn_context.input_tokens,
        "output_tokens": turn_context.output_tokens,
        "total_tokens": turn_context.total_tokens,
        "availability": "unknown",
    }
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
        usage=usage,
        provenance={
            "response_origin": response_origin,
            "source_configuration_status": "verified",
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "evaluation_run_id": run_id,
            "analyst_retrieval_mode": settings.analyst_retrieval_mode,
            "visualizer_retrieval_mode": settings.visualizer_retrieval_mode,
        },
    )


def _diagnostic_text(value: Any) -> str:
    text = str(value)
    for key, secret in os.environ.items():
        if (
            secret
            and len(secret) >= 8
            and any(
                token in key.lower() for token in ("key", "token", "secret", "password")
            )
        ):
            text = text.replace(secret, "[REDACTED]")
    return re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[REDACTED]", text)[:1024]


def _bounded_diagnostic(value: Any, depth: int = 0) -> Any:
    if isinstance(value, dict) and depth < 6:
        return {
            _diagnostic_text(key)[:128]: _bounded_diagnostic(item, depth + 1)
            for key, item in list(_redact_mapping(value).items())[:16]
        }
    if isinstance(value, (list, tuple)) and depth < 6:
        return [_bounded_diagnostic(item, depth + 1) for item in value[:8]]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _diagnostic_text(value)


def _append_diagnostic(
    path: Path, item: dict[str, Any], context: ChatTurnContext
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    budget = current_evaluation_budget()
    item["attempts"] = context.llm_provider_attempts
    item["retries"] = context.retries_used
    item["reservation_ids"] = budget.provenance()["reservation_ids"] if budget else []
    item["job_ids"] = budget.provenance()["job_ids"] if budget else []
    bounded = _bounded_diagnostic(item)
    encoded = json.dumps(bounded, ensure_ascii=False)
    if len(encoded) >= 16000:
        # Preserve useful prefixes and guard evidence even for pathological payloads.
        retained = {
            key: bounded[key]
            for key in (
                "event",
                "name",
                "node",
                "run_id",
                "configured_provider",
                "configured_model",
                "content",
                "routing_output",
                "response_metadata",
                "attempts",
                "retries",
                "reservation_ids",
                "job_ids",
            )
            if key in bounded
        }
        if bounded.get("exception_chain"):
            retained["exception_chain"] = [
                {
                    key: entry[key]
                    for key in ("type", "message", "status")
                    if key in entry
                }
                for entry in bounded["exception_chain"][:5]
            ]
        retained["truncated"] = True
        encoded = json.dumps(retained, ensure_ascii=False)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_live_diagnostic(
    path: Path, event: dict[str, Any], context: ChatTurnContext
) -> None:
    event_type = str(event.get("event") or "")
    if event_type not in {
        "on_chain_start",
        "on_chain_end",
        "on_chain_error",
        "on_chat_model_end",
        "on_chat_model_error",
        "on_tool_start",
        "on_tool_end",
        "on_tool_error",
    }:
        return
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    node = str(metadata.get("langgraph_node") or "")
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    item = {
        "event": event_type,
        "name": _diagnostic_text(event.get("name") or ""),
        "node": node,
        "run_id": str(event.get("run_id") or ""),
        "configured_provider": str(metadata.get("ls_provider") or ""),
        "configured_model": str(metadata.get("ls_model_name") or ""),
    }
    output = data.get("output")
    if event_type == "on_chat_model_end":
        item["content"] = (
            _diagnostic_text(getattr(output, "content", ""))
            if node == "orchestrator"
            else "[not captured]"
        )
        item["tool_calls"] = _redact_mapping(
            {"calls": getattr(output, "tool_calls", []) or []}
        )["calls"]
        item["response_metadata"] = _redact_mapping(
            getattr(output, "response_metadata", {}) or {}
        )
    elif event_type == "on_chain_end" and node == "orchestrator":
        item["routing_output"] = _diagnostic_text(output)
    elif event_type == "on_tool_start":
        item["arguments"] = (
            _redact_mapping(data.get("input"))
            if isinstance(data.get("input"), dict)
            else {}
        )
    elif event_type.endswith("error"):
        error = data.get("error")
        item["error_type"] = type(error).__name__
        if isinstance(error, BaseException):
            item["error_message"] = _diagnostic_text(error)
            item["http_status"] = provider_status_code(error)
    _append_diagnostic(path, item, context)


def _write_live_exception(
    path: Path, exc: BaseException, context: ChatTurnContext
) -> None:
    chain = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(chain) < 5:
        seen.add(id(current))
        chain.append(
            {
                "type": type(current).__name__,
                "message": _diagnostic_text(current),
                "status": provider_status_code(current),
                "stack": [
                    {
                        "file": frame.filename,
                        "line": frame.lineno,
                        "function": frame.name,
                    }
                    for frame in traceback.extract_tb(current.__traceback__)[-8:]
                ],
            }
        )
        current = current.__cause__ or current.__context__
    _append_diagnostic(path, {"event": "exception", "exception_chain": chain}, context)


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
    current_turn_start = input_message_count
    for index in range(len(messages) - 1, -1, -1):
        if getattr(messages[index], "type", None) == "human":
            current_turn_start = index + 1
            break
    for message in reversed(messages[current_turn_start:]):
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
    if isinstance(exc, (EvaluationBudgetExceeded, EvaluationBudgetUnavailable)):
        return exc.reason
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
