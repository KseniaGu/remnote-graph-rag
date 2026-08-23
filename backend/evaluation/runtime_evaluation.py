"""Trace-first evaluation contracts and scorers for the online learner workflow."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "runtime-evaluation"
EVIDENCE_ITEM_LIMIT = 10
EVIDENCE_EXCERPT_LIMIT = 1_200
EVIDENCE_TOTAL_LIMIT = 10_000
GRAPH_NODE_LIMIT = 25
GRAPH_EDGE_LIMIT = 35
WORKFLOW_NODES = {
    "orchestrator",
    "retriever",
    "researcher",
    "analyst",
    "mentor",
    "visualizer",
}
SECRET_KEY_PARTS = (
    "api_key",
    "authorization",
    "password",
    "secret",
    "access_token",
    "bearer",
    "credential",
)
EVIDENCE_ID_RE = re.compile(r"\b(?:chunk|concept|rel)_[A-Za-z0-9_:.-]+\b")
JUDGE_RUBRIC_VERSION = "runtime-semantic-v1"
SEMANTIC_JUDGE_OUTPUT_TOKEN_LIMITS = {
    "claim_faithfulness": 4096,
    "analyst_usefulness": 512,
    "mentor_pedagogy": 512,
    "conversational_continuity": 512,
    "graph_usefulness": 512,
}
SEMANTIC_JUDGE_RUBRICS = {
    "claim_faithfulness": (
        "Classify every externally verifiable factual claim as supported, partially "
        "supported, or unsupported using only the supplied bounded evidence."
    ),
    "analyst_usefulness": (
        "Pass only if the answer directly addresses the request, is clear and useful, "
        "and does not replace the requested topic with a merely adjacent topic."
    ),
    "mentor_pedagogy": (
        "Pass only if the response behaves as a tutor: it responds to the learner state, "
        "gives an appropriate hint or next step, and avoids an unrelated lecture."
    ),
    "conversational_continuity": (
        "Pass only if the response preserves the topic and interaction mode established by "
        "the supplied recent history and session summary."
    ),
    "graph_usefulness": (
        "Pass only if the labeled graph is relevant to the request, contains useful "
        "relationships, and avoids replacing the requested topic with an adjacent topic."
    ),
}
SEMANTIC_JUDGE_REQUIRED_FINAL_WORKERS = {
    "claim_faithfulness": "analyst",
    "analyst_usefulness": "analyst",
    "mentor_pedagogy": "mentor",
    "conversational_continuity": "mentor",
    "graph_usefulness": "visualizer",
}


def _is_runtime_schema(value: Any) -> bool:
    observed = str(value or "")
    version_prefix = f"{SCHEMA_VERSION}-v"
    return observed == SCHEMA_VERSION or (
        observed.startswith(version_prefix)
        and observed.removeprefix(version_prefix).isdigit()
    )


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StructuredOutputMethodJudge(BaseChatModel):
    """Makes OpenEvals use the selected structured-output transport."""

    wrapped: Any = Field(exclude=True)
    structured_output_method: Literal[
        "function_calling", "json_schema", "json_mode"
    ] = "function_calling"

    @property
    def _llm_type(self) -> str:
        return "evaluation-structured-output-adapter"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "wrapped_model": getattr(self.wrapped, "model", ""),
            "structured_output_method": self.structured_output_method,
        }

    def _generate(self, messages: list[Any], **kwargs: Any) -> Any:
        return self.wrapped._generate(messages, **kwargs)

    async def _agenerate(self, messages: list[Any], **kwargs: Any) -> Any:
        return await self.wrapped._agenerate(messages, **kwargs)

    def with_structured_output(self, schema: dict | type, **kwargs: Any) -> Any:
        return self.wrapped.with_structured_output(
            schema,
            method=self.structured_output_method,
            **kwargs,
        )


class ScenarioInput(StrictModel):
    user_message: str
    message_history: list[dict[str, str]] = Field(default_factory=list)
    session_summary: str = ""


class ToolArgumentExpectation(StrictModel):
    tool: str
    argument: str
    required_terms_all: list[str] = Field(default_factory=list)
    required_terms_any: list[str] = Field(default_factory=list)
    forbidden_terms: list[str] = Field(default_factory=list)
    forbidden_exact_values: list[str] = Field(default_factory=list)


class RuntimeBudgets(StrictModel):
    max_worker_steps: int | None = None
    max_logical_llm_calls: int | None = None
    max_provider_attempts: int | None = None
    max_retries: int | None = None
    max_tavily_searches: int | None = None
    max_total_tokens: int | None = None
    max_latency_seconds: float | None = None


class RuntimeExpectations(StrictModel):
    allowed_worker_sequences: list[list[str]] = Field(default_factory=list)
    required_agents: list[str] = Field(default_factory=list)
    forbidden_agents: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    tool_arguments: list[ToolArgumentExpectation] = Field(default_factory=list)
    request_scope: str | None = None
    allowed_request_scopes: list[str] = Field(default_factory=list)
    allowed_retrieval_statuses: list[str] = Field(default_factory=list)
    retrieval_outcome: str | None = None
    sources_exhausted: bool | None = None
    requested_modality: Literal["text", "graph"] | None = None
    final_response_required: bool | None = None
    graph_min_nodes: int | None = None
    graph_max_nodes: int | None = None
    graph_min_edges: int | None = None
    graph_max_edges: int | None = None
    graph_required_node_ids: list[str] = Field(default_factory=list)
    forbidden_failure_types: list[str] = Field(default_factory=list)
    required_failure_types: list[str] = Field(default_factory=list)
    termination_required: bool = True
    one_tool_per_worker: bool = True
    budgets: RuntimeBudgets = Field(default_factory=RuntimeBudgets)
    judge_dimensions: list[str] = Field(default_factory=list)


class RuntimeScenario(StrictModel):
    schema_version: str = SCHEMA_VERSION
    id: str
    version: int = 1
    review_status: Literal["reviewed", "provisional_needs_review"]
    description: str
    trace_ids: list[str] = Field(default_factory=list)
    input: ScenarioInput
    expectations: RuntimeExpectations
    provenance: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("schema_version", mode="before")
    @classmethod
    def normalize_schema_version(cls, value: Any) -> str:
        if not _is_runtime_schema(value):
            raise ValueError("unsupported runtime evaluation schema")
        return SCHEMA_VERSION

    @model_validator(mode="after")
    def validate_scenario(self) -> RuntimeScenario:
        if not self.id.strip():
            raise ValueError("scenario id cannot be empty")
        if self.version <= 0:
            raise ValueError("scenario version must be positive")
        if not self.input.user_message.strip():
            raise ValueError("scenario user_message cannot be empty")
        for message in self.input.message_history:
            if message.get("role") not in {"user", "assistant"}:
                raise ValueError("message_history roles must be user or assistant")
            if not isinstance(message.get("content"), str):
                raise ValueError("message_history content must be text")
        max_steps = self.expectations.budgets.max_worker_steps
        oversized = [
            path
            for path in self.expectations.allowed_worker_sequences
            if max_steps is not None and len(path) > max_steps
        ]
        if oversized:
            raise ValueError("allowed worker sequences cannot exceed max_worker_steps")
        return self


class EvidenceObservation(StrictModel):
    evidence_id: str
    kind: Literal["local_chunk", "local_relation", "web_result", "tool_output"]
    query: str = ""
    rank: int | None = None
    score: float | None = None
    title: str = ""
    source_path: str = ""
    excerpt: str = ""


class FaithfulnessClaim(StrictModel):
    claim: str = Field(max_length=400)
    verdict: Literal["supported", "partial", "unsupported"]
    evidence_ids: list[str] = Field(default_factory=list, max_length=5)
    reason: str = Field(default="", max_length=240)


class FaithfulnessAssessment(StrictModel):
    claims: list[FaithfulnessClaim] = Field(default_factory=list, max_length=32)


class BooleanJudgeAssessment(StrictModel):
    score: bool
    reason: str = Field(default="", max_length=320)


class ToolObservation(StrictModel):
    run_id: str = ""
    name: str
    node: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: str = "success"
    evidence_ids: list[str] = Field(default_factory=list)
    bounded_evidence: list[EvidenceObservation] = Field(default_factory=list)


class ProviderObservation(StrictModel):
    run_id: str = ""
    node: str = ""
    provider: str = ""
    model: str = ""
    status: str = "success"
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    content_present: bool = False
    tool_call_count: int = 0
    configured_output_token_limit: int | None = None


class FailureObservation(StrictModel):
    code: str
    node: str = ""
    run_id: str = ""


class GraphObservation(StrictModel):
    artifact_count: int = 0
    node_ids: list[str] = Field(default_factory=list)
    edges: list[tuple[str, str, str]] = Field(default_factory=list)
    dangling_edge_count: int = 0
    node_labels: dict[str, str] = Field(default_factory=dict)
    labeled_edges: list[tuple[str, str, str]] = Field(default_factory=list)
    capture_status: Literal["captured", "unavailable", "not_applicable"] = (
        "not_applicable"
    )


class RuntimeUsage(StrictModel):
    latency_seconds: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    logical_llm_calls: int = 0
    provider_attempts: int = 0
    retries: int = 0
    tavily_searches: int = 0
    reported_cost: float | None = None


class RuntimeRecord(StrictModel):
    schema_version: str = SCHEMA_VERSION
    scenario_id: str
    scenario_version: int
    run_id: str
    repetition: int = 1
    source_mode: Literal["offline_trace", "controlled_live"]
    trace_id: str | None = None
    trace_status: str = "unknown"
    input: ScenarioInput
    agent_sequence: list[str] = Field(default_factory=list)
    worker_sequence: list[str] = Field(default_factory=list)
    route_decisions: list[str] = Field(default_factory=list)
    tools: list[ToolObservation] = Field(default_factory=list)
    providers: list[ProviderObservation] = Field(default_factory=list)
    retrieval_statuses: list[str] = Field(default_factory=list)
    retrieval_outcome: str = "not_run"
    request_scope: str = "unclassified"
    sources_exhausted: bool = False
    final_response: str = ""
    output_modality: Literal["text", "graph", "error", "none"] = "none"
    graph: GraphObservation = Field(default_factory=GraphObservation)
    evidence_ids: list[str] = Field(default_factory=list)
    bounded_evidence: list[EvidenceObservation] = Field(default_factory=list)
    evidence_capture_status: Literal["captured", "unavailable", "not_applicable"] = (
        "not_applicable"
    )
    action_signatures: list[str] = Field(default_factory=list)
    failures: list[FailureObservation] = Field(default_factory=list)
    terminated: bool = False
    usage: RuntimeUsage = Field(default_factory=RuntimeUsage)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version", mode="before")
    @classmethod
    def normalize_schema_version(cls, value: Any) -> str:
        if not _is_runtime_schema(value):
            raise ValueError("unsupported runtime evaluation schema")
        return SCHEMA_VERSION

    @model_validator(mode="after")
    def validate_observation_bounds(self) -> RuntimeRecord:
        if len(self.bounded_evidence) > EVIDENCE_ITEM_LIMIT:
            raise ValueError("bounded evidence exceeds item limit")
        if any(
            len(item.excerpt) > EVIDENCE_EXCERPT_LIMIT for item in self.bounded_evidence
        ):
            raise ValueError("bounded evidence excerpt exceeds character limit")
        if (
            sum(len(item.excerpt) for item in self.bounded_evidence)
            > EVIDENCE_TOTAL_LIMIT
        ):
            raise ValueError("bounded evidence exceeds total character limit")
        if len(self.graph.node_ids) > GRAPH_NODE_LIMIT:
            raise ValueError("bounded graph exceeds node limit")
        if len(self.graph.edges) > GRAPH_EDGE_LIMIT:
            raise ValueError("bounded graph exceeds edge limit")
        if len(self.graph.labeled_edges) > GRAPH_EDGE_LIMIT:
            raise ValueError("bounded graph exceeds labeled-edge limit")
        return self


class CheckResult(StrictModel):
    dimension: str
    name: str
    status: Literal["pass", "fail", "not_observed", "not_applicable"]
    reason: str
    gating: bool = True


class RuntimeCaseResult(StrictModel):
    scenario_id: str
    scenario_version: int
    run_id: str
    repetition: int
    source_mode: Literal["offline_trace", "controlled_live"]
    trace_id: str | None = None
    passed: bool
    checks: list[CheckResult]
    failure_types: list[str]
    metrics: dict[str, int | float | str | bool | None]


class SemanticJudgeResult(StrictModel):
    scenario_id: str
    scenario_version: int
    run_id: str
    repetition: int
    dimension: str
    rubric_version: str = JUDGE_RUBRIC_VERSION
    framework: str = "openevals"
    judge_provider: str
    judge_model: str
    status: Literal["success", "skipped", "error"]
    score: bool | float | None = None
    reason: str = ""
    provider_call_attempted: bool = False
    error_type: str = ""
    error_cause_types: list[str] = Field(default_factory=list)
    raw_output_omitted: bool = False
    transport_status: Literal["not_attempted", "completed", "failed"] = "not_attempted"
    http_status_code: int | None = None
    stop_reason: str = ""
    configured_output_token_limit: int | None = None
    output_present: bool = False
    output_size_chars: int | None = None
    parser_classification: Literal[
        "not_attempted",
        "success",
        "malformed_json",
        "schema_validation_failure",
        "structured_output_parse_failure",
        "provider_failure",
    ] = "not_attempted"
    confirmed_truncation: bool = False
    evidence_ids: list[str] = Field(default_factory=list)
    supported_claims: int | None = None
    partial_claims: int | None = None
    unsupported_claims: int | None = None
    total_claims: int | None = None
    grounded_claim_rate: float | None = None
    claims: list[FaithfulnessClaim] = Field(default_factory=list)


def load_runtime_scenarios(path: Path) -> list[RuntimeScenario]:
    """Loads the versioned runtime scenario array and rejects duplicate IDs."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("runtime scenario file must contain a JSON array")
    scenarios = [RuntimeScenario.model_validate(item) for item in payload]
    ids = [scenario.id for scenario in scenarios]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate runtime scenario ids: {', '.join(duplicates)}")
    return scenarios


def load_trace_export(path: Path) -> list[dict[str, Any]]:
    """Loads an exported LangSmith run list without contacting LangSmith."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(run, dict) for run in payload
    ):
        raise ValueError("LangSmith export must contain a JSON array of run objects")
    return payload


def load_runtime_records(path: Path) -> list[RuntimeRecord]:
    """Loads canonical JSONL records previously produced by this evaluator."""
    records: list[RuntimeRecord] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            records.append(RuntimeRecord.model_validate_json(raw_line))
        except ValueError as exc:
            raise ValueError(
                f"invalid runtime record at {path}:{line_number}: {exc}"
            ) from exc
    return records


def normalize_trace_export(
    runs: list[dict[str, Any]], scenarios: list[RuntimeScenario]
) -> list[RuntimeRecord]:
    """Normalizes scenario-linked root traces into framework-neutral records."""
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        trace_id = str(run.get("trace_id") or "")
        if trace_id:
            by_trace[trace_id].append(run)

    records: list[RuntimeRecord] = []
    missing: list[str] = []
    for scenario in scenarios:
        for repetition, trace_id in enumerate(scenario.trace_ids, start=1):
            trace_runs = by_trace.get(trace_id)
            if not trace_runs:
                missing.append(f"{scenario.id}:{trace_id}")
                continue
            records.append(
                normalize_trace_runs(trace_runs, scenario, repetition=repetition)
            )
    if missing:
        raise ValueError("missing scenario traces: " + ", ".join(missing))
    return records


def normalize_trace_runs(
    runs: list[dict[str, Any]],
    scenario: RuntimeScenario,
    *,
    repetition: int = 1,
) -> RuntimeRecord:
    """Converts one exported LangSmith trace into the canonical runtime record."""
    ordered = sorted(runs, key=lambda run: str(run.get("start_time") or ""))
    roots = [run for run in ordered if run.get("parent_run_id") is None]
    if len(roots) != 1:
        raise ValueError(f"expected one root run, found {len(roots)}")
    root = roots[0]
    root_id = str(root.get("_id") or "")
    trace_id = str(root.get("trace_id") or root_id)
    root_inputs = root.get("inputs") if isinstance(root.get("inputs"), dict) else {}
    root_outputs = root.get("outputs") if isinstance(root.get("outputs"), dict) else {}

    direct_nodes = [
        run
        for run in ordered
        if str(run.get("parent_run_id") or "") == root_id
        and run.get("name") in WORKFLOW_NODES
    ]
    agent_sequence = [str(run["name"]) for run in direct_nodes]
    worker_sequence = [name for name in agent_sequence if name != "orchestrator"]
    route_decisions = [
        str(run.get("outputs", {}).get("next_step"))
        for run in direct_nodes
        if run.get("name") == "orchestrator"
        and isinstance(run.get("outputs"), dict)
        and run.get("outputs", {}).get("next_step")
    ]

    providers = [
        _provider_observation(run) for run in ordered if run.get("run_type") == "llm"
    ]
    tools = [_tool_observation(run) for run in ordered if run.get("run_type") == "tool"]
    graph = _graph_observation(root_outputs, tools, ordered)
    bounded_evidence = _merge_bounded_evidence(
        item for tool in tools for item in tool.bounded_evidence
    )
    evidence_tools_observed = any(
        tool.name
        in {"search_knowledge_base", "deep_web_research", "get_subgraphs_to_visualize"}
        for tool in tools
    )
    evidence_capture_status = (
        "captured"
        if bounded_evidence
        else ("unavailable" if evidence_tools_observed else "not_applicable")
    )

    retrieval_statuses = [
        str(run.get("outputs", {}).get("retrieval_status"))
        for run in direct_nodes
        if run.get("name") == "retriever"
        and isinstance(run.get("outputs"), dict)
        and run.get("outputs", {}).get("retrieval_status")
    ]
    retrieval_outcome = _retrieval_outcome(retrieval_statuses, tools, providers)
    failures = _failure_observations(
        ordered,
        providers,
        retrieval_outcome=retrieval_outcome,
        sources_exhausted=bool(root_outputs.get("sources_exhausted", False)),
    )
    final_response = _final_response(root_inputs, root_outputs)
    modality: Literal["text", "graph", "error", "none"]
    if graph.artifact_count:
        modality = "graph"
    elif final_response:
        modality = "text"
    elif str(root.get("status")) == "error":
        modality = "error"
    else:
        modality = "none"

    input_messages = _messages(root_inputs.get("messages"))
    latest_human_index = max(
        (index for index, item in enumerate(input_messages) if item["role"] == "user"),
        default=-1,
    )
    if latest_human_index >= 0:
        user_message = input_messages[latest_human_index]["content"]
        history = input_messages[:latest_human_index]
    else:
        user_message = scenario.input.user_message
        history = scenario.input.message_history

    metadata = root.get("extra", {}).get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    evidence_ids = sorted(
        {evidence_id for tool in tools for evidence_id in tool.evidence_ids}
        | set(graph.node_ids)
    )
    logical_calls = len({provider.run_id for provider in providers})
    # One exported leaf LLM run represents one provider attempt. Retried calls share
    # the same parent wrapper, when that information is present.
    logical_parent_ids = {
        str(run.get("parent_run_id") or run.get("_id") or "")
        for run in ordered
        if run.get("run_type") == "llm"
    }
    if logical_parent_ids:
        logical_calls = len(logical_parent_ids)
    provider_attempts = len(providers)
    reported_cost = _float_or_none(root.get("total_cost"))

    return RuntimeRecord(
        scenario_id=scenario.id,
        scenario_version=scenario.version,
        run_id=trace_id,
        repetition=repetition,
        source_mode="offline_trace",
        trace_id=trace_id,
        trace_status=str(root.get("status") or "unknown"),
        input=ScenarioInput(
            user_message=user_message,
            message_history=history,
            session_summary=str(root_inputs.get("session_summary") or ""),
        ),
        agent_sequence=agent_sequence,
        worker_sequence=worker_sequence,
        route_decisions=route_decisions,
        tools=tools,
        providers=providers,
        retrieval_statuses=retrieval_statuses,
        retrieval_outcome=retrieval_outcome,
        request_scope=str(
            root_outputs.get("request_scope")
            or root_inputs.get("request_scope")
            or "unclassified"
        ),
        sources_exhausted=bool(root_outputs.get("sources_exhausted", False)),
        final_response=final_response,
        output_modality=modality,
        graph=graph,
        evidence_ids=evidence_ids,
        bounded_evidence=bounded_evidence,
        evidence_capture_status=evidence_capture_status,
        action_signatures=_action_signatures(worker_sequence, tools),
        failures=failures,
        terminated=(
            str(root.get("status")) == "success"
            and (
                root_outputs.get("next_step") == "__end__"
                or route_decisions[-1:] == ["__end__"]
            )
        ),
        usage=RuntimeUsage(
            latency_seconds=_float_or_none(root.get("latency")),
            input_tokens=_int_or_zero(root.get("prompt_tokens")),
            output_tokens=_int_or_zero(root.get("completion_tokens")),
            total_tokens=_int_or_zero(root.get("total_tokens")),
            logical_llm_calls=logical_calls,
            provider_attempts=provider_attempts,
            retries=max(0, provider_attempts - logical_calls),
            tavily_searches=sum(tool.name == "deep_web_research" for tool in tools),
            reported_cost=reported_cost,
        ),
        provenance={
            "revision_id": str(metadata.get("revision_id") or ""),
            "evaluation_fingerprint": str(metadata.get("evaluation_fingerprint") or ""),
            "source_configuration_status": (
                "verified"
                if metadata.get("evaluation_fingerprint")
                else "needs verification"
            ),
            "langsmith_project": str(metadata.get("LANGSMITH_PROJECT") or ""),
            "thread_id": str(metadata.get("thread_id") or ""),
            "checkpoint_ns": str(metadata.get("checkpoint_ns") or ""),
            "analyst_retrieval_mode": str(
                metadata.get("analyst_retrieval_mode") or "needs verification"
            ),
            "visualizer_retrieval_mode": str(
                metadata.get("visualizer_retrieval_mode") or "needs verification"
            ),
        },
    )


def score_runtime_record(
    scenario: RuntimeScenario, record: RuntimeRecord
) -> RuntimeCaseResult:
    """Scores exact workflow contracts without judging free-form answer prose."""
    expected = scenario.expectations
    checks: list[CheckResult] = []

    def check(
        dimension: str,
        name: str,
        passed: bool,
        reason: str,
        *,
        observed: bool = True,
        gating: bool = True,
    ) -> None:
        checks.append(
            CheckResult(
                dimension=dimension,
                name=name,
                status="pass" if passed else ("fail" if observed else "not_observed"),
                reason=reason,
                gating=gating,
            )
        )

    if expected.allowed_worker_sequences:
        check(
            "routing",
            "worker_sequence",
            record.worker_sequence in expected.allowed_worker_sequences,
            f"actual={record.worker_sequence!r}; allowed={expected.allowed_worker_sequences!r}",
        )
    agent_set = set(record.worker_sequence)
    missing_agents = sorted(set(expected.required_agents) - agent_set)
    forbidden_agents = sorted(set(expected.forbidden_agents) & agent_set)
    if expected.required_agents:
        check(
            "routing",
            "required_agents",
            not missing_agents,
            f"missing={missing_agents}",
        )
    if expected.forbidden_agents:
        check(
            "routing",
            "forbidden_agents",
            not forbidden_agents,
            f"present={forbidden_agents}",
        )

    tool_names = [tool.name for tool in record.tools]
    missing_tools = sorted(set(expected.required_tools) - set(tool_names))
    forbidden_tools = sorted(set(expected.forbidden_tools) & set(tool_names))
    if expected.required_tools:
        check("tools", "required_tools", not missing_tools, f"missing={missing_tools}")
    if expected.forbidden_tools:
        check(
            "tools",
            "forbidden_tools",
            not forbidden_tools,
            f"present={forbidden_tools}",
        )
    for argument_expectation in expected.tool_arguments:
        matches = [
            tool for tool in record.tools if tool.name == argument_expectation.tool
        ]
        observed = bool(matches)
        values = [
            str(tool.arguments.get(argument_expectation.argument, ""))
            for tool in matches
        ]
        normalized_values = [value.casefold() for value in values]
        all_terms = all(
            any(term.casefold() in value for value in normalized_values)
            for term in argument_expectation.required_terms_all
        )
        any_terms = not argument_expectation.required_terms_any or any(
            term.casefold() in value
            for term in argument_expectation.required_terms_any
            for value in normalized_values
        )
        forbidden_terms = [
            term
            for term in argument_expectation.forbidden_terms
            if any(term.casefold() in value for value in normalized_values)
        ]
        forbidden_exact = [
            forbidden
            for forbidden in argument_expectation.forbidden_exact_values
            if forbidden.casefold() in normalized_values
        ]
        check(
            "tools",
            f"{argument_expectation.tool}.{argument_expectation.argument}",
            observed
            and all_terms
            and any_terms
            and not forbidden_terms
            and not forbidden_exact,
            (
                f"values={values!r}; missing_all={not all_terms}; "
                f"missing_any={not any_terms}; forbidden_terms={forbidden_terms}; "
                f"forbidden_exact={forbidden_exact}"
            ),
            observed=observed,
        )
    if expected.one_tool_per_worker:
        tool_counts = Counter(tool.node for tool in record.tools if tool.node)
        excess = {node: count for node, count in tool_counts.items() if count > 1}
        check("tools", "one_tool_per_worker", not excess, f"excess={excess}")

    worker_action_signatures = record.action_signatures[: len(record.worker_sequence)]
    signature_counts = Counter(worker_action_signatures)
    repeated_actions = sum(
        count - 1 for count in signature_counts.values() if count > 1
    )
    loop_detected = repeated_actions > 0
    check(
        "routing",
        "basic_loop_detector",
        not loop_detected,
        (
            f"loop_detected={loop_detected}; repeated_action_count={repeated_actions}; "
            f"signatures={worker_action_signatures!r}"
        ),
    )

    if expected.request_scope is not None:
        check(
            "scope",
            "request_scope",
            record.request_scope == expected.request_scope,
            f"actual={record.request_scope!r}; expected={expected.request_scope!r}",
            observed=record.request_scope != "unclassified",
        )
    if expected.allowed_request_scopes:
        check(
            "scope",
            "request_scope_allowed",
            record.request_scope in expected.allowed_request_scopes,
            (
                f"actual={record.request_scope!r}; "
                f"allowed={expected.allowed_request_scopes!r}"
            ),
            observed=record.request_scope != "unclassified",
        )
    if expected.allowed_retrieval_statuses:
        observed_status = (
            record.retrieval_statuses[-1] if record.retrieval_statuses else "not_run"
        )
        check(
            "retrieval",
            "retrieval_status",
            observed_status in expected.allowed_retrieval_statuses,
            f"actual={observed_status!r}; allowed={expected.allowed_retrieval_statuses!r}",
        )
    if expected.retrieval_outcome is not None:
        check(
            "retrieval",
            "retrieval_outcome",
            record.retrieval_outcome == expected.retrieval_outcome,
            f"actual={record.retrieval_outcome!r}; expected={expected.retrieval_outcome!r}",
        )
    if expected.sources_exhausted is not None:
        check(
            "fallback",
            "sources_exhausted",
            record.sources_exhausted is expected.sources_exhausted,
            f"actual={record.sources_exhausted}; expected={expected.sources_exhausted}",
        )

    if expected.requested_modality is not None:
        check(
            "modality",
            "requested_modality",
            record.output_modality == expected.requested_modality,
            f"actual={record.output_modality!r}; expected={expected.requested_modality!r}",
        )
    if expected.final_response_required is not None:
        has_response = bool(record.final_response.strip())
        check(
            "answer",
            "final_response_present",
            has_response is expected.final_response_required,
            f"present={has_response}; expected={expected.final_response_required}",
        )

    graph_expected = any(
        value is not None
        for value in (
            expected.graph_min_nodes,
            expected.graph_max_nodes,
            expected.graph_min_edges,
            expected.graph_max_edges,
        )
    ) or bool(expected.graph_required_node_ids)
    if graph_expected:
        check(
            "graph",
            "artifact_present",
            record.graph.artifact_count > 0,
            f"artifact_count={record.graph.artifact_count}",
        )
        check(
            "graph",
            "no_dangling_edges",
            record.graph.dangling_edge_count == 0,
            f"dangling_edge_count={record.graph.dangling_edge_count}",
        )
    _check_bound(
        checks,
        "graph",
        "node_count_min",
        len(record.graph.node_ids),
        expected.graph_min_nodes,
        minimum=True,
    )
    _check_bound(
        checks,
        "graph",
        "node_count_max",
        len(record.graph.node_ids),
        expected.graph_max_nodes,
        minimum=False,
    )
    _check_bound(
        checks,
        "graph",
        "edge_count_min",
        len(record.graph.edges),
        expected.graph_min_edges,
        minimum=True,
    )
    _check_bound(
        checks,
        "graph",
        "edge_count_max",
        len(record.graph.edges),
        expected.graph_max_edges,
        minimum=False,
    )
    if expected.graph_required_node_ids:
        missing_nodes = sorted(
            set(expected.graph_required_node_ids) - set(record.graph.node_ids)
        )
        check("graph", "required_nodes", not missing_nodes, f"missing={missing_nodes}")

    failure_types = sorted({failure.code for failure in record.failures})
    forbidden_failures = sorted(
        set(expected.forbidden_failure_types) & set(failure_types)
    )
    required_failures = sorted(
        set(expected.required_failure_types) - set(failure_types)
    )
    if expected.forbidden_failure_types:
        check(
            "reliability",
            "forbidden_failures",
            not forbidden_failures,
            f"present={forbidden_failures}",
        )
    if expected.required_failure_types:
        check(
            "reliability",
            "required_failures",
            not required_failures,
            f"missing={required_failures}",
        )
    if expected.termination_required:
        check(
            "termination",
            "terminated",
            record.terminated,
            f"terminated={record.terminated}",
        )

    budgets = expected.budgets
    _check_bound(
        checks,
        "budget",
        "worker_steps",
        len(record.worker_sequence),
        budgets.max_worker_steps,
        minimum=False,
    )
    _check_bound(
        checks,
        "budget",
        "logical_llm_calls",
        record.usage.logical_llm_calls,
        budgets.max_logical_llm_calls,
        minimum=False,
        gating=False,
    )
    _check_bound(
        checks,
        "budget",
        "provider_attempts",
        record.usage.provider_attempts,
        budgets.max_provider_attempts,
        minimum=False,
        gating=False,
    )
    _check_bound(
        checks,
        "budget",
        "retries",
        record.usage.retries,
        budgets.max_retries,
        minimum=False,
        gating=False,
    )
    _check_bound(
        checks,
        "budget",
        "tavily_searches",
        record.usage.tavily_searches,
        budgets.max_tavily_searches,
        minimum=False,
    )
    _check_bound(
        checks,
        "budget",
        "total_tokens",
        record.usage.total_tokens,
        budgets.max_total_tokens,
        minimum=False,
        gating=False,
    )
    _check_bound(
        checks,
        "budget",
        "latency_seconds",
        record.usage.latency_seconds,
        budgets.max_latency_seconds,
        minimum=False,
        gating=False,
    )

    passed = not any(result.status == "fail" and result.gating for result in checks)
    ideal_steps = (
        min(len(path) for path in expected.allowed_worker_sequences)
        if expected.allowed_worker_sequences
        else None
    )
    path_efficiency = (
        min(1.0, ideal_steps / len(record.worker_sequence))
        if ideal_steps is not None
        and len(record.worker_sequence) >= ideal_steps
        and record.terminated
        and len(record.worker_sequence) > 0
        else (1.0 if ideal_steps == 0 and record.terminated else None)
    )
    tool_checks = [
        item
        for item in checks
        if item.dimension == "tools"
        and item.name in {"required_tools", "forbidden_tools"}
    ]
    argument_checks = [
        item for item in checks if item.dimension == "tools" and "." in item.name
    ]
    return RuntimeCaseResult(
        scenario_id=scenario.id,
        scenario_version=scenario.version,
        run_id=record.run_id,
        repetition=record.repetition,
        source_mode=record.source_mode,
        trace_id=record.trace_id,
        passed=passed,
        checks=checks,
        failure_types=failure_types,
        metrics={
            "worker_steps": len(record.worker_sequence),
            "tool_calls": len(record.tools),
            "graph_nodes": len(record.graph.node_ids),
            "graph_edges": len(record.graph.edges),
            "logical_llm_calls": record.usage.logical_llm_calls,
            "provider_attempts": record.usage.provider_attempts,
            "retries": record.usage.retries,
            "tavily_searches": record.usage.tavily_searches,
            "total_tokens": record.usage.total_tokens,
            "input_tokens": record.usage.input_tokens,
            "output_tokens": record.usage.output_tokens,
            "latency_seconds": record.usage.latency_seconds,
            "path_efficiency_ratio": path_efficiency,
            "loop_detected": loop_detected,
            "repeated_action_count": repeated_actions,
            "tool_selection_correct": (
                all(item.status == "pass" for item in tool_checks)
                if tool_checks
                else None
            ),
            "tool_argument_valid": (
                all(item.status == "pass" for item in argument_checks)
                if argument_checks
                else None
            ),
            "fallback_correct": (
                "deep_web_research" in tool_names
                and "researcher" in record.worker_sequence
                and (
                    expected.sources_exhausted is None
                    or record.sources_exhausted is expected.sources_exhausted
                )
                if "deep_web_research" in expected.required_tools
                else None
            ),
            "unnecessary_web": (
                record.usage.tavily_searches > 0
                if "deep_web_research" in expected.forbidden_tools
                else None
            ),
            "output_limit_hit": any(
                provider.stop_reason == "length" for provider in record.providers
            ),
        },
    )


def summarize_runtime_results(
    records: list[RuntimeRecord], results: list[RuntimeCaseResult]
) -> dict[str, Any]:
    """Builds separate pass/failure and reliability summaries without one score."""
    by_dimension: dict[str, Counter[str]] = defaultdict(Counter)
    gating_by_dimension: dict[str, Counter[str]] = defaultdict(Counter)
    for result in results:
        for check in result.checks:
            by_dimension[check.dimension][check.status] += 1
            if check.gating:
                gating_by_dimension[check.dimension][check.status] += 1

    reliability: dict[str, Any] = {}
    records_by_scenario: dict[str, list[RuntimeRecord]] = defaultdict(list)
    results_by_scenario: dict[str, list[RuntimeCaseResult]] = defaultdict(list)
    for record in records:
        records_by_scenario[record.scenario_id].append(record)
    for result in results:
        results_by_scenario[result.scenario_id].append(result)
    for scenario_id, scenario_records in records_by_scenario.items():
        scenario_results = results_by_scenario[scenario_id]
        routes = [tuple(record.worker_sequence) for record in scenario_records]
        route_consistency = max(Counter(routes).values()) / len(routes)
        evidence_sets = [set(record.evidence_ids) for record in scenario_records]
        overlaps = [
            _jaccard(left, right) for left, right in combinations(evidence_sets, 2)
        ]
        pass_count = sum(result.passed for result in scenario_results)
        run_count = len(scenario_results)
        latencies = [
            record.usage.latency_seconds
            for record in scenario_records
            if record.usage.latency_seconds is not None
        ]
        tokens = [record.usage.total_tokens for record in scenario_records]
        reliability[scenario_id] = {
            "run_count": run_count,
            "pass_rate": pass_count / run_count,
            "pass_rate_wilson_95": list(_wilson_interval(pass_count, run_count)),
            "route_consistency": route_consistency,
            "evidence_set_stability": sum(overlaps) / len(overlaps)
            if overlaps
            else None,
            "evidence_set_sizes": [len(items) for items in evidence_sets],
            "looping_rate": (
                sum(
                    bool(result.metrics.get("loop_detected"))
                    for result in scenario_results
                )
                / run_count
            ),
            "output_limit_hit_rate": (
                sum(
                    bool(result.metrics.get("output_limit_hit"))
                    for result in scenario_results
                )
                / run_count
            ),
            "tokens": _distribution(tokens),
            "latency_seconds": _distribution(latencies),
            "failure_type_frequency": dict(
                Counter(
                    failure_type
                    for result in scenario_results
                    for failure_type in result.failure_types
                )
            ),
        }

    return {
        "record_count": len(records),
        "passed_count": sum(result.passed for result in results),
        "failed_count": sum(not result.passed for result in results),
        "failed_run_ids": [result.run_id for result in results if not result.passed],
        "dimensions": {key: dict(value) for key, value in sorted(by_dimension.items())},
        "gating_dimensions": {
            key: dict(value) for key, value in sorted(gating_by_dimension.items())
        },
        "reliability": reliability,
    }


def write_runtime_artifacts(
    output_dir: Path,
    records: list[RuntimeRecord],
    results: list[RuntimeCaseResult],
    *,
    source: str,
    framework_results: list[dict[str, Any]] | None = None,
    judge_results: list[SemanticJudgeResult] | None = None,
    judge_provider_calls: int = 0,
) -> dict[str, Any]:
    """Writes inspectable JSONL detail plus concise JSON and Markdown summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_runtime_results(records, results)
    created_at = datetime.now(UTC).isoformat()
    _write_jsonl(output_dir / "runtime_records.jsonl", records)
    _write_jsonl(output_dir / "case_results.jsonl", results)
    if framework_results is not None:
        (output_dir / "framework_results.json").write_text(
            json.dumps(framework_results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if judge_results is not None:
        _write_jsonl(output_dir / "judge_results.jsonl", judge_results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "source": source,
        "record_count": len(records),
        "framework": "agentevals" if framework_results is not None else "none",
        "judge_enabled": judge_results is not None,
        "judge_provider_calls": judge_provider_calls,
        "judge_result_count": len(judge_results or []),
        "judge_successful_results": sum(
            item.status == "success" for item in judge_results or []
        ),
        "judge_error_results": sum(
            item.status == "error" for item in judge_results or []
        ),
        "judge_skipped_results": sum(
            item.status == "skipped" for item in judge_results or []
        ),
        "runtime_provider_attempts": sum(
            record.usage.provider_attempts
            for record in records
            if record.source_mode == "controlled_live"
        ),
        "runtime_tavily_attempts": sum(
            record.usage.tavily_searches
            for record in records
            if record.source_mode == "controlled_live"
        ),
        "provider_calls_made_by_offline_scoring": 0,
        "source_revisions": sorted(
            {
                str(record.provenance.get("revision_id"))
                for record in records
                if record.provenance.get("revision_id")
            }
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        _summary_markdown(
            summary,
            results,
            framework_results,
            judge_results,
            judge_provider_calls,
        ),
        encoding="utf-8",
    )
    return summary


def evaluate_with_agentevals(
    scenario: RuntimeScenario, record: RuntimeRecord
) -> dict[str, Any]:
    """Runs AgentEvals strict graph matching as an optional non-gating cross-check."""
    try:
        from agentevals.graph_trajectory.strict import graph_trajectory_strict_match
    except ImportError as exc:
        raise RuntimeError(
            "AgentEvals is optional; install the evaluation dependency group first"
        ) from exc

    allowed = scenario.expectations.allowed_worker_sequences
    if not allowed:
        return {
            "scenario_id": scenario.id,
            "run_id": record.run_id,
            "status": "not_applicable",
            "reason": "scenario has no exact worker trajectory",
        }
    candidate_results = []
    for expected_steps in allowed:
        result = graph_trajectory_strict_match(
            outputs={"results": [], "steps": [record.worker_sequence]},
            reference_outputs={"results": [], "steps": [expected_steps]},
        )
        candidate_results.append(result)
    matched = any(bool(result.get("score")) for result in candidate_results)
    return {
        "scenario_id": scenario.id,
        "run_id": record.run_id,
        "status": "pass" if matched else "fail",
        "metric": "graph_trajectory_strict_match",
        "score": matched,
        "gating": False,
        "candidate_results": candidate_results,
    }


def evaluate_with_openevals_judge(
    scenario: RuntimeScenario,
    record: RuntimeRecord,
    dimension: str,
    *,
    judge: Any,
    judge_provider: str,
    judge_model: str,
    configured_output_token_limit: int | None = None,
    evaluator_factory: Any | None = None,
    structured_output_method: Literal[
        "function_calling", "json_schema", "json_mode"
    ] = "function_calling",
) -> SemanticJudgeResult:
    """Runs one optional semantic dimension using only captured runtime context."""
    rubric = SEMANTIC_JUDGE_RUBRICS.get(dimension)
    if rubric is None:
        supported = ", ".join(sorted(SEMANTIC_JUDGE_RUBRICS))
        raise ValueError(
            f"unsupported semantic judge dimension {dimension!r}; supported: {supported}"
        )
    if dimension not in scenario.expectations.judge_dimensions:
        raise ValueError(
            f"scenario {scenario.id!r} does not opt into judge dimension {dimension!r}"
        )

    def error(
        reason: str,
        *,
        status: Literal["skipped", "error"] = "error",
        provider_call_attempted: bool = False,
        exc: Exception | None = None,
        raw: Any = None,
    ) -> SemanticJudgeResult:
        error_types = _exception_type_chain(exc) if exc is not None else []
        return SemanticJudgeResult(
            scenario_id=scenario.id,
            scenario_version=scenario.version,
            run_id=record.run_id,
            repetition=record.repetition,
            dimension=dimension,
            judge_provider=judge_provider,
            judge_model=judge_model,
            status=status,
            reason=reason,
            provider_call_attempted=provider_call_attempted,
            error_type=error_types[0] if error_types else "PrerequisiteUnavailable",
            error_cause_types=error_types[1:],
            raw_output_omitted=provider_call_attempted,
            configured_output_token_limit=configured_output_token_limit,
            **_judge_diagnostics(exc=exc, raw=raw, attempted=provider_call_attempted),
            evidence_ids=record.evidence_ids,
        )

    required_worker = SEMANTIC_JUDGE_REQUIRED_FINAL_WORKERS[dimension]
    actual_worker = record.worker_sequence[-1] if record.worker_sequence else None
    if actual_worker != required_worker:
        return error(
            f"required final worker {required_worker!r} was not observed; "
            f"actual final worker={actual_worker!r}; judge call was not attempted",
            status="skipped",
        )
    if not record.final_response.strip() and dimension != "graph_usefulness":
        return error(
            "final response is absent; judge call was not attempted", status="skipped"
        )
    if dimension == "claim_faithfulness":
        if record.evidence_capture_status != "captured" or not record.bounded_evidence:
            return error(
                "bounded evidence is unavailable; judge call was not attempted",
                status="skipped",
            )
        raw_response: Any = None
        schema_instruction = _structured_output_instruction(FaithfulnessAssessment)
        try:
            if evaluator_factory is not None:
                raw_assessment = evaluator_factory(
                    rubric=rubric,
                    judge=judge,
                    schema=FaithfulnessAssessment,
                )(
                    request=record.input.user_message,
                    response=record.final_response,
                    evidence=[
                        item.model_dump(mode="json") for item in record.bounded_evidence
                    ],
                )
                raw_response = raw_assessment
            else:
                from langchain_core.messages import HumanMessage, SystemMessage

                structured_judge = judge.with_structured_output(
                    FaithfulnessAssessment,
                    method=structured_output_method,
                    include_raw=True,
                )
                envelope = structured_judge.invoke(
                    [
                        SystemMessage(
                            content=(
                                "Evaluate claim faithfulness using only the supplied "
                                "bounded evidence. Do not use external knowledge. "
                                + rubric
                                + " Return compact results: at most 32 atomic claims; "
                                "claim text at most 400 characters; at most five evidence "
                                "IDs and a 240-character reason per claim. "
                                "Use only supported, partial, or unsupported verdicts. "
                                + schema_instruction
                            )
                        ),
                        HumanMessage(
                            content=json.dumps(
                                {
                                    "request": record.input.user_message,
                                    "response": record.final_response,
                                    "evidence": [
                                        item.model_dump(mode="json")
                                        for item in record.bounded_evidence
                                    ],
                                },
                                ensure_ascii=False,
                            )
                        ),
                    ]
                )
                if not isinstance(envelope, dict):
                    raise TypeError("structured judge returned a non-mapping envelope")
                raw_response = envelope.get("raw")
                parsing_error = envelope.get("parsing_error")
                if parsing_error is not None:
                    return error(
                        _safe_judge_failure_reason(parsing_error),
                        provider_call_attempted=True,
                        exc=parsing_error,
                        raw=raw_response,
                    )
                raw_assessment = envelope.get("parsed")
            assessment = FaithfulnessAssessment.model_validate(raw_assessment)
        except Exception as exc:
            return error(
                _safe_judge_failure_reason(exc),
                provider_call_attempted=True,
                exc=exc,
                raw=raw_response,
            )
        claims = assessment.claims
        supported = sum(item.verdict == "supported" for item in claims)
        partial = sum(item.verdict == "partial" for item in claims)
        unsupported = sum(item.verdict == "unsupported" for item in claims)
        total = len(claims)
        grounded_rate = (supported + 0.5 * partial) / total if total else None
        return SemanticJudgeResult(
            scenario_id=scenario.id,
            scenario_version=scenario.version,
            run_id=record.run_id,
            repetition=record.repetition,
            dimension=dimension,
            judge_provider=judge_provider,
            judge_model=judge_model,
            status="success",
            score=grounded_rate,
            reason="Claim-level faithfulness assessment.",
            provider_call_attempted=True,
            raw_output_omitted=True,
            configured_output_token_limit=configured_output_token_limit,
            **_judge_diagnostics(exc=None, raw=raw_response, attempted=True),
            evidence_ids=record.evidence_ids,
            supported_claims=supported,
            partial_claims=partial,
            unsupported_claims=unsupported,
            total_claims=total,
            grounded_claim_rate=grounded_rate,
            claims=claims,
        )

    if dimension == "graph_usefulness" and record.graph.capture_status != "captured":
        return error(
            "bounded graph labels are unavailable; judge call was not attempted",
            status="skipped",
        )
    if evaluator_factory is None:
        try:
            from openevals.llm import create_llm_as_judge
        except ImportError as exc:
            raise RuntimeError(
                "OpenEvals is optional; install the evaluation dependency group first"
            ) from exc
        evaluator_factory = create_llm_as_judge
        judge = StructuredOutputMethodJudge(
            wrapped=judge,
            structured_output_method=structured_output_method,
        )

    prompt = f"""Evaluate only the named dimension for this Graph RAG response.
Do not use external knowledge. Judge only the supplied request, recent conversation
context, runtime outcome, response, and bounded graph when present.

<dimension>{dimension}</dimension>
<rubric>{rubric}</rubric>
<required_output_schema>
{{"score": true_or_false, "reason": "concise explanation, at most 320 characters"}}
</required_output_schema>
<inputs>{{inputs}}</inputs>
<outputs>{{outputs}}</outputs>
"""
    evaluator = evaluator_factory(
        prompt=prompt,
        feedback_key=dimension,
        judge=judge,
        continuous=False,
        use_reasoning=True,
    )
    try:
        raw_result = evaluator(
            inputs={
                "request": record.input.user_message,
                "recent_history": record.input.message_history,
                "session_summary": record.input.session_summary,
            },
            outputs={
                "response": record.final_response,
                "worker_sequence": record.worker_sequence,
                "request_scope": record.request_scope,
                "output_modality": record.output_modality,
                "evidence_ids": record.evidence_ids,
                "graph": {
                    "node_labels": record.graph.node_labels,
                    "edges": record.graph.labeled_edges,
                },
            },
        )
    except Exception as exc:
        return error(
            _safe_judge_failure_reason(exc),
            provider_call_attempted=True,
            exc=exc,
        )
    if not isinstance(raw_result, dict):
        exc = TypeError("OpenEvals judge returned a non-mapping result")
        return error(
            _safe_judge_failure_reason(exc),
            provider_call_attempted=True,
            exc=exc,
        )
    return SemanticJudgeResult(
        scenario_id=scenario.id,
        scenario_version=scenario.version,
        run_id=record.run_id,
        repetition=record.repetition,
        dimension=dimension,
        judge_provider=judge_provider,
        judge_model=judge_model,
        status="success",
        score=raw_result.get("score"),
        reason=str(raw_result.get("comment") or ""),
        provider_call_attempted=True,
        raw_output_omitted=True,
        configured_output_token_limit=configured_output_token_limit,
        **_judge_diagnostics(exc=None, raw=raw_result, attempted=True),
        evidence_ids=record.evidence_ids,
    )


def _structured_output_instruction(schema: type[BaseModel]) -> str:
    """Returns an explicit compact schema instruction, including for JSON mode."""
    return (
        "Return exactly one JSON object matching this schema; do not add prose or "
        "Markdown: "
        + json.dumps(
            schema.model_json_schema(), ensure_ascii=False, separators=(",", ":")
        )
    )


def _judge_diagnostics(
    *, exc: Exception | None, raw: Any, attempted: bool
) -> dict[str, Any]:
    """Extracts privacy-safe diagnostics without retaining model output."""
    if not attempted:
        return {}
    parser = _judge_parser_classification(exc)
    stop_reason, output_present, output_size, raw_http_status = (
        _judge_output_observation(raw)
    )
    http_status = raw_http_status or _exception_http_status(exc)
    transport_status = (
        "completed" if raw is not None or parser != "provider_failure" else "failed"
    )
    normalized_stop = stop_reason.casefold().replace("-", "_")
    confirmed_truncation = normalized_stop in {
        "length",
        "max_tokens",
        "max_token",
        "token_limit",
    }
    return {
        "transport_status": transport_status,
        "http_status_code": http_status,
        "stop_reason": stop_reason,
        "output_present": output_present,
        "output_size_chars": output_size,
        "parser_classification": parser,
        "confirmed_truncation": confirmed_truncation,
    }


def _judge_output_observation(raw: Any) -> tuple[str, bool, int | None, int | None]:
    metadata = getattr(raw, "response_metadata", {}) if raw is not None else {}
    if not isinstance(metadata, dict):
        metadata = {}
    stop_reason = str(
        metadata.get("done_reason")
        or metadata.get("finish_reason")
        or metadata.get("stop_reason")
        or ""
    )
    content = getattr(raw, "content", None) if raw is not None else None
    if content is None and isinstance(raw, dict):
        content = raw
    if isinstance(content, str):
        output_size = len(content)
        output_present = bool(content)
    elif content is not None:
        try:
            output_size = len(json.dumps(content, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            output_size = None
        output_present = bool(content)
    else:
        output_size = None
        output_present = False
    return (
        stop_reason,
        output_present,
        output_size,
        _first_int(metadata.get("status_code")),
    )


def _judge_parser_classification(exc: Exception | None) -> str:
    chain = _exception_type_chain(exc) if exc is not None else []
    if "JSONDecodeError" in chain:
        return "malformed_json"
    if "ValidationError" in chain:
        return "schema_validation_failure"
    if "OutputParserException" in chain:
        return "structured_output_parse_failure"
    return "provider_failure" if exc is not None else "success"


def _exception_http_status(exc: Exception | None) -> int | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status = _first_int(
            getattr(current, "status_code", None),
            getattr(response, "status_code", None),
        )
        if status is not None:
            return status
        current = current.__cause__ or current.__context__
    return None


def _exception_type_chain(exc: Exception) -> list[str]:
    """Returns exception class names without persisting private model output."""
    names: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        names.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    return names


def _safe_judge_failure_reason(exc: Exception) -> str:
    """Classifies judge failures while deliberately omitting provider output."""
    chain = " -> ".join(_exception_type_chain(exc))
    if "OutputParserException" in chain or "ValidationError" in chain:
        category = "structured judge response could not be parsed or validated"
    else:
        category = "judge invocation failed"
    return f"{category}; error chain={chain}; raw model output omitted"


def _provider_observation(run: dict[str, Any]) -> ProviderObservation:
    kwargs = _llm_message_kwargs(run)
    response_metadata = kwargs.get("response_metadata", {})
    if not isinstance(response_metadata, dict):
        response_metadata = {}
    metadata = run.get("extra", {}).get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    content = kwargs.get("content")
    tool_calls = kwargs.get("tool_calls")
    return ProviderObservation(
        run_id=str(run.get("_id") or ""),
        node=str(metadata.get("langgraph_node") or run.get("name") or ""),
        provider=str(
            metadata.get("ls_provider") or response_metadata.get("model_provider") or ""
        ),
        model=str(
            metadata.get("ls_model_name") or response_metadata.get("model_name") or ""
        ),
        status=str(run.get("status") or "unknown"),
        stop_reason=str(response_metadata.get("done_reason") or ""),
        input_tokens=_int_or_zero(run.get("prompt_tokens")),
        output_tokens=_int_or_zero(run.get("completion_tokens")),
        content_present=bool(str(content or "").strip()),
        tool_call_count=len(tool_calls) if isinstance(tool_calls, list) else 0,
        configured_output_token_limit=_first_int(
            metadata.get("ls_max_tokens"),
            metadata.get("max_tokens"),
            metadata.get("num_predict"),
        ),
    )


def _tool_observation(run: dict[str, Any]) -> ToolObservation:
    metadata = run.get("extra", {}).get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    raw_args = run.get("inputs") if isinstance(run.get("inputs"), dict) else {}
    output = (
        run.get("outputs", {}).get("output")
        if isinstance(run.get("outputs"), dict)
        else None
    )
    name = str(run.get("name") or "")
    safe_args = _redact_mapping(raw_args)
    return ToolObservation(
        run_id=str(run.get("_id") or ""),
        name=name,
        node=str(metadata.get("langgraph_node") or ""),
        arguments=safe_args,
        status=str(run.get("status") or "unknown"),
        evidence_ids=sorted(_find_evidence_ids(output)),
        bounded_evidence=_bounded_evidence_from_tool_output(
            output, tool_name=name, arguments=safe_args
        ),
    )


def _graph_labels_from_artifacts(
    artifacts: Any,
) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    """Extracts bounded display labels and relations from captured Plotly figures."""
    if not isinstance(artifacts, list):
        return {}, []
    labels: list[str] = []
    labeled_edges: list[tuple[str, str, str]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("data"), list):
            continue
        for trace in artifact["data"]:
            if not isinstance(trace, dict):
                continue
            mode = str(trace.get("mode") or "")
            text = trace.get("text")
            if "text" in mode and isinstance(text, (list, tuple)):
                labels.extend(
                    str(value)[:200] for value in text if str(value or "").strip()
                )
            hovertext = trace.get("hovertext")
            if isinstance(hovertext, (list, tuple)):
                for value in hovertext:
                    parts = [part.strip() for part in str(value).split(" → ", 2)]
                    if len(parts) == 3 and all(parts):
                        labeled_edges.append(tuple(parts))
    node_labels = {
        f"artifact_node_{index}": label
        for index, label in enumerate(dict.fromkeys(labels))
    }
    return (
        dict(list(node_labels.items())[:GRAPH_NODE_LIMIT]),
        list(dict.fromkeys(labeled_edges))[:GRAPH_EDGE_LIMIT],
    )


def _graph_observation(
    root_outputs: dict[str, Any],
    tools: list[ToolObservation],
    runs: list[dict[str, Any]],
) -> GraphObservation:
    artifacts = root_outputs.get("visual_artifacts")
    artifact_count = len(artifacts) if isinstance(artifacts, list) else 0
    node_ids: list[str] = []
    edges: list[tuple[str, str, str]] = []
    node_labels: dict[str, str] = {}
    tool_run_ids = {
        tool.run_id for tool in tools if tool.name == "get_subgraphs_to_visualize"
    }
    for run in runs:
        if str(run.get("_id") or "") not in tool_run_ids:
            continue
        output = (
            run.get("outputs", {}).get("output")
            if isinstance(run.get("outputs"), dict)
            else None
        )
        if not isinstance(output, (list, tuple)) or len(output) < 2:
            continue
        raw_nodes, raw_edges = output[0], output[1]
        if isinstance(raw_nodes, list):
            for node in raw_nodes[:GRAPH_NODE_LIMIT]:
                if isinstance(node, dict):
                    node_id = str(node.get("id") or node.get("node_id") or "")
                    label = str(node.get("label") or node.get("name") or "")
                    if node_id:
                        node_ids.append(node_id)
                        if label:
                            node_labels[node_id] = label[:200]
                else:
                    node_ids.append(str(node))
        if isinstance(raw_edges, list):
            for edge in raw_edges[:GRAPH_EDGE_LIMIT]:
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
        if len(output) > 3 and isinstance(output[3], list):
            for detail in output[3][:GRAPH_NODE_LIMIT]:
                if isinstance(detail, dict):
                    node_id = str(detail.get("id") or "")
                    label = str(detail.get("label") or detail.get("name") or "")
                    if node_id and label:
                        node_labels[node_id] = label[:200]
    node_ids = list(dict.fromkeys(node_ids))
    node_set = set(node_ids)
    dangling = sum(
        subject not in node_set or object_ not in node_set
        for subject, _, object_ in edges
    )
    labeled_edges = [
        (
            node_labels.get(subject, subject),
            predicate,
            node_labels.get(object_, object_),
        )
        for subject, predicate, object_ in edges[:GRAPH_EDGE_LIMIT]
        if subject in node_labels and object_ in node_labels
    ]
    artifact_labels, artifact_edges = _graph_labels_from_artifacts(artifacts)
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
        node_ids=node_ids[:GRAPH_NODE_LIMIT],
        edges=edges[:GRAPH_EDGE_LIMIT],
        dangling_edge_count=dangling,
        node_labels=dict(list(node_labels.items())[:GRAPH_NODE_LIMIT]),
        labeled_edges=labeled_edges,
        capture_status=capture_status,
    )


def _retrieval_outcome(
    statuses: list[str],
    tools: list[ToolObservation],
    providers: list[ProviderObservation],
) -> str:
    if not statuses:
        return "not_run"
    status = statuses[-1]
    if status == "adequate":
        return "adequate"
    retriever_providers = [
        provider for provider in providers if provider.node == "retriever"
    ]
    if any(
        provider.stop_reason == "length"
        and not provider.content_present
        and provider.tool_call_count == 0
        for provider in retriever_providers
    ):
        return "invalid_model_output"
    local_tools = {
        "search_knowledge_base",
        "get_subgraphs_to_visualize",
    }
    if status == "no_results" and any(tool.name in local_tools for tool in tools):
        return "true_empty"
    if status == "no_results":
        return "no_tool_call"
    return status


def _failure_observations(
    runs: list[dict[str, Any]],
    providers: list[ProviderObservation],
    *,
    retrieval_outcome: str,
    sources_exhausted: bool,
) -> list[FailureObservation]:
    failures: list[FailureObservation] = []
    for provider in providers:
        if provider.stop_reason == "length":
            code = (
                "model_output_truncated"
                if not provider.content_present and provider.tool_call_count == 0
                else "model_output_length_limit"
            )
            failures.append(
                FailureObservation(
                    code=code, node=provider.node, run_id=provider.run_id
                )
            )
        if provider.status == "error":
            failures.append(
                FailureObservation(
                    code="provider_error", node=provider.node, run_id=provider.run_id
                )
            )
    for run in runs:
        if run.get("status") != "error":
            continue
        name = str(run.get("name") or "")
        error = str(run.get("error") or "").casefold()
        metadata = run.get("extra", {}).get("metadata", {})
        node = (
            str(metadata.get("langgraph_node") or name)
            if isinstance(metadata, dict)
            else name
        )
        if (
            name == "PydanticOutputParser"
            or "outputparser" in error
            or "invalid json output" in error
        ):
            code = "structured_output_parse_error"
        elif any(
            term in error
            for term in (
                "redis",
                "pinecone",
                "neo4j",
                "memgraph",
                "mongodb",
                "storage",
            )
        ):
            code = "storage_error"
        elif "timeout" in error:
            code = "timeout"
        elif "recursion" in error:
            code = "recursion_limit"
        elif run.get("run_type") == "tool":
            code = "tool_error"
        elif name == "LangGraph":
            code = "workflow_error"
        else:
            code = "run_error"
        failures.append(
            FailureObservation(code=code, node=node, run_id=str(run.get("_id") or ""))
        )
    if retrieval_outcome == "invalid_model_output":
        failures.append(
            FailureObservation(code="invalid_retriever_output", node="retriever")
        )
    if retrieval_outcome == "no_tool_call":
        failures.append(
            FailureObservation(code="retriever_no_tool_call", node="retriever")
        )
    if sources_exhausted:
        failures.append(FailureObservation(code="sources_exhausted"))
    unique: dict[tuple[str, str, str], FailureObservation] = {}
    for failure in failures:
        unique[(failure.code, failure.node, failure.run_id)] = failure
    return list(unique.values())


def _messages(raw_messages: Any) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list):
        return []
    messages: list[dict[str, str]] = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        kwargs = raw.get("kwargs") if isinstance(raw.get("kwargs"), dict) else raw
        message_type = str(raw.get("type") or kwargs.get("type") or "")
        role = "user" if message_type in {"human", "user"} else "assistant"
        content = kwargs.get("content", raw.get("content", ""))
        messages.append({"role": role, "content": str(content or "")})
    return messages


def _final_response(root_inputs: dict[str, Any], root_outputs: dict[str, Any]) -> str:
    input_messages = _messages(root_inputs.get("messages"))
    output_raw = root_outputs.get("messages")
    if not isinstance(output_raw, list):
        return ""
    for raw in reversed(output_raw[len(input_messages) :]):
        if not isinstance(raw, dict):
            continue
        kwargs = raw.get("kwargs") if isinstance(raw.get("kwargs"), dict) else raw
        additional = kwargs.get("additional_kwargs", raw.get("additional_kwargs", {}))
        agent = (
            str(additional.get("agent") or "") if isinstance(additional, dict) else ""
        )
        if agent in {"[ANALYST]", "[MENTOR]"}:
            return str(kwargs.get("content", raw.get("content", "")) or "")
    return ""


def _llm_message_kwargs(run: dict[str, Any]) -> dict[str, Any]:
    try:
        kwargs = run["outputs"]["generations"][0][0]["message"]["kwargs"]
    except (KeyError, IndexError, TypeError):
        return {}
    return kwargs if isinstance(kwargs, dict) else {}


def _bounded_evidence_from_tool_output(
    output: Any, *, tool_name: str, arguments: dict[str, Any]
) -> list[EvidenceObservation]:
    """Captures a small, redacted evidence view directly from a tool result."""
    safe_output = _redact_value(output)
    try:
        rendered = (
            safe_output
            if isinstance(safe_output, str)
            else json.dumps(safe_output, ensure_ascii=False, default=str)
        )
    except (TypeError, ValueError):
        rendered = str(safe_output)
    rendered = re.sub(r"\s+", " ", rendered).strip()
    if not rendered:
        return []

    query_value = arguments.get("topic") or arguments.get("queries") or ""
    if isinstance(query_value, list):
        query = " | ".join(str(item) for item in query_value)
    else:
        query = str(query_value)
    evidence_ids = sorted(_find_evidence_ids(safe_output))
    if not evidence_ids and tool_name == "deep_web_research":
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]
        evidence_ids = [f"web_{digest}"]
    if not evidence_ids:
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]
        evidence_ids = [f"tool_{digest}"]

    observations: list[EvidenceObservation] = []
    for rank, evidence_id in enumerate(evidence_ids[:EVIDENCE_ITEM_LIMIT], start=1):
        kind: Literal["local_chunk", "local_relation", "web_result", "tool_output"]
        if evidence_id.startswith("chunk_"):
            kind = "local_chunk"
        elif evidence_id.startswith("rel_"):
            kind = "local_relation"
        elif evidence_id.startswith("web_") or tool_name == "deep_web_research":
            kind = "web_result"
        else:
            kind = "tool_output"
        position = rendered.find(evidence_id)
        start = max(0, position - 250) if position >= 0 else 0
        excerpt = rendered[start : start + EVIDENCE_EXCERPT_LIMIT]
        score_match = re.search(r"\bScore:\s*([0-9.]+)", excerpt)
        observations.append(
            EvidenceObservation(
                evidence_id=evidence_id,
                kind=kind,
                query=query,
                rank=rank,
                score=_float_or_none(score_match.group(1)) if score_match else None,
                excerpt=excerpt,
            )
        )
    return observations


def _merge_bounded_evidence(
    items: Any,
) -> list[EvidenceObservation]:
    merged: list[EvidenceObservation] = []
    seen: set[str] = set()
    total = 0
    for item in items:
        if item.evidence_id in seen or len(merged) >= EVIDENCE_ITEM_LIMIT:
            continue
        remaining = EVIDENCE_TOTAL_LIMIT - total
        if remaining <= 0:
            break
        excerpt = item.excerpt[: min(EVIDENCE_EXCERPT_LIMIT, remaining)]
        merged.append(item.model_copy(update={"excerpt": excerpt}))
        seen.add(item.evidence_id)
        total += len(excerpt)
    return merged


def _action_signatures(
    worker_sequence: list[str], tools: list[ToolObservation]
) -> list[str]:
    remaining = list(tools)
    signatures: list[str] = []
    for worker in worker_sequence:
        tool = next((item for item in remaining if item.node == worker), None)
        if tool is None:
            signatures.append(worker)
            continue
        remaining.remove(tool)
        args = json.dumps(
            tool.arguments, sort_keys=True, ensure_ascii=False, default=str
        )
        digest = hashlib.sha256(args.encode("utf-8")).hexdigest()[:12]
        signatures.append(f"{worker}:{tool.name}:{digest}")
    signatures.extend(
        f"{tool.node or 'unknown'}:{tool.name}:"
        f"{hashlib.sha256(json.dumps(tool.arguments, sort_keys=True, default=str).encode()).hexdigest()[:12]}"
        for tool in remaining
    )
    return signatures


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _redact_mapping(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    return value


def _find_evidence_ids(value: Any) -> set[str]:
    if isinstance(value, str):
        return set(EVIDENCE_ID_RE.findall(value))
    if isinstance(value, dict):
        return (
            set().union(*(_find_evidence_ids(item) for item in value.values()))
            if value
            else set()
        )
    if isinstance(value, (list, tuple)):
        return (
            set().union(*(_find_evidence_ids(item) for item in value))
            if value
            else set()
        )
    return set()


def _redact_mapping(value: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key)
        if any(part in key_text.casefold() for part in SECRET_KEY_PARTS):
            redacted[key_text] = "[REDACTED]"
        elif isinstance(item, dict):
            redacted[key_text] = _redact_mapping(item)
        elif isinstance(item, list):
            redacted[key_text] = [
                _redact_mapping(entry) if isinstance(entry, dict) else entry
                for entry in item
            ]
        else:
            redacted[key_text] = item
    return redacted


def _check_bound(
    checks: list[CheckResult],
    dimension: str,
    name: str,
    actual: int | float | None,
    threshold: int | float | None,
    *,
    minimum: bool,
    gating: bool = True,
) -> None:
    if threshold is None:
        return
    observed = actual is not None
    passed = bool(
        observed and (actual >= threshold if minimum else actual <= threshold)
    )
    relation = ">=" if minimum else "<="
    checks.append(
        CheckResult(
            dimension=dimension,
            name=name,
            status="pass" if passed else ("fail" if observed else "not_observed"),
            reason=f"actual={actual}; expected {relation} {threshold}",
            gating=gating,
        )
    )


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    z = 1.96
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = proportion + z**2 / (2 * total)
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    )
    return (
        max(0.0, (centre - margin) / denominator),
        min(1.0, (centre + margin) / denominator),
    )


def _distribution(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    ordered = sorted(float(value) for value in values)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _write_jsonl(path: Path, models: list[BaseModel]) -> None:
    lines = [model.model_dump_json(exclude_none=True) for model in models]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _summary_markdown(
    summary: dict[str, Any],
    results: list[RuntimeCaseResult],
    framework_results: list[dict[str, Any]] | None,
    judge_results: list[SemanticJudgeResult] | None,
    judge_provider_calls: int,
) -> str:
    judge_successes = sum(item.status == "success" for item in judge_results or [])
    judge_errors = sum(item.status == "error" for item in judge_results or [])
    judge_skipped = sum(item.status == "skipped" for item in judge_results or [])
    lines = [
        "# Runtime Graph RAG Evaluation Summary",
        "",
        f"- Records: {summary['record_count']}",
        f"- Deterministic passes: {summary['passed_count']}",
        f"- Deterministic failures: {summary['failed_count']}",
        f"- Judge calls: {judge_provider_calls}",
        f"- Successful judge results: {judge_successes}",
        f"- Skipped judge results: {judge_skipped}",
        f"- Judge execution errors: {judge_errors}",
        "",
        "## Per-run results",
        "",
        "| Scenario | Run | Result | Failure types |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        failures = ", ".join(result.failure_types) or "none"
        lines.append(
            f"| `{result.scenario_id}` | `{result.run_id}` | "
            f"{'PASS' if result.passed else 'FAIL'} | {failures} |"
        )
    lines.extend(
        [
            "",
            "## Checks by dimension",
            "",
            "| Dimension | Pass | Fail | Not observed |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for dimension, counts in summary["dimensions"].items():
        lines.append(
            f"| {dimension} | {counts.get('pass', 0)} | {counts.get('fail', 0)} | "
            f"{counts.get('not_observed', 0)} |"
        )
    if judge_results is not None:
        lines.extend(
            [
                "",
                "## Optional semantic judge",
                "",
                "Judge dimensions remain separate from deterministic gates.",
                "",
                "| Scenario | Dimension | Status | Score | Reason |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for item in judge_results:
            reason = item.reason.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {item.scenario_id} | {item.dimension} | {item.status} | "
                f"{item.score if item.score is not None else 'n/a'} | {reason} |"
            )
    if framework_results is not None:
        lines.extend(
            [
                "",
                "## Optional AgentEvals cross-check",
                "",
                "These strict trajectory results are inspectable but non-gating.",
                "",
                "| Scenario | Status |",
                "| --- | --- |",
            ]
        )
        lines.extend(
            f"| `{item['scenario_id']}` | {item['status']} |"
            for item in framework_results
        )
    return "\n".join(lines) + "\n"


def _first_int(*values: Any) -> int | None:
    for value in values:
        try:
            if value is not None:
                return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return None


def _int_or_zero(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
