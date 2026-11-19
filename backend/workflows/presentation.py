"""Shared presentation records for Reflex state, quick-action cache and snapshots."""

import json
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StringConstraints,
    field_validator,
    model_validator,
)

from backend.configs.constants import TITLE_MAX_LENGTH

Identity = Annotated[
    str, StringConstraints(pattern=r"^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$")
]
MAX_RECENT_MAPS = 10
MAX_GRAPH_SOURCES = 128


class PresentationRecord(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class TurnIdentity(PresentationRecord):
    session_id: Identity
    turn_id: Identity


StageOutcome = Literal[
    "running", "completed", "failed", "interrupted", "cancelled", "incomplete"
]
TurnOutcome = Literal[
    "running",
    "answer",
    "graph",
    "out_of_scope",
    "no_results",
    "empty",
    "error",
    "interrupted",
    "cancelled",
    "incomplete",
]


class ExecutionStage(TurnIdentity):
    """One observed LangGraph node invocation. Completion describes its boundary."""

    invocation_id: Annotated[str, StringConstraints(min_length=1)]
    sequence: int = Field(ge=0)
    agent: str
    start_offset: float = Field(ge=0, allow_inf_nan=False)
    end_offset: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    outcome: StageOutcome = "running"

    @property
    def duration_seconds(self) -> float | None:
        return None if self.end_offset is None else self.end_offset - self.start_offset

    @model_validator(mode="after")
    def validate_boundary(self) -> Self:
        if (self.end_offset is None) != (self.outcome == "running"):
            raise ValueError("Stage outcome and end boundary disagree")
        if self.end_offset is not None and self.end_offset < self.start_offset:
            raise ValueError("Stage end precedes its start")
        return self


class TurnExecution(TurnIdentity):
    """Presentation history for an admitted turn, including cache replays."""

    origin: Literal["workflow", "cache"]
    outcome: TurnOutcome = "running"
    stages: list[ExecutionStage] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_stages(self) -> Self:
        if (
            (self.origin == "cache" and self.stages)
            or len({stage.invocation_id for stage in self.stages}) != len(self.stages)
            or any(
                (stage.session_id, stage.turn_id, stage.sequence)
                != (self.session_id, self.turn_id, index)
                or (self.outcome != "running" and stage.outcome == "running")
                for index, stage in enumerate(self.stages)
            )
        ):
            raise ValueError("Invalid turn execution stages")
        return self


def validated_web_url(value: str | None) -> str | None:
    if (
        not value
        or len(value) > 2048
        or any(char.isspace() or ord(char) < 32 or char in '\\<>"' for char in value)
    ):
        return None
    try:
        url = urlsplit(value)
        if url.port == 0:
            return None
        if url.scheme in {"http", "https"} and url.hostname and url.username is None:
            return value
    except ValueError:
        pass
    return None


def stored_string(value: object) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def stored_strings(value: object) -> list[str]:
    values = value if isinstance(value, (list, tuple, set)) else (value,)
    return [item for value in values if (item := stored_string(value)) is not None]


def note_source_id(chunk_id: str) -> str:
    return f"note:{chunk_id}"


class ResponseSource(PresentationRecord):
    """An admitted context source, with missing provenance represented by nulls."""

    kind: Literal["note", "web"]
    source_id: str | None = None
    chunk_id: str | None = None
    note_id: str | None = None
    title: str | None = None
    path: list[str] = Field(default_factory=list)
    excerpt: str | None = Field(default=None, max_length=500)
    summary: str | None = None
    retrieval_score: FiniteFloat | None = None
    supports_relationship: bool = False
    url: str | None = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, url: str | None) -> str | None:
        if url is not None and validated_web_url(url) is None:
            raise ValueError("Invalid web source URL")
        return url


class ResponseEvidence(PresentationRecord):
    """Sources and concept identities associated with one response's context."""

    sources: list[ResponseSource] = Field(default_factory=list)
    concept_ids: list[str] = Field(default_factory=list)
    has_relationship_context: bool = False


class Message(TurnIdentity):
    """A user prompt or a finalized assistant response with a separate DOM target."""

    role: Literal["user", "assistant"]
    content: str
    agent: str
    timestamp: str
    dom_id: Annotated[str, StringConstraints(pattern=r"^message-[0-9a-f]{32}$")]
    response_id: Identity | None = None
    evidence: ResponseEvidence | None = None
    invocation_id: str | None = None

    @model_validator(mode="after")
    def validate_role(self) -> Self:
        if (self.role == "assistant") != (self.response_id is not None):
            raise ValueError("Only assistant messages have response IDs")
        if self.role == "user" and (
            self.evidence is not None or self.invocation_id is not None
        ):
            raise ValueError(
                "Only assistant messages have response evidence and invocation IDs"
            )
        return self


class GraphNode(PresentationRecord):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    type: str | None = None
    description: str | None = None
    source_ids: list[str] = Field(default_factory=list)


class GraphEdge(PresentationRecord):
    id: Identity
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    evidence_source_ids: list[str] = Field(default_factory=list)
    relation_id: str | None = None
    synthetic: bool

    @classmethod
    def from_relationship(
        cls, source: str, predicate: str, target: str, **metadata
    ) -> Self:
        key = json.dumps([source, predicate, target], ensure_ascii=False)
        return cls(
            id=str(uuid5(NAMESPACE_URL, f"graph-relationship:{key}")),
            source=source,
            predicate=predicate,
            target=target,
            **metadata,
        )


class GraphPayload(PresentationRecord):
    """Versioned graph data independent of the interactive renderer."""

    schema_version: Literal[1] = 1
    artifact_id: Identity
    title: str
    retrieval_queries: list[str] = Field(
        default_factory=list, exclude_if=lambda queries: not queries
    )
    nodes: list[GraphNode] = Field(min_length=1, max_length=512)
    edges: list[GraphEdge] = Field(max_length=2048)
    sources: list[ResponseSource] = Field(
        default_factory=list, max_length=MAX_GRAPH_SOURCES
    )

    @model_validator(mode="after")
    def validate_topology(self) -> Self:
        node_ids = {node.id for node in self.nodes}
        if (
            len(node_ids) != len(self.nodes)
            or len({edge.id for edge in self.edges}) != len(self.edges)
            or any(
                edge.source not in node_ids or edge.target not in node_ids
                for edge in self.edges
            )
        ):
            raise ValueError("Invalid graph topology")
        source_ids = {source_id for node in self.nodes for source_id in node.source_ids}
        source_ids.update(
            source_id for edge in self.edges for source_id in edge.evidence_source_ids
        )
        supplied_ids = {source.source_id for source in self.sources}
        if len(supplied_ids) != len(self.sources) or not supplied_ids <= source_ids:
            raise ValueError("Invalid graph source association")
        return self


class GraphArtifact(GraphPayload, TurnIdentity):
    timestamp: str


class GraphToolResult(PresentationRecord):
    queries: list[str]
    graph: GraphPayload | None


def graph_title(queries: list[str]) -> str:
    title = " & ".join(queries)
    if len(title) > TITLE_MAX_LENGTH:
        title = title[:TITLE_MAX_LENGTH] + "…"
    return title.title()


def _validate_unique_ids(
    messages: list[Message], artifacts: list[GraphArtifact]
) -> None:
    dom_ids = {message.dom_id for message in messages}
    response_ids = [message.response_id for message in messages if message.response_id]
    if (
        len(dom_ids) != len(messages)
        or len(set(response_ids)) != len(response_ids)
        or len({artifact.artifact_id for artifact in artifacts}) != len(artifacts)
    ):
        raise ValueError("Duplicate presentation identity")


class SessionSnapshot(PresentationRecord):
    session_id: Identity
    messages: list[Message]
    artifacts: list[GraphArtifact]
    executions: list[TurnExecution] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_session(self) -> Self:
        _validate_unique_ids(self.messages, self.artifacts)
        turn_ids = set()
        for message in self.messages:
            if message.session_id != self.session_id:
                raise ValueError("Message belongs to another session")
            if message.role == "user":
                if message.turn_id in turn_ids:
                    raise ValueError("Duplicate turn")
                turn_ids.add(message.turn_id)
            elif message.turn_id not in turn_ids:
                raise ValueError("Response has no preceding user turn")
        if len(self.artifacts) > MAX_RECENT_MAPS or any(
            artifact.session_id != self.session_id or artifact.turn_id not in turn_ids
            for artifact in self.artifacts
        ):
            raise ValueError("Invalid session map history")
        if len({execution.turn_id for execution in self.executions}) != len(
            self.executions
        ) or any(
            execution.session_id != self.session_id
            or execution.turn_id not in turn_ids
            or execution.outcome == "running"
            for execution in self.executions
        ):
            raise ValueError("Invalid session execution history")
        return self


class CachedTurn(TurnIdentity):
    responses: list[Message]
    artifacts: list[GraphArtifact]
    agent_history: list[str]

    @model_validator(mode="after")
    def validate_turn(self) -> Self:
        _validate_unique_ids(self.responses, self.artifacts)
        if (
            not (self.responses or self.artifacts)
            or any(response.role != "assistant" for response in self.responses)
            or any(
                (record.session_id, record.turn_id) != (self.session_id, self.turn_id)
                for record in [*self.responses, *self.artifacts]
            )
        ):
            raise ValueError("Expected the outputs of one cached turn")
        return self

    def replay(self, destination: TurnIdentity, timestamp: str) -> Self:
        """Rebind session-owned identities while preserving response and graph data."""
        turn = {"session_id": destination.session_id, "turn_id": destination.turn_id}
        return type(self)(
            **turn,
            responses=[
                response.model_copy(
                    deep=True,
                    update={
                        **turn,
                        "response_id": str(uuid4()),
                        "invocation_id": None,
                        "dom_id": f"message-{uuid4().hex}",
                        "timestamp": timestamp,
                    },
                )
                for response in self.responses
            ],
            artifacts=[
                artifact.model_copy(
                    deep=True,
                    update={
                        **turn,
                        "artifact_id": str(uuid4()),
                        "timestamp": timestamp,
                    },
                )
                for artifact in self.artifacts
            ],
            agent_history=list(self.agent_history),
        )
