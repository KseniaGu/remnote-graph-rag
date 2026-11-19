import asyncio
from datetime import UTC, datetime, timedelta
from threading import Event
from types import SimpleNamespace
from uuid import uuid4

import pytest
from bson import BSON
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from pymongo.errors import DuplicateKeyError

from app import state as app_state
from app.strings import QUICK_ACTIONS
from backend.configs.chat_limits import ChatLimitsSettings
from backend.configs.constants import (
    CACHE_SCHEMA_VERSION,
    SESSION_MEMORY_SCHEMA_VERSION,
    VISUALIZATION_EMPTY_CONTEXT,
)
from backend.configs.enums import ModelRoleType, WorkflowEventType
from backend.utils.cache import QuickActionAnswerCache, normalize_quick_action_prompt
from backend.utils.chat_limits import ChatCooldownExceeded, ChatLimitService
from backend.utils.session_memory import SessionMemoryStore
from backend.workflows.agents.schemas import RoutingDecision, State
from backend.workflows.agents.tools import get_subgraphs_to_visualize
from backend.workflows.learner import LearnerWorkflow
from backend.workflows.learner_reflex import ReflexLearnerWorkflow, WorkflowEvent
from backend.workflows.presentation import (
    GraphArtifact,
    GraphEdge,
    GraphNode,
    GraphPayload,
    GraphToolResult,
    ResponseEvidence,
    ResponseSource,
)


def graph_payload():
    return GraphPayload(
        artifact_id=str(uuid4()),
        title="Attention map",
        nodes=[GraphNode(id="concept-a", label="Attention")],
        edges=[],
    ).model_dump(mode="json")


MAP_PROMPT = next(
    prompt for _, title, prompt in QUICK_ACTIONS if title == "Map relationships"
)


class Collection:
    """Round-trip documents through BSON without connecting to MongoDB."""

    def __init__(self):
        self.docs = {}
        self.fail = False
        self.reads = 0

    def find_one(self, query):
        self.reads += 1
        if self.fail:
            raise OSError("unavailable")
        doc = self.docs.get(query["_id"])
        return (
            doc
            if doc and all(doc.get(key) == val for key, val in query.items())
            else None
        )

    def replace_one(self, query, doc, *, upsert):
        if self.fail:
            raise OSError("unavailable")
        assert upsert and query["_id"] == doc["_id"]
        if query.get("cleared") == {"$ne": True} and self.docs.get(doc["_id"], {}).get(
            "cleared"
        ):
            raise DuplicateKeyError("term retired")
        self.docs[doc["_id"]] = BSON(BSON.encode(doc)).decode()

    def delete_one(self, query):
        if self.fail:
            raise OSError("unavailable")
        self.docs.pop(query["_id"], None)


@pytest.fixture
def stores():
    cache = QuickActionAnswerCache(settings=object())
    memory = SessionMemoryStore(settings=object())
    cache._collection = Collection()
    memory._collection = Collection()
    return cache, memory


@pytest.fixture
def records():
    from backend.workflows.presentation import (
        CachedTurn,
        Message,
        SessionSnapshot,
    )

    session_id, turn_id, response_id, artifact_id = (str(uuid4()) for _ in range(4))
    turn = {"session_id": session_id, "turn_id": turn_id}
    user = Message(
        **turn,
        role="user",
        agent="",
        content="Explain attention",
        timestamp="12:00",
        dom_id=f"message-{uuid4().hex}",
    )
    response = Message(
        **turn,
        role="assistant",
        agent="analyst",
        content="Attention answer",
        timestamp="12:01",
        dom_id=f"message-{uuid4().hex}",
        response_id=response_id,
    )
    artifact = GraphArtifact(
        **turn, **dict(graph_payload(), artifact_id=artifact_id), timestamp="12:01"
    )
    return (
        SessionSnapshot(
            session_id=session_id, messages=[user, response], artifacts=[artifact]
        ),
        CachedTurn(
            **turn,
            responses=[response],
            artifacts=[artifact],
            agent_history=["analyst"],
        ),
    )


def test_snapshot_round_trip_and_content_only_summary(stores, records):
    _, memory = stores
    snapshot, _ = records
    assert memory.replace_snapshot(snapshot, recent_limit=1)
    assert memory.get(snapshot.session_id) == snapshot
    assert (
        memory.get_prompt_summary(snapshot.session_id, "Summarize the previous topic")
        == "- user: Explain attention"
    )
    assert memory.get_prompt_summary(snapshot.session_id, "New question") == ""
    assert memory.get(str(uuid4())) is None


@pytest.mark.parametrize("store_name", ["cache", "session"])
@pytest.mark.parametrize(
    "damage",
    [
        "old-schema",
        "wrong-session",
        "expired",
        "malformed",
        "duplicate",
        "orphan-map",
        "malformed-map",
        "unrenderable-map",
    ],
)
def test_incompatible_or_expired_records_are_misses(
    stores, records, store_name, damage
):
    cache, memory = stores
    snapshot, cached_turn = records
    if store_name == "cache":
        store, key, schema = cache, "prompt", CACHE_SCHEMA_VERSION
        assert cache.set(key, cached_turn)
        records_key = "responses"
    else:
        store, key, schema = memory, snapshot.session_id, SESSION_MEMORY_SCHEMA_VERSION
        assert memory.replace_snapshot(snapshot)
        records_key = "messages"
    doc = store._collection.docs[key]
    if damage == "old-schema":
        doc["schema_version"] = schema - 1
    elif damage == "wrong-session":
        doc["presentation"][records_key][0]["session_id"] = str(uuid4())
    elif damage == "expired":
        doc["expires_at"] = datetime.now(UTC) - timedelta(seconds=1)
    elif damage == "malformed":
        doc["expires_at"] = "not a date"
    elif damage == "duplicate":
        doc["presentation"][records_key].append(doc["presentation"][records_key][0])
    elif damage == "orphan-map":
        doc["presentation"]["artifacts"][0]["turn_id"] = str(uuid4())
    elif damage == "malformed-map":
        doc["presentation"]["artifacts"][0]["nodes"] = []
    else:
        doc["presentation"]["artifacts"][0]["schema_version"] = 99
    assert store.get(key) is None
    assert store._collection.docs[key] is doc


def test_cache_hit_normalizes_key_and_supports_ttl_override(stores, records):
    cache, _ = stores
    _, cached_turn = records
    assert cache.get("example prompt") is None
    assert cache.set(" Example   Prompt ", cached_turn, ttl_seconds=42)
    doc = cache._collection.docs["example prompt"]
    assert doc["expires_at"] - doc["updated_at"] == timedelta(seconds=42)
    assert cache.get("EXAMPLE prompt") == cached_turn
    assert cache.set("example prompt", cached_turn, ttl_seconds=0)
    assert "expires_at" not in cache._collection.docs["example prompt"]
    assert cache.get("example prompt") == cached_turn
    assert not cache.set(
        "empty", cached_turn.model_copy(update={"responses": [], "artifacts": []})
    )


def test_disposable_graph_cutover_rehearsal(stores, application):
    cache, memory = stores
    calls, admissions = application
    visitor, old_session = (str(uuid4()) for _ in range(2))
    old_keys = [normalize_quick_action_prompt(MAP_PROMPT), old_session]
    old_docs = [
        {
            "_id": key,
            "schema_version": 4,
            "presentation": {"artifacts": [{"figure": {}}]},
        }
        for key in old_keys
    ]
    disposable = {
        "quick_action_answer_cache": cache._collection,
        "chat_session_memory": memory._collection,
    }
    for collection, document in zip(disposable.values(), old_docs, strict=True):
        collection.docs[document["_id"]] = document
    quota = {visitor: {"turns": 7}}
    protected = {
        "chat_limit_state": quota,
        "checkpoints": {"prior": {}},
        "knowledge_index": {"document": {}},
        "evaluation": {"run": {}},
    }
    baseline = BSON(BSON.encode(protected)).decode()

    def reset_disposable(environment):
        assert environment == "synthetic-stage5"
        inventory = {
            name: [
                key
                for key, document in collection.docs.items()
                if document.get("schema_version") == 4
            ]
            for name, collection in disposable.items()
        }
        for name, keys in inventory.items():
            for key in keys:
                disposable[name].delete_one({"_id": key})
        return inventory

    with pytest.raises(AssertionError):
        reset_disposable("production")
    assert [len(collection.docs) for collection in disposable.values()] == [1, 1]
    assert cache.get(MAP_PROMPT) is None and memory.get(old_session) is None
    assert reset_disposable("synthetic-stage5") == dict(
        zip(disposable, [[old_keys[0]], [old_session]], strict=True)
    )
    assert all(not collection.docs for collection in disposable.values())
    for collection, document in zip(disposable.values(), old_docs, strict=True):
        collection.docs[document["_id"]] = document
    assert cache.get(MAP_PROMPT) is None and memory.get(old_session) is None

    fresh = app_state.AppState(_reflex_internal_init=True)
    fresh.visitor_id = visitor
    asyncio.run(app_state.AppState.initialize_session.fn(fresh))
    assert fresh.session_id != old_session and not fresh.messages
    asyncio.run(fresh._process_message(MAP_PROMPT))
    assert calls == ["workflow"]
    assert fresh.selected_graph is not None
    assert admissions[0][0].visitor_id == visitor
    assert admissions[0][0].cache_hit is False

    restored = app_state.AppState(_reflex_internal_init=True)
    restored.session_id, restored.visitor_id = fresh.session_id, visitor
    asyncio.run(app_state.AppState.initialize_session.fn(restored))
    assert restored.selected_graph == fresh.selected_graph
    assert restored.visitor_id == visitor
    assert protected == baseline


def test_unavailable_stores_are_best_effort(stores, records):
    cache, memory = stores
    snapshot, cached_turn = records
    cache._collection.fail = memory._collection.fail = True
    assert cache.get("example") is None
    assert not cache.set("example", cached_turn)
    assert memory.get(snapshot.session_id) is None
    assert not memory.replace_snapshot(snapshot)
    assert not memory.delete(snapshot.session_id)


@pytest.fixture
def application(monkeypatch, stores):
    cache, memory = stores
    calls = []
    admissions = []

    class Admission(ChatLimitService):
        async def finish_turn(self, admission, status):
            admissions.append((admission.context, status))
            await super().finish_turn(admission, status)

    limit_service = Admission(settings=ChatLimitsSettings(shared_quotas_enabled=False))

    class Workflow:
        async def stream_with_tokens(self, **kwargs):
            count = 11 if kwargs["user_message"] == "many maps" else 1
            response_id = str(uuid4())
            yield WorkflowEvent(WorkflowEventType.AGENT_START, {"agent": "analyst"})
            yield WorkflowEvent(
                WorkflowEventType.VISUALIZATION,
                {"artifacts": [graph_payload() for _ in range(count)]},
            )
            yield WorkflowEvent(
                WorkflowEventType.TOKEN,
                {"agent": "analyst", "chunk": "Answer", "response_id": response_id},
            )
            yield WorkflowEvent(
                WorkflowEventType.RESPONSE,
                {
                    "agent": "analyst",
                    "content": "Answer",
                    "response_id": response_id,
                    "evidence": ResponseEvidence(
                        sources=[
                            ResponseSource(
                                kind="note",
                                source_id="note:chunk-a",
                                chunk_id="chunk-a",
                            )
                        ],
                        concept_ids=["concept-a"],
                    ),
                },
            )
            yield WorkflowEvent(
                WorkflowEventType.RESPONSE, {"agent": "mentor", "content": "Follow-up"}
            )
            yield WorkflowEvent(WorkflowEventType.COMPLETE, {"outcome": "answer"})

    def make_workflow():
        calls.append("workflow")
        return Workflow()

    async def enter(self):
        return self

    async def exit_context(self, *exc):
        pass

    monkeypatch.setattr(app_state.AppState, "__aenter__", enter)
    monkeypatch.setattr(app_state.AppState, "__aexit__", exit_context)
    monkeypatch.setattr(app_state, "get_quick_action_cache", lambda: cache)
    monkeypatch.setattr(app_state, "get_session_memory_store", lambda: memory)
    monkeypatch.setattr(app_state, "get_chat_limit_service", lambda: limit_service)
    monkeypatch.setattr(app_state, "get_workflow", make_workflow)
    monkeypatch.setattr(app_state, "CACHED_TOKENS_REPLAY_LATENCY", 0)
    return calls, admissions


def test_cached_turn_replay_and_restoration(stores, application):
    cache, memory = stores
    calls, admissions = application
    live = app_state.AppState(_reflex_internal_init=True)
    asyncio.run(live._process_message(MAP_PROMPT))
    assert calls == ["workflow"]
    assert live.error_message == ""
    cached = cache.get(MAP_PROMPT)
    assert len(cached.responses) == 2 and len(cached.artifacts) == 1
    assert cached.responses[0].evidence.sources[0].chunk_id == "chunk-a"
    assert cached.responses[1].evidence is None

    replay = app_state.AppState(_reflex_internal_init=True)
    asyncio.run(replay._process_message(MAP_PROMPT))
    assert calls == ["workflow"]
    assert [(context.cache_hit, status) for context, status in admissions] == [
        (False, "success"),
        (True, "success"),
    ]
    for (context, _), state in zip(admissions, [live, replay], strict=True):
        assert context.turn_id == state.messages[0].turn_id
        assert context.session_id == state.session_id
    assert replay.error_message == ""
    assert live.session_id != replay.session_id
    assert {m.turn_id for m in live.messages}.isdisjoint(
        m.turn_id for m in replay.messages
    )
    assert {m.response_id for m in live.messages[1:]}.isdisjoint(
        m.response_id for m in replay.messages[1:]
    )
    assert {m.dom_id for m in live.messages}.isdisjoint(
        m.dom_id for m in replay.messages
    )
    assert (
        live._visual_artifacts[0].artifact_id != replay._visual_artifacts[0].artifact_id
    )
    assert replay._visual_artifacts[0].turn_id == replay.messages[0].turn_id
    assert replay._visual_artifacts[0].nodes == live._visual_artifacts[0].nodes
    assert replay.messages[1].evidence == live.messages[1].evidence
    assert replay.messages[2].evidence is None
    assert cache.get(MAP_PROMPT) == cached
    assert [(item.origin, item.outcome) for item in live.executions] == [
        ("workflow", "answer")
    ]
    assert [(item.origin, item.outcome, item.stages) for item in replay.executions] == [
        ("cache", "answer", [])
    ]

    restored = app_state.AppState(_reflex_internal_init=True)
    restored.session_id = replay.session_id
    restored.visitor_id = replay.visitor_id
    asyncio.run(app_state.AppState.initialize_session.fn(restored))
    assert list(restored.messages) == list(replay.messages)
    assert list(restored._visual_artifacts) == list(replay._visual_artifacts)
    assert list(restored.executions) == list(replay.executions)
    restored.select_session_history_item(restored.messages[0].turn_id)
    assert restored.scroll_target_message_dom_id == replay.messages[0].dom_id
    restored.select_recent_map(replay._visual_artifacts[0].artifact_id)
    assert restored.active_view == "graph"
    assert restored.selected_graph.nodes[0].label == "Attention"

    asyncio.run(app_state.AppState.clear_chat.fn(restored))
    assert restored.session_id != replay.session_id
    assert restored.visitor_id == replay.visitor_id
    assert not restored.messages and not restored._visual_artifacts
    assert memory.get(replay.session_id) is None
    assert memory.get(live.session_id) is not None
    assert cache.get(MAP_PROMPT) == cached


def test_structured_graph_only_turn_survives_bridge_cache_and_snapshot(stores, records):
    cache, memory = stores
    template, cached_turn = records
    payloads = [
        GraphPayload(
            artifact_id=str(uuid4()),
            title=f"Map {index}",
            nodes=[
                GraphNode(id="a", label="Alpha", type=None),
                GraphNode(id="b", label="Beta", source_ids=["note:chunk-b"]),
            ],
            edges=[
                GraphEdge.from_relationship("a", "USES", "b", synthetic=False),
                GraphEdge.from_relationship(
                    "b",
                    "RELATED_TO",
                    "a",
                    synthetic=True,
                    evidence_source_ids=["note:chunk-b"],
                ),
            ],
        )
        for index in range(2)
    ]
    workflow = object.__new__(LearnerWorkflow)
    queries = [["Alpha", "Beta relationships"], ["Beta", "Related notes"]]

    class Graph:
        async def astream_events(self, *args, **kwargs):
            yield {"event": "on_chain_start", "name": "LangGraph", "run_id": "root"}
            state = State()
            for payload, graph_queries in zip(payloads, queries, strict=True):
                result = GraphToolResult(queries=graph_queries, graph=payload)
                state.context = (
                    '{"get_subgraphs_to_visualize": ' + result.model_dump_json() + "}"
                )
                update = await workflow.visualizer_node(state)
                state = state.model_copy(update=update)
            yield {
                "event": "on_chain_end",
                "name": "LangGraph",
                "run_id": "root",
                "data": {
                    "output": {
                        "visual_artifacts": state.visual_artifacts,
                        "context": state.context,
                    }
                },
            }

    wrapper = ReflexLearnerWorkflow()
    wrapper._graph = Graph()

    async def run():
        return [event async for event in wrapper.stream_with_tokens("Map", [])]

    events = asyncio.run(run())
    assert all(event.type != WorkflowEventType.RESPONSE for event in events)
    graph_event = next(
        event for event in events if event.type == WorkflowEventType.VISUALIZATION
    )
    identity = {"session_id": cached_turn.session_id, "turn_id": cached_turn.turn_id}
    artifacts = [
        GraphArtifact.model_validate({**identity, **item, "timestamp": "12:01"})
        for item in graph_event.data["artifacts"]
    ]
    assert [artifact.artifact_id for artifact in artifacts] == [
        payload.artifact_id for payload in payloads
    ]
    turn = app_state.CachedTurn(
        **identity, responses=[], artifacts=artifacts, agent_history=["visualizer"]
    )
    assert cache.set("graph", turn)
    destination = app_state.TurnIdentity(session_id=str(uuid4()), turn_id=str(uuid4()))
    cached = cache.get("graph")
    replay = cached.replay(destination, "12:02")
    assert [artifact.edges for artifact in replay.artifacts] == [
        payload.edges for payload in payloads
    ]
    assert {artifact.artifact_id for artifact in replay.artifacts}.isdisjoint(
        artifact.artifact_id for artifact in artifacts
    )
    replay.artifacts[0].nodes[1].source_ids.clear()
    assert cached.artifacts[0].nodes[1].source_ids == ["note:chunk-b"]
    user = template.messages[0].model_copy(update=destination.model_dump())
    snapshot = app_state.SessionSnapshot(
        session_id=destination.session_id, messages=[user], artifacts=replay.artifacts
    )
    assert memory.replace_snapshot(snapshot)
    restored = memory.get(destination.session_id)
    assert restored == snapshot
    assert [artifact.retrieval_queries for artifact in restored.artifacts] == queries
    state = app_state.AppState(_reflex_internal_init=True)
    state.messages = restored.messages
    state._visual_artifacts = restored.artifacts
    assert [item.retrieval_text for item in state.study_turns[0].maps] == [
        " & ".join(graph_queries) for graph_queries in queries
    ]
    assert memory.get(cached_turn.session_id) is None


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        (
            '{"get_subgraphs_to_visualize":{"queries":["Map"],"graph":null}}',
            VISUALIZATION_EMPTY_CONTEXT,
        ),
        (
            '{"get_subgraphs_to_visualize":{"queries":["Map"],"graph":{}}}',
            "Visualization failed",
        ),
        ("malformed", "Visualization failed"),
    ],
)
def test_structured_visualizer_empty_and_malformed_results(context, expected):
    workflow = object.__new__(LearnerWorkflow)
    result = asyncio.run(workflow.visualizer_node(State(context=context)))
    assert result["context"] == expected
    assert "visual_artifacts" not in result


def test_structured_graph_through_the_workflow_tool_and_bridge():
    payload = GraphPayload(
        artifact_id=str(uuid4()),
        title="Map",
        nodes=[GraphNode(id="concept_a", label="Alpha")],
        edges=[],
    )
    calls = []

    def retrieve(queries):
        calls.append(queries)
        return GraphToolResult(queries=queries, graph=payload)

    graph_tool = get_subgraphs_to_visualize(
        visualizer_pipeline=SimpleNamespace(structured_graph=retrieve), structured=True
    )
    workflow = object.__new__(LearnerWorkflow)
    workflow.workflow = StateGraph(State)
    workflow.tools = {graph_tool.name: graph_tool}
    workflow.create_messages_to_pass = lambda *args: []

    async def call_model(messages, role, **kwargs):
        if role == ModelRoleType.orchestrator:
            return RoutingDecision(
                request_scope="in_scope", next_step="retriever", reasoning="fixture"
            )
        assert role == ModelRoleType.retriever
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "graph",
                    "name": graph_tool.name,
                    "args": {"queries": ["Map"]},
                }
            ],
        )

    workflow.call_model = call_model
    workflow._init_nodes()
    wrapper = ReflexLearnerWorkflow()
    wrapper._graph = workflow.workflow.compile()

    async def run():
        return [event async for event in wrapper.stream_with_tokens("Map Alpha", [])]

    events = asyncio.run(run())
    assert calls == [["Map"]]
    assert [
        event.data["artifacts"]
        for event in events
        if event.type == WorkflowEventType.VISUALIZATION
    ] == [[{**payload.model_dump(mode="json"), "retrieval_queries": ["Map"]}]]
    assert all(
        event.type not in {WorkflowEventType.RESPONSE, WorkflowEventType.ERROR}
        for event in events
    )


def test_ineligible_prompts_and_recent_map_eviction(stores, application):
    cache, memory = stores
    calls, _ = application
    state = app_state.AppState(_reflex_internal_init=True)
    asyncio.run(state._process_message("many maps"))
    assert cache._collection.docs == {}
    assert len(state._visual_artifacts) == len(state.recent_maps) == 10
    oldest_map_id = state._visual_artifacts[0].artifact_id
    asyncio.run(state._process_message(MAP_PROMPT))
    assert calls == ["workflow", "workflow"]
    assert cache._collection.docs == {}
    assert cache._collection.reads == 0
    assert len(state.messages) == 6
    assert len(state._visual_artifacts) == len(state.recent_maps) == 10
    assert oldest_map_id not in {item.artifact_id for item in state.recent_maps}
    assert [item.artifact_id for item in state.recent_maps] == [
        artifact.artifact_id for artifact in memory.get(state.session_id).artifacts
    ]
    state.select_recent_map(oldest_map_id)
    assert state.active_view == "chat"
    state.select_recent_map(state.recent_maps[0].artifact_id)
    assert state.selected_map_index == 0
    state.select_recent_map(state.recent_maps[1].artifact_id)
    assert state.selected_recent_map_id == state.recent_maps[1].artifact_id
    state.select_recent_map(state.recent_maps[0].artifact_id)
    assert state.selected_recent_map_id == state.recent_maps[0].artifact_id


def test_graph_only_cache_hit_obeys_admission(
    stores, records, application, monkeypatch
):
    cache, _ = stores
    _, cached_turn = records
    assert cache.set(MAP_PROMPT, cached_turn.model_copy(update={"responses": []}))
    replay = app_state.AppState(_reflex_internal_init=True)
    asyncio.run(replay._process_message(MAP_PROMPT))
    assert [message.role for message in replay.messages] == ["user"]
    assert replay._visual_artifacts[0].turn_id == replay.messages[0].turn_id
    assert application[0] == []

    async def reject(visitor_id, session_id):
        raise ChatCooldownExceeded(10)

    monkeypatch.setattr(app_state.get_chat_limit_service(), "admit_cached_turn", reject)
    denied = app_state.AppState(_reflex_internal_init=True)
    asyncio.run(denied._process_message(MAP_PROMPT))
    assert denied.error_message
    assert not denied.messages and not denied._visual_artifacts
    assert application[0] == []


def test_streaming_agents_keep_their_response_ids_until_finalization(
    application, monkeypatch
):
    state = app_state.AppState(_reflex_internal_init=True)
    streamed_ids = {}

    class Workflow:
        async def stream_with_tokens(self, **kwargs):
            for agent in ("analyst", "mentor"):
                response_id = str(uuid4())
                yield WorkflowEvent(
                    WorkflowEventType.TOKEN,
                    {"agent": agent, "chunk": agent, "response_id": response_id},
                )
                streamed_ids[agent] = state.streaming_response_id
            for agent in streamed_ids:
                yield WorkflowEvent(
                    WorkflowEventType.RESPONSE,
                    {
                        "agent": agent,
                        "content": agent,
                        "response_id": streamed_ids[agent],
                    },
                )
            yield WorkflowEvent(WorkflowEventType.COMPLETE, {"outcome": "answer"})

    monkeypatch.setattr(app_state, "get_workflow", Workflow)
    asyncio.run(state._process_message("Review my understanding"))
    assert len(set(streamed_ids.values())) == 2
    assert {msg.agent: msg.response_id for msg in state.messages[1:]} == streamed_ids
    assert state.streaming_response_id == ""


def test_fresh_session_with_unavailable_or_old_storage(stores, application):
    cache, memory = stores
    state = app_state.AppState(_reflex_internal_init=True)
    visitor = str(uuid4())
    state.visitor_id = visitor.upper()
    state.session_id = "invalid"
    asyncio.run(app_state.AppState.initialize_session.fn(state))
    assert state.visitor_id == visitor
    assert state.session_id != "invalid"
    disposable_id = state.session_id
    memory._collection.docs[disposable_id] = {
        "_id": disposable_id,
        "schema_version": 1,
        "messages": [{"content": "old"}],
    }
    asyncio.run(app_state.AppState.initialize_session.fn(state))
    assert not state.messages and not state.recent_maps
    assert state.session_id == disposable_id
    assert disposable_id in memory._collection.docs
    cache._collection.fail = memory._collection.fail = True
    asyncio.run(state._process_message(MAP_PROMPT))
    assert state.error_message == "" and len(state.messages) == 3
    assert state.visitor_id == visitor and state.session_id == disposable_id


def test_clear_during_streaming_stops_the_workflow_and_preserves_visitor(
    stores, application, monkeypatch
):
    _, memory = stores
    state = app_state.AppState(_reflex_internal_init=True)

    async def run():
        started = asyncio.Event()
        stopped = asyncio.Event()
        resume = asyncio.Event()

        class Workflow:
            async def stream_with_tokens(self, **kwargs):
                try:
                    yield WorkflowEvent(
                        WorkflowEventType.TOKEN,
                        {
                            "agent": "analyst",
                            "chunk": "Partial",
                            "response_id": str(uuid4()),
                        },
                    )
                    started.set()
                    await resume.wait()
                    yield WorkflowEvent(
                        WorkflowEventType.VISUALIZATION,
                        {"artifacts": [graph_payload()]},
                    )
                finally:
                    stopped.set()

        monkeypatch.setattr(app_state, "get_workflow", Workflow)
        task = asyncio.create_task(state._process_message("Study memory"))
        async with asyncio.timeout(3):
            await started.wait()
            old_session, visitor = state.session_id, state.visitor_id
            await app_state.AppState.clear_chat.fn(state)
            resume.set()
            await asyncio.gather(task, return_exceptions=True)
        assert stopped.is_set()
        assert state.session_id != old_session and state.visitor_id == visitor
        assert not state.messages and not state._visual_artifacts
        assert not state.is_processing and not state.streaming_content
        assert memory.get(old_session) is None
        assert [status for _, status in application[1]] == ["cancelled"]

    asyncio.run(run())


def test_clear_during_cached_replay_releases_admission_without_writing_snapshot(
    stores, records, application, monkeypatch
):
    cache, memory = stores
    _, cached = records
    assert cache.set(MAP_PROMPT, cached)
    state = app_state.AppState(_reflex_internal_init=True)

    async def run():
        streamed = asyncio.Event()

        async def pause_replay(_):
            streamed.set()
            await asyncio.Future()

        monkeypatch.setattr(app_state.asyncio, "sleep", pause_replay)
        task = asyncio.create_task(state._process_message(MAP_PROMPT))
        try:
            async with asyncio.timeout(3):
                await streamed.wait()
                assert state.streaming_content
                old_session = state.session_id
                await app_state.AppState.clear_chat.fn(state)
                await asyncio.gather(task, return_exceptions=True)
        finally:
            task.cancel()
        assert state.session_id != old_session
        assert not state.messages and not state._visual_artifacts
        assert memory.get(old_session) is None
        assert not application[0]
        assert [(ctx.cache_hit, status) for ctx, status in application[1]] == [
            (True, "cancelled")
        ]
        assert cache.get(MAP_PROMPT) == cached

    asyncio.run(run())


def test_clear_waits_for_in_flight_snapshot_and_blocks_late_upserts(
    stores, application, monkeypatch
):
    _, memory = stores
    state = app_state.AppState(_reflex_internal_init=True)
    entered, continue_write = Event(), Event()
    replace = memory.replace_snapshot
    submitted = []

    def blocked(snapshot):
        submitted.append(snapshot)
        entered.set()
        assert continue_write.wait(3)
        return replace(snapshot)

    monkeypatch.setattr(memory, "replace_snapshot", blocked)

    async def run():
        retiring = asyncio.Event()
        retire = app_state.AppState._retire_session

        async def observe_retirement(self, *args):
            retiring.set()
            await retire(self, *args)

        monkeypatch.setattr(app_state.AppState, "_retire_session", observe_retirement)
        turn = asyncio.create_task(state._process_message("A question"))
        async with asyncio.timeout(3):
            assert await asyncio.to_thread(entered.wait, 2)
            old_session = state.session_id
            clear = asyncio.create_task(app_state.AppState.clear_chat.fn(state))
            try:
                await retiring.wait()
                assert state.session_id != old_session
                assert not clear.done()
            finally:
                continue_write.set()
            await clear
            await asyncio.gather(turn, return_exceptions=True)
        assert memory.get(old_session) is None
        assert not replace(submitted[0])
        assert memory.get(old_session) is None

    asyncio.run(run())


@pytest.mark.parametrize("terminal", ["timeout", "cancelled", "incomplete"])
def test_unfinished_workflow_is_persisted_with_the_observed_stage(
    terminal, stores, application, monkeypatch
):
    _, memory = stores
    state = app_state.AppState(_reflex_internal_init=True)
    service = app_state.get_chat_limit_service()
    if terminal == "timeout":
        monkeypatch.setattr(service.settings, "workflow_timeout_seconds", 0.05)

    async def run():
        waiting = asyncio.Event()

        class Workflow:
            async def stream_with_tokens(self, turn_context, **kwargs):
                stage = app_state.ExecutionStage(
                    session_id=turn_context.session_id,
                    turn_id=turn_context.turn_id,
                    invocation_id="worker-run",
                    sequence=0,
                    agent="mentor",
                    start_offset=0.0,
                )
                yield WorkflowEvent(
                    WorkflowEventType.AGENT_START, {"agent": "mentor", "stage": stage}
                )
                yield WorkflowEvent(
                    WorkflowEventType.TOKEN,
                    {
                        "agent": "mentor",
                        "chunk": "Unfinished",
                        "response_id": str(uuid4()),
                        "invocation_id": stage.invocation_id,
                    },
                )
                waiting.set()
                if terminal != "incomplete":
                    await asyncio.Future()

        monkeypatch.setattr(app_state, "get_workflow", Workflow)
        task = asyncio.create_task(state._process_message("A question"))
        async with asyncio.timeout(3):
            await waiting.wait()
            if terminal == "cancelled":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await task
        report = memory.get(state.session_id).executions[0]
        [stage] = report.stages
        expected = "interrupted" if terminal == "timeout" else terminal
        assert report.outcome == stage.outcome == expected
        assert stage.duration_seconds >= (0.04 if terminal == "timeout" else 0)
        assert len(state.messages) == 1 and not state.streaming_content
        assert not state.is_processing
        assert application[1][-1][1] != "success"

    asyncio.run(run())


def test_stale_hydration_cannot_restore_a_cleared_session(
    stores, records, application, monkeypatch
):
    _, memory = stores
    snapshot, _ = records
    state = app_state.AppState(_reflex_internal_init=True)
    state.session_id = snapshot.session_id
    entered, complete = Event(), Event()

    def retrieve(session_id, **kwargs):
        assert session_id == snapshot.session_id
        entered.set()
        assert complete.wait(3)
        return snapshot

    monkeypatch.setattr(memory, "get", retrieve)

    async def run():
        hydration = asyncio.create_task(app_state.AppState.initialize_session.fn(state))
        assert await asyncio.to_thread(entered.wait, 2)
        try:
            await app_state.AppState.clear_chat.fn(state)
        finally:
            complete.set()
        await hydration
        assert state.session_id != snapshot.session_id
        assert not state.messages and not state._visual_artifacts

    asyncio.run(run())


def test_old_finalizer_does_not_change_a_new_session_turn(application, monkeypatch):
    state = app_state.AppState(_reflex_internal_init=True)
    service = app_state.get_chat_limit_service()
    finish = service.finish_turn

    async def run():
        old_finalizing = asyncio.Event()
        finish_old = asyncio.Event()
        new_running = asyncio.Event()
        finish_new = asyncio.Event()

        async def hold_old(admission, status):
            if not old_finalizing.is_set():
                old_finalizing.set()
                await finish_old.wait()
            await finish(admission, status)

        monkeypatch.setattr(service, "finish_turn", hold_old)
        original = app_state.get_workflow()

        class Workflow:
            async def stream_with_tokens(self, user_message, **kwargs):
                if user_message == "New question":
                    new_running.set()
                    await finish_new.wait()
                async for event in original.stream_with_tokens(
                    user_message=user_message, **kwargs
                ):
                    yield event

        monkeypatch.setattr(app_state, "get_workflow", Workflow)
        old_turn = asyncio.create_task(state._process_message("Old question"))
        async with asyncio.timeout(3):
            await old_finalizing.wait()
            old_session = state.session_id
            clearing = asyncio.create_task(app_state.AppState.clear_chat.fn(state))
            retired = asyncio.Event()
            retire = app_state.AppState._retire_session

            async def observe_retirement(self, *args):
                retired.set()
                await retire(self, *args)

            monkeypatch.setattr(
                app_state.AppState, "_retire_session", observe_retirement
            )
            await retired.wait()
            assert state.session_id != old_session
            new_turn = asyncio.create_task(state._process_message("New question"))
            await new_running.wait()
            finish_old.set()
            await clearing
            await asyncio.gather(old_turn, return_exceptions=True)
            assert state.is_processing
            assert [msg.content for msg in state.messages] == ["New question"]
            finish_new.set()
            await new_turn
        assert [msg.content for msg in state.messages] == [
            "New question",
            "Answer",
            "Follow-up",
        ]

    asyncio.run(run())


def test_form_submission_runs_as_one_background_event(application):
    state = app_state.AppState(_reflex_internal_init=True)

    async def run():
        assert app_state.AppState.handle_form_submit.is_background
        state.current_input = "Old state value"
        await app_state.AppState.handle_form_submit.fn(
            state, {"message": "Current form value"}
        )
        assert state.messages[0].content == "Current form value"
        assert state.current_input == "" and not state.is_processing
        assert application[0] == ["workflow"]

    asyncio.run(run())


def test_submit_during_hydration_uses_saved_history_and_skips_first_turn_cache(
    stores, records, application, monkeypatch
):
    cache, memory = stores
    snapshot, cached = records
    assert memory.replace_snapshot(snapshot)
    assert cache.set(MAP_PROMPT, cached)
    state = app_state.AppState(_reflex_internal_init=True)
    state.session_id = snapshot.session_id
    entered, release = Event(), Event()
    get = memory.get

    def hold_first_read(session_id, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(3)
        return get(session_id, **kwargs)

    monkeypatch.setattr(memory, "get", hold_first_read)

    async def run():
        hydration = asyncio.create_task(app_state.AppState.initialize_session.fn(state))
        async with asyncio.timeout(3):
            assert await asyncio.to_thread(entered.wait, 2)
            try:
                await state._process_message(MAP_PROMPT)
            finally:
                release.set()
            await hydration
        assert list(state.messages[: len(snapshot.messages)]) == snapshot.messages
        assert state.messages[len(snapshot.messages)].content == MAP_PROMPT
        assert application[0] == ["workflow"]
        assert not application[1][0][0].cache_hit
        assert get(state.session_id).messages == list(state.messages)

    asyncio.run(run())


def test_failed_snapshot_read_cannot_overwrite_existing_history(
    stores, records, application, monkeypatch
):
    cache, memory = stores
    snapshot, cached = records
    assert memory.replace_snapshot(snapshot)
    assert cache.set(MAP_PROMPT, cached)
    state = app_state.AppState(_reflex_internal_init=True)
    state.session_id = snapshot.session_id
    find = memory._collection.find_one

    def unavailable(_query):
        raise OSError("temporary read failure")

    monkeypatch.setattr(memory._collection, "find_one", unavailable)
    asyncio.run(state._process_message(MAP_PROMPT))
    monkeypatch.setattr(memory._collection, "find_one", find)
    assert memory.get(snapshot.session_id) == snapshot
    assert len(state.messages) == 3
    assert not application[1][0][0].cache_hit


def test_clear_suppresses_a_queued_cache_write(stores, application, monkeypatch):
    cache, memory = stores
    state = app_state.AppState(_reflex_internal_init=True)
    to_thread = asyncio.to_thread

    async def run():
        queued, release, retiring = asyncio.Event(), asyncio.Event(), asyncio.Event()
        retire = app_state.AppState._retire_session

        async def pause_before_write(function, *args, **kwargs):
            if function.__name__ == "guarded_write":
                queued.set()
                await release.wait()
            return await to_thread(function, *args, **kwargs)

        async def observe_retirement(self, *args):
            retiring.set()
            await retire(self, *args)

        monkeypatch.setattr(app_state.asyncio, "to_thread", pause_before_write)
        monkeypatch.setattr(app_state.AppState, "_retire_session", observe_retirement)
        turn = asyncio.create_task(state._process_message(MAP_PROMPT))
        async with asyncio.timeout(3):
            await queued.wait()
            old_session = state.session_id
            clear = asyncio.create_task(app_state.AppState.clear_chat.fn(state))
            try:
                await retiring.wait()
            finally:
                release.set()
            await clear
            await asyncio.gather(turn, return_exceptions=True)
        assert state.session_id != old_session
        assert not state.messages and cache.get(MAP_PROMPT) is None
        assert memory.get(old_session) is None

    asyncio.run(run())


def test_concurrent_visitors_keep_separate_turns_reports_and_snapshots(
    stores, application
):
    _, memory = stores
    states = [app_state.AppState(_reflex_internal_init=True) for _ in range(2)]
    prompts = ["First conversation", "Second conversation"]

    async def run():
        await asyncio.gather(
            *(
                state._process_message(prompt)
                for state, prompt in zip(states, prompts, strict=True)
            )
        )

    asyncio.run(run())
    assert len({state.visitor_id for state in states}) == 2
    assert len({state.session_id for state in states}) == 2
    for state, prompt in zip(states, prompts, strict=True):
        snapshot = memory.get(state.session_id)
        assert snapshot.messages[0].content == prompt
        assert {message.turn_id for message in snapshot.messages} == {
            snapshot.executions[0].turn_id
        }
        assert {artifact.session_id for artifact in snapshot.artifacts} == {
            state.session_id
        }
    assert len({ctx.turn_id for ctx, _ in application[1]}) == 2
