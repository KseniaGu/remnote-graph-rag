"""Serve the actual Reflex API with disposable browser test dependencies."""

import asyncio
import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from app import state
from app.app import app
from app.strings import QUICK_ACTIONS
from backend.configs.enums import WorkflowEventType
from backend.utils.chat_limits import ChatCooldownExceeded, ChatLimitService
from backend.workflows import learner_reflex
from backend.workflows.learner_reflex import WorkflowEvent
from backend.workflows.presentation import (
    CachedTurn,
    ExecutionStage,
    GraphPayload,
    Message,
    ResponseEvidence,
    ResponseSource,
    SessionSnapshot,
)

MAP_PROMPT = next(
    prompt for _, title, prompt in QUICK_ACTIONS if title == "Map relationships"
)
GRAPH_FIXTURE_PATH = os.environ.get("PRACTICE_GRAPH_FIXTURE")
GRAPH_FIXTURE = (
    json.loads(Path(GRAPH_FIXTURE_PATH).read_text()) if GRAPH_FIXTURE_PATH else None
)
LONG_MAP_TITLE = (
    "Attention architectures and inference techniques across a collection of "
    "research notes on efficient transformer decoding"
)


def fixture_graph(name):
    graph = deepcopy(GRAPH_FIXTURE["graphs"][name])
    graph["artifact_id"] = str(uuid4())
    source_ids = {
        source_id for node in graph["nodes"] for source_id in node.get("source_ids", [])
    }
    graph["sources"] = [
        source
        for source in GRAPH_FIXTURE["sources"]
        if source["source_id"] in source_ids
    ]
    return GraphPayload.model_validate(graph).model_dump(mode="json")


ANSWER = (
    """
## What the cache stores

The cache stores keys and values. Inline math remains readable: $O(n)$.

$$
K = X W_k
$$

| Technique | Value |
| --- | --- |
| Cache | Avoided projection |

```python
cache = model.allocate_cache()
review = "This line is intentionally long enough to require horizontal scrolling on smaller screens while leaving the conversation width unchanged."
```

```mermaid
graph TD
    Input --> Cache
```
"""
    + "\n\nA decoder uses cached keys and values to process each new token." * 34
)


class Memory:
    def __init__(self):
        self.sessions = {}

    def get(self, session_id, *, require_available=False):
        snapshot = self.sessions.get(session_id)
        return SessionSnapshot.model_validate(deepcopy(snapshot)) if snapshot else None

    def get_prompt_summary(self, session_id, latest_user_message):
        return ""

    def replace_snapshot(self, snapshot):
        stored = snapshot.model_dump()
        turns_without_reports = {
            message.turn_id
            for message in snapshot.messages
            if message.role == "user"
            and message.content == "Study without an execution report"
        }
        stored["executions"] = [
            report
            for report in stored["executions"]
            if report["turn_id"] not in turns_without_reports
        ]
        self.sessions[snapshot.session_id] = deepcopy(stored)

    def delete(self, session_id):
        self.sessions.pop(session_id, None)


class QuickActionCache:
    def __init__(self):
        self.maps = {}
        session_id, turn_id, response_id = (str(uuid4()) for _ in range(3))
        self.turn = CachedTurn(
            session_id=session_id,
            turn_id=turn_id,
            responses=[
                Message(
                    session_id=session_id,
                    turn_id=turn_id,
                    response_id=response_id,
                    dom_id=f"message-{uuid4().hex}",
                    role="assistant",
                    agent="mentor",
                    content="Cached practice question.",
                    evidence=ResponseEvidence(
                        sources=[
                            ResponseSource(
                                kind="note",
                                title="Cached context",
                                excerpt="Diffusion models learn to remove noise.",
                            ),
                            ResponseSource(kind="note", chunk_id="relationship-note"),
                        ]
                    ),
                    timestamp="12:00",
                )
            ],
            artifacts=[],
            agent_history=["mentor"],
        )

    def get(self, prompt):
        return self.turn if prompt == QUICK_ACTIONS[0][2] else self.maps.get(prompt)

    def set(self, prompt, turn, **kwargs):
        if prompt == MAP_PROMPT:
            self.maps[prompt] = turn


class Admission(ChatLimitService):
    reject_next_turn = False

    async def admit_expensive_turn(self, visitor_id, session_id):
        if self.reject_next_turn:
            self.reject_next_turn = False
            raise ChatCooldownExceeded(10)
        return await super().admit_expensive_turn(visitor_id, session_id)


PRACTICE_SCENARIOS = {
    "Study local notes": (
        (
            "orchestrator",
            "retriever",
            "orchestrator",
            "analyst",
            "orchestrator",
            "analyst",
        ),
        "answer",
    ),
    "Study web fallback": (
        ("orchestrator", "retriever", "orchestrator", "researcher", "analyst"),
        "answer",
    ),
    "Study without retrieval": (("orchestrator", "mentor"), "answer"),
    "Study relationship notes": (("orchestrator", "retriever", "analyst"), "answer"),
    "Study relationships without notes": (
        ("orchestrator", "retriever", "analyst"),
        "answer",
    ),
    "Study with empty evidence": (("analyst",), "answer"),
    "Study without an execution report": (("analyst",), "answer"),
    "Refuse outside scope": (("orchestrator",), "out_of_scope"),
    "Finish with no results": (("orchestrator", "retriever"), "no_results"),
    "Fail after one response": (("analyst", "researcher"), "error"),
    "Interrupt a stream": (("analyst",), "interrupted"),
    "Compare agent colors": (
        ("orchestrator", "retriever", "researcher", "analyst", "mentor", "visualizer"),
        "answer",
    ),
}

LOCAL_EVIDENCE = ResponseEvidence(
    sources=[
        ResponseSource(
            kind="note",
            title="Decoder attention notes",
            path=["Transformers", "Decoder"],
            excerpt="The decoder retains key and value vectors.",
            summary="Explains how the decoder caches attention projections.",
            retrieval_score=0.87,
        ),
        ResponseSource(kind="note", chunk_id="missing-metadata"),
        *(
            ResponseSource(
                kind="note",
                title=f"Supplemental Study note {index}",
                excerpt=f"Retrieved context for note {index}.",
                summary=f"Stored summary for note {index}.",
                supports_relationship=index > 3,
            )
            for index in range(2, 9)
        ),
    ],
    concept_ids=["concept-cache"],
)
WEB_EVIDENCE = ResponseEvidence(
    sources=[
        ResponseSource(
            kind="web",
            title="Web research context",
            url="https://example.org/reference",
        ),
        ResponseSource(kind="web", title="Web record without a link"),
    ]
)


async def practice_events(prompt, context):
    agents, outcome = PRACTICE_SCENARIOS[prompt]
    for index, agent in enumerate(agents):
        response_id = str(uuid4())
        stage = ExecutionStage(
            session_id=context.session_id,
            turn_id=context.turn_id,
            sequence=index,
            agent=agent,
            invocation_id=f"invocation-{index}",
            start_offset=float(index),
        )
        yield WorkflowEvent(
            WorkflowEventType.AGENT_START, {"agent": agent, "stage": stage}
        )
        if prompt == "Compare agent colors" and agent == "visualizer":
            await asyncio.sleep(4)
        if prompt in {"Study local notes", "Interrupt a stream"} and agent == "analyst":
            yield WorkflowEvent(
                WorkflowEventType.TOKEN,
                {
                    "agent": agent,
                    "chunk": (
                        "Second response in progress"
                        if index == 5
                        else "Pending evidence preview"
                    ),
                    "response_id": response_id,
                    "invocation_id": stage.invocation_id,
                },
            )
            await asyncio.sleep(1)
        stage_outcome = (
            "interrupted"
            if outcome == "interrupted"
            else "failed"
            if outcome == "error" and agent == "researcher"
            else "completed"
        )
        yield WorkflowEvent(
            WorkflowEventType.AGENT_END,
            {
                "agent": agent,
                "stage": stage.model_copy(
                    update={"end_offset": index + 0.25, "outcome": stage_outcome}
                ),
                "response_ids": [response_id]
                if prompt == "Study local notes" and agent == "analyst"
                else [],
            },
        )
        if stage_outcome == "failed":
            await asyncio.sleep(0.45)
        records = []
        if agent == "analyst" and prompt == "Study local notes":
            if index == 3:
                records = [("Local explanation one", LOCAL_EVIDENCE)]
            else:
                records = [
                    (
                        "Local explanation two",
                        ResponseEvidence(
                            sources=[
                                ResponseSource(
                                    kind="note",
                                    title="Grouped query notes",
                                    excerpt="Query heads can share key and value heads.",
                                )
                            ]
                        ),
                    )
                ]
        elif agent == "analyst" and prompt == "Study web fallback":
            records = [("Web fallback explanation", WEB_EVIDENCE)]
        elif agent == "analyst" and outcome == "error":
            records = [("Completed before failure", LOCAL_EVIDENCE)]
        elif agent == "mentor":
            records = [("Mentor practice reply", None)]
        elif agent == "analyst" and prompt == "Study relationship notes":
            records = [
                (
                    "An answer with graph references only",
                    ResponseEvidence(
                        sources=[
                            ResponseSource(kind="note", chunk_id="relationship-note")
                        ],
                        has_relationship_context=True,
                    ),
                )
            ]
        elif agent == "analyst" and prompt == "Study relationships without notes":
            records = [
                (
                    "An answer without linked notes",
                    ResponseEvidence(has_relationship_context=True),
                )
            ]
        elif prompt in {
            "Study with empty evidence",
            "Study without an execution report",
        }:
            records = [("An answer with explicitly empty evidence", ResponseEvidence())]
        elif outcome == "out_of_scope":
            records = [("This question is outside the study scope.", None)]
        for content, evidence in records:
            yield WorkflowEvent(
                WorkflowEventType.RESPONSE,
                {
                    "agent": agent,
                    "content": content,
                    "evidence": evidence,
                    "response_id": response_id,
                    "invocation_id": stage.invocation_id,
                },
            )
    if outcome == "error":
        yield WorkflowEvent(
            WorkflowEventType.ERROR,
            {"message": "The rest of this turn failed.", "outcome": outcome},
        )
    else:
        yield WorkflowEvent(WorkflowEventType.COMPLETE, {"outcome": outcome})


class Workflow:
    def __init__(self, admission):
        self.admission = admission

    async def stream_with_tokens(self, user_message, **kwargs):
        if user_message in PRACTICE_SCENARIOS:
            async for event in practice_events(user_message, kwargs["turn_context"]):
                yield event
            return
        if user_message == "Map the cache":
            graph = {
                "schema_version": 1,
                "artifact_id": str(uuid4()),
                "title": "Cache map",
                "nodes": [
                    {
                        "id": "concept-cache",
                        "label": "KV caching",
                        "type": "CONCEPT",
                        "source_ids": [
                            "note:graph-cache",
                            "note:cache-rollover",
                            *(f"note:cache-{index}" for index in range(3, 9)),
                        ],
                    },
                    {"id": "concept-keys", "label": "Keys", "type": "CONCEPT"},
                ],
                "edges": [
                    {
                        "id": str(uuid4()),
                        "source": "concept-cache",
                        "target": "concept-keys",
                        "predicate": "USES",
                        "synthetic": False,
                    }
                ],
                "sources": [
                    {
                        "kind": "note",
                        "source_id": "note:graph-cache",
                        "title": "Graph cache note",
                        "excerpt": "Stored independently of the answer's evidence.",
                        "summary": "Describes how key and value projections are cached.",
                    },
                    {
                        "kind": "note",
                        "source_id": "note:cache-rollover",
                        "path": ["Decoding", "Sliding windows"],
                        "summary": "Sliding-window decoding evicts older cache entries as new tokens arrive. The retained window bounds memory usage during generation.",
                    },
                    *(
                        {
                            "kind": "note",
                            "source_id": f"note:cache-{index}",
                            "title": f"Supplemental cache note {index}",
                            "summary": f"Stored summary for cache note {index}.",
                        }
                        for index in range(3, 9)
                    ),
                ],
            }
            yield WorkflowEvent(WorkflowEventType.VISUALIZATION, {"artifacts": [graph]})
            yield WorkflowEvent(WorkflowEventType.COMPLETE, {"outcome": "graph"})
            return
        map_name = {
            MAP_PROMPT: "kv",
            "Map at the rendering ceiling": "ceiling",
            "Map above the rendering ceiling": "oversized",
        }.get(user_message.split("\n", 1)[0])
        if GRAPH_FIXTURE and map_name is not None:
            if user_message == "Map at the rendering ceiling":
                context = kwargs["turn_context"]
                visualizer = ExecutionStage(
                    session_id=context.session_id,
                    turn_id=context.turn_id,
                    sequence=0,
                    invocation_id="visualizer-0",
                    agent="visualizer",
                    start_offset=0,
                )
                yield WorkflowEvent(
                    WorkflowEventType.AGENT_START,
                    {"agent": "visualizer", "stage": visualizer},
                )
                await asyncio.sleep(0.6)
                yield WorkflowEvent(
                    WorkflowEventType.AGENT_END,
                    {
                        "agent": "visualizer",
                        "stage": visualizer.model_copy(
                            update={"end_offset": 0.6, "outcome": "completed"}
                        ),
                    },
                )
            elif user_message != MAP_PROMPT:
                await asyncio.sleep(0.35)
            artifact = fixture_graph(map_name)
            if map_name == "kv":
                artifact["retrieval_queries"] = [
                    "KV cache",
                    "grouped-query attention",
                    "PagedAttention block allocation",
                ]
            elif map_name == "oversized":
                artifact["title"] = LONG_MAP_TITLE
                artifact["retrieval_queries"] = [
                    LONG_MAP_TITLE,
                    "grouped query attention",
                    "transformer training objectives",
                ]
            yield WorkflowEvent(
                WorkflowEventType.VISUALIZATION, {"artifacts": [artifact]}
            )
            yield WorkflowEvent(WorkflowEventType.COMPLETE, {"outcome": "graph"})
            return
        agent = "mentor" if user_message.startswith("Quiz me on ") else "analyst"
        response_id = str(uuid4())
        if user_message == "Clear this streaming turn":
            yield WorkflowEvent(
                WorkflowEventType.TOKEN,
                {
                    "agent": agent,
                    "chunk": "Clear test in progress",
                    "response_id": response_id,
                },
            )
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                yield WorkflowEvent(
                    WorkflowEventType.RESPONSE,
                    {
                        "agent": agent,
                        "content": "Late clear test answer",
                        "response_id": response_id,
                    },
                )
            return
        if user_message == "Reload this streaming turn":
            yield WorkflowEvent(
                WorkflowEventType.TOKEN,
                {
                    "agent": agent,
                    "chunk": "Reload test in progress",
                    "response_id": response_id,
                },
            )
            await asyncio.sleep(1)
            yield WorkflowEvent(
                WorkflowEventType.RESPONSE,
                {
                    "agent": agent,
                    "content": "Reload test completed",
                    "response_id": response_id,
                },
            )
            yield WorkflowEvent(WorkflowEventType.COMPLETE, {"outcome": "answer"})
            return
        yield WorkflowEvent(WorkflowEventType.AGENT_START, {"agent": agent})
        if user_message != "Layout check":
            for part in ANSWER.split("\n\n")[:20]:
                if "$O(" in part:
                    prefix, suffix = part.split("$O(", 1)
                    chunks = (prefix + "$O(", suffix + "\n\n")
                elif part.startswith("```mermaid"):
                    prefix, suffix = part.split("-->", 1)
                    chunks = (prefix + "-->", suffix + "\n\n")
                else:
                    chunks = (part + "\n\n",)
                for chunk in chunks:
                    yield WorkflowEvent(
                        WorkflowEventType.TOKEN,
                        {"agent": agent, "chunk": chunk, "response_id": response_id},
                    )
                    await asyncio.sleep(0.3 if len(chunks) > 1 else 0.12)
        yield WorkflowEvent(
            WorkflowEventType.RESPONSE,
            {"agent": agent, "content": ANSWER, "response_id": response_id},
        )
        if user_message == "Reject the next turn":
            self.admission.reject_next_turn = True
        yield WorkflowEvent(WorkflowEventType.COMPLETE, {"outcome": "answer"})


def create_app():
    memory = Memory()
    admission = Admission()
    workflow = Workflow(admission)
    patch.object(state, "get_session_memory_store", return_value=memory).start()
    patch.object(
        state, "get_quick_action_cache", return_value=QuickActionCache()
    ).start()
    patch.object(state, "get_chat_limit_service", return_value=admission).start()
    patch.object(state, "get_workflow", return_value=workflow).start()

    def get_document(chunk_id, *, raise_error):
        if chunk_id == "relationship-note":
            return SimpleNamespace(
                id_=chunk_id,
                text="Attention uses learned query, key, and value projections.",
                metadata={
                    "path": ["Transformers", "Attention"],
                    "postprocess_chunk_summary": (
                        "Attention projections map each token into query, key, and value vectors. "
                        "Query and key vectors determine which tokens the model attends to. "
                        "The attention weights then combine information from the value vectors."
                    ),
                },
            )
        return None

    patch.object(
        learner_reflex,
        "open_document_store",
        return_value=SimpleNamespace(get_document=get_document),
    ).start()
    return app()
