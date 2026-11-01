import asyncio
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter
from openrouter import OpenRouter
from openrouter.errors import ResponseValidationError
from synthetic_runtime import make_scenario
from synthetic_workflow import FakeGraph, wrapper

from backend.evaluation import runtime_live
from backend.evaluation.runtime_evaluation import (
    RuntimeRecord,
    _judge_diagnostics,
    _judge_transport_boundary,
    score_runtime_record,
)
from backend.utils.chat_limits import get_current_chat_turn

CASES = [
    ({"prompt_tokens": 8}, (8, None, None, None)),
    ({"completion_tokens": 5}, (None, 5, None, None)),
    ({"total_tokens": 12}, (None, None, 12, 12)),
    ({}, (None, None, None, None)),
    ({"prompt_tokens": 0, "completion_tokens": 0}, (0, 0, 0, None)),
    ({"prompt_tokens": 8, "completion_tokens": 2}, (8, 2, 10, None)),
    ({"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}, (8, 2, 10, 10)),
    (
        {"prompt_tokens": "8", "completion_tokens": True, "total_tokens": "9"},
        (None, None, None, None),
    ),
]


def _adapter(usage):
    def respond(_request):
        return httpx.Response(
            200,
            json={
                "id": "wire-1",
                "model": "fixture",
                "object": "chat.completion",
                "created": 0,
                "system_fingerprint": None,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"score": true, "reasoning": "ok"}',
                        },
                    }
                ],
                "usage": usage,
            },
        )

    client = OpenRouter(
        api_key="dummy",
        retry_config=None,
        client=httpx.Client(transport=httpx.MockTransport(respond)),
        async_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )

    # Installed SDK schema rejects partial wire mappings before conversion.  The
    # dummy transport preserves the wire response; only SDK unmarshalling is
    # substituted so the installed adapter conversion can be exercised.
    def send(**_kwargs):
        return client.sdk_configuration.client.post(
            "https://fixture.invalid/chat", json={"stream": False}
        ).json()

    async def send_async(**_kwargs):
        response = await client.sdk_configuration.async_client.post(
            "https://fixture.invalid/chat", json={"stream": False}
        )
        return response.json()

    client.chat.send = send
    client.chat.send_async = send_async
    return ChatOpenRouter(
        model="fixture", api_key="dummy", client=client, max_retries=0
    )


@pytest.mark.parametrize("wire,expected", CASES)
def test_adapter_wire_runtime_and_judge_availability(monkeypatch, wire, expected):
    model = _adapter(wire)

    class Graph(FakeGraph):
        async def astream_events(self, initial, config, version):
            get_current_chat_turn().llm_provider_attempts = 1
            output = await model.ainvoke([HumanMessage(content="dummy")])
            yield {
                "event": "on_chat_model_end",
                "name": "ChatOpenRouter",
                "run_id": "event-1",
                "metadata": {
                    "langgraph_node": "researcher",
                    "ls_provider": "openrouter",
                },
                "data": {"output": output},
            }
            yield {
                "event": "on_chain_end",
                "name": "LangGraph",
                "data": {"output": self.state(initial)},
            }

    w = wrapper(Graph())
    w._workflow = SimpleNamespace(researcher_structured=model)
    monkeypatch.setattr(runtime_live, "get_workflow", lambda: w)
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        record = asyncio.run(runtime_live.run_controlled_live_scenario(make_scenario()))
        with _judge_transport_boundary(model, "openrouter", "fixture") as state:
            raw = model.invoke([HumanMessage(content="dummy")])
        judge = _judge_diagnostics(
            exc=None,
            raw=raw,
            attempted=state["attempted"],
            transport_metadata=state.get("response_metadata"),
        )
    provider = record.providers[0]
    assert (
        provider.input_tokens,
        provider.output_tokens,
        provider.total_tokens,
        provider.reported_total_tokens,
    ) == expected
    assert provider.token_sources["input_tokens"] == (
        "wire_response" if expected[0] is not None else "unavailable"
    )
    assert provider.token_sources["output_tokens"] == (
        "wire_response" if expected[1] is not None else "unavailable"
    )
    assert provider.token_sources["total_tokens"] == (
        "wire_response"
        if expected[3] is not None
        else "derived_input_output"
        if expected[2] is not None
        else "unavailable"
    )
    restored = RuntimeRecord.model_validate_json(record.model_dump_json())
    assert restored.providers[0] == provider
    for index, field in enumerate(("input_tokens", "output_tokens", "total_tokens")):
        cov = restored.usage.token_coverage[field]
        assert cov["observed"] == int(expected[index] is not None)
        assert cov["attempted"] == 1
        assert cov["complete"] == (expected[index] is not None)
    from backend.evaluation.evaluation_reporting import _token_usage_summary

    summary = _token_usage_summary([restored.model_dump(mode="json")])
    for index, field in enumerate(("input_tokens", "output_tokens", "total_tokens")):
        assert summary["token_coverage"][field]["observed_attempts"] == int(
            expected[index] is not None
        )
        assert summary[field]["mean"] == (
            float(expected[index]) if expected[index] is not None else None
        )
    judge_usage = judge["usage"]
    assert (
        tuple(
            getattr(judge_usage, k)
            for k in ("input_tokens", "output_tokens", "total_tokens")
        )
        == expected[:3]
    )
    scenario = make_scenario()
    scenario.expectations.budgets.max_total_tokens = 100
    check = next(
        c
        for c in score_runtime_record(scenario, restored).checks
        if c.name == "total_tokens"
    )
    assert check.status == ("pass" if expected[2] is not None else "not_observed")


def test_partial_wire_rejected_by_real_sdk_keeps_observed_usage(monkeypatch):
    wire = {"prompt_tokens": 8}
    model = _adapter(wire)
    # Restore the SDK's actual unmarshaller; it requires the omitted fields.
    from openrouter.chat import Chat

    model.client.chat.send_async = Chat.send_async.__get__(model.client.chat, Chat)
    model.client.chat.send = Chat.send.__get__(model.client.chat, Chat)

    class Graph(FakeGraph):
        async def astream_events(self, initial, config, version):
            get_current_chat_turn().llm_provider_attempts = 1
            await model.ainvoke([HumanMessage(content="dummy")])
            yield {}

    w = wrapper(Graph())
    w._workflow = SimpleNamespace(researcher_structured=model)
    monkeypatch.setattr(runtime_live, "get_workflow", lambda: w)
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        record = asyncio.run(runtime_live.run_controlled_live_scenario(make_scenario()))
        with _judge_transport_boundary(model, "openrouter", "fixture") as state:
            with pytest.raises(ResponseValidationError):
                model.invoke([HumanMessage(content="dummy")])
        judge = _judge_diagnostics(
            exc=ValueError("sdk rejected"),
            raw=None,
            attempted=state["attempted"],
            transport_metadata=state.get("response_metadata"),
            normalized_untrusted=True,
        )
    assert record.trace_status == "error"
    assert record.usage.provider_attempts == 1
    assert record.usage.input_tokens == 8
    assert record.usage.output_tokens is None and record.usage.total_tokens is None
    assert judge["usage"].input_tokens == 8
    assert judge["usage"].output_tokens is None and judge["usage"].total_tokens is None


def test_normalized_only_openrouter_trace_is_not_wire_evidence():
    from backend.evaluation.runtime_evaluation import _provider_observation

    fields = {
        "_id": "trace-event",
        "name": "ChatOpenRouter",
        "prompt_tokens": 8,
        "completion_tokens": 0,
        "total_tokens": 8,
        "extra": {"metadata": {"ls_provider": "openrouter"}},
        "outputs": {
            "generations": [
                [
                    {
                        "message": {
                            "kwargs": {
                                "response_metadata": {
                                    "model_provider": "openrouter",
                                    "token_usage": {"prompt_tokens": 8},
                                },
                                "usage_metadata": {
                                    "input_tokens": 8,
                                    "output_tokens": 0,
                                    "total_tokens": 8,
                                },
                            }
                        }
                    }
                ]
            ]
        },
    }
    result = _provider_observation(fields)
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (
        None,
        None,
        None,
    )


def test_structured_researcher_envelope_uses_same_wire_capture(monkeypatch):
    from backend.workflows.agents.schemas import ResearchResult

    model = _adapter({"prompt_tokens": 8})
    structured = model.with_structured_output(ResearchResult, include_raw=True)

    class Graph(FakeGraph):
        async def astream_events(self, initial, config, version):
            get_current_chat_turn().llm_provider_attempts = 1
            envelope = await structured.ainvoke([HumanMessage(content="dummy")])
            yield {
                "event": "on_chat_model_end",
                "name": "ChatOpenRouter",
                "run_id": "event-structured",
                "metadata": {
                    "langgraph_node": "researcher",
                    "ls_provider": "openrouter",
                },
                "data": {"output": envelope["raw"]},
            }
            yield {
                "event": "on_chain_end",
                "name": "LangGraph",
                "data": {"output": self.state(initial)},
            }

    w = wrapper(Graph())
    w._workflow = SimpleNamespace(researcher_structured=structured)
    monkeypatch.setattr(runtime_live, "get_workflow", lambda: w)
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        record = asyncio.run(runtime_live.run_controlled_live_scenario(make_scenario()))
    assert record.providers[0].input_tokens == 8
    assert record.providers[0].output_tokens is None
    assert record.usage.total_tokens is None


def test_streaming_response_is_not_buffered_by_evaluation_hook():
    from backend.evaluation.runtime_live import _capture_openrouter_wire_usage

    async def run():
        model = _adapter({})
        client = model.client.sdk_configuration.async_client

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"data: fixture\n\n"

        async def respond(_request):
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=Stream()
            )

        client._transport = httpx.MockTransport(respond)
        w = wrapper(FakeGraph())
        w._workflow = SimpleNamespace(analyst=model)
        with _capture_openrouter_wire_usage(w) as captured:
            async with client.stream(
                "POST", "https://fixture.invalid/chat", json={"stream": True}
            ) as response:
                assert not response.is_stream_consumed
                assert await response.aread() == b"data: fixture\n\n"
            assert captured == {}
        assert client.event_hooks["response"] == []

    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        asyncio.run(run())


@pytest.mark.parametrize(
    "wire,expected",
    [
        ({"prompt_tokens": 8}, (8, None, None)),
        ({"prompt_tokens": 0, "completion_tokens": 0}, (0, 0, 0)),
    ],
)
def test_real_judge_consumer_uses_wire_not_adapter_defaults(wire, expected):
    from synthetic_runtime import root_run

    from backend.evaluation.runtime_evaluation import (
        evaluate_with_openevals_judge,
        normalize_trace_runs,
    )

    model = _adapter(wire)
    scenario = make_scenario()
    scenario.expectations.judge_dimensions = ["analyst_usefulness"]
    record = normalize_trace_runs(root_run(), scenario)
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        result = evaluate_with_openevals_judge(
            scenario,
            record,
            "analyst_usefulness",
            judge=model,
            judge_provider="openrouter",
            judge_model="fixture",
            structured_output_method="json_schema",
            strict_structured_output=True,
        )
    assert result.status == "success"
    restored = type(result).model_validate_json(result.model_dump_json())
    assert (
        tuple(
            getattr(restored.usage, key)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        )
        == expected
    )
    assert restored.usage.provider_attempts == 1
    assert restored.usage_sources["input_tokens"] == "wire_response"
    assert restored.reported_total_tokens is None
    assert restored.usage_sources["total_tokens"] == (
        "derived_input_output" if expected[2] is not None else "unavailable"
    )


def test_normalized_only_live_openrouter_is_unavailable(monkeypatch):
    from langchain_core.messages import AIMessage

    class Graph(FakeGraph):
        async def astream_events(self, initial, config, version):
            get_current_chat_turn().llm_provider_attempts = 1
            yield {
                "event": "on_chat_model_end",
                "name": "ChatOpenRouter",
                "run_id": "e",
                "metadata": {
                    "langgraph_node": "orchestrator",
                    "ls_provider": "openrouter",
                },
                "data": {
                    "output": AIMessage(
                        content="",
                        usage_metadata={
                            "input_tokens": 8,
                            "output_tokens": 0,
                            "total_tokens": 8,
                        },
                    )
                },
            }
            yield {
                "event": "on_chain_end",
                "name": "LangGraph",
                "data": {"output": self.state(initial)},
            }

    monkeypatch.setattr(runtime_live, "get_workflow", lambda: wrapper(Graph()))
    result = asyncio.run(runtime_live.run_controlled_live_scenario(make_scenario()))
    assert result.usage.provider_attempts == 1
    assert (
        result.usage.input_tokens,
        result.usage.output_tokens,
        result.usage.total_tokens,
    ) == (None, None, None)


def test_duplicate_wire_response_ids_are_unavailable_and_hooks_restored():
    from backend.evaluation.runtime_live import _capture_openrouter_wire_usage

    async def run():
        model = _adapter(
            {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}
        )
        client = model.client.sdk_configuration.async_client
        w = wrapper(FakeGraph())
        w._workflow = SimpleNamespace(analyst=model)
        with _capture_openrouter_wire_usage(w) as captured:
            await asyncio.gather(
                *[
                    client.post("https://fixture.invalid/chat", json={"stream": False})
                    for _ in range(2)
                ]
            )
            assert captured == {"wire-1": None}
        assert client.event_hooks["response"] == []

    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        asyncio.run(run())


def test_unread_request_body_and_response_read_error_do_not_abort_request():
    from backend.evaluation.runtime_live import _capture_openrouter_wire_usage

    async def run():
        model = _adapter({})
        client = model.client.sdk_configuration.async_client
        w = wrapper(FakeGraph())
        w._workflow = SimpleNamespace(analyst=model)
        with _capture_openrouter_wire_usage(w) as captured:
            hook = client.event_hooks["response"][-1]

            class Broken:
                headers: ClassVar[dict[str, str]] = {"content-type": "application/json"}

                @property
                def request(self):
                    raise httpx.RequestNotRead()

            await hook(Broken())

            class ReadError:
                headers: ClassVar[dict[str, str]] = {"content-type": "application/json"}
                request = SimpleNamespace(content=b"{}")

                async def aread(self):
                    raise httpx.ResponseNotRead()

            await hook(ReadError())
            assert captured == {}
        assert client.event_hooks["response"] == []

    asyncio.run(run())


def test_judge_stream_hook_does_not_read_or_change_response():
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"data: fixture\n\n"

    model = _adapter({})
    client = model.client.sdk_configuration.client
    client._transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Stream()
        )
    )
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        with _judge_transport_boundary(model, "openrouter", "fixture") as state:
            with client.stream(
                "POST", "https://fixture.invalid/chat", json={"stream": True}
            ) as response:
                assert not response.is_stream_consumed
                assert response.read() == b"data: fixture\n\n"
        assert state["attempted"] is True
        assert state["response_metadata"] == {"http_status_code": 200}
    assert client.event_hooks["response"] == []


@pytest.mark.parametrize("observed_provider", [None, "anthropic"])
def test_identity_unknown_trace_does_not_trust_normalized_usage(observed_provider):
    from backend.evaluation.runtime_evaluation import _provider_observation

    run = {
        "_id": "unknown-event",
        "name": "model",
        "prompt_tokens": 8,
        "completion_tokens": 0,
        "total_tokens": 8,
        "outputs": {
            "generations": [
                [
                    {
                        "message": {
                            "kwargs": {
                                "usage_metadata": {
                                    "input_tokens": 8,
                                    "output_tokens": 0,
                                    "total_tokens": 8,
                                },
                                "response_metadata": {
                                    "model_provider": observed_provider
                                },
                            }
                        }
                    }
                ]
            ]
        },
    }
    provider = _provider_observation(run)
    assert (provider.input_tokens, provider.output_tokens, provider.total_tokens) == (
        None,
        None,
        None,
    )


def test_openrouter_adapter_with_serving_provider_label_still_untrusted(monkeypatch):
    from langchain_core.messages import AIMessage

    class Graph(FakeGraph):
        async def astream_events(self, initial, config, version):
            get_current_chat_turn().llm_provider_attempts = 1
            yield {
                "event": "on_chat_model_end",
                "name": "ChatOpenRouter",
                "run_id": "e",
                "metadata": {
                    "langgraph_node": "orchestrator",
                    "ls_provider": "openrouter",
                },
                "data": {
                    "output": AIMessage(
                        content="",
                        response_metadata={"model_provider": "serving-provider"},
                        usage_metadata={
                            "input_tokens": 8,
                            "output_tokens": 0,
                            "total_tokens": 8,
                        },
                    )
                },
            }
            yield {
                "event": "on_chain_end",
                "name": "LangGraph",
                "data": {"output": self.state(initial)},
            }

    monkeypatch.setattr(runtime_live, "get_workflow", lambda: wrapper(Graph()))
    result = asyncio.run(runtime_live.run_controlled_live_scenario(make_scenario()))
    assert (
        result.usage.input_tokens,
        result.usage.output_tokens,
        result.usage.total_tokens,
    ) == (None, None, None)


def test_identity_unknown_trace_serialization_never_passes_token_budget():
    from synthetic_runtime import root_run

    from backend.evaluation.runtime_evaluation import normalize_trace_runs

    runs = root_run()
    root = next(run for run in runs if not run.get("parent_run_id"))
    root["provider_attempts"] = 1
    run = next(run for run in runs if run.get("run_type") == "llm")
    run["name"] = "unknown-model"
    run["extra"]["metadata"].pop("ls_provider", None)
    run["prompt_tokens"], run["completion_tokens"], run["total_tokens"] = 8, 0, 8
    kwargs = run["outputs"]["generations"][0][0]["message"]["kwargs"]
    kwargs["response_metadata"] = {}
    kwargs["usage_metadata"] = {
        "input_tokens": 8,
        "output_tokens": 0,
        "total_tokens": 8,
    }
    scenario = make_scenario()
    scenario.expectations.budgets.max_total_tokens = 100
    record = normalize_trace_runs(runs, scenario)
    restored = RuntimeRecord.model_validate_json(record.model_dump_json())
    assert restored.usage.provider_attempts == 1
    assert restored.usage.token_coverage["total_tokens"]["observed"] == 0
    assert restored.usage.total_tokens is None
    check = next(
        item
        for item in score_runtime_record(scenario, restored).checks
        if item.name == "total_tokens"
    )
    assert check.status == "not_observed"


def test_duplicate_model_callbacks_for_one_wire_response_count_once(monkeypatch):
    model = _adapter({"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10})

    class Graph(FakeGraph):
        async def astream_events(self, initial, config, version):
            get_current_chat_turn().llm_provider_attempts = 2
            output = await model.ainvoke([HumanMessage(content="dummy")])
            for event_id in ("one", "two"):
                yield {
                    "event": "on_chat_model_end",
                    "name": "ChatOpenRouter",
                    "run_id": event_id,
                    "metadata": {
                        "langgraph_node": "researcher",
                        "ls_provider": "openrouter",
                    },
                    "data": {"output": output},
                }
            yield {
                "event": "on_chain_end",
                "name": "LangGraph",
                "data": {"output": self.state(initial)},
            }

    w = wrapper(Graph())
    w._workflow = SimpleNamespace(researcher_structured=model)
    monkeypatch.setattr(runtime_live, "get_workflow", lambda: w)
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        record = asyncio.run(runtime_live.run_controlled_live_scenario(make_scenario()))
    assert record.usage.provider_attempts == 2
    assert record.usage.total_tokens is None
    assert record.usage.token_coverage["total_tokens"] == {
        "observed": 1,
        "attempted": 2,
        "subtotal": 10,
        "complete": False,
    }


def test_verified_wire_fields_override_even_trusted_canonical_metadata():
    from backend.evaluation.runtime_evaluation import _provider_response_fields

    fields = _provider_response_fields(
        {"token_usage": {"prompt_tokens": 8}},
        {"input_tokens": 8, "output_tokens": 0, "total_tokens": 8},
        wire_captured=True,
    )
    assert (
        fields["input_tokens"],
        fields["output_tokens"],
        fields["total_tokens"],
        fields["reported_total_tokens"],
    ) == (8, None, None, None)


def test_live_verified_wire_overrides_conflicting_adapter_label(monkeypatch):
    model = _adapter({"prompt_tokens": 8})

    class Graph(FakeGraph):
        async def astream_events(self, initial, config, version):
            get_current_chat_turn().llm_provider_attempts = 1
            output = await model.ainvoke([HumanMessage(content="dummy")])
            output.response_metadata["model_provider"] = "ollama"
            yield {
                "event": "on_chat_model_end",
                "name": "ChatOllama",
                "run_id": "event",
                "metadata": {"langgraph_node": "researcher", "ls_provider": "ollama"},
                "data": {"output": output},
            }
            yield {
                "event": "on_chain_end",
                "name": "LangGraph",
                "data": {"output": self.state(initial)},
            }

    w = wrapper(Graph())
    w._workflow = SimpleNamespace(researcher_structured=model)
    monkeypatch.setattr(runtime_live, "get_workflow", lambda: w)
    with patch(
        "socket.socket.connect", side_effect=AssertionError("network forbidden")
    ):
        record = asyncio.run(runtime_live.run_controlled_live_scenario(make_scenario()))
    assert (
        record.usage.input_tokens,
        record.usage.output_tokens,
        record.usage.total_tokens,
    ) == (8, None, None)
