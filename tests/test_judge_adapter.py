import json
from unittest.mock import patch

import httpx
from langchain_openrouter import ChatOpenRouter
from openrouter import OpenRouter
from synthetic_runtime import make_scenario, root_run

from backend.evaluation.runtime_evaluation import (
    evaluate_with_openevals_judge,
    normalize_trace_runs,
)


def test_strict_judge_serializes_boolean_and_captures_metadata():
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "offline-id",
                "model": "offline-model",
                "provider": "Phala",
                "object": "chat.completion",
                "created": 0,
                "system_fingerprint": None,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {"score": True, "reasoning": "direct"}
                            ),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    client = OpenRouter(
        api_key="dummy",
        retry_config=None,
        client=httpx.Client(
            transport=httpx.MockTransport(respond), follow_redirects=False
        ),
    )
    base = ChatOpenRouter(
        model="xingchen-agi/xing4.0-29b-a4b:free",
        api_key="dummy",
        client=client,
        max_retries=0,
    )
    judge = base.model_copy(update={"temperature": 0, "max_tokens": 512})
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
            judge=judge,
            judge_provider="openrouter",
            judge_model="xingchen-agi/xing4.0-29b-a4b:free",
            structured_output_method="json_schema",
            strict_structured_output=True,
        )
    assert result.status == "success", result
    assert result.score is True
    assert result.stop_reason == "stop"
    assert result.usage.usage_method == "event_unique_v1"
    assert result.usage.total_tokens == 2
    assert result.usage.token_coverage["total_tokens"] == {
        "observed": 1,
        "attempted": 1,
        "subtotal": 2,
        "complete": True,
    }
    assert len(requests) == 1
    body = requests[0]
    assert body["temperature"] == 0
    assert body["max_tokens"] == 512
    assert body["response_format"]["json_schema"]["strict"] is True
    assert set(body["response_format"]["json_schema"]["schema"]["required"]) == {
        "score",
        "reasoning",
    }


def test_local_prompt_format_error_does_not_reserve_or_dispatch():
    from backend.evaluation import runtime_evaluation as evaluation

    calls = []
    client = OpenRouter(
        api_key="dummy",
        retry_config=None,
        client=httpx.Client(
            transport=httpx.MockTransport(lambda request: calls.append(request))
        ),
    )
    judge = ChatOpenRouter(
        model="xingchen-agi/xing4.0-29b-a4b:free",
        api_key="dummy",
        client=client,
        max_retries=0,
    )
    scenario = make_scenario()
    scenario.expectations.judge_dimensions = ["analyst_usefulness"]
    record = normalize_trace_runs(root_run(), scenario)
    with patch.dict(
        evaluation.SEMANTIC_JUDGE_RUBRICS,
        {"analyst_usefulness": "Broken {unknown_key}"},
    ):
        result = evaluate_with_openevals_judge(
            scenario,
            record,
            "analyst_usefulness",
            judge=judge,
            judge_provider="openrouter",
            judge_model="xingchen-agi/xing4.0-29b-a4b:free",
            structured_output_method="json_schema",
            strict_structured_output=True,
        )
    assert result.status == "error"
    assert result.error_type == "KeyError"
    assert result.provider_call_attempted is False
    assert result.transport_status == "not_attempted"
    assert result.usage.token_coverage["total_tokens"]["attempted"] == 0
    assert result.usage.total_tokens is None
    assert calls == []
