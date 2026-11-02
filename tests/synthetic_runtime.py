from backend.evaluation.runtime_evaluation import RuntimeScenario


def make_scenario(**overrides):
    payload = {
        "schema_version": "runtime-evaluation",
        "id": "runtime_case",
        "version": 1,
        "review_status": "reviewed",
        "description": "Synthetic runtime contract.",
        "trace_ids": ["trace_1"],
        "input": {"user_message": "Explain Mamba"},
        "expectations": {
            "allowed_worker_sequences": [["retriever", "analyst"]],
            "required_agents": ["retriever", "analyst"],
            "required_tools": ["search_knowledge_base"],
            "request_scope": "in_scope",
            "allowed_retrieval_statuses": ["adequate"],
            "retrieval_outcome": "adequate",
            "sources_exhausted": False,
            "requested_modality": "text",
            "final_response_required": True,
            "forbidden_failure_types": ["provider_error"],
            "budgets": {
                "max_worker_steps": 2,
                "max_provider_attempts": 3,
                "max_tavily_searches": 0,
            },
        },
        "provenance": ["synthetic"],
    }
    payload.update(overrides)
    return RuntimeScenario.model_validate(payload)


def llm_run(*, stop_reason="stop", content="ok", tool_calls=None):
    return {
        "_id": "llm_1",
        "trace_id": "trace_1",
        "parent_run_id": "wrapper_1",
        "name": "ChatOllama",
        "run_type": "llm",
        "status": "success",
        "start_time": "2026-01-01T00:00:03Z",
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "extra": {
            "metadata": {
                "langgraph_node": "retriever",
                "ls_provider": "ollama",
                "ls_model_name": "model",
            }
        },
        "outputs": {
            "generations": [
                [
                    {
                        "message": {
                            "kwargs": {
                                "content": content,
                                "tool_calls": tool_calls or [],
                                "response_metadata": {"done_reason": stop_reason},
                            }
                        }
                    }
                ]
            ]
        },
    }


def root_run(*, retrieval_status="adequate", sources_exhausted=False):
    human = {"type": "human", "kwargs": {"content": "Explain Mamba"}}
    answer = {
        "type": "ai",
        "kwargs": {
            "content": "Grounded answer.",
            "additional_kwargs": {"agent": "[ANALYST]"},
        },
    }
    return [
        {
            "_id": "root_1",
            "trace_id": "trace_1",
            "parent_run_id": None,
            "name": "LangGraph",
            "run_type": "chain",
            "status": "success",
            "start_time": "2026-01-01T00:00:00Z",
            "latency": 1.25,
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "inputs": {"messages": [human]},
            "outputs": {
                "messages": [human, answer],
                "request_scope": "in_scope",
                "sources_exhausted": sources_exhausted,
                "next_step": "__end__",
                "visual_artifacts": [],
            },
            "extra": {
                "metadata": {
                    "thread_id": "thread_1",
                    "revision_id": "fixture-revision",
                }
            },
        },
        {
            "_id": "orchestrator_1",
            "trace_id": "trace_1",
            "parent_run_id": "root_1",
            "name": "orchestrator",
            "run_type": "chain",
            "status": "success",
            "start_time": "2026-01-01T00:00:01Z",
            "outputs": {"next_step": "retriever"},
        },
        {
            "_id": "retriever_1",
            "trace_id": "trace_1",
            "parent_run_id": "root_1",
            "name": "retriever",
            "run_type": "chain",
            "status": "success",
            "start_time": "2026-01-01T00:00:02Z",
            "outputs": {"retrieval_status": retrieval_status},
        },
        llm_run(),
        {
            "_id": "tool_1",
            "trace_id": "trace_1",
            "parent_run_id": "retriever_1",
            "name": "search_knowledge_base",
            "run_type": "tool",
            "status": "success",
            "start_time": "2026-01-01T00:00:04Z",
            "inputs": {"queries": ["Mamba"], "api_key": "never-export"},
            "outputs": {"output": [{"id": "chunk_mamba"}]},
            "extra": {"metadata": {"langgraph_node": "retriever"}},
        },
        {
            "_id": "analyst_1",
            "trace_id": "trace_1",
            "parent_run_id": "root_1",
            "name": "analyst",
            "run_type": "chain",
            "status": "success",
            "start_time": "2026-01-01T00:00:05Z",
            "outputs": {},
        },
    ]
