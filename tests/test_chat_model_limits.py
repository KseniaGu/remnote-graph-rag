from pathlib import Path
from unittest.mock import patch

from backend.configs.models import ModelSettings, _vllm_models
from backend.workflows.agents.factory import AgentsFactory


def test_ollama_timeout_is_forwarded_to_both_clients() -> None:
    settings = ModelSettings().analyst

    with patch("backend.workflows.agents.factory.ChatOllama") as chat_ollama:
        AgentsFactory.get_llm_by_role(settings)

    kwargs = chat_ollama.call_args.kwargs
    assert kwargs["client_kwargs"]["timeout"] == 120.0
    assert kwargs["async_client_kwargs"]["timeout"] == 120.0


def test_analyst_and_mentor_output_caps_for_vllm(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ROUTING_URL", "https://routing.example")
    monkeypatch.setenv("VLLM_GENERATION_URL", "https://generation.example")
    monkeypatch.setenv("VLLM_MODEL_PATH", "test-model")

    settings = _vllm_models()

    assert settings.analyst.max_tokens == 8192
    assert settings.mentor.max_tokens == 4096


def test_active_prompt_metadata_uses_the_runtime_output_cap() -> None:
    prompt_root = (
        Path(__file__).parents[1] / "backend" / "llm" / "prompts" / "learner_workflow"
    )

    active_prompts = (
        (prompt_root / "analyst" / "v6.yaml", 8192),
        (prompt_root / "analyst" / "v4.yaml", 8192),
        (prompt_root / "mentor" / "v4.yaml", 4096),
    )

    for prompt_path, output_cap in active_prompts:
        assert f"num_predict: {output_cap}" in prompt_path.read_text(encoding="utf-8")
