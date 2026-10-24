from backend.configs.enums import LLMProviderType, ModelRoleType
from backend.configs.models import (
    ModelSettings,
    OllamaSettings,
    OpenAISettings,
    OpenRouterSettings,
    RerankerSettings,
    _vllm_models,
    get_model_settings,
)


def test_ollama_chat_params_are_constructor_only_fields() -> None:
    settings = OllamaSettings(
        role=ModelRoleType.analyst,
        model_name="test-model",
        temperature=0.1,
        top_k=12,
        top_p=0.7,
        num_predict=2048,
        base_url="https://ollama.example",
        num_ctx=16384,
        prompt_version="v99",
        reasoning=True,
    )

    assert settings.ollama_chat_params() == {
        "temperature": 0.1,
        "top_k": 12,
        "top_p": 0.7,
        "num_predict": 2048,
        "base_url": "https://ollama.example",
        "num_ctx": 16384,
        "reasoning": True,
    }


def test_openai_style_chat_params_are_provider_specific() -> None:
    settings = OpenAISettings(
        role=ModelRoleType.retriever,
        provider=LLMProviderType.vllm,
        model_name="served-model",
        temperature=0.0,
        top_p=0.4,
        max_tokens=512,
        base_url="https://vllm.example",
    )

    assert settings.vllm_chat_params() == {
        "temperature": 0.0,
        "top_p": 0.4,
        "max_tokens": 512,
    }
    assert settings.openai_chat_params() == {
        "temperature": 0.0,
        "top_p": 0.4,
        "max_tokens": 512,
        "base_url": "https://vllm.example",
    }


def test_default_openrouter_roles_preserve_generation_contract(monkeypatch) -> None:
    from backend.workflows.agents import factory

    monkeypatch.delenv("LLM_PIPELINE", raising=False)
    monkeypatch.setattr(factory, "ChatOpenRouter", lambda **kwargs: kwargs)
    settings = get_model_settings()
    roles = [
        (settings.orchestrator, 2048, 0.0, 0.5, None),
        (settings.retriever, 512, 0.0, 0.3, None),
        (settings.researcher.with_tools, 1024, 0.0, 0.3, None),
        (settings.researcher.structured, 4096, 0.0, 0.5, None),
        (settings.analyst, 8192, 0.1, 0.8, {"effort": "low"}),
        (settings.mentor, 4096, 0.7, 0.8, {"effort": "low"}),
    ]
    for role, limit, temperature, top_p, reasoning in roles:
        assert role.provider == LLMProviderType.openrouter
        assert role.model_name == "openrouter/free"
        expected = {
            "model": "openrouter/free",
            "api_key": None,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": limit,
            "request_timeout": 120000,
        }
        if reasoning is not None:
            expected["reasoning"] = reasoning
        assert factory.AgentsFactory.get_llm_by_role(role) == expected
    assert settings.analyst.prompt_version == "v6"
    assert settings.mentor.prompt_version == "v4"


def test_openrouter_settings_select_provider_and_forward_reasoning(monkeypatch) -> None:
    from backend.workflows.agents import factory

    monkeypatch.setattr(factory, "ChatOpenRouter", lambda **kwargs: kwargs)
    settings = OpenRouterSettings(
        _env_file=None,
        role=ModelRoleType.analyst,
        model_name="provider/model",
        api_key="synthetic-key",
        provider=LLMProviderType.openrouter,
        reasoning={"effort": "low"},
        request_timeout=1.5,
    )
    assert settings.reasoning == {"effort": "low"}
    assert factory.AgentsFactory.get_llm_by_role(settings) == {
        "model": "provider/model",
        "api_key": "synthetic-key",
        "temperature": 0.0,
        "top_p": 0.5,
        "max_tokens": 128,
        "reasoning": {"effort": "low"},
        "request_timeout": 1500,
    }


def test_openrouter_settings_default_to_openrouter_provider():
    settings = OpenRouterSettings(
        _env_file=None, role=ModelRoleType.analyst, model_name="test-model"
    )
    assert settings.provider == LLMProviderType.openrouter


def test_ollama_pipeline_selector_remains_openrouter_compatibility_alias(monkeypatch):
    monkeypatch.setenv("LLM_PIPELINE", "ollama")
    assert get_model_settings().analyst.provider == LLMProviderType.openrouter


def test_runtime_routing_prompt_versions_are_current(monkeypatch) -> None:
    default = ModelSettings()
    assert default.orchestrator.prompt_version["routing"] == "v5"
    assert default.retriever.prompt_version == "v6"
    assert default.researcher.prompt_version == "v5"

    monkeypatch.setenv("VLLM_ROUTING_URL", "https://vllm.example")
    monkeypatch.setenv("VLLM_GENERATION_URL", "https://vllm-generation.example")
    monkeypatch.setenv("VLLM_MODEL_PATH", "test-model")
    vllm = _vllm_models()
    assert vllm.orchestrator.prompt_version["routing"] == "v5"
    assert vllm.retriever.prompt_version == "v6"
    assert vllm.researcher.prompt_version == "v5"


def test_graph_index_ollama_params_include_model_and_indexing_generation_fields() -> (
    None
):
    settings = OllamaSettings(
        role=ModelRoleType.orchestrator,
        model_name="graph-model",
        temperature=0.0,
        top_k=10,
        top_p=0.9,
        base_url="https://ollama.example",
    )

    assert settings.graph_index_ollama_params() == {
        "model": "graph-model",
        "temperature": 0.0,
        "top_k": 10,
        "top_p": 0.9,
        "base_url": "https://ollama.example",
    }


def test_reranker_settings_expose_constructor_specific_params() -> None:
    settings = RerankerSettings(
        model_path="models/qwen-reranker",
        device="cpu",
        batch_size=4,
        top_n=9,
        local_files_only=True,
        trust_remote_code=False,
        base_url="http://localhost:11434",
        request_timeout=30.0,
        choice_batch_size=3,
    )

    assert settings.sentence_transformer_params() == {
        "model_name": "models/qwen-reranker",
        "top_n": 9,
        "batch_size": 4,
        "device": "cpu",
        "local_files_only": True,
        "trust_remote_code": False,
    }
    assert settings.ollama_llm_rerank_params() == {
        "model": "models/qwen-reranker",
        "base_url": "http://localhost:11434",
        "request_timeout": 30.0,
        "choice_batch_size": 3,
        "top_n": 9,
    }
