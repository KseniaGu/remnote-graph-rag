from backend.configs.enums import LLMProviderType, ModelRoleType
from backend.configs.models import (
    ModelSettings,
    OllamaSettings,
    OpenAISettings,
    RerankerSettings,
    _vllm_models,
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


def test_default_analyst_uses_v6_educational_cloud_configuration() -> None:
    analyst = ModelSettings().analyst

    assert analyst.model_name == "qwen3.5:cloud"
    assert analyst.prompt_version == "v6"
    assert analyst.num_predict == 8192
    assert analyst.request_timeout == 120.0
    assert analyst.num_ctx == 32768
    assert analyst.ollama_chat_params()["reasoning"] is False
    assert ModelSettings().mentor.num_predict == 4096

    researcher = ModelSettings().researcher.structured
    assert "reasoning" not in researcher.ollama_chat_params()


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
