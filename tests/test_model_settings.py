from backend.configs.enums import LLMProviderType, ModelRoleType
from backend.configs.models import OllamaSettings, OpenAISettings, RerankerSettings


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
    )

    assert settings.ollama_chat_params() == {
        "temperature": 0.1,
        "top_k": 12,
        "top_p": 0.7,
        "num_predict": 2048,
        "base_url": "https://ollama.example",
        "num_ctx": 16384,
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
