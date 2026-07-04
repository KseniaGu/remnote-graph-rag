from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.configs.constants import ENV_PATH

RetrievalPipelineMode = Literal["optimized", "legacy_vector_context"]


class TavilySettings(BaseSettings):
    """Tavily search engine settings configuration."""
    api_key: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="TAVILY_",
    )


class KnowledgeGraphSearchSettings(BaseSettings):
    """Knowledge graph search settings configuration."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="KG_SEARCH_",
        populate_by_name=True,
    )

    analyst_retrieval_mode: RetrievalPipelineMode = "legacy_vector_context"
    visualizer_retrieval_mode: RetrievalPipelineMode = "optimized"
    retriever_params: dict = {
        "VectorContextRetriever": {
            "include_text": True, "similarity_top_k": 5, "similarity_score": None, "path_depth": 2,
            "include_properties": True
        }
    }
    visualizer_retriever_params: dict = {
        "VectorContextRetriever": {
            "include_text": False, "similarity_top_k": 5, "similarity_score": None, "path_depth": 4,
            "include_properties": False
        },
        # "VectorIndexRetriever": {"similarity_top_k": 10,}
    }
    analyst_source_candidate_k: int = 30
    analyst_source_final_k: int = 6
    analyst_source_min_relative_score: float = 0.35
    analyst_source_min_raw_margin: float = 4.0
    analyst_source_min_keep: int = 3
    analyst_source_max_per_path: int = 2
    analyst_source_exact_topic_boost: float = 0.05
    analyst_source_fill_min_score: float = 0.50
    analyst_source_fill_min_relative_score: float = 0.45
    analyst_relation_final_k: int = 5
    analyst_relation_min_relative_score: float = 0.50
    analyst_relation_min_raw_margin: float = 3.0
    analyst_relation_seed_extra_k: int = 4
    analyst_relation_seed_min_score: float = 0.50
    analyst_context_max_chars: int = 7000
    analyst_graph_depth: int = 1
    analyst_graph_relation_limit: int = 30
    analyst_reranker_mode: Literal["disabled", "sentence_transformers", "ollama_llm_rerank"] = "sentence_transformers"
    analyst_source_rerank_candidate_k: int = 10
    analyst_source_rerank_max_chars: int = 1200
    analyst_relation_reranker_enabled: bool = False
    analyst_relation_rerank_candidate_k: int = 8
    analyst_relation_rerank_max_chars: int = 1000
    analyst_relation_require_source_evidence: bool = True
    visualizer_anchor_top_k: int = 3
    visualizer_anchor_min_score: float = 0.50
    visualizer_source_candidate_k: int = 20
    visualizer_concept_candidate_k: int = 25
    visualizer_max_nodes: int = 25
    visualizer_max_edges: int = 35
    visualizer_min_nodes: int = 3
    visualizer_max_edges_per_node: int = 6
    visualizer_graph_depth: int = 2
    visualizer_allow_synthetic_edges: bool = True
    visualizer_synthetic_edge_limit: int = 6
    visualizer_anchor_source_filter: bool = True
    visualizer_include_isolated_nodes: bool = False
    visualizer_synthetic_edge_label: str = "RELATED_TO"
    visualizer_show_chunks: bool = False
    visualizer_denied_relation_labels: tuple[str, ...] = ("MENTIONS", "PARENT", "CHILD")
