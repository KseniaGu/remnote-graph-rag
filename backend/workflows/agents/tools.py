import asyncio
import json
import random
import re
from typing import Annotated, Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from backend.configs.chat_limits import ChatLimitsSettings
from backend.configs.constants import (
    MIN_RELEVANCE_SCORE,
    RETRIEVAL_BELOW_THRESHOLD,
    RETRIEVAL_NO_RESULTS,
    RETRIEVAL_TOPIC_MISMATCH,
    WORKFLOW_LOGGING,
)
from backend.utils.chat_errors import (
    KnowledgeBaseUnavailable,
    UserFacingChatError,
    WebSearchCapacity,
    WebSearchUnavailable,
)
from backend.utils.chat_limits import (
    ChatLimitService,
    get_chat_limit_service,
    get_current_chat_turn,
    is_transient_provider_error,
    provider_status_code,
    truncate_text,
)
from backend.utils.helpers import get_logger
from backend.workflows.agents.retrieval_evidence import (
    QueryEvidenceResult,
    evidence_from_retrieved_node,
    format_search_results,
    format_visualization_results,
)

logger = get_logger(WORKFLOW_LOGGING)


class QueryListInput(BaseModel):
    """Validated query-list arguments shared by knowledge retrieval tools."""

    queries: list[str] = Field(
        min_length=1,
        max_length=3,
        description=(
            "A native JSON array containing 1 to 3 distinct semantic search queries. "
            "Do not encode the array as a string."
        ),
    )

    @field_validator("queries", mode="before")
    @classmethod
    def normalize_stringified_queries(cls, value: Any) -> Any:
        """Recovers common stringified-list tool calls while preserving array schema."""
        if not isinstance(value, str):
            return value

        raw_value = value.strip()
        if not raw_value:
            return []

        if raw_value.startswith("[") and raw_value.endswith("]"):
            try:
                decoded = json.loads(raw_value)
            except json.JSONDecodeError:
                inner = raw_value[1:-1].strip()
                if not inner:
                    return []
                if "," in inner:
                    return [
                        part.strip().strip("'\"")
                        for part in inner.split(",")
                        if part.strip().strip("'\"")
                    ]
                return [inner.strip("'\"")]
            else:
                return decoded

        return [raw_value]

    @field_validator("queries")
    @classmethod
    def normalize_query_text(cls, queries: list[str]) -> list[str]:
        normalized = list(
            dict.fromkeys(query.strip() for query in queries if query.strip())
        )
        if not normalized:
            raise ValueError("at least one non-empty query is required")
        if any(len(query) > 256 for query in normalized):
            raise ValueError("each query must contain at most 256 characters")
        return normalized


TopicAliasGroup = Annotated[list[str], Field(min_length=1, max_length=3)]


class KnowledgeSearchInput(QueryListInput):
    """Knowledge-search arguments with deterministic named-topic requirements."""

    required_topics: list[TopicAliasGroup] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Up to three required topic groups. At least one alias in every group "
            "must occur in retrieved evidence."
        ),
    )

    @field_validator("required_topics")
    @classmethod
    def normalize_required_topics(
        cls, required_topics: list[list[str]]
    ) -> list[list[str]]:
        normalized_groups: list[list[str]] = []
        for group in required_topics:
            if len(group) > 3:
                raise ValueError(
                    "each required topic group may contain at most 3 aliases"
                )

            normalized_aliases: list[str] = []
            seen_aliases: set[str] = set()
            for alias in group:
                alias = " ".join(alias.strip().split())
                if not alias:
                    continue
                if len(alias) > 80:
                    raise ValueError(
                        "each required topic alias must contain at most 80 characters"
                    )
                dedupe_key = alias.casefold()
                if dedupe_key not in seen_aliases:
                    seen_aliases.add(dedupe_key)
                    normalized_aliases.append(alias)

            if not normalized_aliases:
                raise ValueError("required topic groups must contain a non-empty alias")
            normalized_groups.append(normalized_aliases)

        return normalized_groups


def _topic_tokens(text: str) -> list[str]:
    """Normalizes punctuation and case while retaining exact token boundaries."""
    return re.findall(r"[^\W_]+", text.casefold())


def _contains_token_sequence(tokens: list[str], alias_tokens: list[str]) -> bool:
    if not alias_tokens or len(alias_tokens) > len(tokens):
        return False
    width = len(alias_tokens)
    return any(
        tokens[index : index + width] == alias_tokens
        for index in range(len(tokens) - width + 1)
    )


def retrieval_covers_required_topics(
    formatted_result: str, required_topics: list[list[str]]
) -> bool:
    """Checks that formatted evidence covers every required topic group."""
    if not required_topics:
        return True

    evidence_lines = []
    for line in formatted_result.splitlines():
        stripped = line.strip()
        if stripped.startswith("QUERY:") or stripped == "RETRIEVER RESULTS:":
            continue
        evidence_lines.append(line)

    evidence_tokens = _topic_tokens("\n".join(evidence_lines))
    return all(
        any(
            _contains_token_sequence(evidence_tokens, _topic_tokens(alias))
            for alias in group
        )
        for group in required_topics
    )


class WebResearchInput(BaseModel):
    """Validated arguments for one Tavily search."""

    topic: str = Field(min_length=1, max_length=256)

    @field_validator("topic")
    @classmethod
    def normalize_topic(cls, topic: str) -> str:
        """Rejects whitespace-only topics and searches the normalized value."""
        topic = topic.strip()
        if not topic:
            raise ValueError("a non-empty research topic is required")
        return topic


def search_knowledge_base(
    retriever: Any = None,
    reranker: Any = None,
    analyst_pipeline: Any = None,
    max_context_chars: int | None = None,
):
    context_limit = (
        max_context_chars or ChatLimitsSettings().retrieval_context_max_chars
    )

    @tool("search_knowledge_base", args_schema=KnowledgeSearchInput)
    def _search_knowledge_base(
        queries: list[str], required_topics: list[list[str]] | None = None
    ):
        """Searches the knowledge base using advanced graph and vector retrieval.

        This tool performs semantic search across the entire knowledge base, combining vector similarity search with graph traversal to find relevant information. It retrieves facts, concepts, relationships, and text snippets from stored documents.

        Use this as the primary search method for:
        - Finding definitions, explanations, and detailed concepts
        - Retrieving relationships between topics and entities
        - Accessing stored knowledge from the documents
        - Getting contextual information about specific subjects

        Search results include:
        - Relations: Graph connections between concepts (e.g., "Topic A -> RELATES_TO -> Topic B")
        - Sources: Text snippets from documents with relevance scores
        - Source paths: Hierarchical paths showing document structure

        Args:
            queries: Search queries for relevant knowledge-base information.
            required_topics: Optional alias groups that must all be covered by the
                formatted evidence before it can be used for generation.

        Returns:
            Formatted search results, or a deterministic inadequacy sentinel when
            results are empty, below threshold, or miss a required topic.

        Example output:
            [RELATION] Classical Computer Vision -> FOCUSES_ON -> Handcrafted features (Score: 0.85)
            [SOURCE] Classical computer vision relied on manually engineered features...
            [SOURCE PATH] CV History > Classical Era > Feature Extraction
        """
        required_topics = required_topics or []
        if analyst_pipeline is not None:
            try:
                result = analyst_pipeline.search(queries)
            except UserFacingChatError:
                raise
            except Exception as exc:
                logger.error(
                    "Optimized knowledge retrieval failed",
                    error_type=type(exc).__name__,
                )
                raise KnowledgeBaseUnavailable() from exc
            if result != RETRIEVAL_NO_RESULTS and not retrieval_covers_required_topics(
                result, required_topics
            ):
                return RETRIEVAL_TOPIC_MISMATCH
            return truncate_text(result, context_limit)

        if retriever is None:
            return RETRIEVAL_NO_RESULTS

        query_results: list[QueryEvidenceResult] = []
        saw_raw_nodes = False
        saw_threshold_qualified_node = False

        for query in queries:
            try:
                nodes = retriever.retrieve(query)
            except UserFacingChatError:
                raise
            except Exception as exc:
                logger.error(
                    "Knowledge retrieval failed",
                    error_type=type(exc).__name__,
                )
                raise KnowledgeBaseUnavailable() from exc
            if reranker is not None:
                try:
                    nodes = reranker.postprocess_nodes(nodes, query_str=query)
                except Exception:
                    logger.error(
                        "Rerank failed, using top 10 raw results.", exc_info=True
                    )
                    nodes = nodes[:10]
            else:
                nodes = nodes[:10]

            saw_raw_nodes = saw_raw_nodes or bool(nodes)
            saw_threshold_qualified_node = saw_threshold_qualified_node or any(
                getattr(node, "score", None) is None
                or node.score >= MIN_RELEVANCE_SCORE
                for node in nodes
            )
            nodes = [
                n
                for n in nodes
                if getattr(n, "score", None) is None or n.score >= MIN_RELEVANCE_SCORE
            ]

            evidence_items = []
            for rank, node_with_score in enumerate(nodes, start=1):
                evidence_items.extend(
                    evidence_from_retrieved_node(
                        node_with_score, query=query, rank=rank
                    )
                )

            query_results.append(QueryEvidenceResult(query=query, items=evidence_items))

        formatted_result = format_search_results(query_results)
        if (
            formatted_result == RETRIEVAL_NO_RESULTS
            and saw_raw_nodes
            and not saw_threshold_qualified_node
        ):
            return RETRIEVAL_BELOW_THRESHOLD
        if (
            formatted_result != RETRIEVAL_NO_RESULTS
            and not retrieval_covers_required_topics(formatted_result, required_topics)
        ):
            return RETRIEVAL_TOPIC_MISMATCH
        return truncate_text(formatted_result, context_limit)

    return _search_knowledge_base


def deep_web_research(
    search_engine: Any,
    limit_service: ChatLimitService | None = None,
    settings: ChatLimitsSettings | None = None,
):
    limits = limit_service or get_chat_limit_service()
    chat_settings = settings or limits.settings

    @tool("deep_web_research", args_schema=WebResearchInput)
    async def _deep_web_research(topic: str):
        """Performs comprehensive web research on the given topic using advanced search capabilities.

        This tool conducts deep web research using advanced search to gather
        comprehensive, up-to-date information about any topic.

        Use this when:
        - The knowledge base has no relevant information
        - You need current events or recent developments
        - The topic requires real-time data or external sources
        - User asks for information beyond the stored knowledge

        Args:
            topic (str): The research topic or question to investigate

        Returns:
            str: Formatted search results with content snippets and source metadata for synthesis
        """
        topic = topic.strip()
        context = get_current_chat_turn()
        limits.begin_tavily_search(context)
        attempt_number = 0
        while True:
            attempt_number += 1
            await limits.reserve_tavily_attempt(context)
            try:
                response = await asyncio.to_thread(
                    search_engine.search,
                    query=topic,
                    search_depth="advanced",
                    max_results=chat_settings.tavily_max_results,
                    timeout=chat_settings.tavily_timeout_seconds,
                )
                break
            except UserFacingChatError:
                raise
            except Exception as exc:
                should_retry = (
                    attempt_number == 1
                    and is_transient_provider_error(exc)
                    and limits.claim_retry(context)
                )
                if should_retry:
                    logger.warning(
                        "Retrying transient Tavily failure",
                        error_type=type(exc).__name__,
                    )
                    await asyncio.sleep(1.0 + random.uniform(0.0, 0.25))
                    continue

                status = provider_status_code(exc)
                logger.error(
                    "Tavily search failed",
                    error_type=type(exc).__name__,
                    status_code=status,
                )
                if status == 429:
                    raise WebSearchCapacity() from exc
                raise WebSearchUnavailable() from exc

        results = response.get("results", [])[: chat_settings.tavily_max_results]

        if not results:
            return f"No web search results found for topic: '{topic}'"

        formatted_output = [f"Web Search Results for: '{topic}'", "=" * 80, ""]

        for idx, result in enumerate(results, 1):
            content = str(result.get("content", "No content available"))[
                : chat_settings.tavily_result_max_chars
            ]
            title = result.get("title", "Unknown Title")
            url = result.get("url", "No URL")
            score = result.get("score", 0.0)

            formatted_output.extend(
                [
                    f"[Result {idx}] (Relevance: {score:.2f})",
                    f"Title: {title}",
                    f"URL: {url}",
                    f"Content: {content}",
                    "",
                ]
            )

        formatted_output.append(f"\nTotal sources found: {len(results)}")

        joined = "\n".join(formatted_output)
        joined = joined.encode("utf-8", "ignore").decode("utf-8")
        return truncate_text(joined, chat_settings.tavily_context_max_chars)

    return _deep_web_research


def get_subgraphs_to_visualize(retriever: Any = None, visualizer_pipeline: Any = None):
    @tool("get_subgraphs_to_visualize", args_schema=QueryListInput)
    def _get_subgraphs_to_visualize(queries: list[str]):
        """Retrieves subgraphs for visualization based on multiple search queries.

        This tool extracts graph structures and nodes from the knowledge base to support visualization workflows. It processes multiple queries simultaneously to gather comprehensive subgraph data.

        Use this for:
        - Creating visual representations of knowledge graphs
        - Exploring connections between multiple topics

        The tool extracts:
        - Nodes: Individual text chunks or entities from the knowledge base
        - Triplets: Relationship structures in format (Node1 -> Relationship -> Node2)

        Args:
            queries (list[str]): A list of search queries to find relevant information and relationships in the knowledge base. Each query should target specific topics or concepts you want to visualize.

        Returns:
            A tuple containing:
                - List of unique node IDs found across all queries
                - List of unique relationship triplets (subject, predicate, object) for graph visualization

        Example:
            queries = ["machine learning", "neural networks", "deep learning"]
            Returns: (['node1', 'node2', ...], [('ML', 'USES', 'Neural Networks'), ...])
        """
        if visualizer_pipeline is not None:
            try:
                return visualizer_pipeline.visualize(queries)
            except UserFacingChatError:
                raise
            except Exception as exc:
                logger.error(
                    "Optimized graph visualization retrieval failed",
                    error_type=type(exc).__name__,
                )
                raise KnowledgeBaseUnavailable() from exc

        if retriever is None:
            return [], [], queries

        query_results: list[QueryEvidenceResult] = []

        for query in queries:
            try:
                retrieved_nodes = retriever.retrieve(query)
            except UserFacingChatError:
                raise
            except Exception as exc:
                logger.error(
                    "Graph visualization retrieval failed",
                    error_type=type(exc).__name__,
                )
                raise KnowledgeBaseUnavailable() from exc
            evidence_items = []

            for rank, node in enumerate(retrieved_nodes, start=1):
                evidence_items.extend(
                    evidence_from_retrieved_node(node, query=query, rank=rank)
                )

            query_results.append(QueryEvidenceResult(query=query, items=evidence_items))

        return format_visualization_results(query_results)

    return _get_subgraphs_to_visualize
