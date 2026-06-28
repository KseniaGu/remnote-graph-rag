from typing import Any

from langchain_core.tools import tool

from backend.configs.constants import WORKFLOW_LOGGING, MIN_RELEVANCE_SCORE
from backend.workflows.agents.retrieval_evidence import (
    QueryEvidenceResult,
    evidence_from_retrieved_node,
    format_search_results,
    format_visualization_results,
)
from backend.utils.helpers import get_logger

logger = get_logger(WORKFLOW_LOGGING)


def search_knowledge_base(retriever: Any = None, reranker: Any = None, analyst_pipeline: Any = None):
    @tool("search_knowledge_base")
    def _search_knowledge_base(queries: list[str]):
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
            queries (list[str]): A list of search queries to find relevant information in the knowledge base.
            
        Returns:
            str: Formatted search results with relations, sources, and relevance scores. Returns "No relevant information found" if no matches exist.
            
        Example output:
            [RELATION] Classical Computer Vision -> FOCUSES_ON -> Handcrafted features (Score: 0.85)
            [SOURCE] Classical computer vision relied on manually engineered features...
            [SOURCE PATH] CV History > Classical Era > Feature Extraction
        """
        if analyst_pipeline is not None:
            return analyst_pipeline.search(queries)

        if retriever is None:
            return "No relevant information found."

        query_results: list[QueryEvidenceResult] = []

        for query in queries:
            nodes = retriever.retrieve(query)
            if reranker is not None:
                try:
                    nodes = reranker.postprocess_nodes(nodes, query_str=query)
                except Exception:
                    logger.error("Rerank failed, using top 10 raw results.", exc_info=True)
                    nodes = nodes[:10]
            else:
                nodes = nodes[:10]

            nodes = [
                n for n in nodes
                if getattr(n, "score", None) is None or getattr(n, "score") >= MIN_RELEVANCE_SCORE
            ]

            evidence_items = []
            for rank, node_with_score in enumerate(nodes, start=1):
                evidence_items.extend(
                    evidence_from_retrieved_node(node_with_score, query=query, rank=rank)
                )

            query_results.append(QueryEvidenceResult(query=query, items=evidence_items))

        return format_search_results(query_results)

    return _search_knowledge_base


def deep_web_research(search_engine: Any):
    @tool("deep_web_research")
    def _deep_web_research(topic: str):
        """Performs comprehensive web research on the given topic using advanced search capabilities.
        
        This tool conducts deep web research using advanced search to gather comprehensive, up-to-date information about any topic. It analyzes multiple sources and provides a synthesized summary.
        
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
        response = search_engine.search(topic, depth="advanced")

        # Extract results
        results = response.get('results', [])

        if not results:
            return f"No web search results found for topic: '{topic}'"

        # Format results for LLM consumption
        formatted_output = [f"Web Search Results for: '{topic}'", "=" * 80, ""]

        for idx, result in enumerate(results, 1):
            content = result.get('content', 'No content available')
            title = result.get('title', 'Unknown Title')
            url = result.get('url', 'No URL')
            score = result.get('score', 0.0)

            formatted_output.extend([
                f"[Result {idx}] (Relevance: {score:.2f})",
                f"Title: {title}",
                f"URL: {url}",
                f"Content: {content}",
                ""
            ])

        formatted_output.append(f"\nTotal sources found: {len(results)}")

        joined = "\n".join(formatted_output)
        return joined.encode("utf-8", "ignore").decode("utf-8")

    return _deep_web_research


def get_subgraphs_to_visualize(retriever: Any = None, visualizer_pipeline: Any = None):
    @tool("get_subgraphs_to_visualize")
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
            return visualizer_pipeline.visualize(queries)

        if retriever is None:
            return [], [], queries

        query_results: list[QueryEvidenceResult] = []

        for query in queries:
            retrieved_nodes = retriever.retrieve(query)
            evidence_items = []

            for rank, node in enumerate(retrieved_nodes, start=1):
                evidence_items.extend(evidence_from_retrieved_node(node, query=query, rank=rank))

            query_results.append(QueryEvidenceResult(query=query, items=evidence_items))

        return format_visualization_results(query_results)

    return _get_subgraphs_to_visualize
