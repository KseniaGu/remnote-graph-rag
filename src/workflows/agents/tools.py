import ast
import re
from typing import Any

from langchain_core.tools import tool
from src.utils.helpers import get_logger
from configs.constants import WORKFLOW_LOGGING

logger = get_logger(WORKFLOW_LOGGING)

def search_knowledge_base(retriever: Any, reranker: Any):
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
        evidence_lines = []
        for query in queries:
            evidence_lines.append(f"QUERY: {query}")
            nodes = retriever.retrieve(query)
            try:
                nodes = reranker.postprocess_nodes(nodes, query_str=query)
            except Exception:
                logger.error(f"Rerank failed, using top 10 raw results.", exc_info=True)
                nodes = nodes[:10]

            for node_with_score in nodes:
                node = node_with_score.node
                score = node_with_score.score
                text_to_add = ""

                if "->" in node.text:
                    if "Here are some facts extracted from the provided text:" in node.text:
                        _, relations, source_node_text = node.text.split("\n\n")
                        path = node.metadata.get("path", [])
                        relations = set(relations.split("\n"))
                        for relation in relations:
                            properties = re.findall(r'\(\{.*?\}\)', relation, re.DOTALL)
                            if properties:
                                for property_ in set(properties):
                                    relation = relation.replace(property_, '')
                            text_to_add += f"[RELATION] {relation.replace('  ', ' ').strip()} (Score: {score:.2f})\n"
                        text_to_add += f"[SOURCE] {source_node_text}\n"
                        if path:
                            text_to_add += f"[SOURCE PATH] {' > '.join(path)}\n"
                    else:
                        # We get the same pair from "CHILD" relation, so drop to not duplicate information
                        if "PARENT" in node.text:
                            continue

                        text, path = node.text, []
                        properties = re.findall(r'\(\{.*?\}\)', node.text, re.DOTALL)
                        if properties:
                            text = text.replace(properties[0], '').replace('  ', ' ')
                            path = ast.literal_eval(properties[0]).get("path", [])
                        text_to_add += f"[RELATION] {text} (Score: {score:.2f})\n"
                        if path:
                            text_to_add += f"[SOURCE PATH] {' > '.join(path)}\n"
                else:
                    text_to_add += f"[SOURCE] {node.text} (Score: {score:.2f})\n"

                evidence_lines.append(text_to_add)

        if not evidence_lines:
            return "No relevant information found."

        return "RETRIEVER RESULTS:\n\n" + "\n\n".join(evidence_lines)

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

        return "\n".join(formatted_output)

    return _deep_web_research


def get_sub_graphs_to_visualize(retriever: Any):
    @tool("get_sub_graphs_to_visualize")
    def _get_sub_graphs_to_visualize(queries: list[str]):
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
        all_nodes, all_triplets = [], []

        for query in queries:
            retrieved_nodes = retriever.retrieve(query)

            for node in retrieved_nodes:
                if "->" in node.text:
                    node_1, relation, node_2 = node.text.split(" -> ")
                    all_triplets.append((node_1.strip(), relation.strip(), node_2.strip()))
                else:
                    all_nodes.append(node.node_id)

        return list(set(all_nodes)), list(set(all_triplets))

    return _get_sub_graphs_to_visualize
