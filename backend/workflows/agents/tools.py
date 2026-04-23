import ast
import re
from typing import Any

from langchain_core.tools import tool

from backend.configs.constants import WORKFLOW_LOGGING, MIN_RELEVANCE_SCORE, MAX_SOURCE_CHARS, RELATION_DROP_SCORE
from backend.utils.helpers import get_logger

logger = get_logger(WORKFLOW_LOGGING)


def _smart_truncate(text: str, limit: int) -> str:
    """Truncates `text` to at most `limit` chars, preferring a sentence boundary."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # Prefer last sentence-ending punctuation within the budget, but only if it saves more than a tiny tail — otherwise hard-cut.
    for sep in (". ", "! ", "? ", "\n"):
        idx = cut.rfind(sep)
        if idx >= int(limit * 0.6):
            return cut[: idx + len(sep)].rstrip() + " …[truncated]"
    return cut.rstrip() + " …[truncated]"


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

        def _clean(s: str) -> str:
            # Strip lone UTF-16 surrogates so downstream JSON/httpx encoding never fails, normalize whitespace.
            return s.encode("utf-8", "ignore").decode("utf-8")

        evidence_lines = []
        found_any = False
        seen_sources_global: set[str] = set()  # dedupe source bodies across the whole result
        seen_paths_global: set[str] = set()

        for query in queries:
            nodes = retriever.retrieve(query)
            try:
                nodes = reranker.postprocess_nodes(nodes, query_str=query)
            except Exception:
                logger.error("Rerank failed, using top 10 raw results.", exc_info=True)
                nodes = nodes[:10]

            nodes = [
                n for n in nodes
                if n.score is None or n.score >= MIN_RELEVANCE_SCORE
            ]

            # Collected relations/sources for this query.
            query_lines: list[str] = []

            for node_with_score in nodes:
                node = node_with_score.node
                score = node_with_score.score or 0.0

                relation_text = None
                source_text = None
                paths = []

                if "->" in node.text:
                    if "Here are some facts extracted from the provided text:" in node.text:
                        _, relations, source_node_text = node.text.split("\n\n")
                        node_path = node.metadata.get("path", [])
                        relations = set(relations.split("\n"))
                        parsed_relations: list[str] = []
                        for relation in relations:
                            properties = re.findall(r'\(\{.*?\}\)', relation, re.DOTALL)
                            id_to_name = {}
                            if properties:
                                for property_ in set(properties):
                                    relation = relation.replace(property_, '')
                                    try:
                                        property_ = ast.literal_eval(property_)
                                        if "name" in property_ and "entity_name" in property_:
                                            id_to_name[property_["name"]] = property_["entity_name"]
                                    except Exception:
                                        logger.error("Failed to parse node properties.", exc_info=True)
                                        pass
                                for k, v in id_to_name.items():
                                    relation = relation.replace(k, v)
                            parsed_relations.append(relation.replace("  ", " ").strip())
                        relation_text = "\n".join(
                            f"[RELATION] {r} (Score: {score:.2f})" for r in parsed_relations
                        )
                        source_text = source_node_text.strip()
                        if node_path:
                            paths = [" > ".join(node_path)]
                    else:
                        # CHILD/PARENT entries. The PARENT row is redundant with CHILD.
                        if "PARENT" in node.text:
                            continue
                        # Drop low-score CHILD rows — they're mostly metadata (url:, hostname:).
                        if score < RELATION_DROP_SCORE:
                            continue

                        node_pair: list[str] = []
                        parsed_paths: list[list[str]] = []
                        properties = re.findall(r'\(\{.*?\}\)', node.text, re.DOTALL)
                        text = None

                        if properties:
                            for property_ in properties[::2]:
                                try:
                                    property_ = ast.literal_eval(property_)
                                except Exception:
                                    logger.error("Failed to parse node property.", exc_info=True)
                                    continue

                                if "text" in property_:
                                    node_pair.append(property_["text"])

                                if "path" in property_:
                                    try:
                                        parsed_paths.append(ast.literal_eval(property_["path"]))
                                    except Exception:
                                        logger.error("Failed to parse node property's path.", exc_info=True)
                            if len(node_pair) == 2:
                                text = node_pair[0] + " -> CHILD -> " + node_pair[1]

                        if not text:
                            continue

                        relation_text = f"[RELATION] {text} (Score: {score:.2f})"
                        paths = [" > ".join(p) for p in parsed_paths[:2]]
                else:
                    source_text = node.text
                    relation_text = None

                if relation_text:
                    query_lines.append(_clean(relation_text))

                if source_text:
                    clipped = _smart_truncate(source_text.strip(), MAX_SOURCE_CHARS)
                    dedup_key = clipped[:120]  # first ~120 chars is plenty to spot dupes
                    if dedup_key not in seen_sources_global:
                        seen_sources_global.add(dedup_key)
                        score_tag = f" (Score: {score:.2f})" if "->" not in (node.text or "") else ""
                        query_lines.append(f"[SOURCE]{score_tag} {_clean(clipped)}")

                for p in paths:
                    if p and p not in seen_paths_global:
                        seen_paths_global.add(p)
                        query_lines.append(f"[SOURCE PATH] {_clean(p)}")

                found_any = True

            if query_lines:
                evidence_lines.append(f"QUERY: {query}\n" + "\n".join(query_lines))

        if not found_any:
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

        joined = "\n".join(formatted_output)
        return joined.encode("utf-8", "ignore").decode("utf-8")

    return _deep_web_research


def get_subgraphs_to_visualize(retriever: Any):
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
        all_nodes, all_triplets = [], []

        for query in queries:
            retrieved_nodes = retriever.retrieve(query)

            for node in retrieved_nodes:
                if "->" in node.text:
                    node_1, relation, node_2 = node.text.split(" -> ")
                    all_triplets.append((node_1.strip(), relation.strip(), node_2.strip()))
                else:
                    all_nodes.append(node.node_id)

        return list(set(all_nodes)), list(set(all_triplets)), queries

    return _get_subgraphs_to_visualize
