import asyncio
import json
from itertools import chain
from typing import Any, Optional

import networkx as nx
import plotly.graph_objects as go
from dotenv import load_dotenv, find_dotenv
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.graph_stores import ChunkNode
from llama_index.core.graph_stores.types import EntityNode, Relation, KG_NODES_KEY, KG_RELATIONS_KEY, \
    TRIPLET_SOURCE_KEY, VECTOR_SOURCE_KEY
from llama_index.core.indices import PropertyGraphIndex
from llama_index.core.indices.property_graph import ImplicitPathExtractor
from llama_index.core.indices.property_graph.sub_retrievers.vector import VectorContextRetriever
from llama_index.core.schema import MetadataMode, RelatedNodeInfo, TextNode, NodeRelationship
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.redis import RedisVectorStore
from tqdm import tqdm

from configs.constants import MAX_TOKEN_COUNTS_PER_CALL, TEST_SOURCES, MAX_DOCUMENTS_TO_INDEX, SPRING_LAYOUT_K, \
    SPRING_LAYOUT_ITERATIONS
from configs.enums import KnowledgeGraphEntity, KnowledgeGraphRelation, StorageType
from configs.paths import PathSettings
from configs.search import KnowledgeGraphSearchSettings
from src.utils.helpers import clean_json_markdown, logger, make_json_serializable

load_dotenv(find_dotenv())


class KnowledgeGraphIndexer:
    """Manages knowledge graph indexing, processing, and retrieval operations.
    
    This class handles the creation, processing, and querying of property graph indexes,
    including implicit graph processing, entity/relation extraction, and vector embeddings.
    """

    def __init__(
            self,
            storage_context: StorageContext,
            path_settings: PathSettings,
            document_storage_type: StorageType,
            kg_search_settings: KnowledgeGraphSearchSettings,
            embedder: Any,
            reranker: Any,
            tokenizer: Optional[Any] = None,
            test_setup: bool = True
    ) -> None:
        """Initializes the KnowledgeGraphIndexer.
        
        Args:
            storage_context: Storage context for managing document, vector and graph storage.
            path_settings: Path configuration settings.
            document_storage_type: Type of storage (local or remote).
            kg_search_settings: Knowledge graph search configuration.
            embedder: Embedding model for vector representations.
            reranker: Reranker model for result refinement.
            tokenizer: Optional tokenizer for token counting.
            test_setup: Whether to use test configuration with limited documents.
        """
        self.path_settings = path_settings
        self.storage_context = storage_context
        self.document_storage_type = document_storage_type
        self.kg_search_settings = kg_search_settings

        self.embedder = embedder
        self.reranker = reranker
        self.tokenizer = tokenizer

        self.node_id_to_text = {k: v.text for k, v in self.storage_context.docstore.docs.items()}
        # 'path' parameter contains the hierarchical path to the RemNote text block (page name, headers, sub-headers, etc.)
        self.node_id_to_path = {k: v.metadata['path'] for k, v in self.storage_context.docstore.docs.items()}
        self.node_id_to_line_number = {
            k: v.metadata['line_number'] for k, v in self.storage_context.docstore.docs.items()
        }

        self.index = None
        self.test_setup = test_setup

    def get_document_nodes(self, node_ids: list[str] | str) -> list[TextNode] | TextNode | None:
        """Retrieves document nodes by their IDs.
        
        Args:
            node_ids: Single node ID or list of node IDs.
            
        Returns:
            List of nodes if input is a list, single node or None otherwise.
        """
        nodes = self.index.property_graph_store.get_llama_nodes(node_ids)
        if isinstance(node_ids, list):
            return nodes

        return next(iter(nodes), None)

    def remove_document_nodes(self, nodes_to_remove_labels: list[str], node_label_to_node_id: dict[str, str]) -> None:
        """Removes document nodes and updates their parent relationships.
        
        Args:
            nodes_to_remove_labels: Labels of nodes to remove.
            node_label_to_node_id: Mapping from node labels to node IDs.
        """
        nodes_to_remove = []
        parents_updated = {}

        for node_label in nodes_to_remove_labels:
            node_id = node_label_to_node_id[node_label]
            node = self.get_document_nodes(node_id)
            parent_node_id = node.parent_node.node_id
            if parent_node_id in parents_updated:
                parent_node = parents_updated[parent_node_id]
            else:
                parent_node = self.get_document_nodes(node.parent_node.node_id)
                parents_updated[parent_node_id] = parent_node
            parents_updated[parent_node_id] = self.update_document_child_nodes(
                parent_node, child_ids_to_remove=[node_id]
            )
            nodes_to_remove.append(node_id)

        if nodes_to_remove:
            self.index.property_graph_store.delete_llama_nodes(nodes_to_remove)
        if parents_updated:
            self.index.property_graph_store.upsert_llama_nodes(list(parents_updated.values()))

    def update_document_child_nodes(
            self,
            node: TextNode,
            child_ids_to_add: Optional[list[str]] = None,
            child_ids_to_remove: Optional[list[str]] = None
    ) -> TextNode:
        """Updates child node relationships for a given node.
        
        Args:
            node: Parent node to update.
            child_ids_to_add: Child node IDs to add.
            child_ids_to_remove: Child node IDs to remove.
            
        Returns:
            Updated node with modified child relationships.
        """
        # Reset child nodes
        if child_ids_to_add is None and child_ids_to_remove is None:
            node.metadata.pop("child_ids", None)
            node.relationships.pop(NodeRelationship.CHILD, None)

        if child_ids_to_add:
            child_node_ids = node.metadata.get("child_ids", [])
            child_node_ids += child_ids_to_add
            node.metadata["child_ids"] = child_node_ids
            child_nodes = node.relationships.get(NodeRelationship.CHILD, [])
            for child_id_to_add in child_ids_to_add:
                related_info = RelatedNodeInfo(
                    node_id=child_id_to_add,
                    metadata={"title": self.get_document_nodes(child_id_to_add).metadata["original_text"]}
                )
                child_nodes.append(related_info)
            node.relationships[NodeRelationship.CHILD] = child_nodes

        if child_ids_to_remove:
            if "child_ids" in node.metadata:
                node.metadata["child_ids"] = [
                    child_id for child_id in node.metadata["child_ids"] if child_id not in child_ids_to_remove
                ]
                node.relationships[NodeRelationship.CHILD] = [
                    child for child in node.relationships[NodeRelationship.CHILD]
                    if child.node_id not in child_ids_to_remove
                ]

        return node

    @staticmethod
    def get_subtree_label(node_label: str) -> str:
        """Extracts subtree identifier from node label.
        
        Args:
            node_label: Node label in format 'subtree_X_leaf_Y'.
            
        Returns:
            Subtree identifier.
        """
        return node_label.split("_")[1]

    def merge_multiple_subtrees(
            self,
            merge_combination: list[str],
            first_node_id: str,
            first_node_parent: TextNode,
            nodes_to_remove: list[str],
            node_labels_to_update: dict[str, str],
            node_label_to_node_id: dict[str, str],
            nodes_to_update: list[TextNode]
    ) -> tuple[TextNode, list[str], dict[str, str], list[TextNode]]:
        """Merges multiple subtrees into a single node.
        
        Args:
            merge_combination: List of node labels to merge.
            first_node_id: ID of the first node in the merge.
            first_node_parent: Parent node of the first node.
            nodes_to_remove: Accumulator for nodes to remove.
            node_labels_to_update: Mapping of old to new node IDs.
            node_label_to_node_id: Mapping from labels to node IDs.
            nodes_to_update: Accumulator for nodes to update.
            
        Returns:
            Tuple of (updated parent node, nodes to remove, label updates, nodes to update).
        """
        # We merge all subtrees (both parents and childs) into one node (parent of the first leaf node)
        current_subtree_label = self.get_subtree_label(merge_combination[0])
        merged_text = self.node_id_to_path[first_node_id][-1] + "\n" + self.node_id_to_text[first_node_id]
        nodes_to_remove.append(first_node_id)
        node_labels_to_update[first_node_id] = first_node_parent.node_id
        parents_to_remove = set()

        for node in merge_combination[1:]:
            subtree_label = self.get_subtree_label(node)
            node_id = node_label_to_node_id[node]
            node_text = self.node_id_to_text[node_id]
            parent_node_id = self.get_document_nodes(node_id).parent_node.node_id

            nodes_to_remove.append(node_id)
            # Redirect removed node to the retained one
            node_labels_to_update[node_id] = first_node_parent.node_id
            # We keep only one node (first_node_parent) so other subtrees (both leaves and their parents) are removed
            if parent_node_id != first_node_parent.node_id:
                parents_to_remove.add(parent_node_id)

            if subtree_label == current_subtree_label:
                merged_text += node_text
            else:
                merged_text += self.node_id_to_path[node_id] + "\n" + node_text
                current_subtree_label = subtree_label

        # Remove all child nodes, all their texts are already gathered in merged_text
        first_node_parent = self.update_document_child_nodes(first_node_parent)
        first_node_parent.metadata["original_text"] = merged_text
        first_node_parent.text = merged_text
        nodes_to_update.append(first_node_parent)
        grand_parents_updated = {}

        for parent_to_remove in parents_to_remove:
            grand_parent_id = self.get_document_nodes(parent_to_remove).parent_node.node_id
            if grand_parent_id in grand_parents_updated:
                grand_parent = grand_parents_updated[grand_parent_id]
            else:
                grand_parent = self.get_document_nodes(grand_parent_id)
                grand_parents_updated[grand_parent_id] = grand_parent
            grand_parents_updated[grand_parent_id] = self.update_document_child_nodes(
                grand_parent, child_ids_to_remove=[parent_to_remove]
            )
            nodes_to_remove.append(parent_to_remove)
            # Redirect removed parent node to the retained one
            node_labels_to_update[parent_to_remove] = first_node_parent.node_id
        nodes_to_update.extend(list(grand_parents_updated.values()))

        return first_node_parent, nodes_to_remove, node_labels_to_update, nodes_to_update

    def merge_subtree(
            self,
            merge_combination: list[str],
            first_node: TextNode,
            first_node_parent: TextNode,
            nodes_to_remove: list[str],
            node_labels_to_update: dict[str, str],
            node_label_to_node_id: dict[str, str],
            nodes_to_update: list[TextNode]
    ) -> tuple[TextNode, TextNode, dict[str, str], list[TextNode]]:
        """Merges nodes within a single subtree.
        
        Args:
            merge_combination: List of node labels to merge.
            first_node: First node to merge into.
            first_node_parent: Parent of the first node.
            nodes_to_remove: Accumulator for nodes to remove.
            node_labels_to_update: Mapping of old to new node IDs.
            node_label_to_node_id: Mapping from labels to node IDs.
            nodes_to_update: Accumulator for nodes to update.
            
        Returns:
            Tuple of (merged node, parent node, label updates, nodes to update).
        """
        merged_text = first_node.text

        for node in merge_combination[1:]:
            try:
                node_id = node_label_to_node_id[node]
                node_text = self.node_id_to_text[node_id]
                merged_text += "\n" + node_text
                nodes_to_remove.append(node_id)
                # Redirect removed node to the retained one
                node_labels_to_update[node_id] = first_node.node_id
            except (KeyError, ValueError) as e:
                logger.error(f"Error merging node {node}: {e}", exc_info=True)

        first_node.text = merged_text
        nodes_to_update.append(first_node)
        first_node_parent = self.update_document_child_nodes(first_node_parent, child_ids_to_remove=nodes_to_remove)
        nodes_to_update.append(first_node_parent)
        return first_node, first_node_parent, node_labels_to_update, nodes_to_update

    def merge_nodes(
            self,
            nodes_to_merge_labels: list[list[str]],
            node_label_to_node_id: dict[str, str]
    ) -> dict[str, str]:
        """Merges multiple node groups based on LLM recommendations.
        
        Args:
            nodes_to_merge_labels: List of node label groups to merge.
            node_label_to_node_id: Mapping from labels to node IDs.
            
        Returns:
            Updated node label to node ID mapping.
        """
        nodes_to_remove, nodes_to_update = [], []
        node_labels_to_update = {}

        for merge_combination in nodes_to_merge_labels:
            if len(merge_combination) <= 1:
                continue
            unique_subtrees = set(self.get_subtree_label(x) for x in merge_combination)
            first_node_id = node_label_to_node_id[merge_combination[0]]
            first_node = self.get_document_nodes(first_node_id)
            first_node_parent = self.get_document_nodes(first_node.parent_node.node_id)

            if len(unique_subtrees) > 1:
                first_node_parent, nodes_to_remove, node_labels_to_update, nodes_to_update = self.merge_multiple_subtrees(
                    merge_combination, first_node_id, first_node_parent, nodes_to_remove, node_labels_to_update,
                    node_label_to_node_id, nodes_to_update
                )
            else:
                first_node, first_node_parent, node_labels_to_update, nodes_to_update = self.merge_subtree(
                    merge_combination, first_node, first_node_parent, nodes_to_remove, node_labels_to_update,
                    node_label_to_node_id, nodes_to_update
                )

        if nodes_to_remove:
            self.index.property_graph_store.delete_llama_nodes(nodes_to_remove)
        if nodes_to_update:
            self.index.property_graph_store.upsert_llama_nodes(nodes_to_update)

        # Update node_label_to_node_id dictionary for the nodes that were merged
        if node_labels_to_update:
            for node_label, node_id in node_label_to_node_id.copy().items():
                if node_id in node_labels_to_update:
                    node_label_to_node_id[node_label] = node_labels_to_update[node_id]

        return node_label_to_node_id

    def update_relations(self, triplets: list[dict[str, Any]], node_label_to_node_id: dict[str, str]):
        """Updates graph with entity and relation triplets.
        
        Args:
            triplets: List of triplet dictionaries containing subject, predicate, object information.
            node_label_to_node_id: Mapping from labels to node IDs.
        """
        entity_map = {}
        entities, relations, nodes_to_update = [], [], []

        for triplet in triplets:
            subject_source_id, object_source_id = triplet["subject_source_id"], triplet["object_source_id"]
            if subject_source_id == object_source_id:
                subject_node = object_node = self.get_document_nodes(node_label_to_node_id[subject_source_id])
            else:
                results = self.get_document_nodes(
                    [node_label_to_node_id[subject_source_id], node_label_to_node_id[object_source_id]]
                )
                if len(results) == 1:
                    subject_node = object_node = results[0]
                elif len(results) == 2:
                    subject_node, object_node = results
                else:
                    raise ValueError(f"Unexpected nodes number returned from the Docstore ({len(results)})")

            subject_key = (triplet["subject"], triplet["subject_type"])
            object_key = (triplet["object"], triplet["object_type"])

            if subject_key not in entity_map:
                entity_map[subject_key] = EntityNode(name=triplet["subject"], label=triplet["subject_type"])
            if object_key not in entity_map:
                entity_map[object_key] = EntityNode(name=triplet["object"], label=triplet["object_type"])

            relation = Relation(
                label=triplet["predicate"],
                source_id=entity_map[subject_key].id,
                target_id=entity_map[object_key].id,
            )

            for entity_type, entity_node in zip((subject_key, object_key), (subject_node, object_node)):
                # This is the set of rules that are necessary for the LLamaIndex structures, refer to the source code for more details
                existing_nodes = entity_node.metadata.get(KG_NODES_KEY, [])
                existing_relations = entity_node.metadata.get(KG_RELATIONS_KEY, [])
                existing_nodes.append(entity_map[entity_type])
                entity_node.metadata[KG_NODES_KEY] = existing_nodes

                if relation not in existing_relations:
                    existing_relations.append(relation)
                    entity_node.metadata[KG_RELATIONS_KEY] = existing_relations

                if TRIPLET_SOURCE_KEY not in entity_map[entity_type].properties:
                    entity_map[entity_type].properties[TRIPLET_SOURCE_KEY] = entity_node.id_
                    relation.properties[TRIPLET_SOURCE_KEY] = entity_node.id_

                nodes_to_update.append(entity_node)

            entities.append(entity_map[subject_key])
            if subject_node != object_node:
                entities.append(entity_map[object_key])

            relations.append(relation)

        if entities:
            self.index.property_graph_store.upsert_nodes(entities)
        if relations:
            self.index.property_graph_store.upsert_relations(relations)
        if nodes_to_update:
            self.index.property_graph_store.upsert_llama_nodes(nodes_to_update)

    def get_prompt_input_from_subtrees(self, subgraph_tailing_subtrees: list[list[str]]) -> tuple[str, dict[str, str]]:
        """Generates prompt input from subtree structures.
        
        Args:
            subgraph_tailing_subtrees: List of subtrees containing leaf node IDs.
            
        Returns:
            Tuple of (formatted prompt input, node label to ID mapping).
        """
        texts, node_label_to_node_id = {}, {}
        prompt_input = ""

        for i, subtree in enumerate(subgraph_tailing_subtrees):
            texts[i], line_numbers = {}, {}

            for j, leaf in enumerate(subtree):
                texts[i][j] = self.node_id_to_text[leaf]
                line_numbers[j] = self.node_id_to_line_number[leaf]
                node_label_to_node_id[f"subtree_{i}_leaf_{j}"] = leaf

            prefix = " > ".join(self.node_id_to_path[subtree[0]]) + " >"
            gathered_leaves = '\n'.join(
                f'leaf_{k}: (line {line_numbers[k]}) ' + f'"{text}"' for k, text in
                texts[i].items()
            )
            subtree_text = f'subtree_{i}:\nprefix: "{prefix}"\n{gathered_leaves}\n'
            prompt_input += subtree_text

        return prompt_input, node_label_to_node_id

    def process_sub_graph(self, sub_graph: nx.DiGraph, model: Ollama, prompt: str) -> int:
        """Processes a single subgraph to extract entities and relations.
        
        Args:
            sub_graph: NetworkX directed graph representing document structure.
            model: LLM for processing.
            prompt: Template prompt for the LLM.
            
        Returns:
            Total tokens used in processing.
        """
        leaves = [node for node, degree in sub_graph.out_degree() if degree == 0]
        subgraph_tailing_subtrees = []
        checked_leaves = set()

        for leaf in leaves:
            if leaf in checked_leaves:
                continue

            parents = list(sub_graph.predecessors(leaf))
            assert len(parents) == 1, f"Unexpected relation: one node has multiple parents (node {leaf})"

            # Find subtree that contains current leaf parent and its other leaves
            subtree = sub_graph.successors(parents[0])
            subtree = [x for x in sorted(subtree, key=lambda y: self.node_id_to_line_number[y]) if x in leaves]
            checked_leaves.update(subtree)
            subgraph_tailing_subtrees.append(subtree)

        def get_min_line_number(subtree: list[str]) -> int:
            """Gets minimum line number from subtree nodes."""
            return min(self.node_id_to_line_number[node_id] for node_id in subtree)

        subgraph_tailing_subtrees = sorted(subgraph_tailing_subtrees, key=get_min_line_number)
        prompt_input, node_label_to_node_id = self.get_prompt_input_from_subtrees(subgraph_tailing_subtrees)
        prompt_filled = prompt.format(
            text=prompt_input,
            allowed_entity_types=[x.value for x in KnowledgeGraphEntity],
            allowed_relation_types=[x.value for x in KnowledgeGraphRelation]
        )

        if self.tokenizer is not None and self.test_setup:
            token_counts = len(self.tokenizer.encode(prompt_input))
            if token_counts > MAX_TOKEN_COUNTS_PER_CALL:
                return 0

        model_output = model.complete(prompt_filled)
        usage = model_output.raw.get('usage', {})
        if not model_output:
            return 0
        try:
            structured_output = json.loads(clean_json_markdown(model_output.text))
        except json.decoder.JSONDecodeError:
            return 0

        self.remove_document_nodes(structured_output['to_remove'], node_label_to_node_id)
        node_label_to_node_id = self.merge_nodes(
            structured_output['to_merge'], node_label_to_node_id
        )
        self.update_relations(structured_output['triplets'], node_label_to_node_id)

        return usage.get('total_tokens', 0)

    def generate_nx_graph_from(self, nodes: list[str], relation_triplets: list[tuple[str, str, str]]) -> nx.DiGraph:
        """Generates NetworkX graph from nodes and relation triplets.
        
        Args:
            nodes: List of node IDs.
            relation_triplets: List of (source, relation, target) tuples.
            
        Returns:
            NetworkX directed graph.
        """
        graph = nx.DiGraph()

        nodes_to_add, edges_to_add = [], []
        all_unique_node_ids = set((nodes + list(chain(*[(node_1, node_2) for node_1, _, node_2 in relation_triplets]))))
        all_unique_nodes = self.index.property_graph_store.get(ids=all_unique_node_ids)

        node_id_to_text = {}
        for node in all_unique_nodes:
            if isinstance(node, EntityNode):
                node_id_to_text[node.id] = f"{node.name} (Label: {node.label})"
            elif isinstance(node, ChunkNode):
                node_id_to_text[node.id] = node.text

        for node_id in all_unique_node_ids:
            nodes_to_add.append((node_id, dict(text=node_id_to_text[node_id])))

        for relation in relation_triplets:
            edges_to_add.append((relation[0], relation[2], dict(relation=relation[1])))

        graph.add_nodes_from(nodes_to_add)
        graph.add_edges_from(edges_to_add)
        return graph

    def get_graph_visualization(self, nodes: list[str], relation_triplets: list[tuple[str, str, str]]) -> go.Figure:
        """Generates interactive Plotly visualization of the knowledge graph.
        
        Args:
            nodes: List of node IDs to visualize.
            relation_triplets: List of (source, relation, target) tuples.
            
        Returns:
            Plotly Figure object for visualization.
        """
        graph = self.generate_nx_graph_from(nodes, relation_triplets)
        pos = nx.spring_layout(graph, k=SPRING_LAYOUT_K, iterations=SPRING_LAYOUT_ITERATIONS)

        edge_x, edge_y = [], []
        edge_label_x, edge_label_y, edge_label_text = [], [], []
        for edge in graph.edges(data=True):
            u, v, data = edge
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            edge_label_x.append((x0 + x1) / 2)
            edge_label_y.append((y0 + y1) / 2)

            relation = data.get("relation", "RELATES_TO")
            # Hover text for edges
            if relation == "CHILD":
                edge_label_text.append(u + " → " + "HAS CHILD" + " → " + v)
            elif relation == "PARENT":
                edge_label_text.append("")
            else:
                edge_label_text.append(u + " → " + data.get("relation", "RELATES_TO") + " → " + v)

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
        )
        edge_text_trace = go.Scatter(
            x=edge_label_x,
            y=edge_label_y,
            mode="markers",
            marker=dict(size=5, opacity=0),
            hoverinfo='text',
            hovertext=edge_label_text,
        )

        node_x, node_y, node_text = [], [], []
        for node, data in graph.nodes(data="text"):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            # Hover text for nodes
            node_text.append(data)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            textposition="top center",
            text=[str(n) for n in graph.nodes()],
            hovertext=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=15,
                colorbar=dict(thickness=15, title='Degree', xanchor='left'),
                color=[10] * len(graph.nodes()),
            )
        )

        fig = go.Figure(
            data=[edge_trace, edge_text_trace, node_trace],
            layout=go.Layout(
                title="Knowledge subgraph visualization",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                autosize=True,
            )
        )

        return fig

    def process_implicit_graph(self, model: Ollama, processing_prompt: str):
        """Processes graph built by ImplicitPathExtractor.

        Processing process consists of the following steps:
            1. Get all connected subgraphs.
            2. For each connected subgraph:
                - Get all its leaves and find "tailing" subtrees that contain only these leaves and their parents.
                - Sort all the nodes in the subtrees based on the number of lines of text they represent.
                - Generate a prompt from all the subtrees to send it to the LLM.
                - Ask the LLM to find the leaves (or whole subtrees) that should be merged. We need this merging,
                because some of the graph nodes contain text parts that make more sense as a whole text block, such as
                list elements or formulas split across separate lines, etc.
                - Check whether we need to remove any nodes (e.g., failed OCR results or non-informative website/blog information).
                - Extract entities and relations, such as "Gradient checkpointing" (entity) -> "IS_A" (relation) -> "Memory-efficient backpropagation technique" (entity).
                - Process the information using LLM: merge nodes, delete garbage nodes, add relation triplets to the graph.

        Args:
            model: LLM used to process subgraphs
            processing_prompt: A prompt for the provided LLM
        """
        logger.info("Processing Implicit Graph...")
        nodes = list(self.index.property_graph_store.graph.nodes.keys())
        edges = [
            (x.source_id, x.target_id) for k, x in self.index.property_graph_store.graph.relations.items()
            if x.id == "CHILD"
        ]

        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        graph.remove_nodes_from(list(nx.isolates(graph)))

        # Get all connected subgraphs
        sub_graphs = [graph.subgraph(c).copy() for c in nx.weakly_connected_components(graph)]

        total_tokens = 0
        pbar = tqdm(sub_graphs, desc="Subgraphs processing")
        for sub_graph in pbar:
            try:
                tokens_used = self.process_sub_graph(sub_graph, model, processing_prompt)
                total_tokens += tokens_used
                pbar.set_postfix({"total_tokens": total_tokens})
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Error processing subgraph: {e}", exc_info=True)

        if self.document_storage_type == StorageType.local:
            self.index.storage_context.persist(persist_dir=str(self.path_settings.local_storage_dir))

    def get_embeddings(self):
        """Generates and stores embeddings for document nodes and graph entities."""
        if self.index.vector_store is None:
            raise ValueError("index.vector_store is None. Ensure Vector Store is properly attached to the index.")

        if self.document_storage_type == StorageType.local:
            if self.index.vector_store.data.embedding_dict:
                return
        else:
            if isinstance(self.index.vector_store, RedisVectorStore):
                if asyncio.run(self.index.vector_store.client.ft("llama_index").info()).get("num_docs", 0) > 0:
                    return
            else:
                raise ValueError(f"Unexpected Vector Store type: {type(self.index.vector_store)}")
        self.index.vector_store.stores_text = True
        document_ids = list(self.index.storage_context.docstore.docs.keys())
        documents = self.get_document_nodes(document_ids)
        documents = [document for document in documents if document.embedding is None]
        graph_entities = [x for x in self.index.property_graph_store.get() if isinstance(x, EntityNode)]
        document_texts = [node.get_content(metadata_mode=MetadataMode.EMBED) for node in documents]
        graph_entity_texts = list(map(str, graph_entities))
        nodes_to_update = []

        embeddings = self.embedder.get_text_embedding_batch(document_texts, show_progress=True)
        for node, embedding in tqdm(zip(documents, embeddings), desc="Getting embeddings for llama nodes"):
            node.embedding = embedding
            node = make_json_serializable(node, "metadata")
            nodes_to_update.append(node)

        embeddings = self.embedder.get_text_embedding_batch(graph_entity_texts, show_progress=True)

        for node, embedding in tqdm(zip(graph_entities, embeddings), desc="Getting embeddings for graph entities"):
            node = make_json_serializable(node, "properties")
            nodes_to_update.append(
                TextNode(
                    text=str(node),
                    metadata={VECTOR_SOURCE_KEY: node.id, **node.properties},
                    embedding=[*embedding],
                )
            )
            nodes_to_update[-1].id_ = node.id

        self.index.vector_store.add(nodes_to_update)

        if self.document_storage_type == StorageType.local:
            self.index.storage_context.persist(persist_dir=str(self.path_settings.local_storage_dir))

    def build_index(
            self,
            documents: Optional[list[TextNode]] = None,
            llm: Optional[dict[str, Any]] = None,
            graph_index_prompt: Optional[str] = None
    ) -> None:
        """Builds a new PropertyGraphIndex from documents.
        
        Args:
            documents: List of document nodes to index. If None, uses documents from storage context.
            llm: LLM configuration parameters.
            graph_index_prompt: Prompt for graph processing.
        """
        if documents is None:
            documents = list(self.storage_context.docstore.docs.values())
            assert documents, "The documents are not provided and not found in Storage Context's docstore."

        model = Ollama(**llm, json_mode=True)
        # First path extractor that generates graph based on PARENT/CHILD relations in the documents
        implicit_extractor = ImplicitPathExtractor()
        if self.test_setup:
            documents = [node for node in documents if node.metadata['source'] in TEST_SOURCES]
            assert len(documents) < MAX_DOCUMENTS_TO_INDEX

        logger.info("Start building Property Graph Index...")
        self.index = PropertyGraphIndex(
            nodes=documents,
            storage_context=self.storage_context,
            # vector_store=self.storage_context.vector_store,
            embed_model=self.embedder,
            kg_extractors=[implicit_extractor],
            use_async=False,
            show_progress=True,
            embed_kg_nodes=False,
        )

        # Here is the more intelligent processing that uses LLM on a limited number of generated graph nodes.
        self.process_implicit_graph(model, graph_index_prompt)

        # Get embeddings for nodes generated by the first path extractor and entities extracted with LLM
        self.get_embeddings()

    def load_index(self):
        """Loads the PropertyGraphIndex from the provided storage_context."""
        implicit_extractor = ImplicitPathExtractor()
        self.index = load_index_from_storage(
            self.storage_context, kg_extractors=[implicit_extractor], embed_model=self.embedder, use_async=False,
            vector_store=self.storage_context.vector_store
        )
        self.get_embeddings()

    def get_retriever(self, retriever_params: dict = None) -> BaseRetriever:
        """Gets retriever engine.

        Available options:
            - Only vector similarity search using the documents from the graph.
            - Only the vector context search on the graph: searches for similar nodes and returns the relation triplets
            related to the found nodes.
            - Both options.

        Args:
            retriever_params: Parameters (such as "similarity_top_k", "depth") dict, containing settings for the chosen
                retriever options ("VectorIndexRetriever" key for the first option, "VectorContextRetriever"- for the second)

        Returns:
            BaseRetriever: Retriever engine with chosen options as sub-retrievers.
        """
        assert self.index is not None, "Index not built or loaded."
        sub_retrievers = []

        if retriever_params is None:
            retriever_params = self.kg_search_settings.retriever_params

        if retriever_params.get("VectorIndexRetriever"):
            # Vector search only
            nodes_with_embs = self.get_document_nodes(list(self.index.storage_context.docstore.docs.keys()))
            vector_index = VectorStoreIndex(nodes=nodes_with_embs, embed_model=self.embedder)
            vector_retriever = vector_index.as_retriever(**retriever_params.get("VectorIndexRetriever"))
            sub_retrievers.append(vector_retriever)
        if retriever_params.get("VectorContextRetriever"):
            # Vector search + Graph traversal on the results (returns only triplets)
            # Disable Neo4j's vector_query so retriever uses Redis vector store instead
            self.index.property_graph_store.supports_vector_queries = False
            context_retriever = VectorContextRetriever(
                graph_store=self.index.property_graph_store,
                vector_store=self.index.vector_store,
                embed_model=self.embedder,
                **retriever_params.get("VectorContextRetriever")
            )
            sub_retrievers.append(context_retriever)

        return self.index.as_retriever(sub_retrievers=sub_retrievers, **retriever_params)

    def get_query_engine(self, query_engine_params: Optional[dict[str, Any]] = None) -> Any:
        """Gets query engine for the knowledge graph.
        
        Args:
            query_engine_params: Optional parameters for query engine configuration.
            
        Returns:
            Query engine instance.
        """
        assert self.index is not None, "Index not built or loaded."
        return self.index.as_query_engine(**query_engine_params)


if __name__ == '__main__':
    from src.utils.prompt_engine import PromptEngine
    from configs.storage import StorageSettings
    from configs.models import ModelSettings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from configs.enums import ModelRoleType, PromptType
    from src.workflows.knowledge_graph.storage import KnowledgeGraphStorage
    from configs.storage import LocalStorageSettings

    storage_settings = StorageSettings()
    models_settings = ModelSettings()
    path_settings = PathSettings()
    kg_search_settings = KnowledgeGraphSearchSettings()

    # To set up local storages, comment the settings lines below to activate non-local storages
    storage_settings.document_storage = LocalStorageSettings(storage_path=path_settings.local_storage_dir)
    storage_settings.index_storage = LocalStorageSettings(storage_path=path_settings.local_storage_dir)
    storage_settings.vector_storage = LocalStorageSettings(storage_path=path_settings.local_storage_dir)
    storage_settings.property_graph_storage = LocalStorageSettings(storage_path=path_settings.local_storage_dir)

    kg_storage = KnowledgeGraphStorage(path_settings, storage_settings)
    embedder = HuggingFaceEmbedding(models_settings.embedder.model_path, trust_remote_code=True, embed_batch_size=5)
    # reranker = CohereRerank(api_key=os.environ["COHERE_API_KEY"], top_n=10)
    knowledge_graph_indexer = KnowledgeGraphIndexer(
        kg_storage.storage_context, path_settings, storage_settings.document_storage.storage_type, kg_search_settings,
        embedder, None
    )
    knowledge_graph_indexer.load_index()

    prompt_engine = PromptEngine(path_settings.prompts_dir)
    role_settings = getattr(models_settings, ModelRoleType.orchestrator.name)
    prompt_version = role_settings.prompt_version["graph_index"]
    graph_index_prompt, graph_index_system_prompt = prompt_engine.render(
        PromptType.learner_workflow, ModelRoleType.orchestrator, prompt_version, "graph_index"
    )
    role_params = role_settings.model_dump(include={"temperature", "top_k", "top_p", "base_url"})
    llm_params = {"model": role_settings.model_name, **role_params}

    # Uncomment to start building index
    # knowledge_graph_indexer.build_index(
    #     llm=llm_params,
    #     graph_index_prompt=graph_index_system_prompt["system_instruction"] + "\n" + graph_index_prompt,
    # )
    knowledge_graph_indexer.load_index()

    retriever_params = {
        "VectorContextRetriever": {
            "include_text": False, "similarity_top_k": 5, "similarity_score": None, "depth": 4,
            "include_properties": False
        },
        "VectorIndexRetriever": {"similarity_top_k": 3, }
    }
    retriever = knowledge_graph_indexer.get_retriever(retriever_params)
    # results = retriever.retrieve("AI Agent Frameworks")
    # results = retriever.retrieve('GAN discriminator real fake images')
