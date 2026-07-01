from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from backend.configs.search import KnowledgeGraphSearchSettings
from backend.configs.constants import WORKFLOW_LOGGING
from backend.utils.helpers import get_logger
from backend.workflows.agents.retrieval_access import (
    POSTPROCESSED_CHUNK_KIND,
    POSTPROCESSED_CONCEPT_KIND,
    RetrievalStoreAccess,
)


logger = get_logger(WORKFLOW_LOGGING)


COMPARISON_PATTERN = re.compile(r"\s+(?:vs\.?|versus|compared?\s+to)\s+", re.IGNORECASE)
VISUALIZATION_PREFIX_PATTERN = re.compile(
    r"^\s*(?:please\s+)?(?:visuali[sz]e|show|plot|map|draw)\s+"
    r"(?:(?:my|the)\s+)?(?:(?:knowledge|information|graph|topics?|connections?)\s+)?"
    r"(?:about|of|for)?\s*",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_+-]*")


@dataclass
class ConceptCandidate:
    node_id: str
    node: Any
    label: str
    score: float
    support_chunk_ids: list[str] = field(default_factory=list)
    is_anchor: bool = False


@dataclass
class EdgeCandidate:
    subject_id: str
    predicate: str
    object_id: str
    score: float
    evidence_chunk_ids: list[str] = field(default_factory=list)
    predicate_family: str | None = None
    relation_phrases: list[str] = field(default_factory=list)
    generality_score: float | None = None
    visualization_usefulness: float | None = None
    synthetic: bool = False


class VisualizerRetrievalPipeline:
    """Concept-first graph retrieval for the Visualizer tool."""

    def __init__(self, knowledge_graph_indexer: Any, settings: KnowledgeGraphSearchSettings | None = None) -> None:
        self.indexer = knowledge_graph_indexer
        self.settings = settings or knowledge_graph_indexer.kg_search_settings
        self.access = RetrievalStoreAccess(knowledge_graph_indexer)
        self.embedder = self.access.embedder
        self.storage_context = self.access.storage_context
        self.vector_store = self.access.vector_store
        self.graph_store = self.access.graph_store

    @property
    def health_report(self):
        return self.access.health_report

    def visualize(self, queries: list[str]) -> tuple[list[str], list[tuple[str, str, str]], list[str]]:
        original_queries = list(queries)
        if not queries:
            return [], [], original_queries

        canonical_queries = [
            query_text
            for query in queries
            if (query_text := self._canonical_query(query))
        ]
        multi_query_focus = len(canonical_queries) > 1

        concept_candidates: dict[str, ConceptCandidate] = {}
        edge_candidates: dict[tuple[str, str, str], EdgeCandidate] = {}

        for query_text in canonical_queries:
            comparison_like_query = len(self._anchor_terms(query_text)) > 1
            strict_synthetic_match = multi_query_focus

            source_chunk_ids = self._retrieve_supporting_source_chunk_ids(query_text)
            anchors = self._resolve_anchors(query_text)
            if not anchors:
                continue
            source_chunk_ids = self._filter_source_chunks_for_anchors(source_chunk_ids, anchors)

            mentioned = self._concepts_mentioned_by_chunks(source_chunk_ids, query_text)
            for concept in [*anchors, *mentioned]:
                self._merge_concept_candidate(concept_candidates, concept, query_text, concept.support_chunk_ids)

            query_edges = self._semantic_edges_for_query(
                query_text,
                anchors,
                mentioned,
                source_chunk_ids,
                concept_candidates,
            )
            for edge in query_edges:
                self._merge_edge_candidate(edge_candidates, edge)

            if self.settings.visualizer_allow_synthetic_edges and not comparison_like_query:
                synthetic_edges = self._synthetic_edges_for_query(
                    query_text,
                    anchors,
                    mentioned,
                    source_chunk_ids,
                    concept_candidates,
                    strict_query_match=strict_synthetic_match,
                )
                for edge in synthetic_edges:
                    self._merge_edge_candidate(edge_candidates, edge)

        if not concept_candidates:
            return [], [], original_queries

        nodes, edges = self._shape_graph(concept_candidates, edge_candidates)
        if len(nodes) < self.settings.visualizer_min_nodes and not edges:
            return [], [], original_queries

        triplets = [(edge.subject_id, edge.predicate, edge.object_id) for edge in edges]
        return nodes, triplets, original_queries

    def _resolve_anchors(self, query: str) -> list[ConceptCandidate]:
        anchors: dict[str, ConceptCandidate] = {}
        anchor_terms = self._anchor_terms(query)
        all_concepts = self._all_concepts()

        for term in anchor_terms:
            term_key = self._normalize_text(term)
            if not term_key:
                continue

            for node in all_concepts:
                node_id = self._node_id(node)
                if not node_id:
                    continue
                label = self._node_label(node)
                aliases = self._node_aliases(node)
                score = self._exact_anchor_score(term_key, label, aliases)
                if score <= 0:
                    continue
                candidate = ConceptCandidate(
                    node_id=node_id,
                    node=node,
                    label=label,
                    score=score,
                    support_chunk_ids=self._node_source_chunk_ids(node),
                    is_anchor=True,
                )
                self._merge_concept_candidate(anchors, candidate, query, candidate.support_chunk_ids)

        if not anchors or max((candidate.score for candidate in anchors.values()), default=0.0) < 1.0:
            for candidate in self._vector_concept_candidates(query):
                if candidate.score < self.settings.visualizer_anchor_min_score:
                    continue
                candidate.is_anchor = True
                self._merge_concept_candidate(anchors, candidate, query, candidate.support_chunk_ids)

        return sorted(
            anchors.values(),
            key=lambda item: (item.score, self._salience(item.node), item.label.lower()),
            reverse=True,
        )[: self.settings.visualizer_anchor_top_k]

    def _retrieve_supporting_source_chunk_ids(self, query: str) -> list[str]:
        if self.vector_store is None:
            self.health_report.record(
                "missing_vector_store",
                "Visualizer retrieval has no vector store; skipping source support lookup.",
                component="visualizer",
            )
            return []

        result = self._query_vector_store(
            query,
            top_k=self.settings.visualizer_source_candidate_k,
            node_kind=POSTPROCESSED_CHUNK_KIND,
        )
        ids = list(getattr(result, "ids", None) or [])
        similarities = list(getattr(result, "similarities", None) or [0.0] * len(ids))
        scored_ids: list[tuple[str, float]] = []

        for node_id, similarity in zip(ids, similarities, strict=False):
            node = self._get_docstore_node(node_id)
            if not self._usable_source_node(node):
                continue
            metadata = getattr(node, "metadata", {}) or {}
            text = self._node_text(node)
            score = self._clip_score(float(similarity or 0.0) + self._source_query_boost(query, text, metadata))
            scored_ids.append((str(node_id), score))

        scored_ids.sort(key=lambda item: item[1], reverse=True)
        return self._ordered_unique(node_id for node_id, _ in scored_ids)

    def _filter_source_chunks_for_anchors(
        self,
        chunk_ids: list[str],
        anchors: list[ConceptCandidate],
    ) -> list[str]:
        if not self.settings.visualizer_anchor_source_filter or not chunk_ids or not anchors:
            return chunk_ids

        anchor_source_keys = self._source_scope_for_concepts(anchors)
        if not anchor_source_keys:
            return chunk_ids

        scoped_chunk_ids: list[str] = []
        for chunk_id in chunk_ids:
            node = self._get_docstore_node(chunk_id)
            metadata = getattr(node, "metadata", {}) or {}
            if self._source_scope_key(metadata) in anchor_source_keys:
                scoped_chunk_ids.append(chunk_id)

        return scoped_chunk_ids or chunk_ids[: max(self.settings.visualizer_min_nodes, 1)]

    def _vector_concept_candidates(self, query: str) -> list[ConceptCandidate]:
        if self.vector_store is None:
            self.health_report.record(
                "missing_vector_store",
                "Visualizer retrieval has no vector store; skipping vector concept lookup.",
                component="visualizer",
            )
            return []

        result = self._query_vector_store(
            query,
            top_k=self.settings.visualizer_concept_candidate_k,
            node_kind=POSTPROCESSED_CONCEPT_KIND,
        )
        ids = list(getattr(result, "ids", None) or [])
        similarities = list(getattr(result, "similarities", None) or [0.0] * len(ids))
        candidates: list[ConceptCandidate] = []

        for node_id, similarity in zip(ids, similarities, strict=False):
            node = self._get_graph_node(str(node_id))
            if node is None or not self._is_concept_node(node):
                continue
            label = self._node_label(node)
            score = self._clip_score(0.15 + (float(similarity or 0.0) * 0.85))
            score += self._label_query_boost(query, label, self._node_aliases(node))
            candidates.append(
                ConceptCandidate(
                    node_id=str(node_id),
                    node=node,
                    label=label,
                    score=self._clip_score(score),
                    support_chunk_ids=self._node_source_chunk_ids(node),
                )
            )

        return sorted(candidates, key=lambda item: (item.score, item.label.lower()), reverse=True)

    def _semantic_edges_for_query(
        self,
        query: str,
        anchors: list[ConceptCandidate],
        mentioned: list[ConceptCandidate],
        source_chunk_ids: list[str],
        concept_candidates: dict[str, ConceptCandidate],
    ) -> list[EdgeCandidate]:
        if self.graph_store is None:
            self.health_report.record(
                "missing_graph_store",
                "Visualizer retrieval has no graph store; skipping semantic edge expansion.",
                component="visualizer",
            )
            return []
        if not anchors:
            return []

        seed_candidates = self._semantic_seed_candidates(anchors, mentioned, source_chunk_ids)
        seed_nodes = [candidate.node for candidate in seed_candidates]
        triplets = self._get_relation_map(seed_nodes)
        anchor_ids = {candidate.node_id for candidate in anchors}
        support_chunk_set = set(source_chunk_ids)
        query_terms = self._query_terms(query)
        edges: list[EdgeCandidate] = []

        for subject, relation, object_ in triplets:
            predicate = self._relation_label(relation)
            if predicate in self.settings.visualizer_denied_relation_labels:
                continue
            subject_id = self._node_id(subject)
            object_id = self._node_id(object_)
            if not subject_id or not object_id:
                continue
            if not self._is_concept_node(subject) or not self._is_concept_node(object_):
                continue

            evidence_chunk_ids = self._relation_evidence_chunk_ids(relation)
            directly_anchor_edge = subject_id in anchor_ids or object_id in anchor_ids
            grounded = bool(support_chunk_set.intersection(evidence_chunk_ids))
            if not directly_anchor_edge and not grounded:
                continue

            subject_candidate = self._concept_candidate_from_node(subject, query)
            object_candidate = self._concept_candidate_from_node(object_, query)
            if predicate == self.settings.visualizer_synthetic_edge_label or predicate == "RELATED_TO":
                endpoint_candidate = object_candidate if subject_id in anchor_ids else subject_candidate
                facet_score = self._facet_candidate_score(
                    query_terms,
                    endpoint_candidate.label,
                    evidence_chunk_ids,
                )
                if self._facet_excludes_candidate(query_terms, facet_score):
                    continue
            self._merge_concept_candidate(concept_candidates, subject_candidate, query, evidence_chunk_ids)
            self._merge_concept_candidate(concept_candidates, object_candidate, query, evidence_chunk_ids)

            relation_properties = getattr(relation, "properties", {}) or {}
            confidence = self._float_or_none(
                relation_properties.get("max_confidence")
                or relation_properties.get("confidence")
            )
            predicate_family = self._string_or_none(relation_properties.get("predicate_family"))
            relation_phrases = self._string_list(relation_properties.get("relation_phrases"))
            generality_score = self._float_or_none(relation_properties.get("max_generality_score"))
            visualization_usefulness = self._float_or_none(relation_properties.get("max_visualization_usefulness"))
            generic_penalty = 0.06 if predicate == "RELATED_TO" or predicate_family == "other" else 0.0
            score = (
                0.35
                + max(subject_candidate.score, object_candidate.score) * 0.35
                + (0.12 if directly_anchor_edge else 0.0)
                + (0.10 if grounded else 0.0)
                + ((confidence or 0.0) * 0.08)
                + ((visualization_usefulness or 0.0) * 0.12)
                + ((generality_score or 0.0) * 0.06)
                + self._relation_query_boost(query, subject_candidate.label, predicate, object_candidate.label)
                - generic_penalty
            )
            edges.append(
                EdgeCandidate(
                    subject_id=subject_id,
                    predicate=predicate,
                    object_id=object_id,
                    score=self._clip_score(score),
                    evidence_chunk_ids=evidence_chunk_ids,
                    predicate_family=predicate_family,
                    relation_phrases=relation_phrases,
                    generality_score=generality_score,
                    visualization_usefulness=visualization_usefulness,
                )
            )

        return edges

    def _semantic_seed_candidates(
        self,
        anchors: list[ConceptCandidate],
        mentioned: list[ConceptCandidate],
        source_chunk_ids: list[str],
    ) -> list[ConceptCandidate]:
        source_chunk_set = set(source_chunk_ids)
        seeds: dict[str, ConceptCandidate] = {candidate.node_id: candidate for candidate in anchors}
        mentioned_limit = max(self.settings.visualizer_anchor_top_k, 1)
        for candidate in mentioned:
            if len(seeds) >= len(anchors) + mentioned_limit:
                break
            if candidate.node_id in seeds:
                continue
            if not source_chunk_set.intersection(candidate.support_chunk_ids):
                continue
            seeds[candidate.node_id] = candidate
        return list(seeds.values())

    def _synthetic_edges_for_query(
        self,
        query: str,
        anchors: list[ConceptCandidate],
        mentioned: list[ConceptCandidate],
        source_chunk_ids: list[str],
        concept_candidates: dict[str, ConceptCandidate],
        *,
        strict_query_match: bool = False,
    ) -> list[EdgeCandidate]:
        if not anchors or not mentioned:
            return []

        source_chunk_set = set(source_chunk_ids)
        source_rank = {chunk_id: idx for idx, chunk_id in enumerate(source_chunk_ids)}
        query_terms = self._query_terms(query)
        edges: list[EdgeCandidate] = []
        for anchor in anchors:
            anchor_source_chunks = [
                chunk_id for chunk_id in anchor.support_chunk_ids
                if chunk_id in source_chunk_set
            ]
            anchor_source_chunk_set = set(anchor_source_chunks)
            for concept in mentioned:
                if concept.node_id == anchor.node_id:
                    continue
                concept_source_chunks = [
                    chunk_id for chunk_id in concept.support_chunk_ids
                    if chunk_id in source_chunk_set
                ]
                if not concept_source_chunks:
                    continue
                truly_shared_chunks = [
                    chunk_id for chunk_id in concept_source_chunks
                    if chunk_id in anchor_source_chunk_set
                ]
                shared_chunks = self._ordered_unique(truly_shared_chunks or concept_source_chunks)
                facet_score = self._facet_candidate_score(query_terms, concept.label, shared_chunks)
                if self._facet_excludes_candidate(query_terms, facet_score):
                    continue
                if strict_query_match and not self._has_term_overlap(query_terms, concept.label) and facet_score <= 0:
                    continue
                self._merge_concept_candidate(concept_candidates, concept, "", shared_chunks)
                best_shared_rank = min(source_rank.get(chunk_id, len(source_rank)) for chunk_id in shared_chunks)
                source_rank_boost = max(0.0, 0.10 - (best_shared_rank * 0.015))
                score = self._clip_score(
                    (anchor.score * 0.45)
                    + (concept.score * 0.35)
                    + 0.10
                    + source_rank_boost
                    + facet_score
                )
                edges.append(
                    EdgeCandidate(
                        subject_id=anchor.node_id,
                        predicate=self.settings.visualizer_synthetic_edge_label,
                        object_id=concept.node_id,
                        score=score,
                        evidence_chunk_ids=shared_chunks,
                        synthetic=True,
                    )
                )

        edges.sort(key=lambda item: (item.score, item.object_id), reverse=True)
        return edges[: self.settings.visualizer_synthetic_edge_limit]

    def _concepts_mentioned_by_chunks(self, chunk_ids: list[str], query: str) -> list[ConceptCandidate]:
        if not chunk_ids or self.graph_store is None:
            return []

        candidates: dict[str, ConceptCandidate] = {}
        for source_node, relation, target_node in self._get_triplets(ids=chunk_ids, relation_names=["MENTIONS"]):
            if self._relation_label(relation) != "MENTIONS":
                continue
            target_id = self._node_id(target_node)
            if not target_id or not self._is_concept_node(target_node):
                continue
            evidence_chunk_ids = self._relation_evidence_chunk_ids(relation) or [self._node_id(source_node)]
            candidate = self._concept_candidate_from_node(target_node, query)
            candidate.support_chunk_ids = self._ordered_unique([*candidate.support_chunk_ids, *evidence_chunk_ids])
            self._merge_concept_candidate(candidates, candidate, "", evidence_chunk_ids)

        return sorted(candidates.values(), key=lambda item: (item.score, item.label.lower()), reverse=True)

    def _shape_graph(
        self,
        concepts: dict[str, ConceptCandidate],
        edges: dict[tuple[str, str, str], EdgeCandidate],
    ) -> tuple[list[str], list[EdgeCandidate]]:
        concepts, edges = self._dedupe_concepts_by_label(concepts, edges)
        edges = self._drop_redundant_generic_edges(edges)
        edge_list = [
            edge for edge in edges.values()
            if edge.predicate not in self.settings.visualizer_denied_relation_labels
        ]
        edge_list.sort(
            key=lambda item: (
                not item.synthetic,
                item.predicate not in {self.settings.visualizer_synthetic_edge_label, "RELATED_TO"},
                bool(item.evidence_chunk_ids),
                item.score,
                item.subject_id,
                item.predicate,
                item.object_id,
            ),
            reverse=True,
        )

        max_edges_per_node = max(1, self.settings.visualizer_max_edges_per_node)
        selected_edges: list[EdgeCandidate] = []
        selected_node_ids: list[str] = []
        degree_counts: dict[str, int] = {}

        for edge in edge_list:
            if len(selected_edges) >= self.settings.visualizer_max_edges:
                break
            if edge.subject_id not in concepts or edge.object_id not in concepts:
                continue
            if degree_counts.get(edge.subject_id, 0) >= max_edges_per_node:
                continue
            if degree_counts.get(edge.object_id, 0) >= max_edges_per_node:
                continue
            selected_node_set = set(selected_node_ids)
            new_node_ids = [
                node_id for node_id in (edge.subject_id, edge.object_id)
                if node_id not in selected_node_set
            ]
            if len(selected_node_ids) + len(new_node_ids) > self.settings.visualizer_max_nodes:
                continue
            selected_edges.append(edge)
            selected_node_ids.extend(new_node_ids)
            degree_counts[edge.subject_id] = degree_counts.get(edge.subject_id, 0) + 1
            degree_counts[edge.object_id] = degree_counts.get(edge.object_id, 0) + 1

        anchors = [
            candidate for candidate in concepts.values()
            if candidate.is_anchor and candidate.node_id not in selected_node_ids
        ]
        other_nodes: list[ConceptCandidate] = []
        if self.settings.visualizer_include_isolated_nodes:
            other_nodes = [
                candidate for candidate in concepts.values()
                if not candidate.is_anchor and candidate.node_id not in selected_node_ids
            ]
        ordered_extra_nodes = sorted(
            [*anchors, *other_nodes],
            key=lambda item: (item.is_anchor, item.score, self._salience(item.node), item.label.lower()),
            reverse=True,
        )
        for candidate in ordered_extra_nodes:
            if len(selected_node_ids) >= self.settings.visualizer_max_nodes:
                break
            selected_node_ids.append(candidate.node_id)

        selected_node_set = set(selected_node_ids)
        selected_edges = [
            edge for edge in selected_edges
            if edge.subject_id in selected_node_set and edge.object_id in selected_node_set
        ]
        if not self.settings.visualizer_show_chunks:
            selected_node_ids = [node_id for node_id in selected_node_ids if not node_id.startswith("chunk_")]

        return selected_node_ids, selected_edges

    def _query_vector_store(self, query: str, *, top_k: int, node_kind: str) -> Any:
        query_embedding = self.embedder.get_query_embedding(query)
        return self.access.query_vector(
            query_embedding,
            top_k=top_k,
            node_kind=node_kind,
            component="visualizer",
            fallback_message="Visualizer vector query with metadata filters failed; retrying unfiltered.",
        )

    def _get_relation_map(self, nodes: list[Any]) -> list[tuple[Any, Any, Any]]:
        return self.access.relation_map(
            nodes,
            depth=self.settings.visualizer_graph_depth,
            limit=max(self.settings.visualizer_max_edges * 3, 30),
            ignore_rels=self.settings.visualizer_denied_relation_labels,
            component="visualizer",
            fallback_message="Visualizer graph relation-map lookup failed; retrying with triplet lookup.",
        )

    def _get_triplets(
        self,
        *,
        ids: list[str] | None = None,
        relation_names: list[str] | None = None,
    ) -> list[tuple[Any, Any, Any]]:
        return self.access.triplets(ids=ids, relation_names=relation_names, component="visualizer")

    def _all_concepts(self) -> list[Any]:
        return self.access.all_concepts(component="visualizer")

    def _get_graph_node(self, node_id: str) -> Any | None:
        return self.access.graph_node(node_id, component="visualizer")

    def _get_docstore_node(self, node_id: str) -> Any | None:
        return self.access.docstore_node(node_id)

    def _concept_candidate_from_node(self, node: Any, query: str) -> ConceptCandidate:
        node_id = self._node_id(node) or ""
        label = self._node_label(node)
        score = 0.35 + (self._salience(node) * 0.25) + self._label_query_boost(query, label, self._node_aliases(node))
        return ConceptCandidate(
            node_id=node_id,
            node=node,
            label=label,
            score=self._clip_score(score),
            support_chunk_ids=self._node_source_chunk_ids(node),
        )

    def _merge_concept_candidate(
        self,
        candidates: dict[str, ConceptCandidate],
        candidate: ConceptCandidate,
        query: str,
        support_chunk_ids: list[str],
    ) -> None:
        if not candidate.node_id:
            return
        candidate.support_chunk_ids = self._ordered_unique([*candidate.support_chunk_ids, *support_chunk_ids])
        if query:
            candidate.score = self._clip_score(
                candidate.score + self._label_query_boost(query, candidate.label, self._node_aliases(candidate.node))
            )
        existing = candidates.get(candidate.node_id)
        if existing is None:
            candidates[candidate.node_id] = candidate
            return
        existing.score = max(existing.score, candidate.score)
        existing.is_anchor = existing.is_anchor or candidate.is_anchor
        existing.support_chunk_ids = self._ordered_unique([*existing.support_chunk_ids, *candidate.support_chunk_ids])

    @staticmethod
    def _merge_edge_candidate(candidates: dict[tuple[str, str, str], EdgeCandidate], candidate: EdgeCandidate) -> None:
        key = (candidate.subject_id, candidate.predicate, candidate.object_id)
        existing = candidates.get(key)
        if existing is None or candidate.score > existing.score:
            candidates[key] = candidate

    def _dedupe_concepts_by_label(
        self,
        concepts: dict[str, ConceptCandidate],
        edges: dict[tuple[str, str, str], EdgeCandidate],
    ) -> tuple[dict[str, ConceptCandidate], dict[tuple[str, str, str], EdgeCandidate]]:
        candidates = list(concepts.values())
        parent = {candidate.node_id: candidate.node_id for candidate in candidates}

        def find(node_id: str) -> str:
            while parent[node_id] != node_id:
                parent[node_id] = parent[parent[node_id]]
                node_id = parent[node_id]
            return node_id

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        owner_by_key: dict[str, str] = {}
        for candidate in candidates:
            for key in self._dedupe_keys_for_candidate(candidate):
                existing_owner = owner_by_key.get(key)
                if existing_owner is None:
                    owner_by_key[key] = candidate.node_id
                    continue
                union(candidate.node_id, existing_owner)

        grouped: dict[str, list[ConceptCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(find(candidate.node_id), []).append(candidate)

        merged_concepts: dict[str, ConceptCandidate] = {}
        node_id_map: dict[str, str] = {}
        for candidates in grouped.values():
            canonical = max(
                candidates,
                key=lambda item: (
                    item.is_anchor,
                    item.score,
                    self._salience(item.node),
                    len(self._node_aliases(item.node)),
                    len(item.label),
                    item.label.lower(),
                ),
            )
            for candidate in candidates:
                node_id_map[candidate.node_id] = canonical.node_id
                if candidate.node_id == canonical.node_id:
                    continue
                canonical.score = max(canonical.score, candidate.score)
                canonical.is_anchor = canonical.is_anchor or candidate.is_anchor
                canonical.support_chunk_ids = self._ordered_unique(
                    [*canonical.support_chunk_ids, *candidate.support_chunk_ids]
                )
            merged_concepts[canonical.node_id] = canonical

        remapped_edges: dict[tuple[str, str, str], EdgeCandidate] = {}
        for edge in edges.values():
            subject_id = node_id_map.get(edge.subject_id, edge.subject_id)
            object_id = node_id_map.get(edge.object_id, edge.object_id)
            if subject_id == object_id:
                continue
            remapped_edge = EdgeCandidate(
                subject_id=subject_id,
                predicate=edge.predicate,
                object_id=object_id,
                score=edge.score,
                evidence_chunk_ids=edge.evidence_chunk_ids,
                predicate_family=edge.predicate_family,
                relation_phrases=edge.relation_phrases,
                generality_score=edge.generality_score,
                visualization_usefulness=edge.visualization_usefulness,
                synthetic=edge.synthetic,
            )
            self._merge_edge_candidate(remapped_edges, remapped_edge)

        return merged_concepts, remapped_edges

    def _drop_redundant_generic_edges(
        self,
        edges: dict[tuple[str, str, str], EdgeCandidate],
    ) -> dict[tuple[str, str, str], EdgeCandidate]:
        generic_labels = {self.settings.visualizer_synthetic_edge_label, "RELATED_TO"}
        specific_pairs = {
            (edge.subject_id, edge.object_id)
            for edge in edges.values()
            if edge.predicate not in generic_labels
        }
        return {
            key: edge
            for key, edge in edges.items()
            if edge.predicate not in generic_labels or (edge.subject_id, edge.object_id) not in specific_pairs
        }

    def _dedupe_keys_for_candidate(self, candidate: ConceptCandidate) -> set[str]:
        properties = getattr(candidate.node, "properties", {}) or {}
        raw_values = [
            candidate.label,
            properties.get("entity_name"),
            properties.get("display_name"),
        ]
        keys: set[str] = set()
        for value in raw_values:
            key = self._normalize_text(str(value or ""))
            if not key:
                continue
            keys.add(key)
        return keys or {candidate.node_id}

    def _source_scope_for_concepts(self, concepts: list[ConceptCandidate]) -> set[str]:
        source_keys: set[str] = set()
        for concept in concepts:
            for chunk_id in concept.support_chunk_ids:
                node = self._get_docstore_node(chunk_id)
                metadata = getattr(node, "metadata", {}) or {}
                if source_key := self._source_scope_key(metadata):
                    source_keys.add(source_key)
        return source_keys

    def _source_scope_key(self, metadata: dict[str, Any]) -> str:
        source = str(metadata.get("source") or "").strip()
        if not source:
            path = self._string_list(metadata.get("path"))
            source = path[0] if path else ""
        return self._normalize_text(source)

    def _anchor_terms(self, query: str) -> list[str]:
        terms = [part.strip() for part in COMPARISON_PATTERN.split(query) if part.strip()]
        return terms or [query]

    @staticmethod
    def _canonical_query(query: str) -> str:
        query = " ".join(str(query or "").split())
        query = VISUALIZATION_PREFIX_PATTERN.sub("", query).strip()
        return query

    def _usable_source_node(self, node: Any | None) -> bool:
        if node is None:
            return False
        metadata = getattr(node, "metadata", {}) or {}
        if metadata.get("docstore_node_kind") not in {None, POSTPROCESSED_CHUNK_KIND}:
            return False
        if metadata.get("graph_enabled") is not True:
            return False
        if metadata.get("quarantined") is True:
            return False
        text = self._node_text(node)
        summary = str(metadata.get("postprocess_chunk_summary") or "")
        return bool(text.strip() or summary.strip())

    def _source_query_boost(self, query: str, text: str, metadata: dict[str, Any]) -> float:
        query_terms = self._query_terms(query)
        metadata_text = " ".join(
            [
                *self._string_list(metadata.get("path")),
                *self._string_list(metadata.get("heading_path")),
                str(metadata.get("source") or ""),
                str(metadata.get("postprocess_chunk_summary") or ""),
            ]
        )
        boost = 0.0
        if self._has_term_overlap(query_terms, metadata_text):
            boost += 0.08
        if self._has_term_overlap(query_terms, text):
            boost += 0.05
        boost += self._facet_source_boost(query_terms, f"{metadata_text} {text}")
        if str(metadata.get("postprocess_action") or "").lower() == "metadata_only":
            boost -= 0.12
        return boost

    def _exact_anchor_score(self, term_key: str, label: str, aliases: list[str]) -> float:
        label_key = self._normalize_text(label)
        alias_keys = [self._normalize_text(alias) for alias in aliases]
        if term_key == label_key or term_key in alias_keys:
            return 1.0
        term_tokens = term_key.split()
        label_tokens = label_key.split()
        if self._can_partially_match_key(term_tokens) and self._token_sequence_overlaps(term_tokens, label_tokens):
            return 0.85
        if any(
            alias_key
            and len(alias_key.split()) >= 2
            and self._token_sequence_overlaps(term_tokens, alias_key.split())
            for alias_key in alias_keys
        ):
            return 0.80
        return 0.0

    def _label_query_boost(self, query: str, label: str, aliases: list[str]) -> float:
        if not query:
            return 0.0
        query_terms = self._query_terms(query)
        label_text = " ".join([label, *aliases])
        boost = 0.12 if self._has_term_overlap(query_terms, label_text) else 0.0
        boost += self._facet_label_boost(query_terms, label_text)
        return boost

    def _relation_query_boost(self, query: str, subject: str, predicate: str, object_: str) -> float:
        relation_text = f"{subject} {predicate} {object_}"
        return 0.08 if self._has_term_overlap(self._query_terms(query), relation_text) else 0.0

    def _facet_candidate_score(self, query_terms: set[str], label: str, evidence_chunk_ids: list[str]) -> float:
        evidence_text_parts = [label]
        for chunk_id in evidence_chunk_ids[:3]:
            node = self._get_docstore_node(chunk_id)
            metadata = getattr(node, "metadata", {}) or {}
            evidence_text_parts.extend(
                [
                    *self._string_list(metadata.get("path")),
                    *self._string_list(metadata.get("heading_path")),
                    str(metadata.get("source") or ""),
                    str(metadata.get("postprocess_chunk_summary") or ""),
                ]
            )

        evidence_text = " ".join(part for part in evidence_text_parts if part)
        return self._facet_label_boost(query_terms, label) + self._facet_source_boost(query_terms, evidence_text)

    @staticmethod
    def _facet_excludes_candidate(query_terms: set[str], facet_score: float) -> bool:
        if query_terms & {"dataset", "datasets", "benchmark", "benchmarks", "corpus", "corpora"}:
            return facet_score <= 0.0
        if query_terms & {"architecture", "component", "components"}:
            return facet_score <= 0.0
        if query_terms & {"training", "objective", "loss"}:
            return facet_score <= 0.0
        if query_terms & {"method", "methods", "model", "models", "classifier", "classifiers"}:
            return facet_score < -0.05
        return False

    @staticmethod
    def _facet_source_boost(query_terms: set[str], text: str) -> float:
        lowered = text.lower()
        boost = 0.0
        if query_terms & {"method", "methods", "model", "models", "classifier", "classifiers"}:
            if any(term in lowered for term in ("method", "model", "classifier", "component", "architecture")):
                boost += 0.10
            if any(term in lowered for term in ("dataset", "datasets", "benchmark", "corpus")):
                boost -= 0.12
        if query_terms & {"dataset", "datasets", "benchmark", "benchmarks", "corpus", "corpora"}:
            if any(term in lowered for term in ("dataset", "datasets", "benchmark", "corpus")):
                boost += 0.12
            if any(term in lowered for term in ("classifier examples", "model for", "training using")):
                boost -= 0.08
        if query_terms & {"architecture", "component", "components"}:
            if any(term in lowered for term in ("architecture", "component", "encoder", "decoder")):
                boost += 0.12
        if query_terms & {"training", "objective", "loss"}:
            if any(term in lowered for term in ("training", "objective", "loss", "pretraining")):
                boost += 0.12
        return boost

    @staticmethod
    def _facet_label_boost(query_terms: set[str], label_text: str) -> float:
        lowered = label_text.lower()
        if query_terms & {"method", "methods", "model", "models", "classifier", "classifiers"}:
            if any(term in lowered for term in ("classifier", "model", "method", "component", "extractor")):
                return 0.12
        if query_terms & {"dataset", "datasets", "benchmark", "benchmarks", "corpus", "corpora"}:
            if any(term in lowered for term in ("dataset", "corpus", "ag news", "dbpedia", "yelp")):
                return 0.12
        if query_terms & {"architecture", "component", "components"}:
            if any(term in lowered for term in ("encoder", "decoder", "architecture", "component")):
                return 0.12
        if query_terms & {"training", "objective", "loss"}:
            if any(term in lowered for term in ("objective", "loss", "training", "supervision", "encoder")):
                return 0.10
        return 0.0

    @staticmethod
    def _is_concept_node(node: Any) -> bool:
        return RetrievalStoreAccess.is_concept_node(node)

    @staticmethod
    def _node_id(node: Any) -> str | None:
        return RetrievalStoreAccess.node_id(node)

    @staticmethod
    def _node_label(node: Any) -> str:
        return RetrievalStoreAccess.node_label(node)

    @staticmethod
    def _node_text(node: Any) -> str:
        return RetrievalStoreAccess.node_text(node)

    @staticmethod
    def _node_aliases(node: Any) -> list[str]:
        return RetrievalStoreAccess.node_aliases(node)

    @staticmethod
    def _node_source_chunk_ids(node: Any) -> list[str]:
        return RetrievalStoreAccess.node_source_chunk_ids(node)

    @staticmethod
    def _relation_label(relation: Any) -> str:
        return RetrievalStoreAccess.relation_label(relation)

    @staticmethod
    def _relation_evidence_chunk_ids(relation: Any) -> list[str]:
        return RetrievalStoreAccess.relation_evidence_chunk_ids(relation)

    @staticmethod
    def _salience(node: Any) -> float:
        return RetrievalStoreAccess.salience(node)

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        return {
            token.lower()
            for token in TOKEN_PATTERN.findall(query)
            if len(token) > 2
        }

    @staticmethod
    def _has_term_overlap(query_terms: set[str], text: str) -> bool:
        if not query_terms or not text:
            return False
        lowered = text.lower()
        return any(term in lowered for term in query_terms)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(token.lower() for token in TOKEN_PATTERN.findall(str(text)))

    @staticmethod
    def _can_partially_match_key(tokens: list[str]) -> bool:
        return bool(tokens) and (len(tokens) >= 2 or len(tokens[0]) >= 3)

    @classmethod
    def _token_sequence_overlaps(cls, left_tokens: list[str], right_tokens: list[str]) -> bool:
        if not left_tokens or not right_tokens:
            return False
        if not cls._can_partially_match_key(left_tokens) or not cls._can_partially_match_key(right_tokens):
            return False
        return cls._contains_token_sequence(left_tokens, right_tokens) or cls._contains_token_sequence(
            right_tokens,
            left_tokens,
        )

    @staticmethod
    def _contains_token_sequence(container: list[str], needle: list[str]) -> bool:
        if not needle or len(needle) > len(container):
            return False
        needle_length = len(needle)
        return any(
            container[index : index + needle_length] == needle
            for index in range(len(container) - needle_length + 1)
        )

    @staticmethod
    def _string_or_none(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list | tuple | set):
            return [str(item) for item in value if item is not None]
        return [str(value)] if str(value) else []

    @staticmethod
    def _ordered_unique(values: Any) -> list[Any]:
        seen: set[Any] = set()
        unique_values: list[Any] = []
        for value in values:
            if value is None or value in seen:
                continue
            seen.add(value)
            unique_values.append(value)
        return unique_values

    @staticmethod
    def _clip_score(score: float) -> float:
        return max(0.0, min(1.0, score))

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
