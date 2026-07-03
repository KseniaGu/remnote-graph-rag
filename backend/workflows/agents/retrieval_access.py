from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, VectorStoreQuery

from backend.configs.constants import WORKFLOW_LOGGING
from backend.utils.helpers import get_logger


logger = get_logger(WORKFLOW_LOGGING)


POSTPROCESSED_CHUNK_KIND = "postprocessed_retrieval_chunk"
POSTPROCESSED_PASSAGE_KIND = "postprocessed_embedding_passage"
POSTPROCESSED_CONCEPT_KIND = "postprocessed_concept_node"


@dataclass
class RetrievalHealthEvent:
    """A recoverable retrieval degradation that should be visible to callers."""

    code: str
    message: str
    component: str
    recoverable: bool = True
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalHealthReport:
    """In-memory health report for optimized retrieval fallbacks."""

    events: list[RetrievalHealthEvent] = field(default_factory=list)

    def record(
        self,
        code: str,
        message: str,
        *,
        component: str = "retrieval",
        recoverable: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            RetrievalHealthEvent(
                code=code,
                message=message,
                component=component,
                recoverable=recoverable,
                details=details or {},
            )
        )

    def has_code(self, code: str) -> bool:
        return any(event.code == code for event in self.events)

    def codes(self) -> list[str]:
        return [event.code for event in self.events]


class RetrievalStoreAccess:
    """Shared access layer for retrieval vector, graph, and docstore reads."""

    def __init__(self, knowledge_graph_indexer: Any) -> None:
        self.indexer = knowledge_graph_indexer
        self.storage_context = getattr(knowledge_graph_indexer, "storage_context", None)
        self.embedder = getattr(knowledge_graph_indexer, "embedder", None)
        self.vector_store = self._resolve_vector_store()
        self.graph_store = self._resolve_graph_store()
        self.health_report = RetrievalHealthReport()
        self._concept_nodes: list[Any] | None = None
        self._concept_by_id: dict[str, Any] = {}

    def query_vector(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        node_kind: str | None = None,
        component: str,
        fallback_message: str,
    ) -> Any:
        if self.vector_store is None:
            self.health_report.record(
                "missing_vector_store",
                "Optimized retrieval has no vector store.",
                component=component,
            )
            return SimpleNamespace(ids=[], similarities=[])

        filters = (
            MetadataFilters(filters=[MetadataFilter(key="docstore_node_kind", value=node_kind)])
            if node_kind
            else None
        )
        query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=top_k,
            filters=filters,
        )

        try:
            return self.vector_store.query(query)
        except Exception:
            if not filters:
                self.health_report.record(
                    "vector_query_failed",
                    "Vector store query failed.",
                    component=component,
                    recoverable=False,
                )
                raise

            logger.warning(fallback_message, exc_info=True)
            self.health_report.record(
                "vector_filter_fallback",
                fallback_message,
                component=component,
                details={"node_kind": node_kind},
            )
            fallback_query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=top_k)
            try:
                return self.vector_store.query(fallback_query)
            except Exception:
                self.health_report.record(
                    "vector_query_failed",
                    "Unfiltered vector store fallback query failed.",
                    component=component,
                    recoverable=False,
                )
                raise

    def relation_map(
        self,
        nodes: list[Any],
        *,
        depth: int,
        limit: int,
        ignore_rels: list[str] | tuple[str, ...] | set[str],
        component: str,
        fallback_message: str,
    ) -> list[tuple[Any, Any, Any]]:
        if not nodes:
            return []
        if self.graph_store is None:
            self.health_report.record(
                "missing_graph_store",
                "Optimized retrieval has no graph store.",
                component=component,
            )
            return []

        try:
            return list(
                self.graph_store.get_rel_map(
                    nodes,
                    depth=depth,
                    limit=limit,
                    ignore_rels=list(ignore_rels),
                )
            )
        except Exception:
            logger.warning(fallback_message, exc_info=True)
            self.health_report.record(
                "graph_relation_map_fallback",
                fallback_message,
                component=component,
                details={"node_count": len(nodes), "limit": limit},
            )
            denied = set(ignore_rels)
            ids = [node_id for node in nodes if (node_id := self.node_id(node))]
            return [
                triplet
                for triplet in self.triplets(ids=ids, component=component)
                if self.relation_label(triplet[1]) not in denied
            ][:limit]

    def triplets(
        self,
        *,
        ids: list[str] | None = None,
        relation_names: list[str] | None = None,
        component: str,
    ) -> list[tuple[Any, Any, Any]]:
        if self.graph_store is None:
            self.health_report.record(
                "missing_graph_store",
                "Optimized retrieval has no graph store.",
                component=component,
            )
            return []

        try:
            return list(self.graph_store.get_triplets(ids=ids, relation_names=relation_names))
        except Exception:
            logger.warning(f"{component.capitalize()} graph triplet lookup failed.", exc_info=True)
            self.health_report.record(
                "graph_triplet_lookup_failed",
                "Graph triplet lookup failed.",
                component=component,
                details={"ids": ids or [], "relation_names": relation_names or []},
            )
            return []

    def all_concepts(self, *, component: str) -> list[Any]:
        if self._concept_nodes is not None:
            return self._concept_nodes

        if self.graph_store is None:
            self.health_report.record(
                "missing_graph_store",
                "Optimized retrieval has no graph store.",
                component=component,
            )
            self._concept_nodes = []
            return self._concept_nodes

        try:
            nodes = list(self.graph_store.get())
        except Exception:
            logger.warning(f"{component.capitalize()} concept enumeration failed.", exc_info=True)
            self.health_report.record(
                "concept_enumeration_failed",
                "Graph concept enumeration failed.",
                component=component,
            )
            nodes = []

        self._concept_nodes = [node for node in nodes if self.is_concept_node(node)]
        self._concept_by_id = {
            node_id: node
            for node in self._concept_nodes
            if (node_id := self.node_id(node))
        }
        return self._concept_nodes

    def graph_node(self, node_id: str, *, component: str) -> Any | None:
        if node_id in self._concept_by_id:
            return self._concept_by_id[node_id]
        if self.graph_store is None:
            self.health_report.record(
                "missing_graph_store",
                "Optimized retrieval has no graph store.",
                component=component,
            )
            return None

        try:
            nodes = list(self.graph_store.get(ids=[node_id]))
        except Exception:
            logger.warning(f"{component.capitalize()} graph node lookup failed for {node_id}.", exc_info=True)
            self.health_report.record(
                "graph_node_lookup_failed",
                "Graph node lookup failed.",
                component=component,
                details={"node_id": node_id},
            )
            return None

        node = nodes[0] if nodes else None
        if node is not None and self.is_concept_node(node):
            self._concept_by_id[node_id] = node
        return node

    def docstore_node(self, node_id: str) -> Any | None:
        docstore = getattr(self.storage_context, "docstore", None)
        return getattr(docstore, "docs", {}).get(node_id)

    def _resolve_vector_store(self) -> Any | None:
        index = getattr(self.indexer, "index", None)
        vector_store = getattr(index, "vector_store", None)
        if vector_store is not None:
            return vector_store
        return getattr(self.storage_context, "vector_store", None)

    def _resolve_graph_store(self) -> Any | None:
        index = getattr(self.indexer, "index", None)
        graph_store = getattr(index, "property_graph_store", None)
        if graph_store is not None:
            return graph_store
        return getattr(self.storage_context, "property_graph_store", None)

    @staticmethod
    def is_concept_node(node: Any) -> bool:
        node_id = RetrievalStoreAccess.node_id(node)
        label = str(getattr(node, "label", "") or "").lower()
        properties = getattr(node, "properties", {}) or {}
        if node_id and node_id.startswith("chunk_"):
            return False
        if label == "text_chunk":
            return False
        return bool(properties.get("entity_name") or properties.get("display_name") or (node_id and node_id.startswith("concept_")))

    @staticmethod
    def node_id(node: Any) -> str | None:
        for attr in ("id", "node_id", "id_", "name"):
            value = getattr(node, attr, None)
            if value is not None:
                return str(value)
        return None

    @staticmethod
    def node_label(node: Any) -> str:
        properties = getattr(node, "properties", {}) or {}
        for key in ("entity_name", "display_name", "name", "text"):
            value = properties.get(key)
            if value:
                return str(value)
        name = getattr(node, "name", None)
        if name and not str(name).startswith("concept_"):
            return str(name)
        node_id = RetrievalStoreAccess.node_id(node)
        return str(node_id) if node_id is not None else ""

    @staticmethod
    def node_text(node: Any) -> str:
        text = getattr(node, "text", None)
        if text is not None:
            return str(text)
        get_content = getattr(node, "get_content", None)
        if callable(get_content):
            try:
                return str(get_content())
            except Exception:
                return ""
        return ""

    @staticmethod
    def node_aliases(node: Any) -> list[str]:
        properties = getattr(node, "properties", {}) or {}
        return RetrievalStoreAccess.string_list(properties.get("aliases"))

    @staticmethod
    def node_source_chunk_ids(node: Any) -> list[str]:
        properties = getattr(node, "properties", {}) or {}
        return RetrievalStoreAccess.string_list(properties.get("source_chunk_ids"))

    @staticmethod
    def relation_label(relation: Any) -> str:
        for attr in ("label", "id"):
            value = getattr(relation, attr, None)
            if value:
                return str(value)
        return ""

    @staticmethod
    def relation_evidence_chunk_ids(relation: Any) -> list[str]:
        properties = getattr(relation, "properties", {}) or {}
        return RetrievalStoreAccess.string_list(
            properties.get("evidence_chunk_ids") or properties.get("source_chunk_ids")
        )

    @staticmethod
    def salience(node: Any) -> float:
        properties = getattr(node, "properties", {}) or {}
        value = properties.get("postprocess_max_salience") or properties.get("max_salience") or 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def string_or_none(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list | tuple | set):
            return [str(item) for item in value if item is not None]
        return [str(value)] if str(value) else []

    @staticmethod
    def ordered_unique(values: Any) -> list[Any]:
        seen: set[Any] = set()
        unique_values: list[Any] = []
        for value in values:
            if value is None or value in seen:
                continue
            seen.add(value)
            unique_values.append(value)
        return unique_values

    @staticmethod
    def clip_score(score: float) -> float:
        return max(0.0, min(1.0, score))

    @staticmethod
    def float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
