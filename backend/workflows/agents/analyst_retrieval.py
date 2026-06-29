from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, VectorStoreQuery

from backend.configs.constants import MAX_SOURCE_CHARS, MIN_RELEVANCE_SCORE, WORKFLOW_LOGGING
from backend.configs.models import RerankerSettings
from backend.configs.search import KnowledgeGraphSearchSettings
from backend.utils.helpers import get_logger
from backend.workflows.agents.retrieval_evidence import NormalizedMetadata, normalize_metadata


logger = get_logger(WORKFLOW_LOGGING)


DENIED_ANALYST_RELATIONS = {"MENTIONS", "PARENT", "CHILD"}
POSTPROCESSED_CHUNK_KIND = "postprocessed_retrieval_chunk"


@dataclass
class SourceCandidate:
    node_id: str
    node: Any
    metadata: NormalizedMetadata
    text: str
    base_score: float
    score: float
    rank: int
    rank_score: float | None = None
    reranked: bool = False
    mentioned_concepts: list[Any] = field(default_factory=list)


@dataclass
class RelationCandidate:
    relation_id: str
    subject: str
    predicate: str
    object: str
    score: float
    rank: int
    rank_score: float | None = None
    reranked: bool = False
    confidence: float | None = None
    predicate_family: str | None = None
    relation_phrases: list[str] = field(default_factory=list)
    generality_score: float | None = None
    retrieval_usefulness: float | None = None
    evidence_chunk_ids: list[str] = field(default_factory=list)
    evidence_spans: list[str] = field(default_factory=list)


class SentenceTransformerReranker:
    """Small LlamaIndex-compatible adapter around sentence-transformers CrossEncoder."""

    def __init__(
        self,
        *,
        model_name: str,
        top_n: int,
        batch_size: int,
        device: str | None,
        local_files_only: bool,
        trust_remote_code: bool,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.top_n = top_n
        self.batch_size = batch_size
        self.model = CrossEncoder(
            model_name,
            device=device,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

    def postprocess_nodes(self, nodes: list[NodeWithScore], query_str: str) -> list[NodeWithScore]:
        if not nodes:
            return []

        pairs = [(query_str, node.node.get_content()) for node in nodes]
        raw_scores = self.model.predict(pairs, batch_size=self.batch_size)
        scored_nodes = [
            NodeWithScore(node=node.node, score=float(score))
            for node, score in zip(nodes, raw_scores, strict=False)
        ]
        scored_nodes.sort(key=lambda item: float(item.score or 0.0), reverse=True)
        return scored_nodes[: self.top_n]


class AnalystRetrievalPipeline:
    """Source-first, graph-enriched retrieval for the Analyst tool only."""

    def __init__(
        self,
        knowledge_graph_indexer: Any,
        settings: KnowledgeGraphSearchSettings | None = None,
        reranker_settings: Any | None = None,
        reranker: Any | None = None,
    ) -> None:
        self.indexer = knowledge_graph_indexer
        self.settings = settings or knowledge_graph_indexer.kg_search_settings
        self.reranker_settings = reranker_settings or RerankerSettings()
        self.embedder = knowledge_graph_indexer.embedder
        self.storage_context = knowledge_graph_indexer.storage_context
        self.vector_store = self._resolve_vector_store()
        self.graph_store = self._resolve_graph_store()
        self.reranker = reranker if reranker is not None else self._create_configured_reranker()
        self._reranker_failure_logged = False

    def search(self, queries: list[str]) -> str:
        formatted_blocks: list[str] = []
        remaining_chars = max(0, self.settings.analyst_context_max_chars - len("RETRIEVER RESULTS:\n\n"))

        for query in queries:
            result = self._search_one(query)
            block = self._format_query_result(result["query"], result["sources"], result["relations"], remaining_chars)
            if block:
                formatted_blocks.append(block)
                remaining_chars -= len(block) + 2
            if remaining_chars <= 0:
                break

        if not formatted_blocks:
            return "No relevant information found."

        return "RETRIEVER RESULTS:\n\n" + "\n\n".join(formatted_blocks)

    def _search_one(self, query: str) -> dict[str, Any]:
        source_candidates = self._retrieve_source_candidates(query)
        source_candidates = self._dedupe_sources(source_candidates)
        source_candidates = self._rerank_sources(query, source_candidates)
        self._attach_mentioned_concepts(source_candidates)
        if self.reranker is None:
            source_candidates = self._rescore_sources_with_graph_support(query, source_candidates)
        source_candidates = self._sort_sources(source_candidates)

        final_sources = self._select_final_sources(source_candidates)
        relation_candidates = self._expand_semantic_relations(query, final_sources)
        relation_candidates = self._dedupe_relations(relation_candidates)
        relation_candidates = self._rerank_relations(query, final_sources, relation_candidates)
        final_relations = self._select_final_relations(relation_candidates)

        return {
            "query": query,
            "sources": final_sources,
            "relations": final_relations,
        }

    def _retrieve_source_candidates(self, query: str) -> list[SourceCandidate]:
        if self.vector_store is None:
            logger.warning("Analyst retrieval has no vector store; returning no source candidates.")
            return []

        query_embedding = self.embedder.get_query_embedding(query)
        query_result = self._query_vector_store(query_embedding, with_filters=True)
        if not query_result.ids:
            query_result = self._query_vector_store(query_embedding, with_filters=False)

        ids = list(query_result.ids or [])
        similarities = list(query_result.similarities or [0.0] * len(ids))
        candidates: list[SourceCandidate] = []

        for rank, (node_id, similarity) in enumerate(zip(ids, similarities, strict=False), start=1):
            node = self._get_docstore_node(node_id)
            if node is None:
                continue

            metadata = normalize_metadata(node)
            text = self._node_text(node)
            if not self._is_usable_source(metadata, text):
                continue

            base_score = self._clip_score(similarity)
            score = self._score_source(query, text, metadata, base_score)
            if score < MIN_RELEVANCE_SCORE:
                continue

            candidates.append(
                SourceCandidate(
                    node_id=node_id,
                    node=node,
                    metadata=metadata,
                    text=text,
                    base_score=base_score,
                    score=score,
                    rank=rank,
                    rank_score=score,
                )
            )

        return candidates

    def _query_vector_store(self, query_embedding: list[float], *, with_filters: bool) -> Any:
        filters = None
        if with_filters:
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(key="docstore_node_kind", value=POSTPROCESSED_CHUNK_KIND),
                ]
            )

        query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self.settings.analyst_source_candidate_k,
            filters=filters,
        )

        try:
            return self.vector_store.query(query)
        except Exception:
            if with_filters:
                logger.warning("Analyst vector query with metadata filters failed; retrying unfiltered.", exc_info=True)
                return self._query_vector_store(query_embedding, with_filters=False)
            raise

    def _rerank_sources(self, query: str, candidates: list[SourceCandidate]) -> list[SourceCandidate]:
        if not candidates or self.reranker is None:
            return candidates

        ordered_candidates = self._sort_sources(candidates)
        rerank_limit = max(0, self.settings.analyst_source_rerank_candidate_k)
        rerank_candidates = ordered_candidates[:rerank_limit]
        if not rerank_candidates:
            return ordered_candidates

        candidate_by_id = {candidate.node_id: candidate for candidate in ordered_candidates}
        nodes = [
            NodeWithScore(
                node=TextNode(id_=candidate.node_id, text=self._rerank_text(candidate, query=query)),
                score=candidate.score,
            )
            for candidate in rerank_candidates
        ]

        try:
            reranked_nodes = self.reranker.postprocess_nodes(nodes, query_str=query)
        except Exception:
            self._log_reranker_failure()
            return candidates

        reranked_ids: list[str] = []
        for reranked in reranked_nodes:
            node_id = self._node_id(reranked.node)
            candidate = candidate_by_id.get(node_id)
            if candidate is None:
                continue
            candidate.rank_score = self._raw_reranker_score(reranked.score)
            candidate.reranked = True
            reranked_ids.append(candidate.node_id)

        ordered = self._sort_sources(
            [candidate_by_id[node_id] for node_id in reranked_ids if node_id in candidate_by_id],
        )
        reranked_id_set = set(reranked_ids)
        ordered.extend(candidate for candidate in ordered_candidates if candidate.node_id not in reranked_id_set)
        return ordered

    def _attach_mentioned_concepts(self, candidates: list[SourceCandidate]) -> None:
        if self.graph_store is None or not candidates:
            return

        candidate_by_id = {candidate.node_id: candidate for candidate in candidates}
        for source_node, relation, target_node in self._get_triplets(ids=list(candidate_by_id), relation_names=["MENTIONS"]):
            if self._relation_label(relation) != "MENTIONS":
                continue
            source_id = getattr(source_node, "id", None)
            candidate = candidate_by_id.get(source_id)
            if candidate is not None:
                candidate.mentioned_concepts.append(target_node)

    def _rescore_sources_with_graph_support(
        self,
        query: str,
        candidates: list[SourceCandidate],
    ) -> list[SourceCandidate]:
        query_terms = self._query_terms(query)
        for candidate in candidates:
            concept_labels = [self._node_label(concept) for concept in candidate.mentioned_concepts]
            if any(self._has_term_overlap(query_terms, label) for label in concept_labels):
                candidate.score = self._clip_score(candidate.score + 0.05)
            if concept_labels and self._semantic_relation_count(candidate.mentioned_concepts) > 0:
                candidate.score = self._clip_score(candidate.score + 0.03)
            candidate.rank_score = candidate.score
        return candidates

    def _expand_semantic_relations(
        self,
        query: str,
        seed_sources: list[SourceCandidate],
    ) -> list[RelationCandidate]:
        if self.graph_store is None or not seed_sources:
            return []

        source_score_by_chunk = {source.node_id: source.score for source in seed_sources}
        allowed_source_chunks = set(source_score_by_chunk)
        concepts = self._ordered_unique(
            concept
            for source in seed_sources
            for concept in source.mentioned_concepts
        )
        if not concepts:
            return []

        triplets = self._get_relation_map(concepts)
        relation_candidates: list[RelationCandidate] = []
        for rank, (subject_node, relation, object_node) in enumerate(triplets, start=1):
            predicate = self._relation_label(relation)
            if predicate in DENIED_ANALYST_RELATIONS:
                continue

            evidence_chunk_ids = self._string_list(
                relation.properties.get("evidence_chunk_ids") or relation.properties.get("source_chunk_ids")
            )
            grounded_evidence_chunk_ids = [
                chunk_id for chunk_id in evidence_chunk_ids if chunk_id in allowed_source_chunks
            ]
            if self.settings.analyst_relation_require_source_evidence and not grounded_evidence_chunk_ids:
                continue

            evidence_spans = self._string_list(relation.properties.get("evidence_spans"))
            confidence = self._float_or_none(relation.properties.get("max_confidence") or relation.properties.get("confidence"))
            predicate_family = self._string_or_none(relation.properties.get("predicate_family"))
            relation_phrases = self._string_list(relation.properties.get("relation_phrases"))
            generality_score = self._float_or_none(relation.properties.get("max_generality_score"))
            retrieval_usefulness = self._float_or_none(relation.properties.get("max_retrieval_usefulness"))

            relation_text = " ".join(
                [
                    self._node_label(subject_node),
                    predicate,
                    self._node_label(object_node),
                    predicate_family or "",
                    " ".join(relation_phrases),
                    " ".join(evidence_spans),
                ]
            )
            evidence_score = max((source_score_by_chunk.get(chunk_id, 0.0) for chunk_id in grounded_evidence_chunk_ids), default=0.0)
            query_boost = 0.08 if self._has_term_overlap(self._query_terms(query), relation_text) else 0.0
            confidence_boost = (confidence or 0.0) * 0.04
            usefulness_boost = (retrieval_usefulness or 0.0) * 0.08
            generic_penalty = 0.04 if predicate in {"RELATED_TO"} or predicate_family == "other" else 0.0
            score = self._clip_score(evidence_score + query_boost + confidence_boost + usefulness_boost - generic_penalty)

            if score < MIN_RELEVANCE_SCORE:
                continue

            relation_candidates.append(
                RelationCandidate(
                    relation_id=f"{rank}:{relation_text}",
                    subject=self._node_label(subject_node),
                    predicate=predicate,
                    object=self._node_label(object_node),
                    score=score,
                    rank=rank,
                    rank_score=score,
                    confidence=confidence,
                    predicate_family=predicate_family,
                    relation_phrases=relation_phrases,
                    generality_score=generality_score,
                    retrieval_usefulness=retrieval_usefulness,
                    evidence_chunk_ids=grounded_evidence_chunk_ids or evidence_chunk_ids,
                    evidence_spans=evidence_spans,
                )
            )

        return relation_candidates

    def _rerank_relations(
        self,
        query: str,
        sources: list[SourceCandidate],
        relations: list[RelationCandidate],
    ) -> list[RelationCandidate]:
        if not relations:
            return []
        if self.reranker is None or not self.settings.analyst_relation_reranker_enabled:
            return self._sort_relations(relations)

        relation_by_id = {relation.relation_id: relation for relation in relations}
        ordered_relations = self._sort_relations(relations)
        rerank_limit = max(0, self.settings.analyst_relation_rerank_candidate_k)
        rerank_relations = ordered_relations[:rerank_limit]
        if not rerank_relations:
            return ordered_relations

        source_text_by_chunk = {
            source.node_id: self._source_excerpt(
                query,
                source.text,
                max_chars=max(120, self.settings.analyst_relation_rerank_max_chars // 2),
            )
            for source in sources
        }
        nodes = [
            NodeWithScore(
                node=TextNode(
                    id_=relation.relation_id,
                    text=self._relation_rerank_text(relation, source_text_by_chunk, query=query),
                ),
                score=relation.score,
            )
            for relation in rerank_relations
        ]

        try:
            reranked_nodes = self.reranker.postprocess_nodes(nodes, query_str=query)
        except Exception:
            self._log_reranker_failure()
            return sorted(relations, key=lambda item: (item.score, -item.rank), reverse=True)

        reranked_ids: set[str] = set()
        ordered: list[RelationCandidate] = []
        for reranked in reranked_nodes:
            relation_id = self._node_id(reranked.node)
            relation = relation_by_id.get(str(relation_id))
            if relation is None:
                continue
            relation.rank_score = self._raw_reranker_score(reranked.score)
            relation.reranked = True
            reranked_ids.add(relation.relation_id)
            ordered.append(relation)

        ordered = self._sort_relations(ordered)
        ordered.extend(
            self._sort_relations([relation for relation in ordered_relations if relation.relation_id not in reranked_ids])
        )
        return ordered

    def _select_final_sources(self, candidates: list[SourceCandidate]) -> list[SourceCandidate]:
        if not candidates:
            return []

        ordered = self._sort_sources(candidates)
        reranked_candidates = [candidate for candidate in ordered if candidate.reranked]
        if reranked_candidates:
            self._calibrate_display_scores(
                reranked_candidates,
                raw_margin=self.settings.analyst_source_min_raw_margin,
            )

        selected: list[SourceCandidate] = []
        path_counts: dict[str, int] = {}
        top_rank_score = self._ranking_score(reranked_candidates[0]) if reranked_candidates else 0.0
        bounded_reranker_scores = self._has_bounded_scores([self._ranking_score(candidate) for candidate in reranked_candidates])
        min_keep = min(
            max(0, self.settings.analyst_source_min_keep),
            max(0, self.settings.analyst_source_final_k),
        )
        max_per_path = max(1, self.settings.analyst_source_max_per_path)

        for candidate in ordered:
            if len(selected) >= self.settings.analyst_source_final_k:
                break

            path_key = self._source_path_key(candidate)
            if path_counts.get(path_key, 0) >= max_per_path:
                continue

            if reranked_candidates:
                if not candidate.reranked:
                    continue
                if (
                    candidate.reranked
                    and not self._passes_rerank_threshold(
                        candidate,
                        top_rank_score=top_rank_score,
                        min_relative_score=self.settings.analyst_source_min_relative_score,
                        raw_margin=self.settings.analyst_source_min_raw_margin,
                        bounded_reranker_scores=bounded_reranker_scores,
                    )
                    and (
                        len(selected) >= min_keep
                        or candidate.score < self.settings.analyst_source_min_relative_score
                    )
                ):
                    continue

            selected.append(candidate)
            path_counts[path_key] = path_counts.get(path_key, 0) + 1

        if len(selected) >= min_keep:
            return selected

        return self._fill_minimum_diverse_sources(
            selected=selected,
            ordered=ordered,
            min_keep=min_keep,
            max_per_path=max_per_path,
        )

    def _fill_minimum_diverse_sources(
        self,
        *,
        selected: list[SourceCandidate],
        ordered: list[SourceCandidate],
        min_keep: int,
        max_per_path: int,
    ) -> list[SourceCandidate]:
        if len(selected) >= min_keep:
            return selected

        selected_ids = {candidate.node_id for candidate in selected}
        top_score = max(
            [candidate.score for candidate in selected] or [ordered[0].score if ordered else 0.0]
        )
        min_fill_score = max(
            self.settings.analyst_source_fill_min_score,
            top_score * self.settings.analyst_source_fill_min_relative_score,
        )
        path_counts: dict[str, int] = {}
        for candidate in selected:
            path_key = self._source_path_key(candidate)
            path_counts[path_key] = path_counts.get(path_key, 0) + 1

        def add_candidates(mode: str) -> None:
            for candidate in ordered:
                if len(selected) >= min_keep or len(selected) >= self.settings.analyst_source_final_k:
                    break
                if candidate.node_id in selected_ids:
                    continue
                if candidate.score < min_fill_score:
                    continue

                path_key = self._source_path_key(candidate)
                path_count = path_counts.get(path_key, 0)
                if mode == "new_path" and path_count > 0:
                    continue
                if mode == "under_path_cap" and path_count >= max_per_path:
                    continue

                selected.append(candidate)
                selected_ids.add(candidate.node_id)
                path_counts[path_key] = path_count + 1

        add_candidates("new_path")
        add_candidates("under_path_cap")

        return selected

    def _select_final_relations(self, relations: list[RelationCandidate]) -> list[RelationCandidate]:
        if not relations:
            return []

        ordered = self._sort_relations(relations)
        reranked_relations = [relation for relation in ordered if relation.reranked]
        if not reranked_relations:
            return ordered[: self.settings.analyst_relation_final_k]

        self._calibrate_display_scores(
            reranked_relations,
            raw_margin=self.settings.analyst_relation_min_raw_margin,
        )
        top_rank_score = self._ranking_score(reranked_relations[0])
        bounded_reranker_scores = self._has_bounded_scores([self._ranking_score(relation) for relation in reranked_relations])

        selected: list[RelationCandidate] = []
        for relation in ordered:
            if len(selected) >= self.settings.analyst_relation_final_k:
                break
            if not relation.reranked:
                continue
            if not self._passes_rerank_threshold(
                relation,
                top_rank_score=top_rank_score,
                min_relative_score=self.settings.analyst_relation_min_relative_score,
                raw_margin=self.settings.analyst_relation_min_raw_margin,
                bounded_reranker_scores=bounded_reranker_scores,
            ):
                continue
            selected.append(relation)

        return selected

    def _passes_rerank_threshold(
        self,
        item: SourceCandidate | RelationCandidate,
        *,
        top_rank_score: float,
        min_relative_score: float,
        raw_margin: float,
        bounded_reranker_scores: bool,
    ) -> bool:
        rank_score = self._ranking_score(item)
        if rank_score == top_rank_score:
            return True
        if bounded_reranker_scores:
            return item.score >= min_relative_score or rank_score >= top_rank_score * min_relative_score
        return (top_rank_score - rank_score) <= raw_margin and item.score >= min_relative_score

    def _calibrate_display_scores(
        self,
        items: list[SourceCandidate] | list[RelationCandidate],
        *,
        raw_margin: float,
    ) -> None:
        if not items:
            return

        ordered = sorted(items, key=lambda item: (self._ranking_score(item), -item.rank), reverse=True)
        scores = [self._ranking_score(item) for item in ordered]
        top_score = scores[0]
        bottom_score = scores[-1]

        if len(ordered) > 1 and abs(top_score - bottom_score) < 1e-9:
            for idx, item in enumerate(ordered):
                item.score = self._clip_score(max(0.2, 1.0 - (idx * 0.05)))
            return

        if self._has_bounded_scores(scores):
            for item in ordered:
                item.score = self._clip_score(self._ranking_score(item))
            return

        display_window = max(raw_margin * 1.5, 1e-6)
        for item in ordered:
            distance_from_top = max(0.0, top_score - self._ranking_score(item))
            relative = max(0.0, 1.0 - (distance_from_top / display_window))
            item.score = self._clip_score(0.2 + (0.8 * relative))

    def _sort_sources(self, candidates: list[SourceCandidate]) -> list[SourceCandidate]:
        return sorted(
            candidates,
            key=lambda item: (item.reranked, self._ranking_score(item), -item.rank),
            reverse=True,
        )

    def _sort_relations(self, relations: list[RelationCandidate]) -> list[RelationCandidate]:
        return sorted(
            relations,
            key=lambda item: (item.reranked, self._ranking_score(item), -item.rank),
            reverse=True,
        )

    def _format_query_result(
        self,
        query: str,
        sources: list[SourceCandidate],
        relations: list[RelationCandidate],
        char_budget: int,
    ) -> str:
        if char_budget <= 0 or (not sources and not relations):
            return ""

        source_id_by_chunk = {source.node_id: f"S{idx}" for idx, source in enumerate(sources, start=1)}
        lines = [f"QUERY: {query}"]

        for source in sources:
            source_id = source_id_by_chunk[source.node_id]
            source_text = self._truncate(self._display_source_text(source), MAX_SOURCE_CHARS)
            source_label = f"; Source: {source.metadata.source}" if source.metadata.source else ""
            lines.append(
                f"[SOURCE] [{source_id}] (Score: {source.score:.2f}; Chunk: {source.node_id}{source_label}) "
                f"{self._clean(source_text)}"
            )
            path = self._display_metadata_path(source.metadata)
            if path:
                lines.append(f"[SOURCE PATH] [{source_id}] {self._clean(path)}")

        for idx, relation in enumerate(relations, start=1):
            evidence_ids = [
                source_id_by_chunk[chunk_id]
                for chunk_id in relation.evidence_chunk_ids
                if chunk_id in source_id_by_chunk
            ]
            evidence_label = ", ".join(evidence_ids)
            if evidence_label:
                evidence_display = f"Evidence: {evidence_label}"
            elif relation.evidence_spans:
                evidence_display = f"Evidence span: {self._clean('; '.join(relation.evidence_spans[:2]))}"
            else:
                evidence_display = "Evidence: unavailable"

            confidence_display = (
                f"; Confidence: {relation.confidence:.2f}" if relation.confidence is not None else ""
            )
            lines.append(
                f"[RELATION] [R{idx}] {self._clean(relation.subject)} -> {relation.predicate} -> "
                f"{self._clean(relation.object)} (Score: {relation.score:.2f}; {evidence_display}{confidence_display})"
            )

        return self._fit_lines(lines, char_budget)

    def _create_configured_reranker(self) -> Any | None:
        if self.settings.analyst_reranker_mode == "disabled":
            return None

        if self.settings.analyst_reranker_mode == "sentence_transformers":
            try:
                reranker = SentenceTransformerReranker(**self.reranker_settings.sentence_transformer_params())
            except Exception as exc:
                logger.warning(
                    f"Sentence-transformers Analyst reranker initialization failed; deterministic ranking will be used. Error: {exc}"
                )
                return None

            if not self._reranker_health_check(reranker):
                logger.warning("Sentence-transformers Analyst reranker health check failed; deterministic ranking will be used.")
                return None
            return reranker

        if self.settings.analyst_reranker_mode == "ollama_llm_rerank":
            try:
                from llama_index.core.postprocessor import LLMRerank
                from llama_index.llms.ollama import Ollama
            except Exception:
                logger.warning("LLMRerank/Ollama imports failed; Analyst reranker disabled.", exc_info=True)
                return None

            reranker_params = self.reranker_settings.ollama_llm_rerank_params()
            reranker_llm = Ollama(
                model=reranker_params["model"],
                base_url=reranker_params["base_url"],
                request_timeout=reranker_params["request_timeout"],
                temperature=0.0,
            )
            reranker = LLMRerank(
                llm=reranker_llm,
                choice_batch_size=reranker_params["choice_batch_size"],
                top_n=reranker_params["top_n"],
            )

            if not self._reranker_health_check(reranker):
                logger.warning("Analyst LLM reranker health check failed; deterministic ranking will be used.")
                return None
            return reranker

        logger.warning("Unknown Analyst reranker mode %r; deterministic ranking will be used.", self.settings.analyst_reranker_mode)
        return None

    def _reranker_health_check(self, reranker: Any) -> bool:
        nodes = [
            NodeWithScore(node=TextNode(text="Graph retrieval expands concepts into evidence-backed relations."), score=0.1),
            NodeWithScore(node=TextNode(text="A cooking recipe lists ingredients and oven temperature."), score=0.1),
        ]
        try:
            ranked = reranker.postprocess_nodes(nodes, query_str="graph retrieval")
        except Exception as exc:
            logger.warning(f"Analyst reranker health check raised an exception; deterministic ranking will be used. Error: {exc}")
            return False
        return bool(ranked) and "graph retrieval" in ranked[0].node.get_content().lower()

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

    def _get_docstore_node(self, node_id: str) -> Any | None:
        return getattr(self.storage_context.docstore, "docs", {}).get(node_id)

    def _is_usable_source(self, metadata: NormalizedMetadata, text: str) -> bool:
        if not text.strip():
            return False
        if metadata.raw.get("docstore_node_kind") not in {None, POSTPROCESSED_CHUNK_KIND}:
            return False
        if metadata.retrieval_enabled is not True:
            return False
        if metadata.quarantined is True:
            return False
        return True

    def _score_source(
        self,
        query: str,
        text: str,
        metadata: NormalizedMetadata,
        base_score: float,
    ) -> float:
        query_terms = self._query_terms(query)
        path_text = " ".join([*metadata.path, *metadata.heading_path]).lower()
        summary = str(metadata.raw.get("postprocess_chunk_summary") or "").lower()
        text_lower = text.lower()

        score = min(0.90, 0.15 + (base_score * 0.75))
        if self._has_exact_topic_match(query, path_text):
            score += 0.04
        if self._has_exact_topic_match(query, summary):
            score += 0.06
        if self._has_exact_topic_match(query, text_lower):
            score += 0.04
        if self._has_exact_topic_match(query, self._topic_match_text(metadata, summary)):
            score += self.settings.analyst_source_exact_topic_boost
        if "dataset" in query_terms and ("dataset" in path_text or "dataset" in summary or "dataset" in text_lower):
            score += 0.08
        if self._has_term_overlap(query_terms, path_text):
            score += 0.02
        if self._has_term_overlap(query_terms, summary):
            score += 0.03

        postprocess_action = str(metadata.raw.get("postprocess_action") or "").lower()
        if postprocess_action == "metadata_only":
            score -= 0.20
        if self._looks_like_broad_index_page(text, metadata, query_terms):
            score -= 0.15

        return self._clip_score(min(score, 0.99))

    def _topic_match_text(self, metadata: NormalizedMetadata, summary: str) -> str:
        return " ".join(
            [
                metadata.source,
                *metadata.path,
                *metadata.heading_path,
                summary,
            ]
        )

    def _looks_like_broad_index_page(
        self,
        text: str,
        metadata: NormalizedMetadata,
        query_terms: set[str],
    ) -> bool:
        text_lower = text.lower()
        summary = str(metadata.raw.get("postprocess_chunk_summary") or "").lower()
        broad_markers = ("course", "guide", "lectures", "what's inside", "seminars", "homeworks")
        if len(text) < 3000 or not any(marker in text_lower for marker in broad_markers):
            return False
        focused_metadata = " ".join([*metadata.path, *metadata.heading_path, summary])
        return not self._has_term_overlap(query_terms, focused_metadata)

    def _semantic_relation_count(self, concepts: list[Any]) -> int:
        return len(self._get_relation_map(concepts[:3], limit=5))

    def _get_relation_map(self, concepts: list[Any], limit: int | None = None) -> list[tuple[Any, Any, Any]]:
        if not concepts or self.graph_store is None:
            return []
        try:
            return list(
                self.graph_store.get_rel_map(
                    concepts,
                    depth=self.settings.analyst_graph_depth,
                    limit=limit or self.settings.analyst_graph_relation_limit,
                    ignore_rels=list(DENIED_ANALYST_RELATIONS),
                )
            )
        except Exception:
            ids = [getattr(concept, "id", None) for concept in concepts if getattr(concept, "id", None)]
            return [
                triplet
                for triplet in self._get_triplets(ids=ids)
                if self._relation_label(triplet[1]) not in DENIED_ANALYST_RELATIONS
            ][: limit or self.settings.analyst_graph_relation_limit]

    def _get_triplets(
        self,
        *,
        ids: list[str] | None = None,
        relation_names: list[str] | None = None,
    ) -> list[tuple[Any, Any, Any]]:
        if self.graph_store is None:
            return []
        try:
            return list(self.graph_store.get_triplets(ids=ids, relation_names=relation_names))
        except Exception:
            logger.warning("Analyst graph triplet lookup failed.", exc_info=True)
            return []

    def _dedupe_sources(self, candidates: list[SourceCandidate]) -> list[SourceCandidate]:
        seen_chunks: set[str] = set()
        deduped: list[SourceCandidate] = []
        for candidate in self._sort_sources(candidates):
            if candidate.node_id in seen_chunks:
                continue
            seen_chunks.add(candidate.node_id)
            deduped.append(candidate)
        return deduped

    def _dedupe_relations(self, relations: list[RelationCandidate]) -> list[RelationCandidate]:
        seen: set[tuple[str, str, str]] = set()
        deduped: list[RelationCandidate] = []
        for relation in relations:
            key = (relation.subject, relation.predicate, relation.object)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(relation)
        return deduped

    def _fit_lines(self, lines: list[str], char_budget: int) -> str:
        fitted: list[str] = []
        used = 0
        for line in lines:
            line_len = len(line) + (1 if fitted else 0)
            if used + line_len > char_budget:
                remaining = char_budget - used - (1 if fitted else 0)
                if line.startswith("[SOURCE]") and remaining > 120:
                    fitted.append(self._truncate_to_length(line, remaining))
                    used = char_budget
                break
            fitted.append(line)
            used += line_len
        if len(fitted) <= 1:
            return ""
        return "\n".join(fitted)

    def _rerank_text(self, candidate: SourceCandidate, query: str = "") -> str:
        parts = [
            self._display_metadata_path(candidate.metadata),
            str(candidate.metadata.raw.get("postprocess_chunk_summary") or ""),
            self._source_excerpt(query, candidate.text, max_chars=self.settings.analyst_source_rerank_max_chars),
        ]
        return self._truncate_to_length(
            "\n".join(part for part in parts if part),
            max(120, self.settings.analyst_source_rerank_max_chars),
        )

    def _relation_rerank_text(
        self,
        relation: RelationCandidate,
        source_text_by_chunk: dict[str, str],
        query: str = "",
    ) -> str:
        source_evidence = "\n".join(
            source_text_by_chunk[chunk_id]
            for chunk_id in relation.evidence_chunk_ids
            if chunk_id in source_text_by_chunk
        )
        parts = [
            f"{relation.subject} {relation.predicate} {relation.object}",
            f"Family: {relation.predicate_family}" if relation.predicate_family else "",
            "Relation phrase: " + "; ".join(relation.relation_phrases) if relation.relation_phrases else "",
            " ".join(relation.evidence_spans),
            source_evidence,
        ]
        text = "\n".join(part for part in parts if part)
        if query:
            return self._truncate_to_length(text, max(120, self.settings.analyst_relation_rerank_max_chars))
        return text

    def _source_excerpt(self, query: str, text: str, *, max_chars: int) -> str:
        text = " ".join(str(text or "").split())
        if not text or max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text

        query_lower = query.strip().lower()
        query_terms = self._query_terms(query)
        sentences = self._split_sentences(text)
        if not sentences:
            return self._truncate(text, max_chars)

        best_index = -1
        best_score = 0
        for idx, sentence in enumerate(sentences):
            lowered = sentence.lower()
            score = 0
            if query_lower and query_lower in lowered:
                score += 5
            score += sum(1 for term in query_terms if term in lowered)
            if score > best_score:
                best_score = score
                best_index = idx

        if best_index < 0:
            return self._truncate(text, max_chars)

        selected: list[str] = []
        used = 0
        for idx in self._sentence_window_order(best_index, len(sentences)):
            sentence = sentences[idx]
            sentence_len = len(sentence) + (1 if selected else 0)
            if selected and used + sentence_len > max_chars:
                continue
            if not selected and sentence_len > max_chars:
                return self._truncate(sentence, max_chars)
            selected.append(sentence)
            used += sentence_len
            if used >= max_chars:
                break

        excerpt = " ".join(selected)
        return self._truncate(excerpt or text, max_chars)

    def _display_source_text(self, source: SourceCandidate) -> str:
        text = source.text.strip()
        stripped = self._strip_redundant_source_prefix(text, source.metadata)
        return stripped or text

    def _strip_redundant_source_prefix(self, text: str, metadata: NormalizedMetadata) -> str:
        for prefix in self._metadata_prefix_candidates(metadata):
            if not self._prefix_matches(text, prefix):
                continue

            remainder = text[len(prefix):].lstrip()
            if remainder.startswith(">"):
                remainder = remainder[1:].lstrip()
            if remainder:
                return remainder

        return text

    def _metadata_prefix_candidates(self, metadata: NormalizedMetadata) -> list[str]:
        candidates: list[str] = []
        for path in (metadata.path, metadata.heading_path):
            cleaned_parts = [self._clean_path_part(part) for part in path if self._clean_path_part(part)]
            for idx in range(len(cleaned_parts), 0, -1):
                candidates.append(" > ".join(cleaned_parts[:idx]))

        if metadata.source:
            candidates.append(metadata.source)

        return sorted(self._ordered_unique(candidates), key=len, reverse=True)

    def _metadata_path(self, metadata: NormalizedMetadata) -> str:
        path = metadata.path or metadata.heading_path
        return " > ".join(path)

    def _display_metadata_path(self, metadata: NormalizedMetadata) -> str:
        path = metadata.path or metadata.heading_path
        cleaned_parts = []
        for part in path:
            cleaned = self._clean_path_part(part)
            if not cleaned or cleaned.startswith("external:"):
                continue
            cleaned_parts.append(cleaned)
        return " > ".join(cleaned_parts)

    def _source_path_key(self, candidate: SourceCandidate) -> str:
        return self._display_metadata_path(candidate.metadata) or self._metadata_path(candidate.metadata) or candidate.node_id

    def _node_text(self, node: Any) -> str:
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

    def _node_id(self, node: Any) -> str | None:
        for attr in ("node_id", "id_", "id"):
            value = getattr(node, attr, None)
            if value is not None:
                return str(value)
        return None

    def _node_label(self, node: Any) -> str:
        properties = getattr(node, "properties", {}) or {}
        for key in ("entity_name", "display_name", "text", "name"):
            value = properties.get(key)
            if value:
                return str(value)
        name = getattr(node, "name", None)
        if name and not str(name).startswith("concept_"):
            return str(name)
        node_id = getattr(node, "id", None)
        return str(node_id) if node_id is not None else ""

    def _relation_label(self, relation: Any) -> str:
        for attr in ("label", "id"):
            value = getattr(relation, attr, None)
            if value:
                return str(value)
        return ""

    def _log_reranker_failure(self) -> None:
        if self._reranker_failure_logged:
            return
        self._reranker_failure_logged = True
        logger.warning("Analyst reranker failed; deterministic ranking will be used.")

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        return {term for term in query.lower().replace("/", " ").split() if len(term) > 2}

    @staticmethod
    def _has_term_overlap(query_terms: set[str], text: str) -> bool:
        if not query_terms or not text:
            return False
        lowered = text.lower()
        return any(AnalystRetrievalPipeline._term_matches_text(term, lowered) for term in query_terms)

    @staticmethod
    def _term_matches_text(term: str, text: str) -> bool:
        if len(term) <= 4:
            pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
            return re.search(pattern, text) is not None
        return term in text

    @staticmethod
    def _has_exact_topic_match(query: str, text: str) -> bool:
        normalized_query = " ".join(str(query or "").lower().split())
        normalized_text = " ".join(str(text or "").lower().split())
        if len(normalized_query) <= 2 or not normalized_text:
            return False
        pattern = rf"(?<![a-z0-9]){re.escape(normalized_query)}(?![a-z0-9])"
        return re.search(pattern, normalized_text) is not None

    @staticmethod
    def _clean_path_part(part: str) -> str:
        stripped = str(part).strip()
        while stripped.startswith("#"):
            stripped = stripped[1:].lstrip()
        return " ".join(stripped.split())

    @staticmethod
    def _prefix_matches(text: str, prefix: str) -> bool:
        if not prefix or not text.startswith(prefix):
            return False
        if len(text) == len(prefix):
            return True
        next_char = text[len(prefix)]
        return next_char.isspace() or next_char in {">", ":", "-", ".", ","}

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
    def _float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clip_score(score: float | int | None) -> float:
        if score is None:
            return 0.0
        return max(0.0, min(1.0, float(score)))

    @staticmethod
    def _raw_reranker_score(score: float | int | None) -> float:
        if score is None:
            return 0.0
        return float(score)

    @staticmethod
    def _ranking_score(item: SourceCandidate | RelationCandidate) -> float:
        return float(item.rank_score if item.rank_score is not None else item.score)

    @staticmethod
    def _has_bounded_scores(scores: list[float]) -> bool:
        return bool(scores) and all(0.0 <= score <= 1.0 for score in scores)

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        cut = text[:limit]
        for separator in (". ", "! ", "? ", "\n"):
            index = cut.rfind(separator)
            if index >= int(limit * 0.6):
                return cut[: index + len(separator)].rstrip() + " ...[truncated]"
        return cut.rstrip() + " ...[truncated]"

    @staticmethod
    def _truncate_to_length(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        suffix = " ...[truncated]"
        if limit <= len(suffix):
            return text[:limit]
        return text[: limit - len(suffix)].rstrip() + suffix

    @staticmethod
    def _clean(text: str) -> str:
        return text.encode("utf-8", "ignore").decode("utf-8")

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
            if sentence.strip()
        ]

    @staticmethod
    def _sentence_window_order(center: int, length: int) -> list[int]:
        ordered = [center]
        distance = 1
        while len(ordered) < length:
            left = center - distance
            right = center + distance
            if left >= 0:
                ordered.append(left)
            if right < length:
                ordered.append(right)
            distance += 1
        return ordered

    @staticmethod
    def _ordered_unique(values: Any) -> list[Any]:
        seen: set[Any] = set()
        result: list[Any] = []
        for value in values:
            key = getattr(value, "id", value)
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
        return result
