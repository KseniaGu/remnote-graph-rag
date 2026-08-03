#!/usr/bin/env python
"""Build final LlamaIndex storage from optimized IR and LLM postprocess sidecars.

This script is the experimental optimized pipeline's final materialization step.
It does not call the legacy KnowledgeGraphIndexer.process_implicit_graph path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.configs.models import ModelSettings
from backend.configs.paths import PathSettings
from backend.configs.storage import LocalStorageSettings, StorageSettings
from backend.data_processing.embedding_passages import (
    build_embedding_passages,
    tokenizer_token_counter,
)
from backend.data_processing.llm_postprocess import (
    ChunkAction,
    ChunkEnrichmentDecision,
    ConceptGraphProjection,
    PostprocessFailure,
    build_graph_projection,
    load_concept_resolution_sidecars,
    load_postprocess_decisions,
    load_postprocess_failures,
    read_jsonl,
    sanitize_markup_for_embedding,
    write_json,
)
from backend.data_processing.parser_optimized import (
    ArtifactGateDecision,
    ExternalResource,
    OptimizedParseResult,
    ParsedArtifact,
    RemNoteBlock,
    RemNoteParserOptimized,
    RetrievalChunk,
    SourceDocument,
)

RETRIEVAL_ACTIONS = {ChunkAction.KEEP, ChunkAction.KEEP_WITH_CLEANED_TEXT}
GRAPH_ACTIONS = {
    ChunkAction.KEEP,
    ChunkAction.KEEP_WITH_CLEANED_TEXT,
    ChunkAction.METADATA_ONLY,
    ChunkAction.GRAPH_ONLY,
}
QUARANTINE_ACTIONS = {
    ChunkAction.NEEDS_VISUAL_REPARSE,
    ChunkAction.EXCLUDE_FROM_EMBEDDING,
}
RAW_TEXT_VECTOR_METADATA_DROP_KEYS = {
    "postprocess_original_embedding_text",
    "original_text",
    "display_text",
    "context_text",
}
MANIFEST_FILENAME = "postprocessed_graph_storage_manifest.json"
POSTPROCESSED_PASSAGE_KIND = "postprocessed_embedding_passage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final storage from optimized IR and LLM postprocess sidecars."
    )
    parser.add_argument("--optimized-ir-dir", required=True, type=Path)
    parser.add_argument("--postprocess-dir", required=True, type=Path)
    parser.add_argument("--final-storage-dir", required=True, type=Path)
    parser.add_argument("--raw-data-dir", type=Path, default=None)
    parser.add_argument("--parsed-pdfs-dir", type=Path, default=None)
    parser.add_argument("--parsed-images-dir", type=Path, default=None)
    parser.add_argument("--parsed-texts-dir", type=Path, default=None)
    parser.add_argument("--embedder-model-path", default=None)
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Build docstore/property graph only.",
    )
    parser.add_argument("--force-rebuild-final", action="store_true")
    return parser.parse_args()


def load_optimized_result(ir_dir: Path) -> OptimizedParseResult:
    summary_path = Path(ir_dir) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing optimized parser summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return OptimizedParseResult(
        source_documents=[
            SourceDocument(**row)
            for row in read_jsonl(Path(ir_dir) / "source_documents.jsonl")
        ],
        blocks=[
            RemNoteBlock(**row) for row in read_jsonl(Path(ir_dir) / "blocks.jsonl")
        ],
        external_resources=[
            ExternalResource(**row)
            for row in read_jsonl(Path(ir_dir) / "external_resources.jsonl")
        ],
        parsed_artifacts=[
            ParsedArtifact(**row)
            for row in read_jsonl(Path(ir_dir) / "parsed_artifacts.jsonl")
        ],
        artifact_gate_decisions=[
            ArtifactGateDecision(**row)
            for row in read_jsonl(Path(ir_dir) / "artifact_gate_decisions.jsonl")
        ],
        retrieval_chunks=[
            RetrievalChunk(**row)
            for row in read_jsonl(Path(ir_dir) / "retrieval_chunks.jsonl")
        ],
        summary=summary,
    )


def make_local_settings(
    final_storage_dir: Path, args: argparse.Namespace
) -> tuple[PathSettings, StorageSettings]:
    defaults = PathSettings()
    path_settings = PathSettings(
        raw_data_dir=args.raw_data_dir or defaults.raw_data_dir,
        parsed_pdfs_dir=args.parsed_pdfs_dir or defaults.parsed_pdfs_dir,
        parsed_images_dir=args.parsed_images_dir or defaults.parsed_images_dir,
        parsed_texts_dir=args.parsed_texts_dir or defaults.parsed_texts_dir,
        local_storage_dir=final_storage_dir,
    )
    local = LocalStorageSettings(storage_path=final_storage_dir)
    return path_settings, StorageSettings(
        document_storage=local,
        index_storage=local,
        vector_storage=local,
        property_graph_storage=local,
    )


def decision_failures_by_chunk(
    failures: Iterable[PostprocessFailure],
) -> dict[str, list[PostprocessFailure]]:
    result: dict[str, list[PostprocessFailure]] = {}
    for failure in failures:
        if failure.chunk_id:
            result.setdefault(failure.chunk_id, []).append(failure)
    return result


def materialized_chunk_text(
    base_text: str, decision: ChunkEnrichmentDecision | None
) -> str:
    if (
        decision
        and decision.action == ChunkAction.KEEP_WITH_CLEANED_TEXT
        and decision.cleaned_embedding_text
    ):
        return decision.cleaned_embedding_text
    return base_text


def materialize_final_text_nodes(
    result: OptimizedParseResult,
    decisions: list[ChunkEnrichmentDecision],
    failures: list[PostprocessFailure],
    path_settings: PathSettings,
    storage_settings: StorageSettings,
) -> tuple[list[Any], dict[str, Any]]:
    parser = RemNoteParserOptimized(
        path_settings,
        storage_settings,
        prepare_external_artifacts=False,
        force_rebuild=True,
        write_ir=False,
    )
    base_nodes = parser.to_text_nodes(result)
    decisions_by_chunk = {decision.chunk_id: decision for decision in decisions}
    failures_by_chunk = decision_failures_by_chunk(failures)
    node_by_id = {node.id_: node for node in base_nodes}

    manifest = {
        "chunk_count": len(base_nodes),
        "retrieval_enabled_count": 0,
        "graph_enabled_count": 0,
        "quarantined_count": 0,
        "missing_decision_count": 0,
        "failed_decision_chunk_count": len(failures_by_chunk),
        "action_counts": {},
        "quarantined_chunk_ids": [],
        "markup_sanitized_count": 0,
        "markup_removed_tag_count": 0,
        "markup_removed_image_count": 0,
        "markup_preserved_alt_text_count": 0,
    }

    for node in base_nodes:
        decision = decisions_by_chunk.get(node.id_)
        chunk_failures = failures_by_chunk.get(node.id_, [])
        missing_decision = decision is None
        action = decision.action if decision else None
        should_quarantine = (
            missing_decision or bool(chunk_failures) or action in QUARANTINE_ACTIONS
        )
        retrieval_enabled = bool(
            decision and action in RETRIEVAL_ACTIONS and not should_quarantine
        )
        graph_enabled = bool(
            decision and action in GRAPH_ACTIONS and not should_quarantine
        )
        original_text = node.text
        materialized_text = materialized_chunk_text(original_text, decision)
        sanitization = sanitize_markup_for_embedding(materialized_text)
        node.text = sanitization.text
        if decision and decision.cleaned_display_text:
            node.metadata["display_text"] = sanitize_markup_for_embedding(
                decision.cleaned_display_text
            ).text
        elif sanitization.changed:
            node.metadata["display_text"] = sanitization.text

        node.metadata.update(
            {
                "docstore_node_kind": "postprocessed_retrieval_chunk",
                "postprocess_decision_id": decision.decision_id if decision else None,
                "postprocess_action": action.value if action else "missing_decision",
                "postprocess_prompt_version": decision.prompt_version
                if decision
                else None,
                "postprocess_model_name": decision.model_name if decision else None,
                "postprocess_input_hash": decision.input_hash if decision else None,
                "postprocess_issue_types": list(decision.issue_types)
                if decision
                else [],
                "postprocess_warnings": list(decision.warnings) if decision else [],
                "postprocess_reason": decision.reason if decision else None,
                "postprocess_chunk_summary": decision.chunk_summary
                if decision
                else None,
                "postprocess_original_embedding_text": original_text,
                "markup_sanitized": sanitization.changed,
                "markup_removed_tag_count": sanitization.removed_tag_count,
                "markup_removed_image_count": sanitization.removed_image_count,
                "markup_preserved_alt_texts": sanitization.preserved_alt_texts,
                "retrieval_enabled": retrieval_enabled,
                "graph_enabled": graph_enabled,
                "quarantined": should_quarantine,
                "quarantine_reasons": quarantine_reasons(
                    decision, chunk_failures, missing_decision
                ),
            }
        )
        node.excluded_embed_metadata_keys = list(node.metadata.keys())

        if sanitization.changed:
            manifest["markup_sanitized_count"] += 1
        manifest["markup_removed_tag_count"] += sanitization.removed_tag_count
        manifest["markup_removed_image_count"] += sanitization.removed_image_count
        manifest["markup_preserved_alt_text_count"] += len(
            sanitization.preserved_alt_texts
        )
        if action:
            manifest["action_counts"][action.value] = (
                manifest["action_counts"].get(action.value, 0) + 1
            )
        if missing_decision:
            manifest["missing_decision_count"] += 1
        if retrieval_enabled:
            manifest["retrieval_enabled_count"] += 1
        if graph_enabled:
            manifest["graph_enabled_count"] += 1
        if should_quarantine:
            manifest["quarantined_count"] += 1
            manifest["quarantined_chunk_ids"].append(node.id_)

    # Remove graph evidence for chunks that are not graph-enabled.
    for node_id, node in node_by_id.items():
        if not node.metadata.get("graph_enabled"):
            node.metadata["postprocess_graph_import_excluded"] = True
    return base_nodes, manifest


def quarantine_reasons(
    decision: ChunkEnrichmentDecision | None,
    failures: list[PostprocessFailure],
    missing_decision: bool,
) -> list[str]:
    reasons: list[str] = []
    if missing_decision:
        reasons.append("missing_decision")
    reasons.extend(failure.error_type for failure in failures)
    if decision and decision.action in QUARANTINE_ACTIONS:
        reasons.append(decision.action.value)
    return sorted(set(reasons))


def filtered_graph_projection(
    decisions: list[ChunkEnrichmentDecision],
    concept_resolution: Any,
    graph_enabled_chunk_ids: set[str],
) -> ConceptGraphProjection:
    graph_decisions = [
        decision
        for decision in decisions
        if decision.chunk_id in graph_enabled_chunk_ids
    ]
    return build_graph_projection(
        graph_decisions, concept_resolution=concept_resolution
    )


def make_vector_store_metadata(
    metadata: dict[str, Any], drop_keys: set[str] | None = None
) -> dict[str, Any]:
    """Return flat JSON-safe metadata for vector-store records.

    Property-graph import stores LlamaIndex EntityNode/Relation objects in chunk
    metadata. Those objects are useful in the docstore/property graph, but vector
    stores only need filterable scalar metadata and cannot serialize them.
    """

    skipped = drop_keys or set()
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in skipped:
            continue
        if value is None or isinstance(value, (str, int, float, bool, bytes)):
            safe[key] = value
            continue
        safe[key] = json.dumps(value, ensure_ascii=False, default=str)
    return safe


def copy_node_for_vector_store(
    node: Any, text_node_cls: Any, drop_metadata_keys: set[str]
) -> Any:
    vector_node = text_node_cls(
        text=getattr(node, "text", node.get_content()),
        metadata=make_vector_store_metadata(
            getattr(node, "metadata", {}), drop_metadata_keys
        ),
        excluded_embed_metadata_keys=list(
            getattr(node, "excluded_embed_metadata_keys", [])
        ),
        excluded_llm_metadata_keys=list(
            getattr(node, "excluded_llm_metadata_keys", [])
        ),
    )
    vector_node.id_ = node.id_
    return vector_node


def _node_source_path(node: Any) -> str:
    metadata = getattr(node, "metadata", {}) or {}
    path = metadata.get("path") or metadata.get("heading_path") or []
    if isinstance(path, str):
        return path
    if isinstance(path, (list, tuple)):
        return " > ".join(str(part) for part in path if part)
    return ""


def _resolve_embedder_tokenizer(embedder: Any) -> Any | None:
    for owner in (
        embedder,
        getattr(embedder, "_model", None),
        getattr(embedder, "model", None),
    ):
        if owner is None:
            continue
        tokenizer = getattr(owner, "tokenizer", None) or getattr(
            owner, "_tokenizer", None
        )
        if tokenizer is not None:
            return tokenizer
    return None


def _node_text(node: Any) -> str:
    text = getattr(node, "text", None)
    if text is not None:
        return str(text)
    get_content = getattr(node, "get_content", None)
    return str(get_content() if callable(get_content) else "")


def make_embedding_passage_nodes(
    nodes: list[Any], text_node_cls: Any, embedder: Any
) -> list[Any]:
    """Creates vector-search child passages while preserving parent chunk IDs.

    The parent retrieval chunk stays in the docstore/property graph as the
    evidence unit. Passage nodes are internal vector records that map back to
    the parent through parent_chunk_id and chunk_id metadata.
    """

    token_counter = tokenizer_token_counter(_resolve_embedder_tokenizer(embedder))
    passage_nodes: list[Any] = []
    for node in nodes:
        if not node.metadata.get("retrieval_enabled"):
            continue
        summary = str(node.metadata.get("postprocess_chunk_summary") or "")
        passages = build_embedding_passages(
            parent_chunk_id=node.id_,
            text=_node_text(node),
            source_path=_node_source_path(node),
            summary=summary,
            token_counter=token_counter,
        )
        for passage in passages:
            metadata = dict(getattr(node, "metadata", {}) or {})
            metadata.pop("kg_nodes", None)
            metadata.pop("kg_relations", None)
            metadata.update(
                {
                    "docstore_node_kind": POSTPROCESSED_PASSAGE_KIND,
                    "vector_source": passage.passage_id,
                    "parent_chunk_id": node.id_,
                    "chunk_id": node.id_,
                    "passage_id": passage.passage_id,
                    "passage_index": passage.passage_index,
                    "passage_char_start": passage.char_start,
                    "passage_char_end": passage.char_end,
                    "passage_token_count": passage.token_count,
                    "passage_split_strategy": passage.split_strategy,
                    "retrieval_enabled": True,
                    "graph_enabled": False,
                    "quarantined": False,
                }
            )
            passage_node = text_node_cls(
                text=passage.text,
                metadata=metadata,
                excluded_embed_metadata_keys=list(metadata.keys()),
                excluded_llm_metadata_keys=list(
                    getattr(node, "excluded_llm_metadata_keys", [])
                ),
            )
            passage_node.id_ = passage.passage_id
            passage_nodes.append(passage_node)
    return passage_nodes


def import_projection_to_property_graph(
    storage_context: Any, nodes: list[Any], projection: ConceptGraphProjection
) -> dict[str, int]:
    from llama_index.core.graph_stores.types import (
        KG_NODES_KEY,
        KG_RELATIONS_KEY,
        EntityNode,
        Relation,
    )

    llama_nodes_by_id = {node.id_: node for node in nodes}
    entity_nodes: dict[str, EntityNode] = {}
    for record in projection.nodes:
        concept_id = record["id"]
        entity_nodes[concept_id] = EntityNode(
            name=concept_id,
            label=record.get("type") or "CONCEPT",
            properties={
                "entity_name": record.get("canonical_name") or concept_id,
                "display_name": record.get("display_name")
                or record.get("canonical_name")
                or concept_id,
                "aliases": record.get("aliases", []),
                "source_chunk_ids": record.get("source_chunk_ids", []),
                "evidence_spans": record.get("evidence_spans", []),
                "mention_ids": record.get("mention_ids", []),
                "postprocess_concept_id": concept_id,
                "postprocess_resolution_source": record.get("resolution_source"),
                "postprocess_merge_status": record.get("merge_status"),
                "postprocess_max_salience": record.get("max_salience"),
            },
        )

    relations = []
    evidence_relation_records: dict[tuple[str, str], dict[str, Any]] = {}

    def extend_unique(values: list[Any], additions: list[Any]) -> list[Any]:
        for item in additions:
            if item not in values:
                values.append(item)
        return values

    for record in projection.edges:
        relation = Relation(
            label=record["canonical_predicate"],
            source_id=record["source_concept_id"],
            target_id=record["target_concept_id"],
            properties={
                "postprocess_relation_id": record["id"],
                "raw_predicates": record.get("raw_predicates", []),
                "predicate_statuses": record.get("predicate_statuses", []),
                "predicate_family": record.get("predicate_family"),
                "predicate_definitions": record.get("predicate_definitions", []),
                "relation_phrases": record.get("relation_phrases", []),
                "evidence_chunk_ids": record.get("evidence_chunk_ids", []),
                "evidence_spans": record.get("evidence_spans", []),
                "decision_ids": record.get("decision_ids", []),
                "max_confidence": record.get("max_confidence"),
                "max_generality_score": record.get("max_generality_score"),
                "max_retrieval_usefulness": record.get("max_retrieval_usefulness"),
                "max_visualization_usefulness": record.get(
                    "max_visualization_usefulness"
                ),
            },
        )
        relations.append(relation)
        for chunk_id in record.get("evidence_chunk_ids", []):
            chunk_node = llama_nodes_by_id.get(chunk_id)
            if chunk_node is None:
                continue
            existing = chunk_node.metadata.get(KG_RELATIONS_KEY, [])
            if relation not in existing:
                existing.append(relation)
            chunk_node.metadata[KG_RELATIONS_KEY] = existing

    for link in projection.evidence_links:
        chunk_id = link["chunk_id"]
        concept_id = link["concept_id"]
        chunk_node = llama_nodes_by_id.get(chunk_id)
        concept_node = entity_nodes.get(concept_id)
        if chunk_node is None or concept_node is None:
            continue
        existing = chunk_node.metadata.get(KG_NODES_KEY, [])
        if concept_node not in existing:
            existing.append(concept_node)
        chunk_node.metadata[KG_NODES_KEY] = existing

        evidence_key = (chunk_id, concept_id)
        evidence_record = evidence_relation_records.setdefault(
            evidence_key,
            {
                "source_id": chunk_id,
                "target_id": concept_id,
                "evidence_spans": [],
                "decision_ids": [],
            },
        )
        extend_unique(
            evidence_record["evidence_spans"], list(link.get("evidence_spans", []))
        )
        extend_unique(
            evidence_record["decision_ids"],
            [link["decision_id"]] if link.get("decision_id") else [],
        )

    evidence_relations = [
        Relation(
            label="MENTIONS",
            source_id=record["source_id"],
            target_id=record["target_id"],
            properties={
                "postprocess_relation_type": "evidence_link",
                "evidence_chunk_ids": [record["source_id"]],
                "evidence_spans": record["evidence_spans"],
                "decision_ids": record["decision_ids"],
            },
        )
        for record in evidence_relation_records.values()
    ]

    graph_store = storage_context.property_graph_store
    if entity_nodes:
        graph_store.upsert_nodes(list(entity_nodes.values()))
    if relations:
        graph_store.upsert_relations(relations)
    if evidence_relations:
        graph_store.upsert_relations(evidence_relations)
    updated_llama_nodes = [
        node
        for node in nodes
        if node.metadata.get(KG_NODES_KEY) or node.metadata.get(KG_RELATIONS_KEY)
    ]
    if updated_llama_nodes:
        graph_store.upsert_llama_nodes(updated_llama_nodes)
    return {
        "concept_nodes_imported": len(entity_nodes),
        "semantic_relations_imported": len(relations),
        "evidence_link_count": len(projection.evidence_links),
        "evidence_relations_imported": len(evidence_relations),
    }


def build_index(storage_context: Any, embedder: Any) -> Any:
    from llama_index.core.indices import PropertyGraphIndex
    from llama_index.core.indices.property_graph import ImplicitPathExtractor

    # LlamaIndex treats an empty kg_extractors list as falsy and falls back to
    # SimpleLLMPathExtractor(Settings.llm), which defaults to OpenAI. Build an
    # empty index with a non-LLM extractor so no extraction runs and no LLM is resolved.
    return PropertyGraphIndex.from_existing(
        property_graph_store=storage_context.property_graph_store,
        vector_store=storage_context.vector_store,
        storage_context=storage_context,
        embed_model=embedder,
        kg_extractors=[ImplicitPathExtractor()],
        use_async=False,
        show_progress=True,
        embed_kg_nodes=False,
    )


def embed_final_nodes(
    storage_context: Any,
    final_nodes: list[Any],
    projection: ConceptGraphProjection,
    embedder: Any,
) -> dict[str, int]:
    from llama_index.core.graph_stores.types import KG_NODES_KEY, KG_RELATIONS_KEY
    from llama_index.core.schema import TextNode

    vector_store = storage_context.vector_store
    vector_store.stores_text = True
    nodes_to_embed = [
        node for node in final_nodes if node.metadata.get("retrieval_enabled")
    ]
    passage_nodes = make_embedding_passage_nodes(nodes_to_embed, TextNode, embedder)
    if passage_nodes:
        storage_context.docstore.add_documents(passage_nodes, allow_update=True)

    entity_text_nodes: list[Any] = []
    for record in projection.nodes:
        text = record.get("canonical_name") or record["id"]
        entity_node = TextNode(
            text=text,
            metadata={
                "vector_source": record["id"],
                "docstore_node_kind": "postprocessed_concept_node",
                "concept_id": record["id"],
                "concept_type": record.get("type"),
                "aliases": record.get("aliases", []),
                "source_chunk_ids": record.get("source_chunk_ids", []),
            },
        )
        entity_node.id_ = record["id"]
        entity_text_nodes.append(entity_node)

    source_nodes = [*passage_nodes, *entity_text_nodes]
    if not source_nodes:
        return {
            "embedded_retrieval_chunks": len(nodes_to_embed),
            "embedded_retrieval_passages": 0,
            "embedded_concept_nodes": 0,
        }
    texts = [node.get_content() for node in source_nodes]
    vector_metadata_drop_keys = {
        KG_NODES_KEY,
        KG_RELATIONS_KEY,
        *RAW_TEXT_VECTOR_METADATA_DROP_KEYS,
    }
    vector_nodes = [
        copy_node_for_vector_store(node, TextNode, vector_metadata_drop_keys)
        for node in source_nodes
    ]
    embeddings = embedder.get_text_embedding_batch(texts, show_progress=True)
    for node, embedding in zip(vector_nodes, embeddings):
        node.embedding = [*embedding]
    vector_store.add(vector_nodes)
    return {
        "embedded_retrieval_chunks": len(nodes_to_embed),
        "embedded_retrieval_passages": len(passage_nodes),
        "embedded_concept_nodes": len(entity_text_nodes),
    }


def build_embedder(args: argparse.Namespace) -> Any:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    settings = ModelSettings()
    model_path = args.embedder_model_path or settings.embedder.model_path
    return HuggingFaceEmbedding(
        model_path,
        trust_remote_code=True,
        device=settings.embedder.device,
        embed_batch_size=5,
        local_files_only=bool(args.embedder_model_path),
    )


def main() -> int:
    args = parse_args()
    final_storage_dir = args.final_storage_dir.expanduser().resolve()
    if final_storage_dir.exists() and any(final_storage_dir.iterdir()):
        if not args.force_rebuild_final:
            raise SystemExit(
                f"Final storage dir is not empty; pass --force-rebuild-final: {final_storage_dir}"
            )
        shutil.rmtree(final_storage_dir)
    final_storage_dir.mkdir(parents=True, exist_ok=True)

    result = load_optimized_result(args.optimized_ir_dir)
    decisions = load_postprocess_decisions(args.postprocess_dir)
    failures = load_postprocess_failures(args.postprocess_dir)
    concept_resolution = load_concept_resolution_sidecars(args.postprocess_dir)
    path_settings, storage_settings = make_local_settings(final_storage_dir, args)

    final_nodes, materialization = materialize_final_text_nodes(
        result,
        decisions,
        failures,
        path_settings,
        storage_settings,
    )
    graph_enabled_chunk_ids = {
        node.id_ for node in final_nodes if node.metadata.get("graph_enabled")
    }
    projection = filtered_graph_projection(
        decisions, concept_resolution, graph_enabled_chunk_ids
    )

    from backend.knowledge_graph.storage import KnowledgeGraphStorage

    kg_storage = KnowledgeGraphStorage(
        path_settings, storage_settings, local_storage=final_storage_dir
    )
    embedder = None if args.skip_embeddings else build_embedder(args)
    kg_storage.storage_context.docstore.add_documents(final_nodes, allow_update=True)
    kg_storage.storage_context.property_graph_store.upsert_llama_nodes(final_nodes)
    if embedder is None:
        index_summary = {
            "property_graph_index_built": False,
            "embeddings_skipped": True,
            "implicit_path_extraction_enabled": False,
        }
    else:
        build_index(kg_storage.storage_context, embedder)
        index_summary = {
            "property_graph_index_built": True,
            "embeddings_skipped": False,
            "implicit_path_extraction_enabled": False,
        }
    graph_summary = import_projection_to_property_graph(
        kg_storage.storage_context, final_nodes, projection
    )
    embedding_summary = {"embedded_retrieval_chunks": 0, "embedded_concept_nodes": 0}
    if embedder is not None:
        embedding_summary = embed_final_nodes(
            kg_storage.storage_context, final_nodes, projection, embedder
        )

    kg_storage.storage_context.persist(persist_dir=str(final_storage_dir))
    manifest = {
        "schema_version": "1.0",
        "optimized_ir_dir": str(args.optimized_ir_dir),
        "postprocess_dir": str(args.postprocess_dir),
        "final_storage_dir": str(final_storage_dir),
        "source_retrieval_chunk_count": len(result.retrieval_chunks),
        "decision_count": len(decisions),
        "failure_record_count": len(failures),
        **materialization,
        **graph_summary,
        **embedding_summary,
        **index_summary,
    }
    write_json(final_storage_dir / MANIFEST_FILENAME, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
