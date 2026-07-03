import json
import tempfile
import unittest
from pathlib import Path

try:
    from llama_index.core.schema import TextNode  # noqa: F401
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False

from backend.configs.paths import PathSettings
from backend.configs.storage import LocalStorageSettings, StorageSettings
from backend.data_processing.llm_postprocess import ChunkEnrichmentDecision
from backend.data_processing.llm_postprocess import ConceptGraphProjection
from backend.data_processing.parser_optimized import (
    OptimizedParseResult,
    RemNoteBlock,
    RetrievalChunk,
    SourceDocument,
)
from scripts.build_postprocessed_graph_storage import (
    import_projection_to_property_graph,
    make_embedding_passage_nodes,
    make_vector_store_metadata,
    materialize_final_text_nodes,
)


def make_decision(chunk_id: str, action: str, *, cleaned: str | None = None) -> ChunkEnrichmentDecision:
    return ChunkEnrichmentDecision(
        chunk_id=chunk_id,
        action=action,
        issue_types=[],
        educational_usefulness=0.8,
        confidence=0.9,
        warnings=[],
        cleaned_embedding_text=cleaned,
        cleaned_display_text=None,
        chunk_summary="summary",
        concepts=[],
        relations=[],
        reason="test",
        decision_id=f"decision_{chunk_id}",
        schema_version="1.0",
        prompt_version="v3",
        model_name="fake",
        input_hash=f"hash_{chunk_id}",
    )


def make_result() -> OptimizedParseResult:
    return OptimizedParseResult(
        source_documents=[
            SourceDocument(
                id="doc_1",
                source="AI Research/Test.md",
                relative_path="Test.md",
                path="/tmp/Test.md",
                line_count=2,
                nonempty_line_count=2,
                url_count=0,
            )
        ],
        blocks=[
            RemNoteBlock(
                id="block_1",
                source_document_id="doc_1",
                source="AI Research/Test.md",
                line_number=1,
                block_ordinal=0,
                raw_text="Original noisy text",
                text="Original noisy text",
                depth_level=0,
                path=["Test"],
            ),
            RemNoteBlock(
                id="block_2",
                source_document_id="doc_1",
                source="AI Research/Test.md",
                line_number=2,
                block_ordinal=1,
                raw_text="Boilerplate",
                text="Boilerplate",
                depth_level=0,
                path=["Test"],
            ),
        ],
        external_resources=[],
        parsed_artifacts=[],
        artifact_gate_decisions=[],
        retrieval_chunks=[
            RetrievalChunk(
                id="chunk_1",
                text="Original noisy text",
                chunk_type="remnote_section",
                source="AI Research/Test.md",
                path=["Test"],
                line_start=1,
                line_end=1,
                source_block_ids=["block_1"],
                heading_path=["Test"],
                embedding_text="Original noisy text",
                display_text="Original noisy text",
            ),
            RetrievalChunk(
                id="chunk_2",
                text="Boilerplate",
                chunk_type="remnote_section",
                source="AI Research/Test.md",
                path=["Test"],
                line_start=2,
                line_end=2,
                source_block_ids=["block_2"],
                heading_path=["Test"],
                embedding_text="Boilerplate",
                display_text="Boilerplate",
            ),
        ],
        summary={},
    )


class PostprocessedGraphStorageMetadataTests(unittest.TestCase):
    def test_vector_metadata_drops_graph_objects_and_serializes_nested_values(self) -> None:
        class NonSerializable:
            pass

        metadata = {
            "chunk_id": "chunk_1",
            "aliases": ["Alias"],
            "stats": {"score": 1},
            "original_text": '<div><img src="x.jpg"></div>',
            "postprocess_original_embedding_text": '<table><tr><td>Noisy</td></tr></table>',
            "kg_nodes": [NonSerializable()],
            "kg_relations": [NonSerializable()],
        }

        safe = make_vector_store_metadata(
            metadata,
            {"kg_nodes", "kg_relations", "original_text", "postprocess_original_embedding_text"},
        )

        self.assertNotIn("kg_nodes", safe)
        self.assertNotIn("kg_relations", safe)
        self.assertNotIn("original_text", safe)
        self.assertNotIn("postprocess_original_embedding_text", safe)
        self.assertEqual("chunk_1", safe["chunk_id"])
        self.assertEqual(["Alias"], json.loads(safe["aliases"]))
        self.assertEqual({"score": 1}, json.loads(safe["stats"]))
        json.dumps(safe)


@unittest.skipUnless(LLAMA_INDEX_AVAILABLE, "llama_index is not installed")
class PostprocessedGraphStorageTests(unittest.TestCase):
    def _storage_settings(self, tmp: str) -> tuple[PathSettings, StorageSettings]:
        storage_path = Path(tmp) / "storage"
        local = LocalStorageSettings(storage_path=storage_path)
        path_settings = PathSettings(
            raw_data_dir=Path(tmp) / "raw",
            parsed_pdfs_dir=Path(tmp) / "pdfs",
            parsed_images_dir=Path(tmp) / "images",
            parsed_texts_dir=Path(tmp) / "texts",
            local_storage_dir=storage_path,
        )
        storage_settings = StorageSettings(
            document_storage=local,
            index_storage=local,
            vector_storage=local,
            property_graph_storage=local,
        )
        return path_settings, storage_settings

    def test_materialization_applies_cleaned_text_and_quarantine_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path_settings, storage_settings = self._storage_settings(tmp)
            nodes, manifest = materialize_final_text_nodes(
                make_result(),
                [
                    make_decision("chunk_1", "keep_with_cleaned_text", cleaned="Cleaned text"),
                    make_decision("chunk_2", "exclude_from_embedding"),
                ],
                [],
                path_settings,
                storage_settings,
            )

        nodes_by_id = {node.id_: node for node in nodes}
        self.assertEqual("Cleaned text", nodes_by_id["chunk_1"].text)
        self.assertTrue(nodes_by_id["chunk_1"].metadata["retrieval_enabled"])
        self.assertTrue(nodes_by_id["chunk_1"].metadata["graph_enabled"])
        self.assertFalse(nodes_by_id["chunk_2"].metadata["retrieval_enabled"])
        self.assertFalse(nodes_by_id["chunk_2"].metadata["graph_enabled"])
        self.assertTrue(nodes_by_id["chunk_2"].metadata["quarantined"])
        self.assertEqual(1, manifest["retrieval_enabled_count"])
        self.assertEqual(1, manifest["quarantined_count"])

    def test_materialization_sanitizes_markup_after_llm_decision(self) -> None:
        result = make_result()
        result.retrieval_chunks[0].embedding_text = (
            'Intro <div><img src="x.jpg" alt="Image"></div> Figure 4. Extensions connect Agents.'
        )
        result.retrieval_chunks[0].display_text = result.retrieval_chunks[0].embedding_text

        with tempfile.TemporaryDirectory() as tmp:
            path_settings, storage_settings = self._storage_settings(tmp)
            nodes, manifest = materialize_final_text_nodes(
                result,
                [
                    make_decision("chunk_1", "keep"),
                    make_decision(
                        "chunk_2",
                        "keep_with_cleaned_text",
                        cleaned='Cleaned <span>text</span> <img src="diagram.jpg" alt="architecture diagram">',
                    ),
                ],
                [],
                path_settings,
                storage_settings,
            )

        nodes_by_id = {node.id_: node for node in nodes}
        self.assertEqual("Intro Figure 4. Extensions connect Agents.", nodes_by_id["chunk_1"].text)
        self.assertEqual("Cleaned text architecture diagram", nodes_by_id["chunk_2"].text)
        self.assertNotIn("<div", nodes_by_id["chunk_1"].text)
        self.assertNotIn("<img", nodes_by_id["chunk_2"].text)
        self.assertTrue(nodes_by_id["chunk_1"].metadata["retrieval_enabled"])
        self.assertTrue(nodes_by_id["chunk_1"].metadata["graph_enabled"])
        self.assertTrue(nodes_by_id["chunk_1"].metadata["markup_sanitized"])
        self.assertEqual(1, nodes_by_id["chunk_1"].metadata["markup_removed_image_count"])
        self.assertEqual(["architecture diagram"], nodes_by_id["chunk_2"].metadata["markup_preserved_alt_texts"])
        self.assertEqual(2, manifest["markup_sanitized_count"])
        self.assertEqual(2, manifest["markup_removed_image_count"])
        self.assertGreaterEqual(manifest["markup_removed_tag_count"], 5)

    def test_import_projection_preserves_enriched_relation_properties(self) -> None:
        class FakeGraphStore:
            def __init__(self) -> None:
                self.nodes = []
                self.relations = []
                self.llama_nodes = []

            def upsert_nodes(self, nodes) -> None:
                self.nodes.extend(nodes)

            def upsert_relations(self, relations) -> None:
                self.relations.extend(relations)

            def upsert_llama_nodes(self, nodes) -> None:
                self.llama_nodes.extend(nodes)

        class FakeStorageContext:
            def __init__(self) -> None:
                self.property_graph_store = FakeGraphStore()

        class FakeChunkNode:
            def __init__(self, node_id: str) -> None:
                self.id_ = node_id
                self.metadata = {}

        storage_context = FakeStorageContext()
        projection = ConceptGraphProjection(
            nodes=[
                {"id": "concept_backprop", "canonical_name": "Backpropagation", "type": "METHOD"},
                {"id": "concept_gradient", "canonical_name": "Gradient", "type": "CONCEPT"},
            ],
            edges=[
                {
                    "id": "rel_1",
                    "source_concept_id": "concept_backprop",
                    "target_concept_id": "concept_gradient",
                    "canonical_predicate": "PRODUCES",
                    "raw_predicates": ["PRODUCES"],
                    "predicate_statuses": ["existing"],
                    "predicate_family": "computation",
                    "predicate_definitions": [],
                    "relation_phrases": ["produces gradients for each layer"],
                    "evidence_chunk_ids": ["chunk_1"],
                    "evidence_spans": ["Backpropagation produces gradients"],
                    "decision_ids": ["decision_1"],
                    "max_confidence": 0.9,
                    "max_generality_score": 0.82,
                    "max_retrieval_usefulness": 0.88,
                    "max_visualization_usefulness": 0.76,
                }
            ],
            evidence_links=[],
        )

        summary = import_projection_to_property_graph(storage_context, [FakeChunkNode("chunk_1")], projection)

        semantic_relation = next(
            relation for relation in storage_context.property_graph_store.relations
            if getattr(relation, "label", None) == "PRODUCES"
        )
        self.assertEqual(1, summary["semantic_relations_imported"])
        self.assertEqual(["produces gradients for each layer"], semantic_relation.properties["relation_phrases"])
        self.assertEqual(0.82, semantic_relation.properties["max_generality_score"])
        self.assertEqual(0.88, semantic_relation.properties["max_retrieval_usefulness"])
        self.assertEqual(0.76, semantic_relation.properties["max_visualization_usefulness"])

    def test_embedding_passage_nodes_preserve_parent_chunk_identity(self) -> None:
        from llama_index.core.schema import TextNode

        node = TextNode(
            id_="chunk_parent",
            text=(
                "Sentence one explains SGD. Sentence two explains Momentum. "
                "Sentence three explains Adam. Sentence four explains AdamW."
            ),
            metadata={
                "docstore_node_kind": "postprocessed_retrieval_chunk",
                "chunk_id": "chunk_parent",
                "path": ["Optimization", "Optimizers"],
                "retrieval_enabled": True,
                "graph_enabled": True,
                "quarantined": False,
                "postprocess_chunk_summary": "Optimizer methods overview.",
            },
        )

        passage_nodes = make_embedding_passage_nodes([node], TextNode, embedder=None)

        self.assertGreaterEqual(len(passage_nodes), 1)
        first = passage_nodes[0]
        self.assertTrue(first.id_.startswith("chunk_parent::passage_"))
        self.assertEqual("postprocessed_embedding_passage", first.metadata["docstore_node_kind"])
        self.assertEqual("chunk_parent", first.metadata["parent_chunk_id"])
        self.assertEqual("chunk_parent", first.metadata["chunk_id"])
        self.assertFalse(first.metadata["graph_enabled"])
        self.assertIn("Optimization > Optimizers", first.text)
        self.assertNotIn("Summary: Optimizer methods overview.", first.text)


if __name__ == "__main__":
    unittest.main()
