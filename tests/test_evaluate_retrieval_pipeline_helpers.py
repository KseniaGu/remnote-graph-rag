import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_retrieval_pipeline import load_embedded_source_ids, load_source_metadata_by_id


class EvaluateRetrievalPipelineHelperTests(unittest.TestCase):
    def test_source_metadata_ignores_passage_docstore_nodes(self) -> None:
        payload = {
            "docstore/data": {
                "chunk_parent": {
                    "__data__": {
                        "id_": "chunk_parent",
                        "metadata": {
                            "docstore_node_kind": "postprocessed_retrieval_chunk",
                            "chunk_id": "chunk_parent",
                            "source": "Parent source",
                        },
                    }
                },
                "chunk_parent::passage_000": {
                    "__data__": {
                        "id_": "chunk_parent::passage_000",
                        "metadata": {
                            "docstore_node_kind": "postprocessed_embedding_passage",
                            "chunk_id": "chunk_parent",
                            "parent_chunk_id": "chunk_parent",
                            "source": "Passage source",
                        },
                    }
                },
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "docstore.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            metadata = load_source_metadata_by_id(path)

        self.assertEqual({"chunk_parent"}, set(metadata))
        self.assertEqual("Parent source", metadata["chunk_parent"]["source"])

    def test_embedded_source_ids_include_parents_of_passage_vectors(self) -> None:
        payload = {
            "embedding_dict": {
                "chunk_parent::passage_000": [0.1, 0.2],
                "concept_optimizer": [0.3, 0.4],
            },
            "metadata_dict": {
                "chunk_parent::passage_000": {
                    "docstore_node_kind": "postprocessed_embedding_passage",
                    "parent_chunk_id": "chunk_parent",
                },
                "concept_optimizer": {
                    "docstore_node_kind": "postprocessed_concept_node",
                    "concept_id": "concept_optimizer",
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "default__vector_store.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            source_ids = load_embedded_source_ids(path)

        self.assertEqual({"chunk_parent"}, source_ids)


if __name__ == "__main__":
    unittest.main()
