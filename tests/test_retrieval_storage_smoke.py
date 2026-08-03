import os
import re
import unittest
from pathlib import Path

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    HuggingFaceEmbedding = None
    LLAMA_INDEX_AVAILABLE = False

from backend.configs.models import ModelSettings
from backend.configs.paths import PathSettings
from backend.configs.search import KnowledgeGraphSearchSettings
from backend.configs.storage import LocalStorageSettings, StorageSettings
from backend.knowledge_graph.indexer import KnowledgeGraphIndexer
from backend.knowledge_graph.storage import KnowledgeGraphStorage
from backend.workflows.agents.analyst_retrieval import AnalystRetrievalPipeline
from backend.workflows.agents.tools import (
    get_subgraphs_to_visualize,
    search_knowledge_base,
)
from backend.workflows.agents.visualizer_retrieval import VisualizerRetrievalPipeline

ROOT = Path(__file__).resolve().parents[1]
CURRENT_RUN_ROOT = (
    ROOT
    / "data"
    / "testing"
    / "final_retrieval_optimization_v7_split_prompts"
    / "optimized_pipeline_run"
)
LEGACY_RUN_ROOT = ROOT / "data" / "testing" / "final" / "optimized_pipeline_run"


def resolve_smoke_run_root() -> Path:
    configured = os.environ.get("RETRIEVAL_SMOKE_RUN_ROOT")
    if configured:
        return Path(configured).expanduser()
    for candidate in (CURRENT_RUN_ROOT, LEGACY_RUN_ROOT):
        if (candidate / "final_storage").exists():
            return candidate
    return CURRENT_RUN_ROOT


RUN_ROOT = resolve_smoke_run_root()
FINAL_STORAGE_DIR = RUN_ROOT / "final_storage"
RAW_DATA_DIR = RUN_ROOT.parent / "raw" / "AI Research"
PARSED_PDFS_DIR = RUN_ROOT / "parsed_pdfs"
PARSED_IMAGES_DIR = RUN_ROOT / "parsed_images"
PARSED_TEXTS_DIR = RUN_ROOT / "parsed_texts"
FALLBACK_EMBEDDER_DIR = ROOT / "models" / "all-MiniLM-L6-v2"


class IdentityReranker:
    def postprocess_nodes(self, nodes: list, query_str: str) -> list:
        return nodes


@unittest.skipUnless(
    LLAMA_INDEX_AVAILABLE, "llama_index HuggingFace embeddings are not installed"
)
class RetrievalStorageSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not FINAL_STORAGE_DIR.exists():
            raise unittest.SkipTest(
                f"final storage directory not found: {FINAL_STORAGE_DIR}"
            )

        embedder_path = Path(ModelSettings().embedder.model_path)
        if not embedder_path.exists():
            embedder_path = FALLBACK_EMBEDDER_DIR
        if not embedder_path.exists():
            raise unittest.SkipTest(
                f"local embedder directory not found: {embedder_path}"
            )

        path_settings = PathSettings(
            raw_data_dir=RAW_DATA_DIR,
            parsed_pdfs_dir=PARSED_PDFS_DIR,
            parsed_images_dir=PARSED_IMAGES_DIR,
            parsed_texts_dir=PARSED_TEXTS_DIR,
            local_storage_dir=FINAL_STORAGE_DIR,
        )
        local_storage = LocalStorageSettings(
            storage_path=path_settings.local_storage_dir
        )
        storage_settings = StorageSettings(
            document_storage=local_storage,
            index_storage=local_storage,
            vector_storage=local_storage,
            property_graph_storage=local_storage,
        )
        kg_storage = KnowledgeGraphStorage(path_settings, storage_settings)
        embedder = HuggingFaceEmbedding(
            str(embedder_path),
            trust_remote_code=True,
            embed_batch_size=5,
            local_files_only=True,
        )
        kg_search_settings = KnowledgeGraphSearchSettings(
            analyst_reranker_mode="disabled"
        )
        cls.indexer = KnowledgeGraphIndexer(
            kg_storage.storage_context,
            path_settings,
            storage_settings.document_storage.storage_type,
            kg_search_settings,
            embedder,
            None,
        )
        cls.indexer.load_index()

    def assert_analyst_output_invariants(self, output: str) -> None:
        self.assertNotIn("-> MENTIONS ->", output)
        self.assertLessEqual(
            len(output), self.indexer.kg_search_settings.analyst_context_max_chars
        )

        source_ids = set(
            re.findall(r"^\[SOURCE\] \[(S\d+)\]", output, flags=re.MULTILINE)
        )
        for line in output.splitlines():
            if line.startswith("[SOURCE PATH]"):
                self.assertNotIn("external:", line)
                self.assertNotRegex(line, r"(^| > )#+")
            if line.startswith("[RELATION]"):
                evidence_match = re.search(r"Evidence: ([^;)]+)", line)
                self.assertIsNotNone(evidence_match, line)
                evidence_ids = {
                    evidence_id.strip()
                    for evidence_id in evidence_match.group(1).split(",")
                    if evidence_id.strip()
                }
                self.assertTrue(evidence_ids, line)
                self.assertTrue(evidence_ids <= source_ids, line)

    def graph_labels_for(self, node_ids: list[str]) -> set[str]:
        nodes = self.indexer.index.property_graph_store.get(ids=node_ids)
        labels = set()
        for node in nodes:
            properties = getattr(node, "properties", {}) or {}
            label = (
                properties.get("entity_name")
                or properties.get("display_name")
                or getattr(node, "name", None)
            )
            if label:
                labels.add(str(label))
        return labels

    def assert_visualizer_output_invariants(
        self,
        nodes: list[str],
        triplets: list[tuple[str, str, str]],
    ) -> None:
        denied = set(self.indexer.kg_search_settings.visualizer_denied_relation_labels)
        self.assertTrue(nodes or triplets)
        self.assertLessEqual(
            len(nodes), self.indexer.kg_search_settings.visualizer_max_nodes
        )
        self.assertLessEqual(
            len(triplets), self.indexer.kg_search_settings.visualizer_max_edges
        )
        self.assertTrue(all(len(triplet) == 3 for triplet in triplets))
        self.assertFalse(any(triplet[1] in denied for triplet in triplets))
        self.assertFalse(any(str(node_id).startswith("chunk_") for node_id in nodes))

    def test_search_tool_runs_against_optimized_final_storage(self) -> None:
        tool = search_knowledge_base(
            analyst_pipeline=AnalystRetrievalPipeline(self.indexer)
        )

        output = tool.invoke(
            {"queries": ["Text Classification", "Naive Bayes classifier"]}
        )

        self.assertIn("RETRIEVER RESULTS", output)
        self.assertIn("[SOURCE]", output)
        self.assertRegex(output, r"(?i)(text classification|naive bayes|classifier)")
        self.assert_analyst_output_invariants(output)

    def test_search_tool_prioritizes_dataset_chunk(self) -> None:
        tool = search_knowledge_base(
            analyst_pipeline=AnalystRetrievalPipeline(self.indexer)
        )

        output = tool.invoke({"queries": ["text classification datasets"]})

        self.assertIn("Common datasets", output)
        self.assertIn("AG News", output)
        logistic_index = output.find("Logistic Regression")
        dataset_index = output.find("Common datasets")
        if logistic_index != -1:
            self.assertLess(dataset_index, logistic_index)
        self.assert_analyst_output_invariants(output)

    def test_search_tool_returns_clip_objective_without_text_classification_noise(
        self,
    ) -> None:
        tool = search_knowledge_base(
            analyst_pipeline=AnalystRetrievalPipeline(self.indexer)
        )

        output = tool.invoke({"queries": ["CLIP training objective"]})

        self.assertRegex(output, r"(?i)(CLIP|Contrastive Objective)")
        self.assertNotIn(
            "Text Classification > How it used to be > Multi-label classification > Word Embeddings",
            output,
        )
        self.assert_analyst_output_invariants(output)

    def test_search_tool_returns_naive_bayes_logistic_regression_comparison(
        self,
    ) -> None:
        tool = search_knowledge_base(
            analyst_pipeline=AnalystRetrievalPipeline(self.indexer)
        )

        output = tool.invoke({"queries": ["Naive Bayes vs Logistic Regression"]})

        self.assertRegex(output, r"(?i)Naive Bayes")
        self.assertRegex(output, r"(?i)Logistic Regression")
        self.assertIn("[SOURCE]", output)
        self.assert_analyst_output_invariants(output)

    def test_search_tool_keeps_react_results_focused(self) -> None:
        tool = search_knowledge_base(
            analyst_pipeline=AnalystRetrievalPipeline(self.indexer)
        )

        output = tool.invoke({"queries": ["Kaggle agents / ReAct"]})

        self.assertRegex(output, r"(?i)(ReAct|orchestration)")
        extension_index = output.find("Extension -> CONNECTS")
        react_index = output.lower().find("react")
        if extension_index != -1 and react_index != -1:
            self.assertLess(react_index, extension_index)
        self.assert_analyst_output_invariants(output)

    def test_visualization_tool_runs_against_optimized_final_storage(self) -> None:
        retriever = self.indexer.get_retriever(
            self.indexer.kg_search_settings.visualizer_retriever_params
        )
        tool = get_subgraphs_to_visualize(retriever)

        nodes, triplets, queries = tool.invoke({"queries": ["Text Classification"]})

        self.assertEqual(["Text Classification"], queries)
        self.assertIsInstance(nodes, list)
        self.assertIsInstance(triplets, list)
        self.assertTrue(nodes or triplets)
        self.assertTrue(all(len(triplet) == 3 for triplet in triplets))

    def test_visualizer_pipeline_keeps_text_classification_methods_concept_centered(
        self,
    ) -> None:
        tool = get_subgraphs_to_visualize(
            visualizer_pipeline=VisualizerRetrievalPipeline(self.indexer)
        )

        nodes, triplets, queries = tool.invoke(
            {"queries": ["Text Classification methods"]}
        )

        self.assertEqual(["Text Classification methods"], queries)
        self.assert_visualizer_output_invariants(nodes, triplets)
        labels = self.graph_labels_for(nodes)
        self.assertIn("Text Classification", labels)
        lower_labels = {label.lower() for label in labels}
        self.assertTrue(
            any("classifier" in label for label in lower_labels)
            or any(
                expected in lower_labels
                for expected in {
                    "feature extractor",
                    "linear classifier",
                    "k independent binary classifier",
                    "subword model",
                    "support vector machine",
                    "logistic regression",
                }
            )
        )

    def test_visualizer_pipeline_returns_naive_bayes_logistic_regression_comparison(
        self,
    ) -> None:
        tool = get_subgraphs_to_visualize(
            visualizer_pipeline=VisualizerRetrievalPipeline(self.indexer)
        )

        nodes, triplets, queries = tool.invoke(
            {"queries": ["Naive Bayes vs Logistic Regression"]}
        )

        self.assertEqual(["Naive Bayes vs Logistic Regression"], queries)
        self.assert_visualizer_output_invariants(nodes, triplets)
        labels = self.graph_labels_for(nodes)
        self.assertTrue(
            any("Naive Bayes" in label or "Naive Baye" in label for label in labels)
        )
        self.assertTrue(any("logistic regression" in label.lower() for label in labels))
        self.assertTrue(triplets)

    def test_visualizer_pipeline_keeps_clip_objective_focused(self) -> None:
        tool = get_subgraphs_to_visualize(
            visualizer_pipeline=VisualizerRetrievalPipeline(self.indexer)
        )

        nodes, triplets, queries = tool.invoke({"queries": ["CLIP training objective"]})

        self.assertEqual(["CLIP training objective"], queries)
        self.assert_visualizer_output_invariants(nodes, triplets)
        labels = self.graph_labels_for(nodes)
        self.assertIn("CLIP", labels)
        lower_labels = {label.lower() for label in labels}
        self.assertTrue(
            any(
                expected in lower_labels
                for expected in {
                    "contrastive objective",
                    "contrastive pretraining task",
                    "natural language supervision",
                    "pairwise loss",
                }
            )
        )
        self.assertFalse(any("classifier" in label.lower() for label in labels))


if __name__ == "__main__":
    unittest.main()
