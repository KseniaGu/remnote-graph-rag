import unittest
from unittest.mock import patch

from backend.workflows.agents.retrieval_evidence import (
    FACT_BLOCK_HEADER,
    QueryEvidenceResult,
    RelationEvidence,
    SourceEvidence,
    evidence_from_retrieved_node,
    format_search_results,
    format_visualization_results,
)
from backend.workflows.agents.tools import get_subgraphs_to_visualize, search_knowledge_base


class FakeNode:
    def __init__(self, text: str, node_id: str = "node_1", metadata: dict | None = None) -> None:
        self.text = text
        self.node_id = node_id
        self.metadata = metadata or {}


class FakeNodeWithScore:
    def __init__(self, node: FakeNode, score: float | None = 0.9) -> None:
        self.node = node
        self.score = score


class FakeRetriever:
    def __init__(self, results_by_query: dict[str, list]) -> None:
        self.results_by_query = results_by_query

    def retrieve(self, query: str) -> list:
        return list(self.results_by_query.get(query, []))


class IdentityReranker:
    def postprocess_nodes(self, nodes: list, query_str: str) -> list:
        return nodes


class RaisingReranker:
    def postprocess_nodes(self, nodes: list, query_str: str) -> list:
        raise RuntimeError("reranker unavailable")


class RetrievalEvidenceTests(unittest.TestCase):
    def test_plain_source_node_formats_as_source(self) -> None:
        node = FakeNode(
            "Text classification assigns labels to text examples.",
            metadata={"path": '["Text Classification", "Overview"]'},
        )
        items = evidence_from_retrieved_node(FakeNodeWithScore(node, 0.81), query="text classification", rank=1)

        self.assertEqual(1, len(items))
        self.assertIsInstance(items[0], SourceEvidence)
        output = format_search_results([QueryEvidenceResult(query="text classification", items=items)])

        self.assertIn("RETRIEVER RESULTS", output)
        self.assertIn("[SOURCE] (Score: 0.81) Text classification assigns labels", output)
        self.assertIn("[SOURCE PATH] Text Classification > Overview", output)

    def test_metadata_json_strings_are_normalized(self) -> None:
        node = FakeNode(
            "Metadata normalization test.",
            node_id="chunk_1",
            metadata={
                "path": '["Root", "Child"]',
                "heading_path": '["Root"]',
                "source_block_ids": '["block_1", "block_2"]',
                "external_resource_ids": '["res_1"]',
                "retrieval_enabled": "true",
                "graph_enabled": "false",
                "quarantined": "false",
                "postprocess_decision_id": "decision_1",
            },
        )
        item = evidence_from_retrieved_node(FakeNodeWithScore(node), query="q", rank=1)[0]

        self.assertEqual(["Root", "Child"], item.metadata.path)
        self.assertEqual(["Root"], item.metadata.heading_path)
        self.assertEqual(["block_1", "block_2"], item.metadata.source_block_ids)
        self.assertEqual(["res_1"], item.metadata.external_resource_ids)
        self.assertTrue(item.metadata.retrieval_enabled)
        self.assertFalse(item.metadata.graph_enabled)
        self.assertFalse(item.metadata.quarantined)
        self.assertEqual("decision_1", item.metadata.postprocess_decision_id)

    def test_simple_relation_becomes_relation_evidence(self) -> None:
        node = FakeNode("Naive Bayes -> IS_A -> Classifier")
        items = evidence_from_retrieved_node(FakeNodeWithScore(node, 0.77), query="naive bayes", rank=1)

        self.assertEqual(1, len(items))
        relation = items[0]
        self.assertIsInstance(relation, RelationEvidence)
        self.assertEqual(("Naive Bayes", "IS_A", "Classifier"), relation.as_triplet())

    def test_fact_block_emits_relation_source_and_path(self) -> None:
        relation_line = (
            "concept_a ({'name': 'concept_a', 'entity_name': 'Naive Bayes', "
            "'source_chunk_ids': ['chunk_a'], 'evidence_spans': ['Naive Bayes']}) "
            "-> IS_A -> "
            "concept_b ({'name': 'concept_b', 'entity_name': 'Classifier'})"
        )
        text = f"{FACT_BLOCK_HEADER}\n\n{relation_line}\n\nNaive Bayes is a probabilistic classifier."
        node = FakeNode(text, metadata={"path": ["Text Classification", "Naive Bayes"]})

        items = evidence_from_retrieved_node(FakeNodeWithScore(node, 0.93), query="naive bayes", rank=1)
        relations = [item for item in items if isinstance(item, RelationEvidence)]
        sources = [item for item in items if isinstance(item, SourceEvidence)]

        self.assertEqual("Naive Bayes -> IS_A -> Classifier", relations[0].raw_relation)
        self.assertEqual(["chunk_a"], relations[0].evidence_chunk_ids)
        self.assertEqual(["Naive Bayes"], relations[0].evidence_spans)
        self.assertEqual(1, len(sources))
        self.assertTrue(sources[0].derived_from_relation_node)

        output = format_search_results([QueryEvidenceResult(query="naive bayes", items=items)])
        self.assertIn("[RELATION] Naive Bayes -> IS_A -> Classifier (Score: 0.93)", output)
        self.assertIn("[SOURCE] Naive Bayes is a probabilistic classifier.", output)
        self.assertIn("[SOURCE PATH] Text Classification > Naive Bayes", output)

    def test_child_parent_and_low_score_handling(self) -> None:
        parent_node = FakeNode("Parent -> PARENT -> Child")
        self.assertEqual([], evidence_from_retrieved_node(FakeNodeWithScore(parent_node, 0.95), query="q", rank=1))

        low_score_child = FakeNode("Parent -> CHILD -> Child")
        self.assertEqual([], evidence_from_retrieved_node(FakeNodeWithScore(low_score_child, 0.2), query="q", rank=1))

        high_score_child = FakeNode(
            "({'text': 'Parent', 'path': \"['Root', 'Parent']\"}) -> CHILD -> "
            "({'text': 'Child', 'path': \"['Root', 'Child']\"})"
        )
        items = evidence_from_retrieved_node(FakeNodeWithScore(high_score_child, 0.8), query="q", rank=1)
        self.assertEqual(1, len(items))
        self.assertIsInstance(items[0], RelationEvidence)
        self.assertIn("Parent -> CHILD -> Child", items[0].raw_relation)

    def test_malformed_property_preserves_fallback_relation(self) -> None:
        node = FakeNode("Subject ({bad}) -> USES -> Object")
        items = evidence_from_retrieved_node(FakeNodeWithScore(node, 0.6), query="q", rank=1)

        self.assertEqual(1, len(items))
        relation = items[0]
        self.assertIsInstance(relation, RelationEvidence)
        self.assertEqual("Subject -> USES -> Object", relation.raw_relation)
        self.assertIn("failed to parse relation property", relation.parse_warnings)

    def test_empty_result_uses_sentinel(self) -> None:
        output = format_search_results([QueryEvidenceResult(query="missing", items=[])])
        self.assertEqual("No relevant information found.", output)

    def test_search_tool_uses_reranker_exception_fallback(self) -> None:
        nodes = [
            FakeNodeWithScore(FakeNode(f"Result {idx}", node_id=f"node_{idx}"), 0.9)
            for idx in range(12)
        ]
        tool = search_knowledge_base(FakeRetriever({"q": nodes}), RaisingReranker())
        with patch("backend.workflows.agents.tools.logger.error"):
            output = tool.invoke({"queries": ["q"]})

        self.assertIn("Result 0", output)
        self.assertIn("Result 9", output)
        self.assertNotIn("Result 10", output)

    def test_visualization_formatter_returns_ordered_unique_results(self) -> None:
        result = QueryEvidenceResult(
            query="graph",
            items=[
                *evidence_from_retrieved_node(FakeNode("A -> REL -> B"), query="graph", rank=1),
                *evidence_from_retrieved_node(FakeNode("A -> REL -> B"), query="graph", rank=2),
                *evidence_from_retrieved_node(FakeNode("Plain text", node_id="node_1"), query="graph", rank=3),
                *evidence_from_retrieved_node(FakeNode("More text", node_id="node_1"), query="graph", rank=4),
            ],
        )

        nodes, triplets, queries = format_visualization_results([result])

        self.assertEqual(["node_1"], nodes)
        self.assertEqual([("A", "REL", "B")], triplets)
        self.assertEqual(["graph"], queries)

    def test_visualization_tool_preserves_tuple_contract(self) -> None:
        tool = get_subgraphs_to_visualize(
            FakeRetriever(
                {
                    "graph": [
                        FakeNode("A -> REL -> B"),
                        FakeNode("Plain source", node_id="node_1"),
                    ]
                }
            )
        )
        nodes, triplets, queries = tool.invoke({"queries": ["graph"]})

        self.assertEqual(["node_1"], nodes)
        self.assertEqual([("A", "REL", "B")], triplets)
        self.assertEqual(["graph"], queries)


if __name__ == "__main__":
    unittest.main()
