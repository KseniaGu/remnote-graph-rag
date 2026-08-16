import unittest
from unittest.mock import patch

from pydantic import ValidationError

from backend.configs.constants import (
    RETRIEVAL_BELOW_THRESHOLD,
    RETRIEVAL_TOPIC_MISMATCH,
)
from backend.utils.chat_errors import KnowledgeBaseUnavailable
from backend.workflows.agents.retrieval_evidence import (
    FACT_BLOCK_HEADER,
    QueryEvidenceResult,
    RelationEvidence,
    SourceEvidence,
    evidence_from_retrieved_node,
    format_search_results,
    format_visualization_results,
)
from backend.workflows.agents.tools import (
    KnowledgeSearchInput,
    get_subgraphs_to_visualize,
    search_knowledge_base,
)


class FakeNode:
    def __init__(
        self, text: str, node_id: str = "node_1", metadata: dict | None = None
    ) -> None:
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


class RaisingRetriever:
    def retrieve(self, query: str) -> list:
        raise ConnectionError("mongodb://user:secret@example.test")


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
        items = evidence_from_retrieved_node(
            FakeNodeWithScore(node, 0.81), query="text classification", rank=1
        )

        self.assertEqual(1, len(items))
        self.assertIsInstance(items[0], SourceEvidence)
        output = format_search_results(
            [QueryEvidenceResult(query="text classification", items=items)]
        )

        self.assertIn("RETRIEVER RESULTS", output)
        self.assertIn(
            "[SOURCE] (Score: 0.81) Text classification assigns labels", output
        )
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
        item = evidence_from_retrieved_node(FakeNodeWithScore(node), query="q", rank=1)[
            0
        ]

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
        items = evidence_from_retrieved_node(
            FakeNodeWithScore(node, 0.77), query="naive bayes", rank=1
        )

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

        items = evidence_from_retrieved_node(
            FakeNodeWithScore(node, 0.93), query="naive bayes", rank=1
        )
        relations = [item for item in items if isinstance(item, RelationEvidence)]
        sources = [item for item in items if isinstance(item, SourceEvidence)]

        self.assertEqual("Naive Bayes -> IS_A -> Classifier", relations[0].raw_relation)
        self.assertEqual(["chunk_a"], relations[0].evidence_chunk_ids)
        self.assertEqual(["Naive Bayes"], relations[0].evidence_spans)
        self.assertIsNone(relations[0].subject_id)
        self.assertIsNone(relations[0].relation_id)
        self.assertEqual(1, len(sources))
        self.assertTrue(sources[0].derived_from_relation_node)

        output = format_search_results(
            [QueryEvidenceResult(query="naive bayes", items=items)]
        )
        self.assertIn(
            "[RELATION] Naive Bayes -> IS_A -> Classifier (Score: 0.93)", output
        )
        self.assertIn("[SOURCE] Naive Bayes is a probabilistic classifier.", output)
        self.assertIn("[SOURCE PATH] Text Classification > Naive Bayes", output)

    def test_postprocessed_relation_preserves_ids_and_edge_evidence(self) -> None:
        relation_text = (
            "concept_a ({'name': 'concept_a', 'entity_name': 'Subject (A)', "
            "'postprocess_concept_id': 'concept_a', "
            "'source_chunk_ids': ['chunk_subject']}) -> "
            "USES ({'postprocess_relation_id': 'rel_1', "
            "'evidence_chunk_ids': ['chunk_edge']}) -> "
            "concept_b ({'name': 'concept_b', 'entity_name': 'Object', "
            "'postprocess_concept_id': 'concept_b', "
            "'source_chunk_ids': ['chunk_object']})"
        )

        relation = evidence_from_retrieved_node(
            FakeNodeWithScore(FakeNode(relation_text)), query="q", rank=1
        )[0]

        self.assertIsInstance(relation, RelationEvidence)
        self.assertEqual("concept_a", relation.subject_id)
        self.assertEqual("rel_1", relation.relation_id)
        self.assertEqual("concept_b", relation.object_id)
        self.assertEqual(["chunk_edge"], relation.evidence_chunk_ids)
        self.assertEqual(("Subject (A)", "USES", "Object"), relation.as_triplet())

    def test_child_parent_and_low_score_handling(self) -> None:
        parent_node = FakeNode("Parent -> PARENT -> Child")
        self.assertEqual(
            [],
            evidence_from_retrieved_node(
                FakeNodeWithScore(parent_node, 0.95), query="q", rank=1
            ),
        )

        low_score_child = FakeNode("Parent -> CHILD -> Child")
        self.assertEqual(
            [],
            evidence_from_retrieved_node(
                FakeNodeWithScore(low_score_child, 0.2), query="q", rank=1
            ),
        )

        high_score_child = FakeNode(
            "({'text': 'Parent', 'path': \"['Root', 'Parent']\"}) -> CHILD -> "
            "({'text': 'Child', 'path': \"['Root', 'Child']\"})"
        )
        items = evidence_from_retrieved_node(
            FakeNodeWithScore(high_score_child, 0.8), query="q", rank=1
        )
        self.assertEqual(1, len(items))
        self.assertIsInstance(items[0], RelationEvidence)
        self.assertIn("Parent -> CHILD -> Child", items[0].raw_relation)

    def test_malformed_property_preserves_fallback_relation(self) -> None:
        node = FakeNode("Subject ({bad}) -> USES -> Object")
        items = evidence_from_retrieved_node(
            FakeNodeWithScore(node, 0.6), query="q", rank=1
        )

        self.assertEqual(1, len(items))
        relation = items[0]
        self.assertIsInstance(relation, RelationEvidence)
        self.assertEqual("Subject -> USES -> Object", relation.raw_relation)
        self.assertIn("failed to parse relation property", relation.parse_warnings)

    def test_empty_result_uses_sentinel(self) -> None:
        output = format_search_results([QueryEvidenceResult(query="missing", items=[])])
        self.assertEqual("No relevant information found.", output)

    def test_search_tool_recovers_stringified_query_list(self) -> None:
        query = "optimizers machine learning gradient descent"
        tool = search_knowledge_base(
            FakeRetriever(
                {query: [FakeNodeWithScore(FakeNode("Optimizer source"), 0.9)]}
            )
        )

        output = tool.invoke({"queries": f"[{query}]"})
        query_schema = tool.args_schema.model_json_schema()["properties"]["queries"]

        self.assertIn("Optimizer source", output)
        self.assertEqual("array", query_schema["type"])
        self.assertEqual(3, query_schema["maxItems"])

    def test_required_topics_are_normalized_and_deduplicated(self) -> None:
        value = KnowledgeSearchInput(
            queries=["RetNet"],
            required_topics=[[" RetNet ", "retnet", "Retentive   Network"]],
        )

        self.assertEqual([["RetNet", "Retentive Network"]], value.required_topics)

    def test_required_topic_contract_rejects_invalid_groups(self) -> None:
        invalid_topics = [
            [[]],
            [["a", "b", "c", "d"]],
            [["x" * 81]],
            [["a"], ["b"], ["c"], ["d"]],
        ]

        for required_topics in invalid_topics:
            with self.subTest(required_topics=required_topics):
                with self.assertRaises(ValidationError):
                    KnowledgeSearchInput(
                        queries=["topic"], required_topics=required_topics
                    )

    def test_retnet_requirement_is_not_satisfied_by_resnet_or_query_header(
        self,
    ) -> None:
        tool = search_knowledge_base(
            FakeRetriever(
                {
                    "RetNet architecture": [
                        FakeNodeWithScore(
                            FakeNode("ResNet is a residual convolutional network."),
                            0.9,
                        )
                    ]
                }
            )
        )

        output = tool.invoke(
            {
                "queries": ["RetNet architecture"],
                "required_topics": [["RetNet", "Retentive Network"]],
            }
        )

        self.assertEqual(RETRIEVAL_TOPIC_MISMATCH, output)

    def test_mla_requirement_is_not_satisfied_by_multi_head_attention(self) -> None:
        tool = search_knowledge_base(
            FakeRetriever(
                {
                    "MLA": [
                        FakeNodeWithScore(
                            FakeNode("Multi-Head Attention projects queries and keys."),
                            0.9,
                        )
                    ]
                }
            )
        )

        output = tool.invoke(
            {
                "queries": ["MLA"],
                "required_topics": [["Multi-Head Latent Attention", "MLA"]],
            }
        )

        self.assertEqual(RETRIEVAL_TOPIC_MISMATCH, output)

    def test_unknown_acronym_requires_exact_evidence_identity(self) -> None:
        tool = search_knowledge_base(
            FakeRetriever(
                {
                    "EAGLT": [
                        FakeNodeWithScore(FakeNode("EAGLE is a decoding method."), 0.9)
                    ]
                }
            )
        )

        output = tool.invoke({"queries": ["EAGLT"], "required_topics": [["EAGLT"]]})

        self.assertEqual(RETRIEVAL_TOPIC_MISMATCH, output)

    def test_hyphen_space_and_case_variants_satisfy_exact_token_sequence(self) -> None:
        tool = search_knowledge_base(
            FakeRetriever(
                {
                    "MLA": [
                        FakeNodeWithScore(
                            FakeNode(
                                "MULTI HEAD LATENT ATTENTION compresses key-value states."
                            ),
                            0.9,
                        )
                    ]
                }
            )
        )

        output = tool.invoke(
            {
                "queries": ["MLA"],
                "required_topics": [["Multi-Head Latent Attention"]],
            }
        )

        self.assertIn("MULTI HEAD LATENT ATTENTION", output)

    def test_source_paths_count_as_required_topic_evidence(self) -> None:
        tool = search_knowledge_base(
            FakeRetriever(
                {
                    "retentive models": [
                        FakeNodeWithScore(
                            FakeNode(
                                "The architecture supports parallel and recurrent forms.",
                                metadata={"path": ["RetNet", "Architecture"]},
                            ),
                            0.9,
                        )
                    ]
                }
            )
        )

        output = tool.invoke(
            {
                "queries": ["retentive models"],
                "required_topics": [["RetNet"]],
            }
        )

        self.assertIn("[SOURCE PATH] RetNet > Architecture", output)

    def test_relation_text_counts_as_required_topic_evidence(self) -> None:
        tool = search_knowledge_base(
            FakeRetriever(
                {
                    "retentive model": [
                        FakeNodeWithScore(
                            FakeNode("RetNet -> IS_A -> Sequence Model"), 0.9
                        )
                    ]
                }
            )
        )

        output = tool.invoke(
            {
                "queries": ["retentive model"],
                "required_topics": [["RetNet"]],
            }
        )

        self.assertIn("[RELATION] RetNet -> IS_A -> Sequence Model", output)

    def test_optimized_formatted_output_uses_same_topic_coverage_check(self) -> None:
        class FakeAnalystPipeline:
            def search(self, queries: list[str]) -> str:
                return (
                    "RETRIEVER RESULTS:\n\nQUERY: "
                    f"{queries[0]}\n[SOURCE] (Score: 0.90) ResNet is a CNN."
                )

        tool = search_knowledge_base(analyst_pipeline=FakeAnalystPipeline())

        output = tool.invoke(
            {
                "queries": ["RetNet"],
                "required_topics": [["RetNet", "Retentive Network"]],
            }
        )

        self.assertEqual(RETRIEVAL_TOPIC_MISMATCH, output)

    def test_every_comparison_group_must_be_covered(self) -> None:
        tool = search_knowledge_base(
            FakeRetriever(
                {
                    "RetNet versus Mamba": [
                        FakeNodeWithScore(FakeNode("RetNet is a sequence model."), 0.9)
                    ]
                }
            )
        )

        output = tool.invoke(
            {
                "queries": ["RetNet versus Mamba"],
                "required_topics": [["RetNet"], ["Mamba"]],
            }
        )

        self.assertEqual(RETRIEVAL_TOPIC_MISMATCH, output)

    def test_empty_requirements_preserve_broad_query_behavior(self) -> None:
        tool = search_knowledge_base(
            FakeRetriever(
                {
                    "optimizers": [
                        FakeNodeWithScore(FakeNode("Adam uses adaptive moments."), 0.9)
                    ]
                }
            )
        )

        output = tool.invoke({"queries": ["optimizers"], "required_topics": []})

        self.assertIn("Adam uses adaptive moments", output)

    def test_all_raw_results_below_tool_threshold_have_dedicated_sentinel(self) -> None:
        tool = search_knowledge_base(
            FakeRetriever(
                {"RetNet": [FakeNodeWithScore(FakeNode("RetNet details."), 0.05)]}
            )
        )

        output = tool.invoke({"queries": ["RetNet"], "required_topics": [["RetNet"]]})

        self.assertEqual(RETRIEVAL_BELOW_THRESHOLD, output)

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

    def test_retrieval_failure_is_not_reported_as_no_results(self) -> None:
        tool = search_knowledge_base(RaisingRetriever())

        with self.assertRaises(KnowledgeBaseUnavailable) as raised:
            tool.invoke({"queries": ["q"]})

        self.assertEqual(
            "I couldn't access the knowledge base right now. Please try again shortly.",
            raised.exception.user_message,
        )
        self.assertNotIn("secret", str(raised.exception))

    def test_visualization_formatter_returns_ordered_unique_results(self) -> None:
        result = QueryEvidenceResult(
            query="graph",
            items=[
                *evidence_from_retrieved_node(
                    FakeNode("A -> REL -> B"), query="graph", rank=1
                ),
                *evidence_from_retrieved_node(
                    FakeNode("A -> REL -> B"), query="graph", rank=2
                ),
                *evidence_from_retrieved_node(
                    FakeNode("Plain text", node_id="node_1"), query="graph", rank=3
                ),
                *evidence_from_retrieved_node(
                    FakeNode("More text", node_id="node_1"), query="graph", rank=4
                ),
            ],
        )

        nodes, triplets, queries = format_visualization_results([result])

        self.assertEqual(["A", "B", "node_1"], nodes)
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

        self.assertEqual(["A", "B", "node_1"], nodes)
        self.assertEqual([("A", "REL", "B")], triplets)
        self.assertEqual(["graph"], queries)


if __name__ == "__main__":
    unittest.main()
