import unittest
from types import SimpleNamespace

from llama_index.core.vector_stores import VectorStoreQueryResult

from backend.configs.search import KnowledgeGraphSearchSettings
from backend.workflows.agents.visualizer_retrieval import (
    ConceptCandidate,
    EdgeCandidate,
    VisualizerRetrievalPipeline,
)


class FakeEmbedder:
    def get_query_embedding(self, query: str) -> list[float]:
        return [1.0]


class FakeVectorStore:
    def __init__(
        self,
        ids_by_kind: dict[str | None, list[str]] | None = None,
        *,
        fail_filtered: bool = False,
    ) -> None:
        self.ids_by_kind = ids_by_kind or {}
        self.fail_filtered = fail_filtered
        self.queries = []

    def query(self, query):
        self.queries.append(query)
        if self.fail_filtered and getattr(query, "filters", None):
            raise RuntimeError("filtered query failed")
        node_kind = None
        if getattr(query, "filters", None):
            filters = getattr(query.filters, "filters", []) or []
            if filters:
                node_kind = getattr(filters[0], "value", None)
        ids = self.ids_by_kind.get(node_kind, self.ids_by_kind.get(None, []))
        return VectorStoreQueryResult(ids=ids, similarities=[0.9 for _ in ids])


class FakeTextNode:
    def __init__(self, node_id: str, text: str, metadata: dict | None = None) -> None:
        self.id_ = node_id
        self.node_id = node_id
        self.text = text
        self.metadata = metadata or {}


class FakeGraphNode:
    def __init__(
        self,
        node_id: str,
        label: str,
        *,
        aliases: list[str] | None = None,
        source_chunk_ids: list[str] | None = None,
        salience: float = 0.9,
        node_label: str = "CONCEPT",
    ) -> None:
        self.id = node_id
        self.label = node_label
        self.properties = {
            "entity_name": label,
            "display_name": label,
            "aliases": aliases or [],
            "source_chunk_ids": source_chunk_ids or [],
            "postprocess_max_salience": salience,
        }


class FakeRelation:
    def __init__(self, label: str, properties: dict | None = None) -> None:
        self.id = label
        self.label = label
        self.properties = properties or {}


class FakeGraphStore:
    def __init__(
        self, nodes: list[FakeGraphNode], triplets: list[tuple] | None = None
    ) -> None:
        self.nodes = {node.id: node for node in nodes}
        self.triplets = triplets or []

    def get(self, properties=None, ids=None):
        if ids:
            return [self.nodes[node_id] for node_id in ids if node_id in self.nodes]
        return list(self.nodes.values())

    def get_triplets(
        self, entity_names=None, relation_names=None, properties=None, ids=None
    ):
        triplets = list(self.triplets)
        if relation_names:
            triplets = [
                triplet for triplet in triplets if triplet[1].label in relation_names
            ]
        if ids:
            id_set = set(ids)
            triplets = [
                triplet
                for triplet in triplets
                if getattr(triplet[0], "id", None) in id_set
                or getattr(triplet[2], "id", None) in id_set
            ]
        return triplets

    def get_rel_map(self, graph_nodes, depth=1, limit=30, ignore_rels=None):
        ids = {node.id for node in graph_nodes}
        ignore_rels = set(ignore_rels or [])
        triplets = [
            triplet
            for triplet in self.triplets
            if triplet[1].label not in ignore_rels
            and (triplet[0].id in ids or triplet[2].id in ids)
        ]
        return triplets[:limit]


def make_source_metadata(**overrides):
    metadata = {
        "docstore_node_kind": "postprocessed_retrieval_chunk",
        "retrieval_enabled": True,
        "graph_enabled": True,
        "quarantined": False,
        "source": "Text Classification",
        "path": ["Text Classification"],
        "heading_path": ["Text Classification"],
        "postprocess_chunk_summary": "Text classification methods.",
    }
    metadata.update(overrides)
    return metadata


def make_indexer(vector_store, graph_store, docs=None, settings=None):
    storage_context = SimpleNamespace(
        docstore=SimpleNamespace(docs=docs or {}), vector_store=vector_store
    )
    index = SimpleNamespace(vector_store=vector_store, property_graph_store=graph_store)
    return SimpleNamespace(
        index=index,
        storage_context=storage_context,
        embedder=FakeEmbedder(),
        kg_search_settings=settings
        or KnowledgeGraphSearchSettings(analyst_reranker_mode="disabled"),
    )


class VisualizerRetrievalPipelineTests(unittest.TestCase):
    def test_exact_anchor_resolution_emits_semantic_concept_relation(self) -> None:
        text_classification = FakeGraphNode(
            "concept_text", "Text Classification", aliases=["text classification"]
        )
        classifier = FakeGraphNode("concept_classifier", "Classifier")
        relation = FakeRelation(
            "USES", {"evidence_chunk_ids": ["chunk_1"], "max_confidence": 0.9}
        )
        graph_store = FakeGraphStore(
            [text_classification, classifier],
            [(text_classification, relation, classifier)],
        )
        pipeline = VisualizerRetrievalPipeline(
            make_indexer(FakeVectorStore(), graph_store)
        )

        nodes, triplets, queries = pipeline.visualize(["Text Classification"])

        self.assertEqual(["Text Classification"], queries)
        self.assertIn("concept_text", nodes)
        self.assertIn(("concept_text", "USES", "concept_classifier"), triplets)

    def test_mentions_discover_concepts_but_do_not_render_mentions_or_chunks(
        self,
    ) -> None:
        text_classification = FakeGraphNode(
            "concept_text", "Text Classification", source_chunk_ids=["chunk_1"]
        )
        classifier = FakeGraphNode(
            "concept_classifier", "Classifier", source_chunk_ids=["chunk_1"]
        )
        chunk = FakeGraphNode("chunk_1", "Chunk", node_label="text_chunk")
        graph_store = FakeGraphStore(
            [text_classification, classifier, chunk],
            [
                (
                    chunk,
                    FakeRelation("MENTIONS", {"evidence_chunk_ids": ["chunk_1"]}),
                    text_classification,
                ),
                (
                    chunk,
                    FakeRelation("MENTIONS", {"evidence_chunk_ids": ["chunk_1"]}),
                    classifier,
                ),
            ],
        )
        docs = {
            "chunk_1": FakeTextNode(
                "chunk_1",
                "Text classifiers use a classifier component.",
                make_source_metadata(),
            )
        }
        vector_store = FakeVectorStore({"postprocessed_retrieval_chunk": ["chunk_1"]})
        pipeline = VisualizerRetrievalPipeline(
            make_indexer(vector_store, graph_store, docs)
        )

        nodes, triplets, _ = pipeline.visualize(["Text Classification methods"])

        self.assertNotIn("chunk_1", nodes)
        self.assertNotIn(("chunk_1", "MENTIONS", "concept_text"), triplets)
        self.assertIn(("concept_text", "RELATED_TO", "concept_classifier"), triplets)

    def test_passage_source_hits_resolve_to_parent_chunks(self) -> None:
        text_classification = FakeGraphNode(
            "concept_text", "Text Classification", source_chunk_ids=["chunk_1"]
        )
        classifier = FakeGraphNode(
            "concept_classifier", "Classifier", source_chunk_ids=["chunk_1"]
        )
        chunk = FakeGraphNode("chunk_1", "Chunk", node_label="text_chunk")
        graph_store = FakeGraphStore(
            [text_classification, classifier, chunk],
            [
                (
                    chunk,
                    FakeRelation("MENTIONS", {"evidence_chunk_ids": ["chunk_1"]}),
                    classifier,
                ),
            ],
        )
        docs = {
            "chunk_1": FakeTextNode(
                "chunk_1",
                "Text classification methods use classifiers.",
                make_source_metadata(),
            ),
            "chunk_1::passage_000": FakeTextNode(
                "chunk_1::passage_000",
                "Text classification methods use classifiers.",
                make_source_metadata(
                    docstore_node_kind="postprocessed_embedding_passage",
                    chunk_id="chunk_1",
                    parent_chunk_id="chunk_1",
                    graph_enabled=False,
                ),
            ),
        }
        vector_store = FakeVectorStore(
            {"postprocessed_embedding_passage": ["chunk_1::passage_000"]}
        )
        pipeline = VisualizerRetrievalPipeline(
            make_indexer(vector_store, graph_store, docs)
        )

        nodes, triplets, _ = pipeline.visualize(["Text Classification methods"])

        self.assertIn("concept_text", nodes)
        self.assertIn(("concept_text", "RELATED_TO", "concept_classifier"), triplets)

    def test_comparison_query_keeps_both_anchors_and_prefers_compares_to(self) -> None:
        naive_bayes = FakeGraphNode("concept_nb", "Naive Bayes")
        logistic = FakeGraphNode("concept_lr", "Logistic Regression")
        graph_store = FakeGraphStore(
            [naive_bayes, logistic],
            [
                (
                    logistic,
                    FakeRelation(
                        "COMPARES_TO",
                        {
                            "evidence_chunk_ids": ["chunk_compare"],
                            "max_confidence": 0.9,
                        },
                    ),
                    naive_bayes,
                )
            ],
        )
        pipeline = VisualizerRetrievalPipeline(
            make_indexer(FakeVectorStore(), graph_store)
        )

        nodes, triplets, _ = pipeline.visualize(["Naive Bayes vs Logistic Regression"])

        self.assertIn("concept_nb", nodes)
        self.assertIn("concept_lr", nodes)
        self.assertIn(("concept_lr", "COMPARES_TO", "concept_nb"), triplets)

    def test_shape_graph_drops_isolated_non_anchor_candidates_by_default(self) -> None:
        settings = KnowledgeGraphSearchSettings(
            analyst_reranker_mode="disabled",
            visualizer_allow_synthetic_edges=False,
        )
        text_classification = FakeGraphNode(
            "concept_text", "Text Classification", source_chunk_ids=["chunk_1"]
        )
        classifier = FakeGraphNode(
            "concept_classifier", "Classifier", source_chunk_ids=["chunk_1"]
        )
        word_dropout = FakeGraphNode(
            "concept_dropout", "Word Dropout", source_chunk_ids=["chunk_1"]
        )
        chunk = FakeGraphNode("chunk_1", "Chunk", node_label="text_chunk")
        graph_store = FakeGraphStore(
            [text_classification, classifier, word_dropout, chunk],
            [
                (
                    text_classification,
                    FakeRelation("USES", {"evidence_chunk_ids": ["chunk_1"]}),
                    classifier,
                ),
                (
                    chunk,
                    FakeRelation("MENTIONS", {"evidence_chunk_ids": ["chunk_1"]}),
                    word_dropout,
                ),
            ],
        )
        docs = {
            "chunk_1": FakeTextNode(
                "chunk_1",
                "Text classification methods mention word dropout in passing.",
                make_source_metadata(),
            )
        }
        vector_store = FakeVectorStore({"postprocessed_retrieval_chunk": ["chunk_1"]})
        pipeline = VisualizerRetrievalPipeline(
            make_indexer(vector_store, graph_store, docs, settings=settings)
        )

        nodes, triplets, _ = pipeline.visualize(["Text Classification methods"])

        self.assertIn("concept_text", nodes)
        self.assertIn("concept_classifier", nodes)
        self.assertNotIn("concept_dropout", nodes)
        self.assertEqual([("concept_text", "USES", "concept_classifier")], triplets)

    def test_anchor_source_filter_prevents_cross_document_mentions(self) -> None:
        text_classification = FakeGraphNode(
            "concept_text", "Text Classification", source_chunk_ids=["chunk_text"]
        )
        classifier = FakeGraphNode(
            "concept_classifier", "Classifier", source_chunk_ids=["chunk_text"]
        )
        clip = FakeGraphNode("concept_clip", "CLIP", source_chunk_ids=["chunk_clip"])
        chunk_text = FakeGraphNode("chunk_text", "Text Chunk", node_label="text_chunk")
        chunk_clip = FakeGraphNode("chunk_clip", "CLIP Chunk", node_label="text_chunk")
        graph_store = FakeGraphStore(
            [text_classification, classifier, clip, chunk_text, chunk_clip],
            [
                (
                    chunk_text,
                    FakeRelation("MENTIONS", {"evidence_chunk_ids": ["chunk_text"]}),
                    classifier,
                ),
                (
                    chunk_clip,
                    FakeRelation("MENTIONS", {"evidence_chunk_ids": ["chunk_clip"]}),
                    clip,
                ),
            ],
        )
        docs = {
            "chunk_text": FakeTextNode(
                "chunk_text",
                "Text classification methods use classifiers.",
                make_source_metadata(source="Text Classification"),
            ),
            "chunk_clip": FakeTextNode(
                "chunk_clip",
                "CLIP uses image and text encoders.",
                make_source_metadata(source="CLIP Paper", path=["CLIP Paper"]),
            ),
        }
        vector_store = FakeVectorStore(
            {"postprocessed_retrieval_chunk": ["chunk_clip", "chunk_text"]}
        )
        pipeline = VisualizerRetrievalPipeline(
            make_indexer(vector_store, graph_store, docs)
        )

        nodes, triplets, _ = pipeline.visualize(["Text Classification methods"])

        self.assertIn(("concept_text", "RELATED_TO", "concept_classifier"), triplets)
        self.assertNotIn("concept_clip", nodes)

    def test_multi_query_focus_does_not_add_synthetic_edges_to_mentioned_noise(
        self,
    ) -> None:
        naive_bayes = FakeGraphNode(
            "concept_nb", "Naive Bayes", source_chunk_ids=["chunk_compare"]
        )
        logistic = FakeGraphNode(
            "concept_lr", "Logistic Regression", source_chunk_ids=["chunk_compare"]
        )
        ag_news = FakeGraphNode(
            "concept_ag", "AG News", source_chunk_ids=["chunk_compare"]
        )
        chunk = FakeGraphNode(
            "chunk_compare", "Comparison Chunk", node_label="text_chunk"
        )
        graph_store = FakeGraphStore(
            [naive_bayes, logistic, ag_news, chunk],
            [
                (
                    logistic,
                    FakeRelation(
                        "COMPARES_TO",
                        {
                            "evidence_chunk_ids": ["chunk_compare"],
                            "max_confidence": 0.9,
                        },
                    ),
                    naive_bayes,
                ),
                (
                    chunk,
                    FakeRelation("MENTIONS", {"evidence_chunk_ids": ["chunk_compare"]}),
                    ag_news,
                ),
            ],
        )
        docs = {
            "chunk_compare": FakeTextNode(
                "chunk_compare",
                "Naive Bayes and Logistic Regression are compared near AG News examples.",
                make_source_metadata(),
            )
        }
        vector_store = FakeVectorStore(
            {"postprocessed_retrieval_chunk": ["chunk_compare"]}
        )
        pipeline = VisualizerRetrievalPipeline(
            make_indexer(vector_store, graph_store, docs)
        )

        nodes, triplets, _ = pipeline.visualize(["Naive Bayes", "Logistic Regression"])

        self.assertIn(("concept_lr", "COMPARES_TO", "concept_nb"), triplets)
        self.assertNotIn(("concept_lr", "RELATED_TO", "concept_ag"), triplets)
        self.assertNotIn(("concept_nb", "RELATED_TO", "concept_ag"), triplets)
        self.assertNotIn("concept_ag", nodes)

    def test_generic_related_to_is_dropped_when_specific_edge_exists_for_same_pair(
        self,
    ) -> None:
        clip = FakeGraphNode("concept_clip", "CLIP")
        supervision = FakeGraphNode(
            "concept_supervision", "Natural Language Supervision"
        )
        graph_store = FakeGraphStore(
            [clip, supervision],
            [
                (
                    clip,
                    FakeRelation("RELATED_TO", {"evidence_chunk_ids": ["chunk_1"]}),
                    supervision,
                ),
                (
                    clip,
                    FakeRelation("USES", {"evidence_chunk_ids": ["chunk_1"]}),
                    supervision,
                ),
            ],
        )
        pipeline = VisualizerRetrievalPipeline(
            make_indexer(FakeVectorStore(), graph_store)
        )

        _, triplets, _ = pipeline.visualize(["CLIP"])

        self.assertIn(("concept_clip", "USES", "concept_supervision"), triplets)
        self.assertNotIn(
            ("concept_clip", "RELATED_TO", "concept_supervision"), triplets
        )

    def test_visualization_usefulness_and_generality_affect_edge_selection(
        self,
    ) -> None:
        settings = KnowledgeGraphSearchSettings(
            analyst_reranker_mode="disabled",
            visualizer_max_edges=1,
            visualizer_max_nodes=3,
            visualizer_max_edges_per_node=3,
            visualizer_allow_synthetic_edges=False,
        )
        anchor = FakeGraphNode("concept_anchor", "CLIP")
        useful = FakeGraphNode("concept_useful", "Contrastive Objective")
        weak = FakeGraphNode("concept_weak", "Training Detail")
        graph_store = FakeGraphStore(
            [anchor, useful, weak],
            [
                (
                    anchor,
                    FakeRelation(
                        "USES",
                        {
                            "evidence_chunk_ids": ["chunk_1"],
                            "max_confidence": 0.7,
                            "predicate_family": "training",
                            "relation_phrases": ["uses a contrastive objective"],
                            "max_generality_score": 1.0,
                            "max_visualization_usefulness": 1.0,
                        },
                    ),
                    useful,
                ),
                (
                    anchor,
                    FakeRelation(
                        "USES",
                        {
                            "evidence_chunk_ids": ["chunk_1"],
                            "max_confidence": 0.7,
                            "predicate_family": "other",
                            "relation_phrases": ["mentions a training detail"],
                            "max_generality_score": 0.0,
                            "max_visualization_usefulness": 0.0,
                        },
                    ),
                    weak,
                ),
            ],
        )
        pipeline = VisualizerRetrievalPipeline(
            make_indexer(FakeVectorStore(), graph_store, settings=settings)
        )

        _, triplets, _ = pipeline.visualize(["CLIP"])

        self.assertEqual([("concept_anchor", "USES", "concept_useful")], triplets)

    def test_short_aliases_do_not_create_unrelated_exact_anchors(self) -> None:
        text_classification = FakeGraphNode(
            "concept_text", "Text Classification", source_chunk_ids=["chunk_text"]
        )
        loss_function = FakeGraphNode(
            "concept_loss",
            "Loss Function",
            aliases=["L"],
            source_chunk_ids=["chunk_loss"],
        )
        hidden_state = FakeGraphNode(
            "concept_h", "$\\mathbf{h}$", aliases=["h"], source_chunk_ids=["chunk_h"]
        )
        classifier = FakeGraphNode(
            "concept_classifier", "Classifier", source_chunk_ids=["chunk_text"]
        )
        graph_store = FakeGraphStore(
            [text_classification, loss_function, hidden_state, classifier],
            [
                (
                    text_classification,
                    FakeRelation("USES", {"evidence_chunk_ids": ["chunk_text"]}),
                    classifier,
                )
            ],
        )
        vector_store = FakeVectorStore({"postprocessed_concept_node": ["concept_text"]})
        pipeline = VisualizerRetrievalPipeline(make_indexer(vector_store, graph_store))

        nodes, triplets, _ = pipeline.visualize(["Classification Methods"])

        self.assertIn("concept_text", nodes)
        self.assertIn(("concept_text", "USES", "concept_classifier"), triplets)
        self.assertNotIn("concept_loss", nodes)
        self.assertNotIn("concept_h", nodes)

    def test_short_alias_substring_does_not_block_vector_anchor_fallback(self) -> None:
        fasttext = FakeGraphNode(
            "concept_fasttext", "FastText", source_chunk_ids=["chunk_fasttext"]
        )
        input_x = FakeGraphNode(
            "concept_input_x",
            "Input x",
            aliases=["x"],
            source_chunk_ids=["chunk_input"],
        )
        subword_sum = FakeGraphNode(
            "concept_subword", "subword sum", source_chunk_ids=["chunk_fasttext"]
        )
        graph_store = FakeGraphStore(
            [fasttext, input_x, subword_sum],
            [
                (
                    fasttext,
                    FakeRelation("USES", {"evidence_chunk_ids": ["chunk_fasttext"]}),
                    subword_sum,
                )
            ],
        )
        vector_store = FakeVectorStore(
            {"postprocessed_concept_node": ["concept_fasttext"]}
        )
        pipeline = VisualizerRetrievalPipeline(make_indexer(vector_store, graph_store))

        nodes, triplets, _ = pipeline.visualize(["Fast Text"])

        self.assertIn("concept_fasttext", nodes)
        self.assertIn(("concept_fasttext", "USES", "concept_subword"), triplets)
        self.assertNotIn("concept_input_x", nodes)

    def test_vector_filter_fallback_records_health_event(self) -> None:
        fasttext = FakeGraphNode(
            "concept_fasttext", "FastText", source_chunk_ids=["chunk_fasttext"]
        )
        subword_sum = FakeGraphNode(
            "concept_subword", "subword sum", source_chunk_ids=["chunk_fasttext"]
        )
        graph_store = FakeGraphStore(
            [fasttext, subword_sum],
            [
                (
                    fasttext,
                    FakeRelation("USES", {"evidence_chunk_ids": ["chunk_fasttext"]}),
                    subword_sum,
                )
            ],
        )
        vector_store = FakeVectorStore({None: ["concept_fasttext"]}, fail_filtered=True)
        pipeline = VisualizerRetrievalPipeline(make_indexer(vector_store, graph_store))

        nodes, triplets, _ = pipeline.visualize(["Fast Text"])

        self.assertIn("concept_fasttext", nodes)
        self.assertIn(("concept_fasttext", "USES", "concept_subword"), triplets)
        self.assertTrue(pipeline.health_report.has_code("vector_filter_fallback"))

    def test_semantic_expansion_uses_grounded_mentioned_concepts(self) -> None:
        settings = KnowledgeGraphSearchSettings(
            analyst_reranker_mode="disabled",
            visualizer_allow_synthetic_edges=False,
        )
        text_classification = FakeGraphNode(
            "concept_text",
            "Text Classification",
            source_chunk_ids=["chunk_text"],
        )
        multi_label = FakeGraphNode(
            "concept_multi",
            "Multi-label Classification",
            source_chunk_ids=["chunk_text"],
        )
        bce = FakeGraphNode(
            "concept_bce",
            "Binary Cross-Entropy Loss",
            source_chunk_ids=["chunk_text"],
        )
        chunk = FakeGraphNode("chunk_text", "Chunk", node_label="text_chunk")
        graph_store = FakeGraphStore(
            [text_classification, multi_label, bce, chunk],
            [
                (
                    chunk,
                    FakeRelation("MENTIONS", {"evidence_chunk_ids": ["chunk_text"]}),
                    multi_label,
                ),
                (
                    multi_label,
                    FakeRelation("USES", {"evidence_chunk_ids": ["chunk_text"]}),
                    bce,
                ),
            ],
        )
        docs = {
            "chunk_text": FakeTextNode(
                "chunk_text",
                "Text classification methods include multi-label classification with binary cross-entropy.",
                make_source_metadata(),
            )
        }
        vector_store = FakeVectorStore(
            {"postprocessed_retrieval_chunk": ["chunk_text"]}
        )
        pipeline = VisualizerRetrievalPipeline(
            make_indexer(vector_store, graph_store, docs, settings=settings)
        )

        nodes, triplets, _ = pipeline.visualize(["Text Classification Methods"])

        self.assertIn("concept_text", nodes)
        self.assertIn("concept_multi", nodes)
        self.assertIn("concept_bce", nodes)
        self.assertIn(("concept_multi", "USES", "concept_bce"), triplets)

    def test_grounded_typed_edges_beat_synthetic_fallback_edges(self) -> None:
        settings = KnowledgeGraphSearchSettings(
            analyst_reranker_mode="disabled",
            visualizer_max_edges=1,
            visualizer_max_nodes=2,
            visualizer_max_edges_per_node=3,
        )
        anchor = FakeGraphNode("concept_anchor", "Anchor", source_chunk_ids=["chunk_1"])
        useful = FakeGraphNode(
            "concept_useful", "Useful Method", source_chunk_ids=["chunk_1"]
        )
        detail = FakeGraphNode(
            "concept_detail", "Useful Detail", source_chunk_ids=["chunk_1"]
        )
        generic = FakeGraphNode(
            "concept_generic", "Generic Concept", source_chunk_ids=["chunk_1"]
        )
        chunk = FakeGraphNode("chunk_1", "Chunk", node_label="text_chunk")
        graph_store = FakeGraphStore(
            [anchor, useful, detail, generic, chunk],
            [
                (
                    chunk,
                    FakeRelation("MENTIONS", {"evidence_chunk_ids": ["chunk_1"]}),
                    useful,
                ),
                (
                    chunk,
                    FakeRelation("MENTIONS", {"evidence_chunk_ids": ["chunk_1"]}),
                    generic,
                ),
                (
                    useful,
                    FakeRelation("USES", {"evidence_chunk_ids": ["chunk_1"]}),
                    detail,
                ),
            ],
        )
        docs = {
            "chunk_1": FakeTextNode(
                "chunk_1",
                "Anchor discusses useful method and generic concept.",
                make_source_metadata(),
            )
        }
        vector_store = FakeVectorStore({"postprocessed_retrieval_chunk": ["chunk_1"]})
        pipeline = VisualizerRetrievalPipeline(
            make_indexer(vector_store, graph_store, docs, settings=settings)
        )

        nodes, triplets, _ = pipeline.visualize(["Anchor"])

        self.assertEqual(["concept_useful", "concept_detail"], nodes)
        self.assertEqual([("concept_useful", "USES", "concept_detail")], triplets)

    def test_missing_anchor_returns_empty_tuple_with_original_queries(self) -> None:
        graph_store = FakeGraphStore([FakeGraphNode("concept_clip", "CLIP")])
        pipeline = VisualizerRetrievalPipeline(
            make_indexer(FakeVectorStore(), graph_store)
        )

        nodes, triplets, queries = pipeline.visualize(["Unknown Topic"])

        self.assertEqual([], nodes)
        self.assertEqual([], triplets)
        self.assertEqual(["Unknown Topic"], queries)

    def test_label_dedupe_collapses_exact_labels_without_using_aliases(self) -> None:
        graph_store = FakeGraphStore([])
        pipeline = VisualizerRetrievalPipeline(
            make_indexer(FakeVectorStore(), graph_store)
        )
        method = FakeGraphNode(
            "concept_method",
            "Logistic regression",
            aliases=["Latent Diffusion Model"],
            node_label="METHOD",
        )
        model = FakeGraphNode(
            "concept_model",
            "Latent Diffusion Model",
            aliases=["Logistic regression"],
            node_label="MODEL",
        )
        same_label_other_type = FakeGraphNode(
            "concept_same_label_model",
            "Logistic regression",
            node_label="MODEL",
        )
        concepts = {
            candidate.node_id: candidate
            for candidate in [
                ConceptCandidate("concept_method", method, "Logistic regression", 0.9),
                ConceptCandidate("concept_model", model, "Latent Diffusion Model", 0.8),
                ConceptCandidate(
                    "concept_same_label_model",
                    same_label_other_type,
                    "Logistic regression",
                    0.7,
                ),
            ]
        }
        edges = {
            ("concept_method", "COMPARES_TO", "concept_model"): EdgeCandidate(
                "concept_method",
                "COMPARES_TO",
                "concept_model",
                0.9,
            )
        }

        merged_concepts, remapped_edges = pipeline._dedupe_concepts_by_label(
            concepts, edges
        )

        self.assertEqual({"concept_method", "concept_model"}, set(merged_concepts))
        self.assertEqual(
            ("concept_method", "COMPARES_TO", "concept_model"),
            next(iter(remapped_edges)),
        )

    def test_node_and_edge_budgets_are_enforced(self) -> None:
        settings = KnowledgeGraphSearchSettings(
            analyst_reranker_mode="disabled",
            visualizer_max_nodes=4,
            visualizer_max_edges=2,
            visualizer_max_edges_per_node=3,
            visualizer_min_nodes=1,
        )
        anchor = FakeGraphNode("concept_anchor", "Anchor")
        neighbors = [
            FakeGraphNode(f"concept_{idx}", f"Neighbor {idx}") for idx in range(6)
        ]
        graph_store = FakeGraphStore(
            [anchor, *neighbors],
            [
                (
                    anchor,
                    FakeRelation("USES", {"evidence_chunk_ids": [f"chunk_{idx}"]}),
                    neighbor,
                )
                for idx, neighbor in enumerate(neighbors)
            ],
        )
        pipeline = VisualizerRetrievalPipeline(
            make_indexer(FakeVectorStore(), graph_store, settings=settings)
        )

        nodes, triplets, _ = pipeline.visualize(["Anchor"])

        self.assertLessEqual(len(nodes), 4)
        self.assertLessEqual(len(triplets), 2)


if __name__ == "__main__":
    unittest.main()
