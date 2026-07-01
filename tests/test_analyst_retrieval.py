import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llama_index.core.vector_stores import VectorStoreQueryResult

from backend.configs.search import KnowledgeGraphSearchSettings
from backend.workflows.agents.analyst_retrieval import AnalystRetrievalPipeline, RelationCandidate


class FakeEmbedder:
    def get_query_embedding(self, query: str) -> list[float]:
        return [1.0]


class FakeVectorStore:
    def __init__(self, ids: list[str], similarities: list[float] | None = None) -> None:
        self.ids = ids
        self.similarities = similarities or [0.9 for _ in ids]
        self.queries = []

    def query(self, query):
        self.queries.append(query)
        return VectorStoreQueryResult(ids=self.ids, similarities=self.similarities)


class FakeTextNode:
    def __init__(self, node_id: str, text: str, metadata: dict) -> None:
        self.node_id = node_id
        self.id_ = node_id
        self.text = text
        self.metadata = metadata


class FakeGraphNode:
    def __init__(self, node_id: str, label: str) -> None:
        self.id = node_id
        self.properties = {"entity_name": label}


class FakeRelation:
    def __init__(self, label: str, properties: dict | None = None) -> None:
        self.id = label
        self.label = label
        self.properties = properties or {}


class FakeGraphStore:
    def __init__(self, mention_triplets=None, relation_triplets=None) -> None:
        self.mention_triplets = mention_triplets or []
        self.relation_triplets = relation_triplets or []

    def get_triplets(self, entity_names=None, relation_names=None, properties=None, ids=None):
        triplets = [*self.mention_triplets, *self.relation_triplets]
        if relation_names:
            triplets = [triplet for triplet in triplets if triplet[1].id in relation_names]
        if ids:
            id_set = set(ids)
            triplets = [triplet for triplet in triplets if triplet[0].id in id_set or triplet[2].id in id_set]
        return triplets

    def get_rel_map(self, graph_nodes, depth=1, limit=30, ignore_rels=None):
        ids = {node.id for node in graph_nodes}
        ignore_rels = set(ignore_rels or [])
        triplets = [
            triplet for triplet in self.relation_triplets
            if triplet[1].id not in ignore_rels and (triplet[0].id in ids or triplet[2].id in ids)
        ]
        return triplets[:limit]


class RaisingReranker:
    def postprocess_nodes(self, nodes, query_str: str):
        raise RuntimeError("reranker unavailable")


class EmptyHealthCheckReranker:
    def postprocess_nodes(self, nodes, query_str: str):
        return []


class OrderedScoreReranker:
    def __init__(self, scores_by_id: dict[str, float], top_n: int | None = None) -> None:
        self.scores_by_id = scores_by_id
        self.top_n = top_n

    def postprocess_nodes(self, nodes, query_str: str):
        scored = []
        for node in nodes:
            node_id = node.node.id_
            content = node.node.get_content()
            score = self.scores_by_id.get(node_id, 0.0)
            for key, value in self.scores_by_id.items():
                if key in content:
                    score = max(score, value)
            scored.append(SimpleNamespace(node=node.node, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: self.top_n] if self.top_n is not None else scored


class RecordingReranker:
    def __init__(self) -> None:
        self.calls = []

    def postprocess_nodes(self, nodes, query_str: str):
        self.calls.append((query_str, list(nodes)))
        return list(nodes)


def make_metadata(**overrides):
    metadata = {
        "docstore_node_kind": "postprocessed_retrieval_chunk",
        "chunk_id": overrides.get("chunk_id", "chunk_good"),
        "source": "Text Classification",
        "path": ["Text Classification", "Common datasets"],
        "heading_path": ["Text Classification", "Common datasets"],
        "retrieval_enabled": True,
        "graph_enabled": True,
        "quarantined": False,
        "postprocess_action": "keep",
        "postprocess_chunk_summary": "Common text classification datasets.",
    }
    metadata.update(overrides)
    return metadata


def make_indexer(vector_store, docs, graph_store=None, settings=None):
    storage_context = SimpleNamespace(docstore=SimpleNamespace(docs=docs), vector_store=vector_store)
    index = SimpleNamespace(vector_store=vector_store, property_graph_store=graph_store or FakeGraphStore())
    return SimpleNamespace(
        index=index,
        storage_context=storage_context,
        embedder=FakeEmbedder(),
        kg_search_settings=settings or KnowledgeGraphSearchSettings(analyst_reranker_mode="disabled"),
    )


class AnalystRetrievalPipelineTests(unittest.TestCase):
    def test_source_filtering_excludes_disabled_quarantined_empty_and_non_docstore_nodes(self) -> None:
        docs = {
            "chunk_good": FakeTextNode(
                "chunk_good",
                "Text classification datasets include AG News and DBpedia.",
                make_metadata(chunk_id="chunk_good"),
            ),
            "chunk_disabled": FakeTextNode(
                "chunk_disabled",
                "Disabled source should not appear.",
                make_metadata(chunk_id="chunk_disabled", retrieval_enabled=False),
            ),
            "chunk_quarantined": FakeTextNode(
                "chunk_quarantined",
                "Quarantined source should not appear.",
                make_metadata(chunk_id="chunk_quarantined", quarantined=True),
            ),
            "chunk_empty": FakeTextNode(
                "chunk_empty",
                "",
                make_metadata(chunk_id="chunk_empty"),
            ),
        }
        vector_store = FakeVectorStore(
            ["chunk_good", "chunk_disabled", "chunk_quarantined", "chunk_empty", "concept_clip"],
            [0.8, 0.95, 0.96, 0.97, 0.99],
        )
        pipeline = AnalystRetrievalPipeline(make_indexer(vector_store, docs))

        output = pipeline.search(["text classification datasets"])

        self.assertIn("[SOURCE] [S1]", output)
        self.assertIn("AG News", output)
        self.assertNotIn("Disabled source", output)
        self.assertNotIn("Quarantined source", output)
        self.assertIsNotNone(vector_store.queries[0].filters)

    def test_mentions_link_concepts_but_do_not_leak_as_relations(self) -> None:
        clip = FakeGraphNode("concept_clip", "CLIP")
        objective = FakeGraphNode("concept_objective", "Contrastive Objective")
        source = FakeGraphNode("chunk_clip", "CLIP source")
        graph_store = FakeGraphStore(
            mention_triplets=[(source, FakeRelation("MENTIONS"), clip)],
            relation_triplets=[
                (
                    clip,
                    FakeRelation(
                        "TRAINS",
                        {
                            "evidence_chunk_ids": ["chunk_clip"],
                            "evidence_spans": ["contrastive objective"],
                            "max_confidence": 0.9,
                        },
                    ),
                    objective,
                )
            ],
        )
        docs = {
            "chunk_clip": FakeTextNode(
                "chunk_clip",
                "CLIP trains with a contrastive objective over image-text pairs.",
                make_metadata(
                    chunk_id="chunk_clip",
                    source="CLIP paper",
                    path=["CLIP paper", "Training objective"],
                    heading_path=["CLIP paper", "Training objective"],
                    postprocess_chunk_summary="CLIP contrastive training objective.",
                ),
            )
        }
        pipeline = AnalystRetrievalPipeline(
            make_indexer(FakeVectorStore(["chunk_clip"], [0.72]), docs, graph_store)
        )

        output = pipeline.search(["CLIP training objective"])

        self.assertIn("[SOURCE] [S1]", output)
        self.assertIn("[RELATION] [R1] CLIP -> TRAINS -> Contrastive Objective", output)
        self.assertIn("Evidence: S1", output)
        self.assertNotIn("-> MENTIONS ->", output)

    def test_source_text_strips_redundant_path_prefix_but_keeps_source_path(self) -> None:
        docs = {
            "chunk_dataset": FakeTextNode(
                "chunk_dataset",
                "Text Classification > Common datasets 1. AG News 2. DBpedia.",
                make_metadata(
                    chunk_id="chunk_dataset",
                    path=["Text Classification", "## Common datasets"],
                    heading_path=["Text Classification", "## Common datasets"],
                ),
            )
        }
        pipeline = AnalystRetrievalPipeline(make_indexer(FakeVectorStore(["chunk_dataset"], [0.9]), docs))

        output = pipeline.search(["text classification datasets"])
        source_line = next(line for line in output.splitlines() if line.startswith("[SOURCE] [S1]"))

        self.assertIn("1. AG News", source_line)
        self.assertNotIn("Text Classification > Common datasets", source_line)
        self.assertIn("[SOURCE PATH] [S1] Text Classification > Common datasets", output)

    def test_source_path_display_removes_external_fragments(self) -> None:
        docs = {
            "chunk_external": FakeTextNode(
                "chunk_external",
                "Useful source text.",
                make_metadata(
                    chunk_id="chunk_external",
                    path=["Root", "## Heading", "external:abc123"],
                    heading_path=["Root", "## Heading", "external:abc123"],
                ),
            )
        }
        pipeline = AnalystRetrievalPipeline(make_indexer(FakeVectorStore(["chunk_external"], [0.9]), docs))

        output = pipeline.search(["source"])

        self.assertIn("[SOURCE PATH] [S1] Root > Heading", output)
        self.assertNotIn("external:abc123", output)

    def test_reranker_score_is_primary_and_reorders_sources(self) -> None:
        docs = {
            "chunk_bad": FakeTextNode(
                "chunk_bad",
                "Broad background source.",
                make_metadata(chunk_id="chunk_bad", path=["Root", "Broad"], heading_path=["Root", "Broad"]),
            ),
            "chunk_good": FakeTextNode(
                "chunk_good",
                "Exact answer source.",
                make_metadata(chunk_id="chunk_good", path=["Root", "Exact"], heading_path=["Root", "Exact"]),
            ),
        }
        pipeline = AnalystRetrievalPipeline(
            make_indexer(FakeVectorStore(["chunk_bad", "chunk_good"], [0.99, 0.40]), docs),
            reranker=OrderedScoreReranker({"chunk_bad": 0.2, "chunk_good": 0.8}),
        )

        output = pipeline.search(["exact answer"])
        first_source = next(line for line in output.splitlines() if line.startswith("[SOURCE] [S1]"))

        self.assertIn("chunk_good", first_source)
        self.assertIn("Score: 0.80", first_source)

    def test_unbounded_reranker_scores_are_calibrated_without_sigmoid_saturation(self) -> None:
        docs = {
            "chunk_top": FakeTextNode(
                "chunk_top",
                "Best source for exact answer.",
                make_metadata(chunk_id="chunk_top", path=["Root", "Top"], heading_path=["Root", "Top"]),
            ),
            "chunk_close": FakeTextNode(
                "chunk_close",
                "Close supporting source for exact answer.",
                make_metadata(chunk_id="chunk_close", path=["Root", "Close"], heading_path=["Root", "Close"]),
            ),
            "chunk_tail": FakeTextNode(
                "chunk_tail",
                "Weak tail source.",
                make_metadata(chunk_id="chunk_tail", path=["Root", "Tail"], heading_path=["Root", "Tail"]),
            ),
        }
        settings = KnowledgeGraphSearchSettings(
            analyst_reranker_mode="disabled",
            analyst_source_min_raw_margin=4.0,
        )
        pipeline = AnalystRetrievalPipeline(
            make_indexer(
                FakeVectorStore(["chunk_tail", "chunk_close", "chunk_top"], [0.95, 0.9, 0.85]),
                docs,
                settings=settings,
            ),
            settings=settings,
            reranker=OrderedScoreReranker({"chunk_top": 10.0, "chunk_close": 8.0, "chunk_tail": 0.0}),
        )

        output = pipeline.search(["exact answer"])

        self.assertIn("chunk_top", output)
        self.assertIn("chunk_close", output)
        self.assertNotIn("chunk_tail", output)
        self.assertIn("Score: 1.00", output)
        self.assertIn("Score: 0.73", output)

    def test_source_pruning_drops_reranked_tail_outside_margin(self) -> None:
        docs = {
            "chunk_top": FakeTextNode(
                "chunk_top",
                "Focused answer source.",
                make_metadata(chunk_id="chunk_top", path=["Root", "Top"], heading_path=["Root", "Top"]),
            ),
            "chunk_tail": FakeTextNode(
                "chunk_tail",
                "Off-topic tail source.",
                make_metadata(chunk_id="chunk_tail", path=["Root", "Tail"], heading_path=["Root", "Tail"]),
            ),
        }
        settings = KnowledgeGraphSearchSettings(
            analyst_reranker_mode="disabled",
            analyst_source_final_k=5,
            analyst_source_min_raw_margin=2.0,
        )
        pipeline = AnalystRetrievalPipeline(
            make_indexer(FakeVectorStore(["chunk_tail", "chunk_top"], [0.99, 0.50]), docs, settings=settings),
            settings=settings,
            reranker=OrderedScoreReranker({"chunk_top": 9.0, "chunk_tail": 3.0}),
        )

        output = pipeline.search(["focused answer"])

        self.assertIn("Focused answer source", output)
        self.assertNotIn("Off-topic tail source", output)

    def test_minimum_source_fill_prefers_new_paths_after_reranking(self) -> None:
        docs = {
            "chunk_top": FakeTextNode(
                "chunk_top",
                "Best reranked source.",
                make_metadata(chunk_id="chunk_top", path=["Root", "Same"], heading_path=["Root", "Same"]),
            ),
            "chunk_same": FakeTextNode(
                "chunk_same",
                "Same path fallback source.",
                make_metadata(chunk_id="chunk_same", path=["Root", "Same"], heading_path=["Root", "Same"]),
            ),
            "chunk_other_a": FakeTextNode(
                "chunk_other_a",
                "First diverse fallback source.",
                make_metadata(chunk_id="chunk_other_a", path=["Root", "Other A"], heading_path=["Root", "Other A"]),
            ),
            "chunk_other_b": FakeTextNode(
                "chunk_other_b",
                "Second diverse fallback source.",
                make_metadata(chunk_id="chunk_other_b", path=["Root", "Other B"], heading_path=["Root", "Other B"]),
            ),
        }
        settings = KnowledgeGraphSearchSettings(
            analyst_reranker_mode="disabled",
            analyst_source_min_keep=3,
            analyst_source_final_k=6,
        )
        pipeline = AnalystRetrievalPipeline(
            make_indexer(
                FakeVectorStore(
                    ["chunk_top", "chunk_same", "chunk_other_a", "chunk_other_b"],
                    [0.99, 0.98, 0.97, 0.96],
                ),
                docs,
                settings=settings,
            ),
            settings=settings,
            reranker=OrderedScoreReranker({"chunk_top": 10.0}, top_n=1),
        )

        output = pipeline.search(["best source"])

        self.assertIn("Best reranked source", output)
        self.assertIn("First diverse fallback source", output)
        self.assertIn("Second diverse fallback source", output)
        self.assertNotIn("Same path fallback source", output)

    def test_minimum_source_fill_does_not_force_weak_fallback_sources(self) -> None:
        docs = {
            "chunk_top": FakeTextNode(
                "chunk_top",
                "Strong focused source.",
                make_metadata(chunk_id="chunk_top", path=["Root", "Top"], heading_path=["Root", "Top"]),
            ),
            "chunk_weak_a": FakeTextNode(
                "chunk_weak_a",
                "Weak diverse fallback A.",
                make_metadata(chunk_id="chunk_weak_a", path=["Root", "Weak A"], heading_path=["Root", "Weak A"]),
            ),
            "chunk_weak_b": FakeTextNode(
                "chunk_weak_b",
                "Weak diverse fallback B.",
                make_metadata(chunk_id="chunk_weak_b", path=["Root", "Weak B"], heading_path=["Root", "Weak B"]),
            ),
        }
        settings = KnowledgeGraphSearchSettings(
            analyst_reranker_mode="disabled",
            analyst_source_min_keep=3,
            analyst_source_fill_min_score=0.50,
        )
        pipeline = AnalystRetrievalPipeline(
            make_indexer(
                FakeVectorStore(["chunk_top", "chunk_weak_a", "chunk_weak_b"], [0.99, 0.35, 0.34]),
                docs,
                settings=settings,
            ),
            settings=settings,
            reranker=OrderedScoreReranker({"chunk_top": 10.0}, top_n=1),
        )

        output = pipeline.search(["focused source"])

        self.assertIn("Strong focused source", output)
        self.assertNotIn("Weak diverse fallback A", output)
        self.assertNotIn("Weak diverse fallback B", output)

    def test_exact_topic_match_uses_phrase_boundaries(self) -> None:
        pipeline = AnalystRetrievalPipeline(make_indexer(FakeVectorStore([]), {}))

        self.assertTrue(pipeline._has_exact_topic_match("Fast Text", "Models > Fast Text > Notes"))
        self.assertTrue(pipeline._has_exact_topic_match("BERT", "BERT architecture"))
        self.assertFalse(pipeline._has_exact_topic_match("BERT", "ModernBERT architecture"))
        self.assertFalse(pipeline._has_exact_topic_match("CLIP", "image clipping utilities"))

    def test_short_query_terms_use_phrase_boundaries(self) -> None:
        pipeline = AnalystRetrievalPipeline(make_indexer(FakeVectorStore([]), {}))

        self.assertTrue(pipeline._has_term_overlap({"bert"}, "BERT architecture"))
        self.assertFalse(pipeline._has_term_overlap({"bert"}, "ModernBERT architecture"))
        self.assertFalse(pipeline._has_term_overlap({"clip"}, "image clipping utilities"))
        self.assertTrue(pipeline._has_term_overlap({"diffusion"}, "diffusions and score models"))

    def test_source_reranker_uses_limited_candidate_pool(self) -> None:
        docs = {
            f"chunk_{idx}": FakeTextNode(
                f"chunk_{idx}",
                f"Candidate source {idx}.",
                make_metadata(chunk_id=f"chunk_{idx}", path=["Root", f"Chunk {idx}"], heading_path=["Root", f"Chunk {idx}"]),
            )
            for idx in range(12)
        }
        settings = KnowledgeGraphSearchSettings(
            analyst_reranker_mode="disabled",
            analyst_source_candidate_k=12,
            analyst_source_rerank_candidate_k=3,
        )
        reranker = RecordingReranker()
        pipeline = AnalystRetrievalPipeline(
            make_indexer(
                FakeVectorStore([f"chunk_{idx}" for idx in range(12)], [0.99 - (idx * 0.01) for idx in range(12)]),
                docs,
                settings=settings,
            ),
            settings=settings,
            reranker=reranker,
        )

        candidates = pipeline._dedupe_sources(pipeline._retrieve_source_candidates("candidate source"))
        reranked = pipeline._rerank_sources("candidate source", candidates)

        self.assertEqual(1, len(reranker.calls))
        self.assertEqual(3, len(reranker.calls[0][1]))
        self.assertEqual(12, len(reranked))
        self.assertEqual(["chunk_0", "chunk_1", "chunk_2"], [node.node.id_ for node in reranker.calls[0][1]])

    def test_source_rerank_text_uses_query_focused_capped_excerpt(self) -> None:
        long_prefix = "Irrelevant background sentence. " * 50
        long_suffix = "Trailing unrelated sentence. " * 50
        docs = {
            "chunk_long": FakeTextNode(
                "chunk_long",
                f"{long_prefix}The FastText model uses subword information for out-of-vocabulary words. {long_suffix}",
                make_metadata(
                    chunk_id="chunk_long",
                    path=["FastText", "Subword model"],
                    heading_path=["FastText", "Subword model"],
                    postprocess_chunk_summary="FastText subword information.",
                ),
            )
        }
        settings = KnowledgeGraphSearchSettings(
            analyst_reranker_mode="disabled",
            analyst_source_rerank_max_chars=220,
        )
        pipeline = AnalystRetrievalPipeline(make_indexer(FakeVectorStore(["chunk_long"], [0.9]), docs, settings=settings), settings)
        candidate = pipeline._retrieve_source_candidates("FastText subword")[0]

        text = pipeline._rerank_text(candidate, query="FastText subword")

        self.assertLessEqual(len(text), 220)
        self.assertIn("FastText", text)
        self.assertIn("subword", text)
        self.assertNotIn("Trailing unrelated sentence. Trailing unrelated sentence.", text)

    def test_duplicate_chunks_and_paths_are_collapsed_deterministically(self) -> None:
        docs = {
            f"chunk_{idx}": FakeTextNode(
                f"chunk_{idx}",
                f"Duplicate path source {idx}.",
                make_metadata(chunk_id=f"chunk_{idx}", path=["Root", "Same"], heading_path=["Root", "Same"]),
            )
            for idx in range(3)
        }
        vector_store = FakeVectorStore(["chunk_0", "chunk_0", "chunk_1", "chunk_2"], [0.9, 0.89, 0.88, 0.87])
        pipeline = AnalystRetrievalPipeline(make_indexer(vector_store, docs))

        output = pipeline.search(["duplicate"])

        self.assertIn("Duplicate path source 0", output)
        self.assertIn("Duplicate path source 1", output)
        self.assertNotIn("Duplicate path source 2", output)

    def test_path_diversity_cap_is_applied_after_reranking(self) -> None:
        docs = {
            "chunk_0": FakeTextNode(
                "chunk_0",
                "Same path weakest source.",
                make_metadata(chunk_id="chunk_0", path=["Root", "Same"], heading_path=["Root", "Same"]),
            ),
            "chunk_1": FakeTextNode(
                "chunk_1",
                "Same path strongest source.",
                make_metadata(chunk_id="chunk_1", path=["Root", "Same"], heading_path=["Root", "Same"]),
            ),
            "chunk_2": FakeTextNode(
                "chunk_2",
                "Same path second strongest source.",
                make_metadata(chunk_id="chunk_2", path=["Root", "Same"], heading_path=["Root", "Same"]),
            ),
        }
        settings = KnowledgeGraphSearchSettings(
            analyst_reranker_mode="disabled",
            analyst_source_max_per_path=2,
        )
        pipeline = AnalystRetrievalPipeline(
            make_indexer(
                FakeVectorStore(["chunk_0", "chunk_1", "chunk_2"], [0.99, 0.50, 0.49]),
                docs,
                settings=settings,
            ),
            settings=settings,
            reranker=OrderedScoreReranker({"chunk_0": 1.0, "chunk_1": 10.0, "chunk_2": 9.0}),
        )

        output = pipeline.search(["same path"])

        self.assertIn("Same path strongest source", output)
        self.assertIn("Same path second strongest source", output)
        self.assertNotIn("Same path weakest source", output)

    def test_context_budget_is_enforced(self) -> None:
        docs = {
            "chunk_long": FakeTextNode(
                "chunk_long",
                "Long source. " * 200,
                make_metadata(chunk_id="chunk_long"),
            )
        }
        settings = KnowledgeGraphSearchSettings(analyst_context_max_chars=350, analyst_reranker_mode="disabled")
        pipeline = AnalystRetrievalPipeline(make_indexer(FakeVectorStore(["chunk_long"], [0.9]), docs), settings)

        output = pipeline.search(["long source"])

        self.assertLessEqual(len(output), 350)
        self.assertIn("RETRIEVER RESULTS", output)

    def test_ungrounded_relations_are_dropped_by_default(self) -> None:
        clip = FakeGraphNode("concept_clip", "CLIP")
        objective = FakeGraphNode("concept_objective", "Contrastive Objective")
        unrelated = FakeGraphNode("concept_unrelated", "Unrelated")
        source = FakeGraphNode("chunk_clip", "CLIP source")
        graph_store = FakeGraphStore(
            mention_triplets=[(source, FakeRelation("MENTIONS"), clip)],
            relation_triplets=[
                (
                    clip,
                    FakeRelation(
                        "TRAINS",
                        {
                            "evidence_chunk_ids": ["chunk_not_returned"],
                            "evidence_spans": ["contrastive objective"],
                            "max_confidence": 0.9,
                        },
                    ),
                    objective,
                ),
                (
                    clip,
                    FakeRelation(
                        "USES",
                        {
                            "evidence_chunk_ids": ["chunk_clip"],
                            "evidence_spans": ["natural language supervision"],
                            "max_confidence": 0.8,
                        },
                    ),
                    unrelated,
                ),
            ],
        )
        docs = {
            "chunk_clip": FakeTextNode(
                "chunk_clip",
                "CLIP uses natural language supervision.",
                make_metadata(
                    chunk_id="chunk_clip",
                    source="CLIP paper",
                    path=["CLIP paper", "Training objective"],
                    heading_path=["CLIP paper", "Training objective"],
                ),
            )
        }
        pipeline = AnalystRetrievalPipeline(
            make_indexer(FakeVectorStore(["chunk_clip"], [0.72]), docs, graph_store)
        )

        output = pipeline.search(["CLIP training objective"])

        self.assertIn("CLIP -> USES -> Unrelated", output)
        self.assertNotIn("CLIP -> TRAINS -> Contrastive Objective", output)
        self.assertNotIn("Evidence span:", output)

    def test_relation_reranker_orders_grounded_relations(self) -> None:
        clip = FakeGraphNode("concept_clip", "CLIP")
        objective = FakeGraphNode("concept_objective", "Contrastive Objective")
        supervision = FakeGraphNode("concept_supervision", "Natural Language Supervision")
        source = FakeGraphNode("chunk_clip", "CLIP source")
        graph_store = FakeGraphStore(
            mention_triplets=[(source, FakeRelation("MENTIONS"), clip)],
            relation_triplets=[
                (
                    clip,
                    FakeRelation("USES", {"evidence_chunk_ids": ["chunk_clip"], "max_confidence": 0.9}),
                    supervision,
                ),
                (
                    clip,
                    FakeRelation("TRAINS", {"evidence_chunk_ids": ["chunk_clip"], "max_confidence": 0.9}),
                    objective,
                ),
            ],
        )
        docs = {
            "chunk_clip": FakeTextNode(
                "chunk_clip",
                "CLIP trains with a contrastive objective and uses natural language supervision.",
                make_metadata(chunk_id="chunk_clip", source="CLIP paper"),
            )
        }
        settings = KnowledgeGraphSearchSettings(
            analyst_reranker_mode="disabled",
            analyst_relation_reranker_enabled=True,
        )
        pipeline = AnalystRetrievalPipeline(
            make_indexer(FakeVectorStore(["chunk_clip"], [0.72]), docs, graph_store, settings=settings),
            settings=settings,
            reranker=OrderedScoreReranker(
                {
                    "chunk_clip": 0.9,
                    "USES": 0.7,
                    "TRAINS": 0.8,
                }
            ),
        )

        output = pipeline.search(["CLIP training objective"])

        self.assertLess(output.find("CLIP -> TRAINS"), output.find("CLIP -> USES"))

    def test_relation_rerank_text_includes_family_phrase_and_source_evidence(self) -> None:
        pipeline = AnalystRetrievalPipeline(make_indexer(FakeVectorStore([]), {}))
        relation = RelationCandidate(
            relation_id="rel_1",
            subject="Backpropagation",
            predicate="PRODUCES",
            object="Gradient",
            score=0.7,
            rank=1,
            predicate_family="computation",
            relation_phrases=["produces gradients for each layer"],
            evidence_chunk_ids=["chunk_1"],
            evidence_spans=["Backpropagation produces gradients"],
        )

        text = pipeline._relation_rerank_text(relation, {"chunk_1": "Source text about training gradients."})

        self.assertIn("Backpropagation PRODUCES Gradient", text)
        self.assertIn("Family: computation", text)
        self.assertIn("Relation phrase: produces gradients for each layer", text)
        self.assertIn("Source text about training gradients.", text)

    def test_relation_reranking_is_disabled_by_default(self) -> None:
        clip = FakeGraphNode("concept_clip", "CLIP")
        objective = FakeGraphNode("concept_objective", "Contrastive Objective")
        source = FakeGraphNode("chunk_clip", "CLIP source")
        graph_store = FakeGraphStore(
            mention_triplets=[(source, FakeRelation("MENTIONS"), clip)],
            relation_triplets=[
                (
                    clip,
                    FakeRelation("TRAINS", {"evidence_chunk_ids": ["chunk_clip"], "max_confidence": 0.9}),
                    objective,
                )
            ],
        )
        docs = {
            "chunk_clip": FakeTextNode(
                "chunk_clip",
                "CLIP trains with a contrastive objective.",
                make_metadata(chunk_id="chunk_clip", source="CLIP paper"),
            )
        }
        reranker = RecordingReranker()
        pipeline = AnalystRetrievalPipeline(
            make_indexer(FakeVectorStore(["chunk_clip"], [0.72]), docs, graph_store),
            reranker=reranker,
        )

        output = pipeline.search(["CLIP training objective"])

        self.assertIn("CLIP -> TRAINS -> Contrastive Objective", output)
        self.assertEqual(1, len(reranker.calls))

    def test_relation_reranker_uses_limited_capped_candidates_when_enabled(self) -> None:
        clip = FakeGraphNode("concept_clip", "CLIP")
        source = FakeGraphNode("chunk_clip", "CLIP source")
        relation_triplets = [
            (
                clip,
                FakeRelation(
                    f"REL_{idx}",
                    {
                        "evidence_chunk_ids": ["chunk_clip"],
                        "evidence_spans": [f"evidence span {idx}"],
                        "max_confidence": 0.9 - (idx * 0.01),
                    },
                ),
                FakeGraphNode(f"concept_{idx}", f"Concept {idx}"),
            )
            for idx in range(6)
        ]
        graph_store = FakeGraphStore(
            mention_triplets=[(source, FakeRelation("MENTIONS"), clip)],
            relation_triplets=relation_triplets,
        )
        docs = {
            "chunk_clip": FakeTextNode(
                "chunk_clip",
                "CLIP relation evidence. " * 200,
                make_metadata(chunk_id="chunk_clip", source="CLIP paper"),
            )
        }
        settings = KnowledgeGraphSearchSettings(
            analyst_reranker_mode="disabled",
            analyst_relation_reranker_enabled=True,
            analyst_relation_rerank_candidate_k=2,
            analyst_relation_rerank_max_chars=180,
        )
        reranker = RecordingReranker()
        pipeline = AnalystRetrievalPipeline(
            make_indexer(FakeVectorStore(["chunk_clip"], [0.72]), docs, graph_store, settings=settings),
            settings=settings,
            reranker=reranker,
        )

        pipeline.search(["CLIP relation"])

        self.assertEqual(2, len(reranker.calls))
        relation_nodes = reranker.calls[1][1]
        self.assertEqual(2, len(relation_nodes))
        self.assertTrue(all(len(node.node.get_content()) <= 180 for node in relation_nodes))

    def test_retrieval_usefulness_affects_relation_order_without_bypassing_grounding(self) -> None:
        clip = FakeGraphNode("concept_clip", "CLIP")
        useful = FakeGraphNode("concept_useful", "Contrastive Objective")
        generic = FakeGraphNode("concept_generic", "Natural Language Supervision")
        source = FakeGraphNode("chunk_clip", "CLIP source")
        graph_store = FakeGraphStore(
            mention_triplets=[(source, FakeRelation("MENTIONS"), clip)],
            relation_triplets=[
                (
                    clip,
                    FakeRelation(
                        "RELATED_TO",
                        {
                            "predicate_family": "other",
                            "relation_phrases": ["is broadly associated with supervision"],
                            "evidence_chunk_ids": ["chunk_clip"],
                            "max_confidence": 0.9,
                            "max_retrieval_usefulness": 0.1,
                        },
                    ),
                    generic,
                ),
                (
                    clip,
                    FakeRelation(
                        "TRAINS",
                        {
                            "predicate_family": "training",
                            "relation_phrases": ["trains with a contrastive objective"],
                            "evidence_chunk_ids": ["chunk_clip"],
                            "max_confidence": 0.9,
                            "max_retrieval_usefulness": 1.0,
                        },
                    ),
                    useful,
                ),
            ],
        )
        docs = {
            "chunk_clip": FakeTextNode(
                "chunk_clip",
                "CLIP trains with a contrastive objective and uses natural language supervision.",
                make_metadata(chunk_id="chunk_clip", source="CLIP paper"),
            )
        }
        pipeline = AnalystRetrievalPipeline(make_indexer(FakeVectorStore(["chunk_clip"], [0.72]), docs, graph_store))

        output = pipeline.search(["CLIP training objective"])

        self.assertLess(output.find("CLIP -> TRAINS"), output.find("CLIP -> RELATED_TO"))

    def test_relation_reranker_drops_low_relative_relation(self) -> None:
        clip = FakeGraphNode("concept_clip", "CLIP")
        objective = FakeGraphNode("concept_objective", "Contrastive Objective")
        unrelated = FakeGraphNode("concept_unrelated", "Unrelated Mechanism")
        source = FakeGraphNode("chunk_clip", "CLIP source")
        graph_store = FakeGraphStore(
            mention_triplets=[(source, FakeRelation("MENTIONS"), clip)],
            relation_triplets=[
                (
                    clip,
                    FakeRelation("TRAINS", {"evidence_chunk_ids": ["chunk_clip"], "max_confidence": 0.9}),
                    objective,
                ),
                (
                    clip,
                    FakeRelation("USES", {"evidence_chunk_ids": ["chunk_clip"], "max_confidence": 0.9}),
                    unrelated,
                ),
            ],
        )
        docs = {
            "chunk_clip": FakeTextNode(
                "chunk_clip",
                "CLIP trains with a contrastive objective.",
                make_metadata(chunk_id="chunk_clip", source="CLIP paper"),
            )
        }
        settings = KnowledgeGraphSearchSettings(
            analyst_reranker_mode="disabled",
            analyst_relation_reranker_enabled=True,
        )
        pipeline = AnalystRetrievalPipeline(
            make_indexer(FakeVectorStore(["chunk_clip"], [0.72]), docs, graph_store, settings=settings),
            settings=settings,
            reranker=OrderedScoreReranker(
                {
                    "chunk_clip": 0.9,
                    "TRAINS": 0.8,
                    "USES": 0.19,
                }
            ),
        )

        output = pipeline.search(["CLIP training objective"])

        self.assertIn("CLIP -> TRAINS -> Contrastive Objective", output)
        self.assertNotIn("CLIP -> USES -> Unrelated Mechanism", output)

    def test_reranker_exception_preserves_deterministic_results(self) -> None:
        docs = {
            "chunk_good": FakeTextNode(
                "chunk_good",
                "Reranker fallback source remains available.",
                make_metadata(chunk_id="chunk_good"),
            )
        }
        pipeline = AnalystRetrievalPipeline(
            make_indexer(FakeVectorStore(["chunk_good"], [0.9]), docs),
            reranker=RaisingReranker(),
        )

        output = pipeline.search(["fallback"])

        self.assertIn("Reranker fallback source", output)
        self.assertTrue(pipeline.health_report.has_code("analyst_reranker_runtime_failed"))

    def test_failed_ollama_reranker_health_check_disables_reranker(self) -> None:
        settings = KnowledgeGraphSearchSettings(analyst_reranker_mode="ollama_llm_rerank")
        indexer = make_indexer(FakeVectorStore([]), {})

        with (
            patch("llama_index.llms.ollama.Ollama", return_value=object()),
            patch("llama_index.core.postprocessor.LLMRerank", return_value=EmptyHealthCheckReranker()),
        ):
            pipeline = AnalystRetrievalPipeline(indexer, settings)

        self.assertIsNone(pipeline.reranker)

    def test_failed_sentence_transformer_reranker_initialization_disables_reranker(self) -> None:
        settings = KnowledgeGraphSearchSettings(analyst_reranker_mode="sentence_transformers")
        indexer = make_indexer(FakeVectorStore([]), {}, settings=settings)

        with patch("sentence_transformers.CrossEncoder", side_effect=RuntimeError("missing model")):
            pipeline = AnalystRetrievalPipeline(indexer, settings)

        self.assertIsNone(pipeline.reranker)


if __name__ == "__main__":
    unittest.main()
