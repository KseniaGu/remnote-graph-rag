from synthetic_analyst import (
    FakeTextNode,
    FakeVectorStore,
    RaisingReranker,
    RecordingReranker,
    make_indexer,
    make_metadata,
)

from backend.configs.search import KnowledgeGraphSearchSettings
from backend.workflows.agents.analyst_retrieval import AnalystRetrievalPipeline


def _pipeline(*, reranker=None, mode="disabled", source_only=False):
    node = FakeTextNode("chunk_good", "Adam uses adaptive moments.", make_metadata())
    settings = KnowledgeGraphSearchSettings(analyst_reranker_mode=mode)
    return AnalystRetrievalPipeline(
        make_indexer(
            FakeVectorStore(["chunk_good"]), {"chunk_good": node}, settings=settings
        ),
        reranker=reranker,
        source_only=source_only,
    )


def test_source_only_observation_counts_real_calls_and_no_reranker():
    pipeline = _pipeline(source_only=True)
    pipeline.start_observation()
    result = pipeline._search_one("Adam")
    counts = pipeline.finish_observation()
    assert result["diagnostics"]["candidate_count"] == 1
    assert result["diagnostics"]["final_count"] == 1
    assert counts == {
        "embedding_operations": 1,
        "reranking_operations": 0,
        "reranking_failures": 0,
        "reranking_applied_ids": 0,
        "reranking_incomplete_operations": 0,
        "requested_reranker_mode": "disabled",
        "effective_reranker_mode": "disabled",
    }


def test_injected_reranker_counts_success_and_failure_without_hiding_fallback():
    good = _pipeline(reranker=RecordingReranker(), mode="sentence_transformers")
    good.start_observation()
    good._search_one("Adam")
    counts = good.finish_observation()
    assert counts["embedding_operations"] == 1
    assert counts["reranking_operations"] == 1
    assert counts["reranking_failures"] == 0
    assert counts["effective_reranker_mode"] == "sentence_transformers"

    injected = _pipeline(reranker=RecordingReranker(), mode="disabled")
    injected.start_observation()
    injected._search_one("Adam")
    counts = injected.finish_observation()
    assert counts["requested_reranker_mode"] == "disabled"
    assert counts["reranking_operations"] == 1
    assert counts["effective_reranker_mode"] == "injected"

    bad = _pipeline(reranker=RaisingReranker(), mode="sentence_transformers")
    bad.start_observation()
    assert bad._search_one("Adam")["sources"]
    counts = bad.finish_observation()
    assert counts["reranking_operations"] == 1
    assert counts["reranking_failures"] == 1
    assert counts["effective_reranker_mode"] == "fallback"


def test_empty_and_failed_embedding_calls_have_observed_counts():
    empty = _pipeline(reranker=RecordingReranker(), mode="sentence_transformers")
    empty.vector_store = FakeVectorStore([])
    empty.access.vector_store = empty.vector_store
    empty.start_observation()
    assert not empty._search_one("Unknown")["sources"]
    counts = empty.finish_observation()
    assert counts["embedding_operations"] == 1
    assert counts["reranking_operations"] == 0
    assert counts["effective_reranker_mode"] == "not_invoked"

    failed = _pipeline(source_only=True)

    class BrokenEmbedder:
        def get_query_embedding(self, query):
            raise RuntimeError("offline fixture failure")

    failed.embedder = BrokenEmbedder()
    failed.start_observation()
    try:
        failed._search_one("Adam")
    except RuntimeError:
        pass
    else:
        raise AssertionError("failed embedding must propagate")
    counts = failed.finish_observation()
    assert counts["embedding_operations"] == 1
    assert counts["reranking_operations"] == 0


def test_context_counts_match_actual_rendering_and_do_not_retrieve_again():
    pipeline = _pipeline(source_only=True)
    raw = [pipeline._search_one("Adam"), pipeline._search_one("Adam again")]
    pipeline.start_observation()
    measured = pipeline.observe_context(raw)
    assert pipeline.finish_observation()["embedding_operations"] == 0
    actual = pipeline.search(["Adam", "Adam again"])
    assert measured["context_source_count"] == sum(
        line.startswith("[SOURCE] ") for line in actual.splitlines()
    )
    assert measured["context_relation_count"] == sum(
        line.startswith("[RELATION] ") for line in actual.splitlines()
    )
    assert (
        measured["context_count"]
        == measured["context_source_count"] + measured["context_relation_count"]
    )
    assert measured["context_characters"] == len(actual)
    assert measured["context_counts_by_query"] == [1, 0]


def test_context_count_honors_render_budget():
    pipeline = _pipeline(source_only=True)
    pipeline.settings.analyst_context_max_chars = 30
    raw = [pipeline._search_one("Adam")]
    measured = pipeline.observe_context(raw)
    actual = pipeline.search(["Adam"])
    assert measured["context_source_count"] == sum(
        line.startswith("[SOURCE] ") for line in actual.splitlines()
    )
    assert measured["context_characters"] == len(actual)


def test_reranker_empty_unmatched_and_partial_outputs_are_not_full_success():
    from llama_index.core.schema import NodeWithScore, TextNode

    from backend.workflows.agents.analyst_retrieval import RelationCandidate

    class OutputReranker:
        def __init__(self, mode):
            self.mode = mode

        def postprocess_nodes(self, nodes, query_str):
            if self.mode == "empty":
                return []
            if self.mode == "unmatched":
                return [
                    NodeWithScore(node=TextNode(id_="unmatched", text="x"), score=1)
                ]
            return nodes[:1]

    for mode, expected, applied in (
        ("empty", "fallback", 0),
        ("unmatched", "fallback", 0),
        ("partial", "partial", 1),
    ):
        pipeline = _pipeline(
            reranker=OutputReranker(mode), mode="sentence_transformers"
        )
        candidates = pipeline._retrieve_source_candidates("Adam")
        second = pipeline._retrieve_source_candidates("Adam")[0]
        second.node_id = "another"
        candidates.append(second)
        pipeline.start_observation()
        result = pipeline._rerank_sources("Adam", candidates)
        counts = pipeline.finish_observation()
        assert sum(item.reranked for item in result) == applied
        assert counts["reranking_applied_ids"] == applied
        assert counts["reranking_incomplete_operations"] == 1
        assert counts["effective_reranker_mode"] == expected

        pipeline.settings.analyst_relation_reranker_enabled = True
        relations = [
            RelationCandidate(str(i), "a", "rel", "b", 0.5, i) for i in range(2)
        ]
        pipeline.start_observation()
        result = pipeline._rerank_relations("Adam", candidates, relations)
        counts = pipeline.finish_observation()
        assert sum(item.reranked for item in result) == applied
        assert counts["reranking_applied_ids"] == applied
        assert counts["reranking_incomplete_operations"] == 1
        assert counts["effective_reranker_mode"] == expected


def test_unsubmitted_candidate_ids_do_not_count_as_reranker_applied():
    from llama_index.core.schema import NodeWithScore, TextNode

    from backend.workflows.agents.analyst_retrieval import RelationCandidate

    class ReturnsUnsubmitted:
        def __init__(self, unsubmitted_id, submitted_count):
            self.unsubmitted_id = unsubmitted_id
            self.submitted_count = submitted_count

        def postprocess_nodes(self, nodes, query_str):
            return [
                *nodes[: self.submitted_count],
                NodeWithScore(
                    node=TextNode(id_=self.unsubmitted_id, text="x"), score=1
                ),
            ]

    for submitted_count, expected in (
        (0, "fallback"),
        (1, "partial"),
        (2, "sentence_transformers"),
    ):
        pipeline = _pipeline(reranker=RecordingReranker(), mode="sentence_transformers")
        pipeline.settings.analyst_source_rerank_candidate_k = 2
        pipeline.settings.analyst_relation_rerank_candidate_k = 2
        pipeline.settings.analyst_relation_reranker_enabled = True
        candidates = []
        for idx in range(3):
            candidate = pipeline._retrieve_source_candidates("Adam")[0]
            candidate.node_id = str(idx)
            candidate.rank = idx
            candidates.append(candidate)
        excluded = pipeline._sort_sources(candidates)[2].node_id
        pipeline.reranker = ReturnsUnsubmitted(excluded, submitted_count)
        pipeline.start_observation()
        pipeline._rerank_sources("Adam", candidates)
        counts = pipeline.finish_observation()
        assert counts["reranking_applied_ids"] == submitted_count
        assert counts["effective_reranker_mode"] == expected

        relations = [
            RelationCandidate(str(i), "a", "rel", "b", 0.5, i) for i in range(3)
        ]
        excluded = pipeline._sort_relations(relations)[2].relation_id
        pipeline.reranker = ReturnsUnsubmitted(excluded, submitted_count)
        pipeline.start_observation()
        pipeline._rerank_relations("Adam", candidates, relations)
        counts = pipeline.finish_observation()
        assert counts["reranking_applied_ids"] == submitted_count
        assert counts["effective_reranker_mode"] == expected
