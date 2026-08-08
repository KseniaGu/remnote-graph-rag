import unittest
from types import SimpleNamespace

from llama_index.core.vector_stores import VectorStoreQueryResult

from backend.workflows.agents.retrieval_access import RetrievalStoreAccess


class FakeVectorStore:
    def __init__(self, *, fail_filtered: bool = False, fail_all: bool = False) -> None:
        self.fail_filtered = fail_filtered
        self.fail_all = fail_all
        self.queries = []

    def query(self, query):
        self.queries.append(query)
        if self.fail_all or (self.fail_filtered and getattr(query, "filters", None)):
            raise RuntimeError("vector query failed")
        return VectorStoreQueryResult(ids=["chunk_1"], similarities=[0.9])


class FakeGraphNode:
    def __init__(
        self, node_id: str, label: str, *, node_label: str = "CONCEPT"
    ) -> None:
        self.id = node_id
        self.label = node_label
        self.properties = {
            "entity_name": label,
            "aliases": ["alias"],
            "source_chunk_ids": ["chunk_1"],
            "postprocess_max_salience": 0.75,
        }


class FakeRelation:
    def __init__(self, label: str, properties: dict | None = None) -> None:
        self.id = label
        self.label = label
        self.properties = properties or {}


class FakeGraphStore:
    def __init__(
        self,
        *,
        nodes: list[FakeGraphNode] | None = None,
        triplets: list[tuple] | None = None,
        fail_rel_map: bool = False,
        fail_triplets: bool = False,
    ) -> None:
        self.nodes = {node.id: node for node in nodes or []}
        self.triplets = triplets or []
        self.fail_rel_map = fail_rel_map
        self.fail_triplets = fail_triplets

    def get(self, properties=None, ids=None):
        if ids:
            return [self.nodes[node_id] for node_id in ids if node_id in self.nodes]
        return list(self.nodes.values())

    def get_rel_map(self, graph_nodes, depth=1, limit=30, ignore_rels=None):
        if self.fail_rel_map:
            raise RuntimeError("relation map failed")
        ids = {node.id for node in graph_nodes}
        denied = set(ignore_rels or [])
        return [
            triplet
            for triplet in self.triplets
            if triplet[1].label not in denied
            and (triplet[0].id in ids or triplet[2].id in ids)
        ][:limit]

    def get_triplets(
        self, entity_names=None, relation_names=None, properties=None, ids=None
    ):
        if self.fail_triplets:
            raise RuntimeError("triplets failed")
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


def make_access(vector_store=None, graph_store=None, docs=None) -> RetrievalStoreAccess:
    storage_context = SimpleNamespace(
        docstore=SimpleNamespace(docs=docs or {}),
        vector_store=vector_store,
        property_graph_store=graph_store,
    )
    index = SimpleNamespace(vector_store=vector_store, property_graph_store=graph_store)
    indexer = SimpleNamespace(
        index=index, storage_context=storage_context, embedder=object()
    )
    return RetrievalStoreAccess(indexer)


class RetrievalStoreAccessTests(unittest.TestCase):
    def test_filtered_vector_query_falls_back_and_records_health_event(self) -> None:
        vector_store = FakeVectorStore(fail_filtered=True)
        access = make_access(vector_store=vector_store)

        result = access.query_vector(
            [1.0],
            top_k=5,
            node_kind="postprocessed_retrieval_chunk",
            component="analyst",
            fallback_message="filtered query failed",
        )

        self.assertEqual(["chunk_1"], result.ids)
        self.assertEqual(2, len(vector_store.queries))
        self.assertIsNotNone(vector_store.queries[0].filters)
        self.assertIsNone(vector_store.queries[1].filters)
        self.assertTrue(access.health_report.has_code("vector_filter_fallback"))

    def test_relation_map_falls_back_to_triplets_and_filters_denied_relations(
        self,
    ) -> None:
        source = FakeGraphNode("concept_source", "Source")
        kept = FakeGraphNode("concept_kept", "Kept")
        denied = FakeGraphNode("concept_denied", "Denied")
        graph_store = FakeGraphStore(
            nodes=[source, kept, denied],
            triplets=[
                (source, FakeRelation("USES"), kept),
                (source, FakeRelation("MENTIONS"), denied),
            ],
            fail_rel_map=True,
        )
        access = make_access(graph_store=graph_store)

        triplets = access.relation_map(
            [source],
            depth=1,
            limit=10,
            ignore_rels={"MENTIONS"},
            component="visualizer",
            fallback_message="relation map failed",
        )

        self.assertEqual([(source, graph_store.triplets[0][1], kept)], triplets)
        self.assertTrue(access.health_report.has_code("graph_relation_map_fallback"))

    def test_triplet_failure_returns_empty_list_and_records_health_event(self) -> None:
        access = make_access(graph_store=FakeGraphStore(fail_triplets=True))

        self.assertEqual([], access.triplets(ids=["concept_1"], component="analyst"))
        self.assertTrue(access.health_report.has_code("graph_triplet_lookup_failed"))

    def test_concept_enumeration_filters_chunk_nodes_and_caches_by_id(self) -> None:
        concept = FakeGraphNode("concept_1", "Concept")
        chunk = FakeGraphNode("chunk_1", "Chunk", node_label="text_chunk")
        access = make_access(graph_store=FakeGraphStore(nodes=[concept, chunk]))

        concepts = access.all_concepts(component="visualizer")

        self.assertEqual([concept], concepts)
        self.assertIs(concept, access.graph_node("concept_1", component="visualizer"))

    def test_docstore_and_property_helpers_match_retrieval_node_shapes(self) -> None:
        node = FakeGraphNode("concept_1", "Concept")
        relation = FakeRelation("USES", {"evidence_chunk_ids": ["chunk_1"]})
        access = make_access(docs={"chunk_1": "stored"})

        self.assertEqual("stored", access.docstore_node("chunk_1"))
        self.assertEqual("concept_1", access.node_id(node))
        self.assertEqual("Concept", access.node_label(node))
        self.assertEqual(["alias"], access.node_aliases(node))
        self.assertEqual(["chunk_1"], access.node_source_chunk_ids(node))
        self.assertEqual("USES", access.relation_label(relation))
        self.assertEqual(["chunk_1"], access.relation_evidence_chunk_ids(relation))
        self.assertEqual(0.75, access.salience(node))


if __name__ == "__main__":
    unittest.main()
