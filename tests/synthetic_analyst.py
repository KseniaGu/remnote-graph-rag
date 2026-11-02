from types import SimpleNamespace

from llama_index.core.vector_stores import VectorStoreQueryResult

from backend.configs.search import KnowledgeGraphSearchSettings


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


class SequencedVectorStore:
    def __init__(self, results: list[tuple[list[str], list[float]]]) -> None:
        self.results = results
        self.queries = []

    def query(self, query):
        self.queries.append(query)
        index = min(len(self.queries) - 1, len(self.results) - 1)
        ids, similarities = self.results[index]
        return VectorStoreQueryResult(ids=ids, similarities=similarities)


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

    def get_triplets(
        self, entity_names=None, relation_names=None, properties=None, ids=None
    ):
        triplets = [*self.mention_triplets, *self.relation_triplets]
        if relation_names:
            triplets = [
                triplet for triplet in triplets if triplet[1].id in relation_names
            ]
        if ids:
            id_set = set(ids)
            triplets = [
                triplet
                for triplet in triplets
                if triplet[0].id in id_set or triplet[2].id in id_set
            ]
        return triplets

    def get_rel_map(self, graph_nodes, depth=1, limit=30, ignore_rels=None):
        ids = {node.id for node in graph_nodes}
        ignore_rels = set(ignore_rels or [])
        triplets = [
            triplet
            for triplet in self.relation_triplets
            if triplet[1].id not in ignore_rels
            and (triplet[0].id in ids or triplet[2].id in ids)
        ]
        return triplets[:limit]


class RaisingReranker:
    def postprocess_nodes(self, nodes, query_str: str):
        raise RuntimeError("reranker unavailable")


class EmptyHealthCheckReranker:
    def postprocess_nodes(self, nodes, query_str: str):
        return []


class OrderedScoreReranker:
    def __init__(
        self, scores_by_id: dict[str, float], top_n: int | None = None
    ) -> None:
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
    storage_context = SimpleNamespace(
        docstore=SimpleNamespace(docs=docs), vector_store=vector_store
    )
    index = SimpleNamespace(
        vector_store=vector_store, property_graph_store=graph_store or FakeGraphStore()
    )
    return SimpleNamespace(
        index=index,
        storage_context=storage_context,
        embedder=FakeEmbedder(),
        kg_search_settings=settings
        or KnowledgeGraphSearchSettings(analyst_reranker_mode="disabled"),
    )
