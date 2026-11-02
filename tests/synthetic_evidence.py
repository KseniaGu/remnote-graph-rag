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
