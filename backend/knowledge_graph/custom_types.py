import hashlib
from typing import Optional

from llama_index.core.graph_stores import LabelledNode, EntityNode, Relation, ChunkNode
from llama_index.core.graph_stores.types import Triplet
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.graph_stores.neo4j.neo4j_property_graph import BASE_NODE_LABEL, BASE_ENTITY_LABEL, remove_empty_values


class CustomNeo4jPropertyGraphStore(Neo4jPropertyGraphStore):
    """Custom Neo4j Property Graph Store with support for ChunkNode relationships."""

    def get_rel_map(
            self,
            graph_nodes: list[LabelledNode],
            depth: int = 2,
            limit: int = 30,
            ignore_rels: Optional[list[str]] = None,
    ) -> list[Triplet]:
        """Gets depth-aware relationship map for all node types including ChunkNodes.

        Args:
            graph_nodes: List of graph nodes to get relationships for.
            depth: Maximum depth for relationship traversal.
            limit: Maximum number of relationships to return.
            ignore_rels: List of relationship types to ignore.

        Returns:
            List of triplets (source, relation, target).
        """
        triples = []

        ids = [node.id for node in graph_nodes]
        # Modified query: use __Node__ instead of __Entity__ to include ChunkNodes
        response = self.structured_query(
            f"""
            WITH $ids AS id_list
            UNWIND range(0, size(id_list) - 1) AS idx
            MATCH (e:`{BASE_NODE_LABEL}`)
            WHERE e.id = id_list[idx]
            MATCH p=(e)-[r*1..{depth}]-(other)
            WHERE ALL(rel in relationships(p) WHERE type(rel) <> 'MENTIONS')
            UNWIND relationships(p) AS rel
            WITH distinct rel, idx
            WITH startNode(rel) AS source,
                type(rel) AS type,
                rel{{.*}} AS rel_properties,
                endNode(rel) AS endNode,
                idx
            LIMIT toInteger($limit)
            RETURN source.id AS source_id, [l in labels(source)
                   WHERE NOT l IN ['{BASE_ENTITY_LABEL}', '{BASE_NODE_LABEL}'] | l][0] AS source_type,
                source{{.* , embedding: Null, id: Null}} AS source_properties,
                type,
                rel_properties,
                endNode.id AS target_id, [l in labels(endNode)
                   WHERE NOT l IN ['{BASE_ENTITY_LABEL}', '{BASE_NODE_LABEL}'] | l][0] AS target_type,
                endNode{{.* , embedding: Null, id: Null}} AS target_properties,
                idx
            ORDER BY idx
            LIMIT toInteger($limit)
            """,
            param_map={"ids": ids, "limit": limit},
        )
        response = response if response else []

        ignore_rels = ignore_rels or []
        for record in response:
            if record["type"] in ignore_rels:
                continue

            source = EntityNode(
                name=record["source_id"],
                label=record["source_type"],
                properties=remove_empty_values(record["source_properties"]),
            )
            target = EntityNode(
                name=record["target_id"],
                label=record["target_type"],
                properties=remove_empty_values(record["target_properties"]),
            )
            rel = Relation(
                source_id=record["source_id"],
                target_id=record["target_id"],
                label=record["type"],
                properties=remove_empty_values(record["rel_properties"]),
            )
            triples.append([source, rel, target])

        return triples

    def get(self, properties: Optional[dict] = None, ids: Optional[list[str]] = None) -> list[LabelledNode]:
        """Get nodes with correct label extraction.

        Adds support for returning actual entity label.
        """
        cypher_statement = f"MATCH (e: {BASE_NODE_LABEL}) "

        params = {}
        cypher_statement += "WHERE e.id IS NOT NULL "

        if ids:
            cypher_statement += "AND e.id in $ids "
            params["ids"] = ids

        if properties:
            prop_list = []
            for i, prop in enumerate(properties):
                prop_list.append(f"e.`{prop}` = $property_{i}")
                params[f"property_{i}"] = properties[prop]
            cypher_statement += " AND " + " AND ".join(prop_list)

        # Filter out both __Entity__ and __Node__ to get the actual label
        return_statement = f"""
        WITH e
        RETURN e.id AS name,
               [l in labels(e) WHERE NOT l IN ['{BASE_ENTITY_LABEL}', '{BASE_NODE_LABEL}'] | l][0] AS type,
               e{{.* , embedding: Null, id: Null}} AS properties
        """
        cypher_statement += return_statement

        response = self.structured_query(cypher_statement, param_map=params)
        response = response if response else []

        nodes = []
        for record in response:
            # text indicates a chunk node
            # none on the type indicates an implicit node, likely a chunk node
            if "text" in record["properties"] or record["type"] is None:
                text = record["properties"].pop("text", "")
                nodes.append(
                    ChunkNode(
                        id_=record["name"],
                        text=text,
                        properties=remove_empty_values(record["properties"]),
                    )
                )
            else:
                nodes.append(
                    EntityNode(
                        name=record["name"],
                        label=record["type"],
                        properties=remove_empty_values(record["properties"]),
                    )
                )

        return nodes


class CustomEntityNode(EntityNode):
    """Custom EntityNode that generates ASCII-safe IDs for compatibility with Pinecone and other vector stores.

    This addresses the limitation where some vector stores (like Pinecone) require ASCII-only IDs,
    but entity names may contain Unicode characters (e.g., '×', '→', etc.).

    Uses SHA256 hash for consistent, ASCII-safe IDs computed once at initialization.
    Original entity name is always preserved in the 'name' attribute.
    """

    def __init__(self, name: str, label: str = "", properties: dict = None):
        """Initializes entity node with cached ASCII-safe ID.

        Args:
            name: Entity name (may contain Unicode)
            label: Entity label/type
            properties: Optional entity properties
        """
        props = (properties or {}).copy()
        if "entity_name" not in props:
            props["entity_name"] = name

        id_source = f"{label}:{name}" if label else name
        ascii_id = hashlib.sha256(id_source.encode("utf-8")).hexdigest()[:16]
        super().__init__(name=ascii_id, label=label, properties=props)

    @property
    def id(self) -> str:
        """Returns cached ASCII-safe ID.

        Returns:
            ASCII-safe identifier suitable for all vector stores.
        """
        return self.name
