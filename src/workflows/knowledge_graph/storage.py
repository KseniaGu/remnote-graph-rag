from pathlib import Path
from typing import Any, Optional

from llama_index.core import StorageContext
from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import LabelledNode, EntityNode, Relation, Triplet
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.graph_stores.neo4j.neo4j_property_graph import remove_empty_values, BASE_NODE_LABEL, BASE_ENTITY_LABEL
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.vector_stores.redis.schema import RedisVectorStoreSchema

from configs.constants import DEFAULT_EMBEDDING_DIM
from configs.enums import StorageType
from configs.paths import PathSettings
from configs.storage import StorageSettings
from src.utils.helpers import make_json_serializable


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


class KnowledgeGraphStorage:
    """Manages storage backends for knowledge graph components.
    
    Supports local, Redis, and Neo4j storage backends for documents, indices, vectors, and property graphs.
    """

    def __init__(self, path_settings: PathSettings, storage_settings: StorageSettings, **kwargs) -> None:
        """Initializes the knowledge graph storage.
        
        Args:
            path_settings: Path configuration settings.
            storage_settings: Storage backend configuration settings.
            **kwargs: Additional arguments passed to storage initialization.
        """
        self.path_settings = path_settings
        self.storage_settings = storage_settings

        self._init_storage_context(**kwargs)

    @staticmethod
    def not_empty(storage_path: Path) -> bool:
        """Checks if a storage directory is not empty.
        
        Args:
            storage_path: Path to check.
            
        Returns:
            True if directory contains files, False otherwise.
        """
        return any(storage_path.iterdir())

    def get_document_storage(self, **kwargs) -> Any:
        """Gets document storage backend.
        
        Supports local and Redis storage. Can initialize from local storage if configured.
        
        Args:
            **kwargs: Additional arguments including optional local_storage path.
            
        Returns:
            Document storage instance (SimpleDocumentStore or RedisDocumentStore).
        """
        if self.storage_settings.document_storage.storage_type == StorageType.local:
            storage_path = self.storage_settings.document_storage.storage_path
            if self.not_empty(storage_path):
                document_storage = SimpleDocumentStore.from_persist_dir(persist_dir=str(storage_path))
            else:
                document_storage = SimpleDocumentStore()
        elif self.storage_settings.document_storage.storage_type == StorageType.redis:
            document_storage = RedisDocumentStore.from_host_and_port(
                host=self.storage_settings.document_storage.host,
                port=self.storage_settings.document_storage.port,
                namespace="llama_index",
            )
            if self.storage_settings.document_storage.init_from_local and not document_storage.docs:
                local_path_to_check = kwargs.get("local_storage", self.path_settings.local_storage_dir)
                if self.not_empty(local_path_to_check):
                    local_document_storage = SimpleDocumentStore.from_persist_dir(
                        persist_dir=str(local_path_to_check)
                    )
                    document_storage.add_documents(list(local_document_storage.docs.values()))
        else:
            raise ValueError("Only Local and Redis storage types are supported for Document Storage")

        return document_storage

    def get_index_storage(self, **kwargs) -> Any:
        """Gets index storage backend.
        
        Supports local and Redis storage. Can initialize from local storage if configured.
        
        Args:
            **kwargs: Additional arguments including optional local_storage path.
            
        Returns:
            Index storage instance (SimpleIndexStore or RedisIndexStore).
        """
        if self.storage_settings.index_storage.storage_type == StorageType.local:
            storage_path = self.storage_settings.index_storage.storage_path
            if self.not_empty(storage_path) and (storage_path / "index_store.json").exists():
                index_storage = SimpleIndexStore.from_persist_dir(persist_dir=str(storage_path))
            else:
                index_storage = SimpleIndexStore()
        elif self.storage_settings.index_storage.storage_type == StorageType.redis:
            index_storage = RedisIndexStore.from_host_and_port(
                host=self.storage_settings.index_storage.host,
                port=self.storage_settings.index_storage.port,
                namespace="llama_index",
            )
            if self.storage_settings.index_storage.init_from_local and not index_storage.index_structs():
                local_path_to_check = kwargs.get("local_storage", self.path_settings.local_storage_dir)
                local_index_storage = SimpleIndexStore.from_persist_dir(persist_dir=str(local_path_to_check))
                index_storage.add_index_struct(local_index_storage.get_index_struct())
        else:
            raise ValueError("Only Local and Redis storage types are supported for Index Storage")

        return index_storage

    def get_vector_storage(self, **kwargs) -> Any:
        """Gets vector storage backend.
        
        Supports local and Redis storage. Configures embedding dimensions for Redis.
        
        Args:
            **kwargs: Additional arguments including optional embedding_dim.
            
        Returns:
            Vector storage instance (SimpleVectorStore or RedisVectorStore).
        """
        if self.storage_settings.vector_storage.storage_type == StorageType.local:
            storage_path = self.storage_settings.vector_storage.storage_path
            if self.not_empty(storage_path) and (storage_path / "default__vector_store.json").exists():
                vector_storage = SimpleVectorStore.from_persist_dir(persist_dir=str(storage_path))
            else:
                vector_storage = SimpleVectorStore()
        elif self.storage_settings.vector_storage.storage_type == StorageType.redis:
            embedding_dim = kwargs.get("embedding_dim", DEFAULT_EMBEDDING_DIM)
            schema = RedisVectorStoreSchema()
            schema.fields["vector"].attrs.dims = embedding_dim

            redis_connection = self.storage_settings.vector_storage.get_connection_url()
            vector_storage = RedisVectorStore(redis_url=redis_connection, overwrite=False, schema=schema)
        else:
            raise ValueError("Only Local and Redis storage types are supported for Index Storage")

        return vector_storage

    def get_property_graph_storage(self, **kwargs) -> Any:
        """Gets property graph storage backend.
        
        Supports local and Neo4j storage. Uses custom Neo4j store for ChunkNode support.
        Can initialize from local storage if configured.
        
        Args:
            **kwargs: Additional arguments including optional local_storage path.
            
        Returns:
            Property graph storage instance (SimplePropertyGraphStore or CustomNeo4jPropertyGraphStore).
        """
        if self.storage_settings.property_graph_storage.storage_type == StorageType.local:
            storage_path = self.storage_settings.property_graph_storage.storage_path
            if self.not_empty(storage_path) and (storage_path / "property_graph_store.json").exists():
                property_graph_storage = SimplePropertyGraphStore.from_persist_dir(persist_dir=str(storage_path))
            else:
                property_graph_storage = SimplePropertyGraphStore()
        elif self.storage_settings.property_graph_storage.storage_type == StorageType.neo4j:
            # We use a custom class in order to be able to retrieve relationships with chunk nodes (containing document texts)
            # while making a retrieval combined with graph traversal (VectorContextRetriever).
            property_graph_storage = CustomNeo4jPropertyGraphStore(
                username=self.storage_settings.property_graph_storage.username.get_secret_value(),
                password=self.storage_settings.property_graph_storage.password.get_secret_value(),
                url=self.storage_settings.property_graph_storage.get_connection_url(),
                database=self.storage_settings.property_graph_storage.database,
            )
            if self.storage_settings.property_graph_storage.init_from_local:
                local_path_to_check = kwargs.get("local_storage", self.path_settings.local_storage_dir)
                if self.not_empty(local_path_to_check):
                    local_property_graph_storage = SimplePropertyGraphStore.from_persist_dir(
                        persist_dir=str(local_path_to_check)
                    )
                    nodes = local_property_graph_storage.get()
                    for node in nodes:
                        node = make_json_serializable(node, "properties")

                    property_graph_storage.upsert_nodes(nodes)
                    property_graph_storage.upsert_relations(list(local_property_graph_storage.graph.relations.values()))
        else:
            raise ValueError("Only Local and Neo4j storage types are supported for Index Storage")

        return property_graph_storage

    def _init_storage_context(self, **kwargs):
        """Initializes the storage context with all storage backends.
        
        Args:
            **kwargs: Additional arguments passed to storage initialization methods.
        """
        self.storage_context = StorageContext.from_defaults(
            docstore=self.get_document_storage(**kwargs),
            index_store=self.get_index_storage(**kwargs),
            vector_store=self.get_vector_storage(**kwargs),
            property_graph_store=self.get_property_graph_storage(**kwargs),
        )
