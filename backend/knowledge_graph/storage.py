from pathlib import Path
from typing import Any

import redis
from llama_index.core import StorageContext
from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.storage.kvstore.redis import RedisKVStore
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.vector_stores.redis.schema import RedisVectorStoreSchema
from pinecone import Pinecone

from backend.configs.constants import DEFAULT_EMBEDDING_DIM
from backend.configs.enums import StorageType
from backend.configs.paths import PathSettings
from backend.configs.storage import StorageSettings
from backend.knowledge_graph.custom_types import CustomNeo4jPropertyGraphStore
from backend.utils.helpers import make_json_serializable


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
            if self.storage_settings.document_storage.password:
                redis_url = self.storage_settings.document_storage.get_connection_url(driver="rediss")
                redis_client = redis.from_url(redis_url)
                kv_store = RedisKVStore(redis_client=redis_client, namespace="llama_index")
                document_storage = KVDocumentStore(kv_store)
            else:
                document_storage = RedisDocumentStore.from_host_and_port(
                    host=self.storage_settings.document_storage.host,
                    port=self.storage_settings.document_storage.port,
                    namespace="llama_index",
                )
            # The least resource demanding way to check whether Redis cloud storage is empty
            storage_is_empty = document_storage._kvstore._redis_client.dbsize() == 0

            if self.storage_settings.document_storage.init_from_local and storage_is_empty:
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
            if self.storage_settings.index_storage.password:
                redis_url = self.storage_settings.index_storage.get_connection_url(driver="rediss")
                redis_client = redis.from_url(redis_url)
                kv_store = RedisKVStore(redis_client=redis_client, namespace="llama_index")
                index_storage = KVIndexStore(kv_store)
            else:
                index_storage = RedisIndexStore.from_host_and_port(
                    host=self.storage_settings.index_storage.host,
                    port=self.storage_settings.index_storage.port,
                    namespace="llama_index",
                )
            try:
                index_storage.get_index_struct()
                storage_is_empty = False
            except AssertionError:
                storage_is_empty = True

            if self.storage_settings.index_storage.init_from_local and storage_is_empty:
                local_path_to_check = kwargs.get("local_storage", self.path_settings.local_storage_dir)
                local_index_storage = SimpleIndexStore.from_persist_dir(persist_dir=str(local_path_to_check))
                index_storage.add_index_struct(local_index_storage.get_index_struct())
        else:
            raise ValueError("Only Local and Redis storage types are supported for Index Storage")

        return index_storage

    def get_vector_storage(self, **kwargs) -> Any:
        """Gets vector storage backend.
        
        Supports local, Redis, and Pinecone storage. Configures embedding dimensions.
        
        Args:
            **kwargs: Additional arguments including optional embedding_dim.
            
        Returns:
            Vector storage instance (SimpleVectorStore, RedisVectorStore, or PineconeVectorStore).
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
            if self.storage_settings.index_storage.password:
                redis_connection = self.storage_settings.vector_storage.get_connection_url(driver="rediss")
            else:
                redis_connection = self.storage_settings.vector_storage.get_connection_url()
            vector_storage = RedisVectorStore(
                redis_url=redis_connection, overwrite=self.storage_settings.vector_storage.overwrite_index,
                schema=schema
            )
        elif self.storage_settings.vector_storage.storage_type == StorageType.pinecone:
            pc = Pinecone(api_key=self.storage_settings.vector_storage.api_key.get_secret_value())
            pinecone_index = pc.Index(self.storage_settings.vector_storage.index_name)
            vector_storage = PineconeVectorStore(pinecone_index=pinecone_index)
        else:
            raise ValueError("Only Local, Redis, and Pinecone storage types are supported for Vector Storage")

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
            not_empty = property_graph_storage.structured_query("MATCH (n) RETURN count(n) > 0 AS not_empty LIMIT 1")[0]
            not_empty = not_empty.get("not_empty", False)

            if self.storage_settings.property_graph_storage.init_from_local and not not_empty:
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
