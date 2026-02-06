from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from backend.configs.models import ModelSettings
from backend.configs.paths import PathSettings
from backend.configs.search import KnowledgeGraphSearchSettings
from backend.configs.storage import LocalStorageSettings
from backend.configs.storage import StorageSettings, RedisSettings, Neo4jSettings
from backend.knowledge_graph.indexer import KnowledgeGraphIndexer
from backend.knowledge_graph.storage import KnowledgeGraphStorage
from backend.utils.helpers import get_logger

logger = get_logger("Migrating data to cloud storages")


def main():
    """Migrates local data to cloud databases."""
    logger.info("Starting cloud database migration...")
    logger.info("Make sure REDIS_* and NEO4J_* environment variables are set for cloud databases.")

    path_settings = PathSettings()

    from backend.configs.storage import PineconeSettings

    storage_settings = StorageSettings(
        document_storage=RedisSettings(),
        index_storage=RedisSettings(),
        # There is no convenient method to migrate the nodes with embeddings from local vector store to the cloud one
        # (as far as I know). So we migrate documents, index and graph, and further migrate vectors when loading the index
        vector_storage=PineconeSettings(),
        property_graph_storage=Neo4jSettings(),
    )
    models_settings = ModelSettings()
    kg_search_settings = KnowledgeGraphSearchSettings()

    logger.info(f"Local storage path: {path_settings.local_storage_dir}")
    logger.info("Initializing KnowledgeGraphStorage to trigger migration...")

    # Migrate documents, graph and its index
    kg_storage = KnowledgeGraphStorage(path_settings, storage_settings)

    # Make new KG Storage with local stores for docs, index and graph, but cloud for vector store, to migrate embeddings properly
    storage_settings.document_storage = LocalStorageSettings(storage_path=path_settings.local_storage_dir)
    storage_settings.index_storage = LocalStorageSettings(storage_path=path_settings.local_storage_dir)
    storage_settings.property_graph_storage = LocalStorageSettings(storage_path=path_settings.local_storage_dir)
    kg_storage = KnowledgeGraphStorage(path_settings, storage_settings)

    embedder = HuggingFaceEmbedding(models_settings.embedder.model_path, trust_remote_code=True, embed_batch_size=5)
    # We need to initialize indexer to calculate and add embeddings as SimpleVectorStore doesn't allow to "get nodes" directly
    knowledge_graph_indexer = KnowledgeGraphIndexer(
        kg_storage.storage_context, path_settings, storage_settings.document_storage.storage_type, kg_search_settings,
        embedder, None
    )
    # Here embeddings will be generated and pushed to
    knowledge_graph_indexer.load_index()
    logger.info("Migration complete!")


if __name__ == "__main__":
    main()
