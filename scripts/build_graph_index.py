from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from backend.configs.models import ModelSettings
from backend.configs.paths import PathSettings
from backend.configs.search import KnowledgeGraphSearchSettings
from backend.configs.storage import StorageSettings, LocalStorageSettings
from backend.knowledge_graph.indexer import KnowledgeGraphIndexer
from backend.knowledge_graph.storage import KnowledgeGraphStorage

if __name__ == '__main__':
    storage_settings = StorageSettings()
    models_settings = ModelSettings()
    path_settings = PathSettings()
    kg_search_settings = KnowledgeGraphSearchSettings()

    # This is the local storage setup, comment the settings lines below to activate non-local storages
    storage_settings.document_storage = LocalStorageSettings(storage_path=path_settings.local_storage_dir)
    storage_settings.index_storage = LocalStorageSettings(storage_path=path_settings.local_storage_dir)
    storage_settings.vector_storage = LocalStorageSettings(storage_path=path_settings.local_storage_dir)
    storage_settings.property_graph_storage = LocalStorageSettings(storage_path=path_settings.local_storage_dir)

    kg_storage = KnowledgeGraphStorage(path_settings, storage_settings)
    embedder = HuggingFaceEmbedding(models_settings.embedder.model_path, trust_remote_code=True, embed_batch_size=5)
    # reranker = CohereRerank(api_key=os.environ["COHERE_API_KEY"], top_n=10)
    knowledge_graph_indexer = KnowledgeGraphIndexer(
        kg_storage.storage_context, path_settings, storage_settings.document_storage.storage_type, kg_search_settings,
        embedder, None
    )
