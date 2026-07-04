import json
from itertools import islice
from typing import Any, Iterable

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone import Pinecone

from backend.configs.models import ModelSettings
from backend.configs.paths import PathSettings
from backend.configs.search import KnowledgeGraphSearchSettings
from backend.configs.storage import (
    LocalStorageSettings,
    StorageSettings,
    RedisSettings,
    Neo4jSettings,
    MemgraphSettings,
    PineconeSettings,
)
from backend.knowledge_graph.indexer import KnowledgeGraphIndexer
from backend.knowledge_graph.storage import KnowledgeGraphStorage
from backend.utils.helpers import get_logger

logger = get_logger("Migrating data to cloud storages")


PINECONE_VECTOR_BATCH_SIZE = 100

PINECONE_COMMON_METADATA_KEYS = {
    "docstore_node_kind",
    "vector_source",
    "document_id",
    "doc_id",
    "ref_doc_id",
}

PINECONE_PASSAGE_METADATA_KEYS = {
    *PINECONE_COMMON_METADATA_KEYS,
    "chunk_id",
    "parent_chunk_id",
    "passage_id",
    "passage_index",
    "passage_char_start",
    "passage_char_end",
    "passage_token_count",
    "passage_split_strategy",
}

PINECONE_CONCEPT_METADATA_KEYS = {
    *PINECONE_COMMON_METADATA_KEYS,
    "concept_id",
    "concept_type",
}


def batched(iterable: Iterable[Any], batch_size: int) -> Iterable[list[Any]]:
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def pinecone_vector_metadata(
    node_id: str, metadata: dict[str, Any], ref_doc_id: str | None = None
) -> dict[str, Any]:
    """Returns Pinecone-safe metadata needed by optimized runtime retrieval.

    Pinecone metadata is deliberately much smaller than local vector metadata.
    Runtime retrieval only filters by docstore_node_kind and then resolves full
    source data through the docstore/property graph.
    """

    node_kind = str(metadata.get("docstore_node_kind") or "")
    if node_kind == "postprocessed_embedding_passage":
        allowed_keys = PINECONE_PASSAGE_METADATA_KEYS
    elif node_kind == "postprocessed_concept_node":
        allowed_keys = PINECONE_CONCEPT_METADATA_KEYS
    else:
        allowed_keys = PINECONE_COMMON_METADATA_KEYS

    safe_metadata: dict[str, Any] = {}
    for key in allowed_keys:
        if key not in metadata:
            continue
        value = _pinecone_metadata_value(metadata[key])
        if value is not None:
            safe_metadata[key] = value

    safe_ref_doc_id = str(
        ref_doc_id or metadata.get("ref_doc_id") or metadata.get("doc_id") or node_id
    )
    safe_metadata.setdefault("ref_doc_id", safe_ref_doc_id)
    safe_metadata.setdefault("doc_id", safe_ref_doc_id)
    safe_metadata.setdefault("document_id", safe_ref_doc_id)
    safe_metadata.setdefault(
        "vector_source", str(metadata.get("vector_source") or node_id)
    )

    # LlamaIndex's PineconeVectorStore.query tries to reconstruct nodes from
    # metadata. We do not use returned nodes, but a tiny text field keeps the
    # legacy fallback parser from failing before ids/scores are returned.
    safe_metadata.setdefault("text", " ")
    return safe_metadata


def _pinecone_metadata_value(value: Any) -> str | int | float | bool | list[str] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, list):
        string_values = [
            str(item)
            for item in value
            if item is not None and isinstance(item, (str, int, float, bool))
        ]
        return string_values or None
    return str(value)


def migrate_vector_store_to_pinecone():
    """Migrates local vectore store data to remote Pinecone storage."""
    logger.info("Starting vecstore migration...")
    path_settings = PathSettings()

    data = json.loads((path_settings.local_storage_dir / "default__vector_store.json").read_text())
    pinecone_settings = PineconeSettings()
    embeddings = data["embedding_dict"]
    metadata = data.get("metadata_dict", {})
    ref_doc_ids = data.get("text_id_to_ref_doc_id", {})

    pc = Pinecone(api_key=pinecone_settings.api_key.get_secret_value())
    index = pc.Index(pinecone_settings.index_name)

    total_upserted = 0
    for batch in batched(embeddings.items(), PINECONE_VECTOR_BATCH_SIZE):
        vectors = []
        for node_id, values in batch:
            vectors.append(
                {
                    "id": node_id,
                    "values": values,
                    "metadata": pinecone_vector_metadata(
                        node_id,
                        metadata.get(node_id, {}),
                        ref_doc_ids.get(node_id),
                    ),
                }
            )
        index.upsert(vectors=vectors)
        total_upserted += len(vectors)
        logger.info(f"Upserted {total_upserted}/{len(embeddings)} vectors to Pinecone.")


def main():
    """Migrates local data to cloud databases."""
    logger.info("Starting cloud database migration...")
    logger.info("Make sure REDIS_* and NEO4J_* environment variables are set for cloud databases.")

    path_settings = PathSettings()

    storage_settings = StorageSettings(
        document_storage=LocalStorageSettings(),
        index_storage=LocalStorageSettings(),
        # There is no convenient method to migrate the nodes with embeddings from local vector store to the cloud one
        # (as far as I know). So we migrate documents, index and graph, and further migrate vectors when loading the index
        vector_storage=LocalStorageSettings(),
        # property_graph_storage=Neo4jSettings(),
        property_graph_storage=MemgraphSettings(),
        # property_graph_storage=LocalStorageSettings(),
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
        kg_storage.storage_context,
        path_settings,
        storage_settings.document_storage.storage_type,
        kg_search_settings,
        embedder,
        None,
    )
    # Here embeddings will be generated and pushed to
    knowledge_graph_indexer.load_index()
    logger.info("Migration complete!")


if __name__ == "__main__":
    # migrate_vector_store_to_pinecone()
    main()
