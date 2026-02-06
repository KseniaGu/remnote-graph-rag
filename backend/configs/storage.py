from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.configs.constants import ENV_PATH, DATA_DIR
from backend.configs.enums import StorageType


class LocalStorageSettings(BaseSettings):
    """Local file system storage settings configuration."""
    storage_type: StorageType = StorageType.local
    storage_path: Path = DATA_DIR / "storage"


class RedisSettings(BaseSettings):
    """Redis settings configuration."""
    storage_type: StorageType = StorageType.redis
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    init_from_local: bool = False
    overwrite_index: bool = False

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="REDIS_",
    )

    def get_connection_url(self, driver: str = "redis") -> str:
        if self.password:
            return f"{driver}://default:{self.password}@{self.host}:{self.port}"
        return f"{driver}://{self.host}:{self.port}"


class PineconeSettings(BaseSettings):
    """Pinecone vector database settings configuration."""
    storage_type: StorageType = StorageType.pinecone
    api_key: SecretStr = SecretStr("")
    environment: str = ""  # e.g., "us-east-1-aws"
    index_name: str = "remnote-graph-rag"
    init_from_local: bool = False

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="PINECONE_",
    )


class Neo4jSettings(BaseSettings):
    """Neo4j graph database settings configuration."""
    storage_type: StorageType = StorageType.neo4j
    username: SecretStr = SecretStr("neo4j")
    password: SecretStr = SecretStr("12345678")
    database: str = "neo4j"
    uri: str = ""  # Full URI (e.g., neo4j+s://xxx.databases.neo4j.io) - takes precedence
    host: str = "localhost"
    port: int = 7687
    init_from_local: bool = False

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="NEO4J_",
    )

    def get_connection_url(self, driver: str = "bolt") -> str:
        # If URI is provided (cloud), convert to bolt format
        if self.uri:
            return self.uri.replace("neo4j+s://", "bolt+s://").replace("neo4j://", "bolt://")
        return f"{driver}://{self.host}:{self.port}"


class StorageSettings(BaseSettings):
    """Aggregates storage configuration settings for different storage backends."""
    document_storage: LocalStorageSettings | RedisSettings = RedisSettings()
    index_storage: LocalStorageSettings | RedisSettings = RedisSettings()
    vector_storage: LocalStorageSettings | RedisSettings | PineconeSettings = PineconeSettings()
    property_graph_storage: LocalStorageSettings | Neo4jSettings = Neo4jSettings()
