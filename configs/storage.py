from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from configs.constants import ENV_PATH, DATA_DIR
from configs.enums import StorageType


class LocalStorageSettings(BaseSettings):
    """Local file system storage settings configuration."""
    storage_type: StorageType = StorageType.local
    storage_path: Path = DATA_DIR / "storage"


class RedisSettings(BaseSettings):
    """Redis settings configuration."""
    storage_type: StorageType = StorageType.redis
    host: str = "localhost"
    port: int = 6379
    init_from_local: bool = True

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="REDIS_",
    )

    def get_connection_url(self, driver: str = "redis") -> str:
        return f"{driver}://{self.host}:{self.port}"


class Neo4jSettings(BaseSettings):
    """Neo4j graph database settings configuration."""
    storage_type: StorageType = StorageType.neo4j
    username: SecretStr = "neo4j"
    password: SecretStr = "12345678"
    database: str = "neo4j"
    host: str = "localhost"
    port: int = 7687
    init_from_local: bool = True

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="NEO4J_",
    )

    def get_connection_url(self, driver: str = "bolt") -> str:
        return f"{driver}://{self.host}:{self.port}"


class StorageSettings(BaseSettings):
    """Aggregates storage configuration settings for different storage backends."""
    document_storage: LocalStorageSettings | RedisSettings = RedisSettings(init_from_local=True)
    index_storage: LocalStorageSettings | RedisSettings = RedisSettings(init_from_local=True)
    vector_storage: LocalStorageSettings | RedisSettings = RedisSettings(init_from_local=True)
    property_graph_storage: LocalStorageSettings | Neo4jSettings = Neo4jSettings(init_from_local=True)
