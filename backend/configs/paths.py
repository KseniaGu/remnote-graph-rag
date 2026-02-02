from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings

from backend.configs.constants import DATA_DIR, ROOT_DIR, REMNOTE_FOLDER_NAME


class PathSettings(BaseSettings):
    """Configuration for file system paths used throughout the application.

    Manages all directory paths for storing and organizing application data.
    """
    raw_data_dir: Path = DATA_DIR / "raw" / REMNOTE_FOLDER_NAME
    parsed_pdfs_dir: Path = DATA_DIR / "raw/parsed_pdfs"
    parsed_images_dir: Path = DATA_DIR / "raw/parsed_images"
    parsed_texts_dir: Path = DATA_DIR / "raw/parsed_texts"
    local_storage_dir: Path = ROOT_DIR / "storage"
    prompts_dir: Path = ROOT_DIR / "backend/llm/prompts"

    @model_validator(mode='after')
    def ensure_directory_existence(self):
        """Ensures all required directories exist.

        Creates any missing directories in the required paths if they don't exist.
        """
        required_dirs = [self.parsed_pdfs_dir, self.parsed_images_dir, self.parsed_texts_dir, self.local_storage_dir]

        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)

        return self
