import os
from pathlib import Path


EMBEDDER_MODEL_PATH_ENV = "EMBEDDER_MODEL_PATH"
DEFAULT_LOCAL_EMBEDDER_MODEL_PATH = Path("/app/models/all-MiniLM-L6-v2")
REQUIRED_EMBEDDER_MODEL_FILES = ("modules.json", "config.json")


def resolve_embedder_model_path(configured_model_path: str) -> str:
    """Returns the model path that should be passed to HuggingFaceEmbedding.

    Production images set EMBEDDER_MODEL_PATH to a concrete local model directory.
    When that variable is set, fail loudly if the directory does not contain the
    SentenceTransformers files expected for offline startup.
    """
    explicit_model_path = os.environ.get(EMBEDDER_MODEL_PATH_ENV, "").strip()
    if explicit_model_path:
        model_path = Path(explicit_model_path)
        _ensure_local_embedder_model(model_path)
        return str(model_path)

    if _is_complete_local_embedder_model(DEFAULT_LOCAL_EMBEDDER_MODEL_PATH):
        return str(DEFAULT_LOCAL_EMBEDDER_MODEL_PATH)

    return configured_model_path


def describe_embedder_model_path(model_path: str) -> dict:
    """Returns safe diagnostics for startup logs."""
    path = Path(model_path)
    exists = path.exists()
    entries = []
    if exists and path.is_dir():
        entries = sorted(child.name for child in path.iterdir())[:30]

    return {
        "embedder_model_path": model_path,
        "embedder_model_path_exists": exists,
        "embedder_model_path_is_dir": path.is_dir(),
        "embedder_model_modules_json_exists": (path / "modules.json").exists(),
        "embedder_model_config_json_exists": (path / "config.json").exists(),
        "embedder_model_entries": entries,
        "embedder_model_path_env": os.environ.get(EMBEDDER_MODEL_PATH_ENV, ""),
        "hf_home": os.environ.get("HF_HOME", ""),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", ""),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE", ""),
    }


def _ensure_local_embedder_model(model_path: Path) -> None:
    missing_files = _missing_required_files(model_path)
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(
            f"{EMBEDDER_MODEL_PATH_ENV}={model_path} is missing required embedder files: {missing}. "
            "The Docker image should copy the bucket model directory to this path before startup."
        )


def _is_complete_local_embedder_model(model_path: Path) -> bool:
    return model_path.is_dir() and not _missing_required_files(model_path)


def _missing_required_files(model_path: Path) -> list[str]:
    return [file_name for file_name in REQUIRED_EMBEDDER_MODEL_FILES if not (model_path / file_name).is_file()]
