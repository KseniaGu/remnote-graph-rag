import json
import logging
import os
import pickle
import threading
from pathlib import Path
from typing import Any

import structlog
import yaml


def write_file(file: Any, path: Path):
    """Writes data to a file with the specified format.

    Args:
        file: The data to be written to the file.
        path: The path where the file should be written.
    """
    extension = os.path.splitext(path)[1]
    if extension == ".pickle":
        with open(path, "wb") as f:
            pickle.dump(file, f)
    elif extension in (".txt", ".md"):
        with open(path, "w", encoding='utf-8') as f:
            f.write(file)
    elif extension == ".json":
        with open(path, "w") as f:
            json.dump(file, f)
    else:
        raise ValueError(f"Unsupported extension: {extension} (.pickle, .txt, .md, .json are only available for now)")


def read_file(path: Path) -> Any:
    """Reads data from a file with the specified format.

    Args:
        path: The path to the file to be read.

    Returns:
        The data read from the file.
    """
    extension = os.path.splitext(path)[1]
    try:
        if extension == ".pickle":
            with open(path, "rb") as f:
                file = pickle.load(f)
        elif extension == ".yaml":
            with open(path, "r") as f:
                file = yaml.safe_load(f)
        elif extension == ".txt":
            file = open(path, "r").read()
        elif extension == ".json":
            with open(path, "r") as f:
                file = json.load(f)
        else:
            print(f"Unsupported extension: {extension} (.pickle, .yaml, .txt, .json are only available)")
            return None
    except FileNotFoundError:
        print(f"File {path} not found")
        return None
    return file


_configure_lock = threading.Lock()
_configured = False

_shared_processors = [
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.ExceptionRenderer(),
]


def _configure_logging() -> None:
    """Configures structlog globally with async-compatible JSON output."""
    global _configured
    if _configured:
        return
    with _configure_lock:
        if _configured:
            return

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
                foreign_pre_chain=_shared_processors,
            )
        )

        logging.root.setLevel(logging.INFO)
        logging.root.addHandler(stream_handler)

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                *_shared_processors,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        _configured = True


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Gets a configured structlog bound logger that emits structured JSON log lines.

    Args:
        name: The name of the logger. Defaults to the module name.
    """
    _configure_logging()
    return structlog.get_logger(name)
