import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

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


def get_logger(name: str = __name__) -> logging.Logger:
    """Gets a configured logger instance with a memory handler.

    Args:
        name: The name of the logger. Defaults to the module name.
    """
    logger = logging.getLogger(name)
    base_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=base_format, handlers=[logging.StreamHandler()])

    return logger
