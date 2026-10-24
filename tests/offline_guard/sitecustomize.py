"""Block and record network attempts before test or child-process imports."""

import os
import sys
from pathlib import Path


def _deny_network(event: str, args: tuple) -> None:
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
        with Path(os.environ["OFFLINE_NETWORK_LOG"]).open("a") as log:
            log.write(f"{event}\n")
        raise PermissionError(f"Offline tests forbid network access: {event}")
    if event == "open" and isinstance(args[0], (str, bytes)):
        path = Path(os.fsdecode(args[0]))
        resolved_path = path.resolve()
        if any(
            candidate.name == ".env" or candidate.name.startswith(".env.")
            for candidate in (path, resolved_path)
        ):
            raise PermissionError("Offline tests forbid dotenv file access")
        root = Path(os.environ["OFFLINE_REPOSITORY_ROOT"]).resolve()
        if any(
            resolved_path.is_relative_to((root / name).resolve())
            for name in ("data", "storage", "models")
        ):
            raise PermissionError("Offline tests forbid private asset access")


if os.environ.get("OFFLINE_NETWORK_LOG"):
    sys.addaudithook(_deny_network)
