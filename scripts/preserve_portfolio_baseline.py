"""Preserve and restore a private B0 source/data checkpoint without service calls."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], stderr=subprocess.PIPE
    )


def safe_path(relative: str, *, source_file: bool = False) -> Path:
    path = Path(relative)
    if source_file and path == Path(".env.example"):
        return path
    if (
        path.is_absolute()
        or ".." in path.parts
        or any(
            part.lower() in {".env", "credentials.json", "secrets.yml", ".git"}
            or part.lower().startswith(".env.")
            for part in path.parts
        )
    ):
        raise ValueError("Unsafe or credential-bearing snapshot path")
    return path


def inventory(root: Path) -> dict:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Snapshot inputs must not contain symlinks")
        if path.is_file() and path.name != ".DS_Store":
            relative = path.relative_to(root).as_posix()
            safe_path(relative)
            files[relative] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return files


def canonical_ledger_binding(source: Path) -> str:
    """Carries the original ledger location without reading or creating campaign state."""
    common = Path(git(source, "rev-parse", "--git-common-dir").decode().strip())
    common = (source / common).resolve()
    marker = common / "portfolio-restored-source"
    if marker.exists():
        binding = json.loads(marker.read_text())["canonical_ledger"]
        if not isinstance(binding, str) or not Path(binding).is_absolute():
            raise ValueError(
                "Restored source lacks its original canonical ledger binding"
            )
        return str(Path(binding).resolve())
    return str(common.parent / "data/evaluation/portfolio_budget/ledger.json")


def capture(args) -> None:
    source = args.source_root.resolve()
    asset_root = args.asset_root.resolve()
    paths = [safe_path(name) for name in args.asset]
    for relative in paths:
        candidate = asset_root / relative
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(asset_root):
            raise ValueError("Snapshot asset escapes the asset root")
        if not candidate.exists():
            raise FileNotFoundError(relative)
        if candidate.is_dir():
            inventory(candidate)
    if git(source, "ls-files", "--others", "--exclude-standard", "-z"):
        raise ValueError("Stage untracked source files before preservation")
    tracked = git(source, "ls-files", "-z").decode().split("\0")
    for name in filter(None, tracked):
        safe_path(name, source_file=True)
    archived_names = (
        git(source, "ls-tree", "--full-tree", "-r", "--name-only", "-z", "HEAD")
        .decode()
        .split("\0")
    )
    for name in filter(None, archived_names):
        safe_path(name, source_file=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "assets").mkdir(mode=0o700)
    commit = git(source, "rev-parse", "HEAD").decode().strip()
    (output / "source.tar").write_bytes(git(source, "archive", "--format=tar", "HEAD"))
    (output / "source-commit.object").write_bytes(
        git(source, "cat-file", "commit", "HEAD")
    )
    (output / "source.patch").write_bytes(git(source, "diff", "--binary", "HEAD", "--"))
    (output / "source-status.txt").write_bytes(
        git(source, "status", "--porcelain=v1", "--untracked-files=all")
    )
    source_files = {}
    for name in filter(None, tracked):
        path = source / name
        if path.is_symlink():
            raise ValueError("Tracked symlinks require explicit preservation review")
        if path.is_file():
            source_files[name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    for relative in paths:
        original = asset_root / relative
        destination = output / "assets" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if original.is_dir():
            shutil.copytree(
                original, destination, ignore=shutil.ignore_patterns(".DS_Store")
            )
        else:
            shutil.copy2(original, destination)
    if (
        git(source, "diff", "--binary", "HEAD", "--")
        != (output / "source.patch").read_bytes()
    ):
        raise ValueError("Source changed during capture; discard and recapture")
    for name, expected in source_files.items():
        if digest(source / name) != expected["sha256"]:
            raise ValueError("Source changed during capture; discard and recapture")
    write_json(output / "source-files.json", source_files)
    manifest = {
        "schema": "portfolio-b0-snapshot-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_commit": commit,
        "source_tree": git(source, "rev-parse", "HEAD^{tree}").decode().strip(),
        "source_tree_clean": not bool(git(source, "status", "--porcelain=v1")),
        "source_root_at_capture": str(source),
        "canonical_ledger_binding": canonical_ledger_binding(source),
        "asset_root_at_capture": str(asset_root),
        "private_access": "Requires filesystem access to the complete private snapshot; never publish assets or raw outputs.",
        "files": inventory(output),
    }
    write_json(output / "manifest.json", manifest)
    (output / "manifest.sha256").write_text(digest(output / "manifest.json") + "\n")
    print(
        json.dumps(
            {
                "snapshot": str(output),
                "source_commit": commit,
                "source_tree": git(source, "rev-parse", "HEAD^{tree}").decode().strip(),
                "files": len(manifest["files"]),
            }
        )
    )


def verify(snapshot: Path, restored_root: Path | None = None) -> dict:
    if (
        digest(snapshot / "manifest.json")
        != (snapshot / "manifest.sha256").read_text().strip()
    ):
        raise ValueError("Snapshot manifest checksum mismatch")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = snapshot / safe_path(relative)
        if (
            path.is_symlink()
            or not path.is_file()
            or digest(path) != expected["sha256"]
        ):
            raise ValueError(f"Snapshot checksum mismatch: {relative}")
    if restored_root is not None:
        source_files = json.loads((snapshot / "source-files.json").read_text())
        asset_files = {
            name.removeprefix("assets/"): entry
            for name, entry in manifest["files"].items()
            if name.startswith("assets/")
        }
        for source_file, expected_files in ((True, source_files), (False, asset_files)):
            for relative, expected in expected_files.items():
                path = restored_root / safe_path(relative, source_file=source_file)
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or digest(path) != expected["sha256"]
                ):
                    raise ValueError(f"Restored checksum mismatch: {relative}")
    return manifest


def restore(args) -> None:
    snapshot = args.snapshot.resolve()
    manifest = verify(snapshot)
    destination = args.destination.resolve()
    if destination.exists():
        raise ValueError("Restore destination must not exist")
    destination.mkdir(parents=True)
    with tarfile.open(snapshot / "source.tar") as archive:
        archive.extractall(destination, filter="data")
    git(destination, "init", "-q")
    write_json(
        destination / ".git" / "portfolio-restored-source",
        {
            "canonical_ledger": manifest.get("canonical_ledger_binding"),
            "snapshot_manifest_sha256": digest(snapshot / "manifest.json"),
        },
    )
    git(destination, "add", "-A")
    if git(destination, "write-tree").decode().strip() != manifest["source_tree"]:
        raise ValueError("Restored source tree differs from original commit")
    restored_commit = (
        git(
            destination,
            "hash-object",
            "-t",
            "commit",
            "-w",
            str(snapshot / "source-commit.object"),
        )
        .decode()
        .strip()
    )
    if restored_commit != manifest["source_commit"]:
        raise ValueError("Restored source commit identity differs")
    git(destination, "update-ref", "HEAD", restored_commit)
    if (snapshot / "source.patch").stat().st_size:
        git(destination, "apply", "--binary", str(snapshot / "source.patch"))
    shutil.copytree(snapshot / "assets", destination, dirs_exist_ok=True)
    verify(snapshot, destination)
    print(json.dumps({"restored": str(destination), "verified": True}))


def deny_network() -> None:
    """Deny Internet sockets in this process and descendants before application imports."""
    library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_release.argtypes = [ctypes.c_void_p]

    class Comparison(ctypes.Structure):
        _fields_ = [
            ("arg", ctypes.c_uint),
            ("op", ctypes.c_int),
            ("value", ctypes.c_uint64),
            ("unused", ctypes.c_uint64),
        ]

    context = library.seccomp_init(0x7FFF0000)
    if not context:
        raise RuntimeError("Cannot initialize network denial")
    try:
        syscall = library.seccomp_syscall_resolve_name(b"socket")
        for domain in (2, 10):
            result = library.seccomp_rule_add(
                context,
                0x00050000 | errno.EPERM,
                syscall,
                1,
                Comparison(0, 4, domain, 0),
            )
            if result != 0:
                raise RuntimeError("Cannot install network denial rule")
        if library.seccomp_load(context) != 0:
            raise RuntimeError("Cannot activate network denial")
    finally:
        library.seccomp_release(context)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "LANGCHAIN_TRACING_V2": "false",
            "LANGSMITH_TRACING": "false",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": "offline",
        }
    )


def offline_run(command: list[str]) -> None:
    deny_network()
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("An offline Python script or -m module is required")
    sys.path.insert(0, str(Path.cwd()))
    if command[0] == "-m":
        sys.argv = command[1:]
        runpy.run_module(command[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = command
        runpy.run_path(command[0], run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--source-root", type=Path, required=True)
    capture_parser.add_argument("--asset-root", type=Path, required=True)
    capture_parser.add_argument("--asset", action="append", required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--snapshot", type=Path, required=True)
    verify_parser.add_argument("--restored-root", type=Path)
    restore_parser = commands.add_parser("restore")
    restore_parser.add_argument("--snapshot", type=Path, required=True)
    restore_parser.add_argument("--destination", type=Path, required=True)
    offline_parser = commands.add_parser("offline-run")
    offline_parser.add_argument("python_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "capture":
        capture(args)
    elif args.command == "restore":
        restore(args)
    elif args.command == "verify":
        verify(args.snapshot, args.restored_root)
        print("Snapshot integrity verified")
    else:
        offline_run(args.python_args)


if __name__ == "__main__":
    main()
