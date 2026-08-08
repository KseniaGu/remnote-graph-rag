"""External artifact preparation for the optimized RemNote parser."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any


def parsed_roots_from_settings(path_settings: Any) -> list[Path]:
    """Returns configured parsed artifact roots in resolver priority order."""

    roots: list[Path] = []
    for attr in ("parsed_images_dir", "parsed_pdfs_dir", "parsed_texts_dir"):
        value = getattr(path_settings, attr, None)
        if value is not None:
            roots.append(Path(value))
    return roots


def ensure_parsed_dirs(path_settings: Any) -> None:
    """Creates configured parsed artifact roots."""

    for root in parsed_roots_from_settings(path_settings):
        root.mkdir(parents=True, exist_ok=True)


def copy_existing_artifacts(source_dir: Path | None, target_dir: Path) -> int:
    """Copies reviewed cached OCR Markdown artifacts into an isolated parsed-image cache.

    Only Markdown files are copied, relative paths are preserved, and existing
    non-empty target files are left intact.
    """

    if source_dir is None:
        return 0
    source_root = Path(source_dir)
    target_root = Path(target_dir)
    if not source_root.exists():
        return 0
    try:
        if source_root.resolve() == target_root.resolve():
            return 0
    except OSError:
        pass

    target_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source_path in sorted(source_root.rglob("*.md")):
        if not source_path.is_file():
            continue
        relative_path = source_path.relative_to(source_root)
        target_path = target_root / relative_path
        if target_path.exists() and target_path.stat().st_size > 0:
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied += 1
    return copied


def prepare_external_artifacts(
    path_settings: Any,
    *,
    get_ocr_pipeline: Callable[[], Any],
) -> int:
    """Downloads and parse unresolved external resources into the configured caches."""

    import requests

    from backend.configs.enums import (
        ImageParsingStatus,
        PDFParsingStatus,
        TextParsingStatus,
    )
    from backend.data_processing.parser_optimized import (
        OptimizedRemNoteParser,
        is_bad_artifact_path,
    )
    from backend.data_processing.utils import (
        save_image_by_url,
        save_pdf_by_url,
        save_text_by_url,
    )

    ensure_parsed_dirs(path_settings)
    preliminary = OptimizedRemNoteParser(
        Path(path_settings.raw_data_dir),
        parsed_roots=parsed_roots_from_settings(path_settings),
    ).run()
    unresolved_resources = [
        resource
        for resource in preliminary.external_resources
        if resource.parse_status == "not_resolved"
    ]
    if not unresolved_resources:
        return 0

    prepared_count = 0
    for resource in unresolved_resources:
        try:
            response = requests.get(resource.url, timeout=30)
            response.raise_for_status()
            content_type = (
                response.headers.get("Content-Type", "")
                .split(";", 1)[0]
                .strip()
                .casefold()
            )
            artifact_path: Path | None = None

            if "image" in content_type or resource.content_type_hint == "image":
                status, saved_file_path = save_image_by_url(
                    response,
                    content_type or "image/jpeg",
                    resource.label,
                    resource.url,
                    Path(path_settings.parsed_images_dir),
                )
                if status in (
                    ImageParsingStatus.file_exists,
                    ImageParsingStatus.success,
                ) and not is_bad_artifact_path(str(saved_file_path)):
                    artifact_path = get_ocr_pipeline().parse_image(saved_file_path)

            elif (
                "pdf" in content_type or resource.content_type_hint == "application/pdf"
            ):
                status, saved_file_path = save_pdf_by_url(
                    response,
                    resource.label,
                    resource.url,
                    Path(path_settings.parsed_pdfs_dir),
                )
                if status in (
                    PDFParsingStatus.file_exists,
                    PDFParsingStatus.success,
                ) and not is_bad_artifact_path(str(saved_file_path)):
                    artifact_path = get_ocr_pipeline().parse_pdf(saved_file_path)

            elif content_type.startswith("text/") or content_type in {
                "application/xhtml+xml",
                "application/xml",
            }:
                status, saved_file_path = save_text_by_url(
                    resource.label,
                    resource.url,
                    Path(path_settings.parsed_texts_dir),
                )
                if status in (
                    TextParsingStatus.file_exists,
                    TextParsingStatus.success,
                ) and not is_bad_artifact_path(str(saved_file_path)):
                    artifact_path = saved_file_path

            if artifact_path and Path(artifact_path).exists():
                prepared_count += 1
        except Exception:
            continue

    return prepared_count
