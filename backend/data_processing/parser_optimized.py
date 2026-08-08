"""Shadow RemNote ingestion and chunking pipeline.

This module intentionally does not replace the production parser. It is a
copy-inspired, provenance-first experiment that reads RemNote markdown exports
and cached external parse artifacts into a typed intermediate representation.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from backend.data_processing.parser_outputs import (
    result_to_jsonable,
    write_comparison_markdown,
    write_optimized_parser_ir,
)
from backend.utils.common_funcs import write_json, write_jsonl

REMNOTE_IMAGE_HOST_MARKER = "remnote-user-data.s3.amazonaws.com"
IMAGE_PLACEHOLDER = "[IMG URL]"
FILENAME_LENGTH_MAX = 100
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif")

MARKDOWN_URL_RE = re.compile(r"(!?)\[(?P<name>[^\]]*)\]\((?P<url>https?://[^)]+)\)")
RAW_URL_RE = re.compile(r"https?://[^\s)]+")
HEADER_RE = re.compile(r"^[\s-]*(#+)")
HEADER_TEXT_RE = re.compile(r"^#{1,6}\s+")
ORDERED_LIST_ITEM_RE = re.compile(r"^\d+\.\s+")
ONLY_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
CODE_FENCE_RE = re.compile(r"^(?:`{3,}|~{3,})(?:\s*[A-Za-z0-9_+.#-]+)?\s*$")
ARTIFACT_URL_RE = re.compile(
    r"^url:\s*(?P<url>https?://\S+)\s*$", re.MULTILINE | re.IGNORECASE
)
TRACKING_QUERY_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "source"}

READABLE_ENGLISH_WORDS = {
    "a",
    "about",
    "agent",
    "agents",
    "all",
    "allow",
    "allows",
    "an",
    "and",
    "api",
    "approach",
    "are",
    "arguments",
    "as",
    "at",
    "bayes",
    "be",
    "between",
    "bridges",
    "by",
    "call",
    "can",
    "class",
    "compute",
    "configuration",
    "connect",
    "could",
    "decide",
    "endpoint",
    "estimate",
    "estimation",
    "example",
    "examples",
    "extension",
    "extensions",
    "external",
    "feature",
    "figure",
    "for",
    "from",
    "gap",
    "gaussian",
    "goal",
    "how",
    "if",
    "image",
    "in",
    "independently",
    "into",
    "is",
    "it",
    "likelihood",
    "mean",
    "model",
    "more",
    "naive",
    "needed",
    "new",
    "of",
    "on",
    "or",
    "parameters",
    "part",
    "per",
    "prior",
    "priors",
    "provided",
    "query",
    "resilient",
    "rule",
    "runtime",
    "sample",
    "should",
    "solving",
    "steps",
    "successfully",
    "suitable",
    "teaching",
    "text",
    "that",
    "the",
    "this",
    "to",
    "training",
    "use",
    "used",
    "user",
    "uses",
    "using",
    "variance",
    "via",
    "what",
    "which",
    "with",
    "would",
}
FORMULA_WORDS = {
    "alpha",
    "begin",
    "beta",
    "cdot",
    "delta",
    "ell",
    "end",
    "exp",
    "frac",
    "gamma",
    "geq",
    "infty",
    "lambda",
    "langle",
    "ldots",
    "leq",
    "left",
    "mathbb",
    "mathcal",
    "mathrm",
    "min",
    "max",
    "mid",
    "mu",
    "operatorname",
    "phi",
    "pi",
    "prod",
    "quad",
    "rangle",
    "right",
    "sigma",
    "sqrt",
    "sum",
    "text",
    "theta",
    "times",
    "varepsilon",
    "xi",
}
HTML_WORDS = {
    "alt",
    "center",
    "div",
    "height",
    "image",
    "img",
    "src",
    "style",
    "text-align",
    "width",
}
QUALITY_TOKEN_RE = re.compile(r"[A-Za-zΑ-Ωα-ωА-Яа-яЁё][A-Za-zΑ-Ωα-ωА-Яа-яЁё'-]{1,}")

__all__ = [
    "ArtifactGateDecision",
    "CachedArtifactResolver",
    "ExternalResource",
    "OptimizedParseResult",
    "OptimizedRemNoteParser",
    "ParsedArtifact",
    "RemNoteBlock",
    "RemNoteParserOptimized",
    "RetrievalChunk",
    "SourceDocument",
    "UrlMatch",
    "result_to_jsonable",
    "write_comparison_markdown",
    "write_json",
    "write_jsonl",
]


@dataclass(frozen=True)
class UrlMatch:
    """One URL occurrence found in a RemNote markdown line."""

    name: str | None
    url: str
    kind: str
    start: int
    end: int
    ordinal: int


@dataclass
class SourceDocument:
    """One normalized RemNote markdown file."""

    id: str
    source: str
    relative_path: str
    path: str
    line_count: int
    nonempty_line_count: int
    url_count: int


@dataclass
class RemNoteBlock:
    """A raw RemNote line/block preserved for provenance."""

    id: str
    source_document_id: str
    source: str
    line_number: int
    block_ordinal: int
    raw_text: str
    text: str
    depth_level: int
    path: list[str]
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)
    external_resource_ids: list[str] = field(default_factory=list)


@dataclass
class ExternalResource:
    """A URL/resource occurrence linked to its parent RemNote block."""

    id: str
    parent_block_id: str
    source: str
    line_number: int
    url: str
    url_hash: str
    label: str | None
    kind: str
    content_type_hint: str
    parse_status: str
    artifact_path: str | None = None
    artifact_type: str | None = None
    error: str | None = None


@dataclass
class ParsedArtifact:
    """A cached parsed external artifact discovered during the shadow run."""

    id: str
    external_resource_id: str
    artifact_path: str
    artifact_type: str
    line_count: int
    nonempty_line_count: int
    char_count: int
    text_preview: str


@dataclass
class ArtifactGateDecision:
    """Deterministic admission decision for one cached external artifact."""

    id: str
    external_resource_id: str
    artifact_path: str
    policy: str
    reason_codes: list[str]
    content_hash: str
    normalized_source_url: str
    declared_artifact_url: str | None
    normalized_declared_artifact_url: str | None
    stats: dict[str, Any]
    emitted_chunk_count: int = 0


@dataclass
class RetrievalChunk:
    """A coherent retrieval unit derived from raw blocks or parsed artifacts."""

    id: str
    text: str
    chunk_type: str
    source: str
    path: list[str]
    line_start: int
    line_end: int
    source_block_ids: list[str]
    external_resource_ids: list[str] = field(default_factory=list)
    parent_block_id: str | None = None
    context_block_ids: list[str] = field(default_factory=list)
    context_text: str | None = None
    source_relation: str | None = None
    artifact_path: str | None = None
    artifact_line_start: int | None = None
    artifact_line_end: int | None = None
    chunk_role: str = "paragraph_group"
    heading_path: list[str] = field(default_factory=list)
    display_text: str | None = None
    embedding_text: str | None = None
    chunk_quality_flags: list[str] = field(default_factory=list)


@dataclass
class ChunkCandidate:
    """Internal atomic retrieval candidate before sibling packing."""

    text: str
    source: str
    heading_path: list[str]
    blocks: list[RemNoteBlock]
    external_resource_ids: list[str]
    role: str
    quality_flags: list[str] = field(default_factory=list)


@dataclass
class OptimizedParseResult:
    """Completes output of the shadow ingestion experiment."""

    source_documents: list[SourceDocument]
    blocks: list[RemNoteBlock]
    external_resources: list[ExternalResource]
    parsed_artifacts: list[ParsedArtifact]
    artifact_gate_decisions: list[ArtifactGateDecision]
    retrieval_chunks: list[RetrievalChunk]
    summary: dict[str, Any]


def normalize_nfc(value: str) -> str:
    """Normalizes user/export text for stable metadata and comparisons."""

    return unicodedata.normalize("NFC", value)


def stable_hash(*parts: Any, length: int = 16) -> str:
    joined = "\x1f".join(normalize_nfc(str(part)) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]


def stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}_{stable_hash(*parts)}"


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def clean_remnote_references(text: str) -> str:
    text = re.sub(r"\[\[(.*?)\|.*?\]\]", r"\1", text)
    return re.sub(r"\[\[(.*?)\]\]", r"\1", text)


def clean_clozes(text: str) -> str:
    return re.sub(r"\{\{c\d+::(.*?)(::.*?)?\}\}", r"\1", text)


def strip_markdown_formatting(text: str) -> str:
    text = text.replace("**", "").replace("__", "")
    return re.sub(r"!\[(.*?)\]\(.*?\)", r"\1", text)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = normalize_nfc(text)
    text = normalize_whitespace(text)
    text = clean_remnote_references(text)
    text = clean_clozes(text)
    text = strip_markdown_formatting(text)
    return text.strip()


def extract_url_matches(line: str) -> list[UrlMatch]:
    """Extracts all Markdown and raw URLs from a single line."""

    matches: list[UrlMatch] = []
    occupied_spans: list[tuple[int, int]] = []

    for match in MARKDOWN_URL_RE.finditer(line):
        name = match.group("name") or None
        url = match.group("url")
        if name == url:
            name = None
        kind = "image" if match.group(1) else "link"
        matches.append(
            UrlMatch(
                name=normalize_nfc(name) if name else None,
                url=url,
                kind=kind,
                start=match.start(),
                end=match.end(),
                ordinal=len(matches),
            )
        )
        occupied_spans.append((match.start("url"), match.end("url")))

    def inside_markdown_span(start: int, end: int) -> bool:
        return any(
            start >= span_start and end <= span_end
            for span_start, span_end in occupied_spans
        )

    for match in RAW_URL_RE.finditer(line):
        if inside_markdown_span(match.start(), match.end()):
            continue
        matches.append(
            UrlMatch(
                name=None,
                url=match.group(0),
                kind="raw",
                start=match.start(),
                end=match.end(),
                ordinal=len(matches),
            )
        )

    return sorted(matches, key=lambda item: (item.start, item.end))


def infer_depth(
    stripped_line: str,
    found_headers: set[int],
    indent_level: int,
    header_bonus: int | None,
) -> tuple[int, int | None]:
    """Mirrors the production parser's depth heuristic without touching it."""

    header_match = HEADER_RE.search(stripped_line)
    if header_match:
        header_length = len(header_match.group(1))
        header_bonus = header_length
        for i in reversed(range(1, header_length)):
            if i not in found_headers:
                header_bonus -= 1
        found_headers.add(header_length)
        header_bonus -= 1
        return indent_level + header_bonus, header_bonus

    bonus = header_bonus + 1 if header_bonus is not None else 0
    return indent_level + bonus, header_bonus


def guess_content_type_hint(url: str, kind: str) -> str:
    path = urlparse(url).path.casefold()
    if kind == "image" or any(path.endswith(ext) for ext in IMAGE_EXTENSIONS):
        return "image"
    if path.endswith(".pdf"):
        return "application/pdf"
    if path.endswith((".md", ".txt", ".html", ".htm")):
        return "text"
    return "unknown"


def sanitize_artifact_name(name: str | None, url: str) -> str:
    if name:
        clean_name = re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_")
    else:
        clean_name = Path(urlparse(url).path).name
    if not clean_name:
        clean_name = stable_hash(url)
    return clean_name[-FILENAME_LENGTH_MAX:]


def is_bad_artifact_path(path: str | None) -> bool:
    if not path:
        return False
    return path in {".", str(Path())}


def is_code_fence_marker(text: str) -> bool:
    return bool(CODE_FENCE_RE.match(normalize_whitespace(text).strip()))


def normalize_url_for_gate(url: str | None) -> str:
    if not url:
        return ""
    parsed = urlparse(url.strip().rstrip("),."))
    if not parsed.netloc:
        return url.strip().casefold().rstrip("/")
    path = re.sub(r"/+", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.casefold().startswith("utm_")
        and key.casefold() not in TRACKING_QUERY_PARAMS
    ]
    query = urlencode(query_items, doseq=True)
    return urlunparse(("", parsed.netloc.casefold(), path, "", query, ""))


def extract_declared_artifact_url(text: str) -> str | None:
    match = ARTIFACT_URL_RE.search(text)
    return normalize_nfc(match.group("url").strip()) if match else None


def is_dataset_like_url(url: str | None) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    host = parsed.netloc.casefold()
    path = parsed.path.casefold()
    if "paperswithcode.com" in host and "/dataset" in path:
        return True
    if "huggingface.co" in host and "/datasets" in path:
        return True
    if "kaggle.com" in host and "/datasets" in path:
        return True
    return bool(re.search(r"(?:^|/|[_-])datasets?(?:/|[_-]|\.|$)", path))


def artifact_text_looks_like_dataset(text: str) -> bool:
    sample = text[:4000].casefold()
    markers = (
        "datasets at hugging face",
        "dataset card",
        "training samples",
        "testing samples",
        "classification dataset",
    )
    return any(marker in sample for marker in markers)


def is_generic_navigation_artifact(declared_url: str | None, text: str) -> bool:
    normalized = normalize_url_for_gate(declared_url)
    sample = text[:1000].casefold()
    return (
        normalized.endswith("//huggingface.co/papers/trending")
        or "title: trending papers - hugging face" in sample
    )


def has_cyrillic(text: str) -> bool:
    return any("CYRILLIC" in unicodedata.name(char, "") for char in text)


def script_stats(text: str) -> dict[str, Any]:
    alpha_count = 0
    cyrillic_count = 0
    latin_count = 0
    greek_count = 0
    for char in text:
        if not char.isalpha():
            continue
        alpha_count += 1
        name = unicodedata.name(char, "")
        if "CYRILLIC" in name:
            cyrillic_count += 1
        elif "LATIN" in name:
            latin_count += 1
        elif "GREEK" in name:
            greek_count += 1
    latin_greek_count = latin_count + greek_count
    return {
        "char_count": len(text),
        "line_count": len(text.splitlines()),
        "nonempty_line_count": sum(1 for line in text.splitlines() if line.strip()),
        "alpha_count": alpha_count,
        "cyrillic_count": cyrillic_count,
        "latin_count": latin_count,
        "greek_count": greek_count,
        "cyrillic_ratio": cyrillic_count / alpha_count if alpha_count else 0.0,
        "latin_greek_ratio": latin_greek_count / alpha_count if alpha_count else 0.0,
    }


def strip_markup_and_formulas(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$.*?\$", " ", text, flags=re.DOTALL)
    return text


def ocr_text_quality_stats(text: str) -> dict[str, Any]:
    natural_text = strip_markup_and_formulas(text)
    tokens = QUALITY_TOKEN_RE.findall(natural_text)
    content_tokens = [
        token
        for token in tokens
        if token.casefold().strip("'-") not in FORMULA_WORDS | HTML_WORDS
    ]
    token_count = len(content_tokens)
    common_english_count = sum(
        token.casefold().strip("'-") in READABLE_ENGLISH_WORDS
        for token in content_tokens
    )
    mixed_case_count = sum(
        len(token) >= 5
        and any(char.islower() for char in token[1:])
        and any(char.isupper() for char in token[1:])
        for token in content_tokens
    )
    greek_token_count = sum(
        any("GREEK" in unicodedata.name(char, "") for char in token)
        for token in content_tokens
    )
    suspicious_lookalike_count = sum(
        bool(re.search(r"(II|JI|IO|YI|BB|MH|KH|HH|CT|KJ|Π|φ|μ|Γ)", token))
        for token in content_tokens
    )
    return {
        "ocr_quality_token_count": token_count,
        "ocr_quality_common_english_count": common_english_count,
        "ocr_quality_common_english_ratio": common_english_count / token_count
        if token_count
        else 0.0,
        "ocr_quality_mixed_case_token_count": mixed_case_count,
        "ocr_quality_mixed_case_token_ratio": mixed_case_count / token_count
        if token_count
        else 0.0,
        "ocr_quality_greek_token_count": greek_token_count,
        "ocr_quality_greek_token_ratio": greek_token_count / token_count
        if token_count
        else 0.0,
        "ocr_quality_suspicious_lookalike_token_count": suspicious_lookalike_count,
        "ocr_quality_suspicious_lookalike_token_ratio": suspicious_lookalike_count
        / token_count
        if token_count
        else 0.0,
    }


def is_low_quality_ocr(
    text: str, resource: ExternalResource, artifact_path: Path, stats: dict[str, Any]
) -> bool:
    is_image_artifact = (
        resource.content_type_hint == "image" or "parsed_images" in artifact_path.parts
    )
    if not is_image_artifact:
        return False
    if stats["char_count"] < 250 or stats["alpha_count"] < 120:
        return False

    quality_stats = ocr_text_quality_stats(text)
    stats.update(quality_stats)
    token_count = quality_stats["ocr_quality_token_count"]
    if token_count < 12:
        return False

    low_readable_english = quality_stats["ocr_quality_common_english_ratio"] < 0.15
    lookalike_noise = (
        quality_stats["ocr_quality_mixed_case_token_ratio"] >= 0.18
        or quality_stats["ocr_quality_suspicious_lookalike_token_ratio"] >= 0.16
        or quality_stats["ocr_quality_greek_token_ratio"] >= 0.06
    )
    return low_readable_english and lookalike_noise


def is_noise_text(text: str) -> bool:
    stripped = normalize_whitespace(text).strip()
    if not stripped:
        return True
    if stripped == IMAGE_PLACEHOLDER:
        return True
    if len(stripped) <= 3:
        return True
    if is_code_fence_marker(stripped) or stripped in {'"""', "'''", "---", "***"}:
        return True
    if ONLY_PUNCT_RE.match(stripped):
        return True
    return False


def is_header_text(text: str) -> bool:
    return bool(HEADER_TEXT_RE.match(normalize_whitespace(text).strip()))


def clean_embedding_heading(text: str) -> str:
    return HEADER_TEXT_RE.sub("", normalize_whitespace(text).strip()).strip()


def semantic_path_text(path: Iterable[str]) -> str:
    parts = []
    for item in path:
        cleaned = clean_embedding_heading(item)
        if not cleaned or cleaned.startswith("external:"):
            continue
        parts.append(cleaned)
    return " > ".join(parts)


def semantic_context_text(context_text: str | None) -> str | None:
    if not context_text:
        return None
    lines = []
    for raw_line in context_text.splitlines():
        line = normalize_whitespace(raw_line).strip()
        if not line or is_noise_text(line) or is_header_text(line):
            continue
        lines.append(clean_embedding_heading(line))
    return "\n".join(lines) or None


def is_ordered_list_item(text: str) -> bool:
    return bool(ORDERED_LIST_ITEM_RE.match(normalize_whitespace(text).strip()))


def common_path(paths: Iterable[list[str]]) -> list[str]:
    normalized_paths = [list(path) for path in paths if path]
    if not normalized_paths:
        return []
    prefix: list[str] = []
    for values in zip(*normalized_paths):
        if len(set(values)) != 1:
            break
        prefix.append(values[0])
    return prefix


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


class CachedArtifactResolver:
    """Resolves external URLs to already cached parse artifacts without network I/O."""

    def __init__(self, parsed_roots: Iterable[Path]):
        self._by_name: dict[str, Path] = {}
        for root in parsed_roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                self._by_name.setdefault(path.name, path)
                self._by_name.setdefault(path.stem, path)

    def resolve(self, match: UrlMatch) -> tuple[Path | None, str | None]:
        clean_name = sanitize_artifact_name(match.name, match.url)
        suffix_name = Path(urlparse(match.url).path).name[-FILENAME_LENGTH_MAX:]
        suffix = Path(suffix_name).suffix.casefold()
        is_image_url = match.kind == "image" or suffix in IMAGE_EXTENSIONS
        candidates = []
        if is_image_url:
            candidates.extend(
                [
                    Path(clean_name).with_suffix(".md").name if clean_name else "",
                    Path(suffix_name).with_suffix(".md").name if suffix_name else "",
                ]
            )
        candidates.extend(
            [
                f"{clean_name}.md",
                clean_name,
                f"{suffix_name}.md",
                suffix_name,
                Path(suffix_name).with_suffix(".md").name if suffix_name else "",
            ]
        )
        for candidate in unique_preserving_order(candidates):
            if candidate and candidate in self._by_name:
                path = self._by_name[candidate]
                artifact_type = "markdown" if path.suffix == ".md" else "binary"
                return path, artifact_type
        return None, None


class OptimizedRemNoteParser:
    """Shadows parser that produces provenance blocks and retrieval chunks."""

    def __init__(self, raw_data_dir: Path, parsed_roots: Iterable[Path] = ()):
        self.raw_data_dir = Path(raw_data_dir)
        self.artifact_resolver = CachedArtifactResolver(parsed_roots)

    def run(
        self, baseline_summary: dict[str, Any] | None = None
    ) -> OptimizedParseResult:
        source_documents: list[SourceDocument] = []
        blocks: list[RemNoteBlock] = []
        external_resources: list[ExternalResource] = []
        parsed_artifacts: list[ParsedArtifact] = []

        for file_path in sorted(self.raw_data_dir.rglob("*.md")):
            doc, doc_blocks, doc_resources, doc_artifacts = self._parse_source_document(
                file_path
            )
            source_documents.append(doc)
            blocks.extend(doc_blocks)
            external_resources.extend(doc_resources)
            parsed_artifacts.extend(doc_artifacts)

        retrieval_chunks, artifact_gate_decisions = self._build_retrieval_chunks(
            blocks, parsed_artifacts, external_resources
        )
        summary = self._build_summary(
            source_documents,
            blocks,
            external_resources,
            parsed_artifacts,
            artifact_gate_decisions,
            retrieval_chunks,
            baseline_summary,
        )
        return OptimizedParseResult(
            source_documents=source_documents,
            blocks=blocks,
            external_resources=external_resources,
            parsed_artifacts=parsed_artifacts,
            artifact_gate_decisions=artifact_gate_decisions,
            retrieval_chunks=retrieval_chunks,
            summary=summary,
        )

    def _parse_source_document(
        self, file_path: Path
    ) -> tuple[
        SourceDocument, list[RemNoteBlock], list[ExternalResource], list[ParsedArtifact]
    ]:
        raw_lines = file_path.read_text(encoding="utf-8").splitlines()
        relative_path = normalize_nfc(str(file_path.relative_to(self.raw_data_dir)))
        source = normalize_nfc(file_path.stem)
        source_document_id = stable_id("src", relative_path, source)

        blocks: list[RemNoteBlock] = []
        resources: list[ExternalResource] = []
        artifacts: list[ParsedArtifact] = []
        node_stack: list[RemNoteBlock | None] = []
        context_stack: list[str] = []
        found_headers: set[int] = set()
        header_bonus: int | None = None
        block_ordinal = 0
        url_count = 0

        for line_number, line in enumerate(raw_lines):
            stripped_line = normalize_nfc(line.strip())
            if not stripped_line or stripped_line.startswith("---"):
                continue

            indent_spaces = len(line) - len(line.lstrip())
            indent_level = indent_spaces // 2
            depth_level, header_bonus = infer_depth(
                stripped_line, found_headers, indent_level, header_bonus
            )

            raw_content = stripped_line.lstrip("-*").strip()
            if not raw_content:
                continue

            cleaned_content = clean_text(raw_content)
            if not cleaned_content.strip() and REMNOTE_IMAGE_HOST_MARKER in raw_content:
                cleaned_content = IMAGE_PLACEHOLDER

            if depth_level < len(node_stack):
                node_stack = node_stack[:depth_level]
                context_stack = context_stack[:depth_level]
            while len(node_stack) < depth_level:
                node_stack.append(node_stack[-1] if node_stack else None)
                context_stack.append("...")

            path = [source] + [
                item
                for item in context_stack
                if item != "..." and not is_code_fence_marker(item)
            ]
            parent = next(
                (item for item in reversed(node_stack) if item is not None), None
            )
            block_id = stable_id(
                "block", relative_path, line_number, block_ordinal, raw_content
            )
            block = RemNoteBlock(
                id=block_id,
                source_document_id=source_document_id,
                source=source,
                line_number=line_number,
                block_ordinal=block_ordinal,
                raw_text=raw_content,
                text=cleaned_content,
                depth_level=depth_level,
                path=path,
                parent_id=parent.id if parent else None,
            )

            if parent:
                parent.child_ids.append(block.id)

            for match in extract_url_matches(raw_content):
                resource, artifact = self._resource_from_match(match, block)
                block.external_resource_ids.append(resource.id)
                resources.append(resource)
                if artifact:
                    artifacts.append(artifact)
                url_count += 1

            blocks.append(block)
            context_stack.append(cleaned_content)
            node_stack.append(block)
            block_ordinal += 1

        doc = SourceDocument(
            id=source_document_id,
            source=source,
            relative_path=relative_path,
            path=normalize_nfc(str(file_path)),
            line_count=len(raw_lines),
            nonempty_line_count=sum(1 for line in raw_lines if line.strip()),
            url_count=url_count,
        )
        return doc, blocks, resources, artifacts

    def _resource_from_match(
        self, match: UrlMatch, block: RemNoteBlock
    ) -> tuple[ExternalResource, ParsedArtifact | None]:
        url_hash = stable_hash(match.url)
        resource_id = stable_id("res", block.id, match.ordinal, match.url)
        artifact_path, artifact_type = self.artifact_resolver.resolve(match)
        parse_status = "cached" if artifact_path else "not_resolved"
        content_type_hint = guess_content_type_hint(match.url, match.kind)

        resource = ExternalResource(
            id=resource_id,
            parent_block_id=block.id,
            source=block.source,
            line_number=block.line_number,
            url=match.url,
            url_hash=url_hash,
            label=match.name,
            kind=match.kind,
            content_type_hint=content_type_hint,
            parse_status=parse_status,
            artifact_path=normalize_nfc(str(artifact_path)) if artifact_path else None,
            artifact_type=artifact_type,
        )

        if not artifact_path or artifact_path.suffix != ".md":
            return resource, None

        text = artifact_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        preview = normalize_whitespace(text)[:240]
        artifact = ParsedArtifact(
            id=stable_id("artifact", resource.id, artifact_path),
            external_resource_id=resource.id,
            artifact_path=normalize_nfc(str(artifact_path)),
            artifact_type="markdown",
            line_count=len(lines),
            nonempty_line_count=sum(1 for line in lines if line.strip()),
            char_count=len(text),
            text_preview=preview,
        )
        return resource, artifact

    def _build_retrieval_chunks(
        self,
        blocks: list[RemNoteBlock],
        parsed_artifacts: list[ParsedArtifact],
        external_resources: list[ExternalResource],
        min_chars: int = 180,
        max_chars: int = 5500,
    ) -> tuple[list[RetrievalChunk], list[ArtifactGateDecision]]:
        chunks: list[RetrievalChunk] = []
        artifact_gate_decisions: list[ArtifactGateDecision] = []
        seen_embedded_artifact_hashes: set[str] = set()
        resources_by_id = {resource.id: resource for resource in external_resources}
        blocks_by_id = {block.id: block for block in blocks}
        children_by_id = {
            block.id: [
                blocks_by_id[child_id]
                for child_id in block.child_ids
                if child_id in blocks_by_id
            ]
            for block in blocks
        }

        def ordered_children(block: RemNoteBlock) -> list[RemNoteBlock]:
            return sorted(
                children_by_id.get(block.id, []),
                key=lambda item: (item.line_number, item.block_ordinal),
            )

        def walk_subtree(block: RemNoteBlock) -> list[RemNoteBlock]:
            subtree = [block]
            for child in ordered_children(block):
                subtree.extend(walk_subtree(child))
            return subtree

        def candidate_role(block: RemNoteBlock) -> str:
            if is_ordered_list_item(block.text):
                return "list_item"
            if block.child_ids:
                return "subtree"
            return "paragraph"

        def make_candidate(block: RemNoteBlock) -> ChunkCandidate | None:
            subtree = walk_subtree(block)
            text_blocks = [
                item
                for item in subtree
                if not is_noise_text(item.text) and not is_header_text(item.text)
            ]
            resource_ids = unique_preserving_order(
                resource_id
                for item in subtree
                for resource_id in item.external_resource_ids
            )
            if not text_blocks:
                return None

            text = "\n".join(item.text for item in text_blocks).strip()
            heading_path = common_path(item.path for item in text_blocks) or [
                block.source
            ]
            quality_flags: list[str] = []
            if resource_ids:
                quality_flags.append("resource_attached")
            if len(text) > max_chars:
                quality_flags.append("oversized_subtree")
            return ChunkCandidate(
                text=text,
                source=block.source,
                heading_path=heading_path,
                blocks=subtree,
                external_resource_ids=resource_ids,
                role=candidate_role(block),
                quality_flags=quality_flags,
            )

        emitted_block_ids: set[str] = set()

        def collect_candidates(block: RemNoteBlock) -> list[ChunkCandidate]:
            if block.id in emitted_block_ids:
                return []
            if is_header_text(block.text):
                emitted_block_ids.add(block.id)
                candidates: list[ChunkCandidate] = []
                for child in ordered_children(block):
                    candidates.extend(collect_candidates(child))
                return candidates

            subtree = walk_subtree(block)
            emitted_block_ids.update(item.id for item in subtree)
            candidate = make_candidate(block)
            return [candidate] if candidate else []

        def chunk_role(candidates: list[ChunkCandidate]) -> str:
            roles = {candidate.role for candidate in candidates}
            if roles == {"list_item"}:
                return "list_item_group"
            if "list_item" in roles:
                return "section_with_list_items"
            if roles == {"paragraph"}:
                return "paragraph_group"
            return "subtree_group"

        def chunk_from_candidates(candidates: list[ChunkCandidate]) -> RetrievalChunk:
            ordered_blocks: list[RemNoteBlock] = []
            seen_blocks: set[str] = set()
            for candidate in candidates:
                for block in candidate.blocks:
                    if block.id in seen_blocks:
                        continue
                    seen_blocks.add(block.id)
                    ordered_blocks.append(block)
            ordered_blocks.sort(key=lambda item: (item.line_number, item.block_ordinal))

            text = "\n".join(candidate.text for candidate in candidates).strip()
            first, last = ordered_blocks[0], ordered_blocks[-1]
            heading_path = (
                common_path(candidate.heading_path for candidate in candidates)
                or candidates[0].heading_path
            )
            resource_ids = unique_preserving_order(
                resource_id
                for candidate in candidates
                for resource_id in candidate.external_resource_ids
            )
            quality_flags = unique_preserving_order(
                flag for candidate in candidates for flag in candidate.quality_flags
            )
            if len(text) < min_chars and "small_but_kept" not in quality_flags:
                quality_flags.append("small_but_kept")
            if len(text) > max_chars and "oversized_chunk" not in quality_flags:
                quality_flags.append("oversized_chunk")

            breadcrumb = semantic_path_text(heading_path)
            embedding_text = f"{breadcrumb}\n{text}" if breadcrumb else text
            return RetrievalChunk(
                id=stable_id(
                    "chunk",
                    "blocks_tree",
                    first.source,
                    first.line_number,
                    last.line_number,
                    text,
                ),
                text=text,
                chunk_type="remnote_section",
                source=first.source,
                path=heading_path or first.path,
                line_start=first.line_number,
                line_end=last.line_number,
                source_block_ids=[block.id for block in ordered_blocks],
                external_resource_ids=resource_ids,
                parent_block_id=candidates[0].blocks[0].parent_id,
                chunk_role=chunk_role(candidates),
                heading_path=heading_path,
                display_text=text,
                embedding_text=embedding_text,
                chunk_quality_flags=quality_flags,
            )

        def flush_candidates(buffer: list[ChunkCandidate]) -> None:
            if buffer:
                chunks.append(chunk_from_candidates(buffer))

        source_order = unique_preserving_order(block.source for block in blocks)
        for source in source_order:
            source_blocks = [block for block in blocks if block.source == source]
            source_block_ids = {block.id for block in source_blocks}
            roots = [
                block
                for block in source_blocks
                if block.parent_id not in source_block_ids
            ]
            source_candidates: list[ChunkCandidate] = []
            for root in sorted(
                roots, key=lambda item: (item.line_number, item.block_ordinal)
            ):
                source_candidates.extend(collect_candidates(root))

            current_candidates: list[ChunkCandidate] = []
            current_key: tuple[str, tuple[str, ...]] | None = None
            for candidate in source_candidates:
                key = (candidate.source, tuple(candidate.heading_path))
                candidate_text = "\n".join(
                    item.text for item in current_candidates + [candidate]
                )
                key_changed = current_key is not None and key != current_key
                over_max_chars = (
                    bool(current_candidates) and len(candidate_text) > max_chars
                )
                if key_changed or over_max_chars:
                    flush_candidates(current_candidates)
                    current_candidates = []

                current_key = key
                current_candidates.append(candidate)
            flush_candidates(current_candidates)

        def artifact_context(resource: ExternalResource) -> dict[str, Any]:
            parent_block = blocks_by_id.get(resource.parent_block_id)
            if not parent_block:
                return {
                    "path": [resource.source],
                    "context_block_ids": [],
                    "context_text": None,
                }

            def ancestors(block: RemNoteBlock) -> list[RemNoteBlock]:
                chain: list[RemNoteBlock] = []
                current = block
                while current.parent_id and current.parent_id in blocks_by_id:
                    current = blocks_by_id[current.parent_id]
                    chain.append(current)
                return list(reversed(chain))

            def siblings_for(block: RemNoteBlock) -> list[RemNoteBlock]:
                if block.parent_id:
                    siblings = children_by_id.get(block.parent_id, [])
                else:
                    siblings = [
                        candidate
                        for candidate in blocks
                        if candidate.source_document_id == block.source_document_id
                        and candidate.parent_id is None
                    ]
                return sorted(
                    siblings, key=lambda item: (item.line_number, item.block_ordinal)
                )

            def immediate_previous_text_sibling(
                block: RemNoteBlock,
            ) -> RemNoteBlock | None:
                siblings = siblings_for(block)
                for index, sibling in enumerate(siblings):
                    if sibling.id != block.id:
                        continue
                    if index == 0:
                        return None
                    candidate = siblings[index - 1]
                    if is_noise_text(candidate.text) or is_header_text(candidate.text):
                        return None
                    return candidate
                return None

            context_blocks = [
                block
                for block in ancestors(parent_block)
                if is_header_text(block.text) and not is_noise_text(block.text)
            ]
            if is_header_text(parent_block.text) and not is_noise_text(
                parent_block.text
            ):
                context_blocks.append(parent_block)

            direct_previous_text = immediate_previous_text_sibling(parent_block)
            if direct_previous_text:
                context_blocks.append(direct_previous_text)

            ordered_context: list[RemNoteBlock] = []
            seen_context_ids: set[str] = set()
            for block in sorted(
                context_blocks, key=lambda item: (item.line_number, item.block_ordinal)
            ):
                if block.id in seen_context_ids:
                    continue
                seen_context_ids.add(block.id)
                ordered_context.append(block)

            context_text = (
                "\n".join(block.text for block in ordered_context if block.text.strip())
                or None
            )
            return {
                "path": parent_block.path or [resource.source],
                "context_block_ids": [block.id for block in ordered_context],
                "context_text": context_text,
            }

        for artifact in parsed_artifacts:
            resource = resources_by_id.get(artifact.external_resource_id)
            if not resource:
                continue
            artifact_path = Path(artifact.artifact_path)
            decision = self._gate_artifact(
                artifact, resource, artifact_path, seen_embedded_artifact_hashes
            )
            artifact_gate_decisions.append(decision)
            if decision.policy != "embed_full":
                continue
            artifact_chunks = self._chunks_from_artifact(
                artifact_path,
                resource,
                context=artifact_context(resource),
                min_chars=min_chars,
                max_chars=max_chars,
            )
            decision.emitted_chunk_count = len(artifact_chunks)
            if artifact_chunks:
                seen_embedded_artifact_hashes.add(decision.content_hash)
            chunks.extend(artifact_chunks)

        clean_chunks = [chunk for chunk in chunks if not is_noise_text(chunk.text)]
        return clean_chunks, artifact_gate_decisions

    def _gate_artifact(
        self,
        artifact: ParsedArtifact,
        resource: ExternalResource,
        artifact_path: Path,
        seen_embedded_artifact_hashes: set[str],
    ) -> ArtifactGateDecision:
        reason_codes: list[str] = []
        text = ""
        read_error: str | None = None
        try:
            if artifact_path.suffix != ".md":
                read_error = "unsupported_artifact_type"
            else:
                text = artifact_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            read_error = exc.__class__.__name__

        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""
        declared_url = extract_declared_artifact_url(text) if text else None
        normalized_source_url = normalize_url_for_gate(resource.url)
        normalized_declared_url = normalize_url_for_gate(declared_url)
        stats = (
            script_stats(text)
            if text
            else {
                "char_count": 0,
                "line_count": 0,
                "nonempty_line_count": 0,
                "alpha_count": 0,
                "cyrillic_count": 0,
                "latin_count": 0,
                "greek_count": 0,
                "cyrillic_ratio": 0.0,
                "latin_greek_ratio": 0.0,
            }
        )
        source_is_dataset = is_dataset_like_url(resource.url)
        declared_is_dataset = is_dataset_like_url(declared_url)
        text_is_dataset = artifact_text_looks_like_dataset(text) if text else False
        is_dataset_artifact = (
            source_is_dataset or declared_is_dataset or text_is_dataset
        )
        generic_navigation = (
            is_generic_navigation_artifact(declared_url, text) if text else False
        )
        url_mismatch = bool(
            normalized_declared_url
            and normalized_source_url
            and normalized_declared_url != normalized_source_url
            and not (source_is_dataset and declared_is_dataset)
        )
        low_quality_ocr = (
            is_low_quality_ocr(text, resource, artifact_path, stats) if text else False
        )

        stats.update(
            {
                "source_url_is_dataset": source_is_dataset,
                "declared_url_is_dataset": declared_is_dataset,
                "artifact_text_looks_like_dataset": text_is_dataset,
                "generic_navigation_artifact": generic_navigation,
                "url_mismatch": url_mismatch,
                "low_quality_ocr": low_quality_ocr,
                "read_error": read_error,
            }
        )

        if read_error:
            policy = "quarantine"
            reason_codes.append(read_error)
        elif not text.strip():
            policy = "quarantine"
            reason_codes.append("empty_artifact")
        elif url_mismatch or generic_navigation:
            policy = "quarantine"
            if url_mismatch:
                reason_codes.append("url_mismatch")
            if generic_navigation:
                reason_codes.append("generic_navigation_artifact")
        elif low_quality_ocr:
            policy = "quarantine"
            reason_codes.append("low_quality_ocr")
        elif is_dataset_artifact:
            policy = "metadata_only"
            reason_codes.append("dataset_artifact")
        elif content_hash in seen_embedded_artifact_hashes:
            policy = "metadata_only"
            reason_codes.append("duplicate_content_hash")
        else:
            policy = "embed_full"
            reason_codes.append("accepted")

        return ArtifactGateDecision(
            id=stable_id(
                "gate", artifact.id, resource.id, content_hash or artifact_path
            ),
            external_resource_id=resource.id,
            artifact_path=normalize_nfc(str(artifact_path)),
            policy=policy,
            reason_codes=reason_codes,
            content_hash=content_hash,
            normalized_source_url=normalized_source_url,
            declared_artifact_url=declared_url,
            normalized_declared_artifact_url=normalized_declared_url or None,
            stats=stats,
        )

    def _chunks_from_artifact(
        self,
        artifact_path: Path,
        resource: ExternalResource,
        context: dict[str, Any],
        min_chars: int,
        max_chars: int,
    ) -> list[RetrievalChunk]:
        lines = artifact_path.read_text(encoding="utf-8", errors="replace").splitlines()
        chunks: list[RetrievalChunk] = []
        buffer: list[str] = []
        start_line: int | None = None
        context_path = list(context.get("path") or [resource.source])
        context_text = context.get("context_text")
        context_block_ids = list(context.get("context_block_ids") or [])
        artifact_path_label = f"external:{resource.url_hash}"
        chunk_path = context_path + [artifact_path_label]
        breadcrumb = semantic_path_text(context_path)
        semantic_context = semantic_context_text(context_text)

        def flush(end_line: int) -> None:
            nonlocal buffer, start_line
            text = "\n".join(buffer).strip()
            if text and not is_noise_text(text):
                embedding_parts: list[str] = []
                if breadcrumb:
                    embedding_parts.append(breadcrumb)
                if semantic_context:
                    embedding_parts.append(semantic_context)
                embedding_parts.append(text)
                quality_flags = ["context_attached"] if context_text else []
                if len(text) < min_chars:
                    quality_flags.append("small_but_kept")
                chunks.append(
                    RetrievalChunk(
                        id=stable_id(
                            "chunk", "artifact", resource.id, start_line, end_line, text
                        ),
                        text=text,
                        chunk_type="external_artifact",
                        source=resource.source,
                        path=chunk_path,
                        line_start=resource.line_number,
                        line_end=resource.line_number,
                        source_block_ids=[resource.parent_block_id],
                        external_resource_ids=[resource.id],
                        parent_block_id=resource.parent_block_id,
                        context_block_ids=context_block_ids,
                        context_text=context_text,
                        source_relation="parsed_external_resource",
                        artifact_path=normalize_nfc(str(artifact_path)),
                        artifact_line_start=start_line,
                        artifact_line_end=end_line,
                        chunk_role="external_artifact",
                        heading_path=context_path,
                        display_text=text,
                        embedding_text="\n\n".join(embedding_parts),
                        chunk_quality_flags=quality_flags,
                    )
                )
            buffer, start_line = [], None

        for line_number, line in enumerate(lines):
            text = clean_text(line.strip().lstrip("-*").strip())
            if is_noise_text(text):
                continue
            if start_line is None:
                start_line = line_number
            candidate = "\n".join(buffer + [text])
            if buffer and len(candidate) > max_chars:
                flush(line_number - 1)
                start_line = line_number
            buffer.append(text)
            if len("\n".join(buffer)) >= max_chars:
                flush(line_number)

        if buffer:
            flush(len(lines) - 1)

        return [
            chunk
            for chunk in chunks
            if len(chunk.text) >= min_chars or len(chunks) == 1
        ]

    def _build_summary(
        self,
        source_documents: list[SourceDocument],
        blocks: list[RemNoteBlock],
        external_resources: list[ExternalResource],
        parsed_artifacts: list[ParsedArtifact],
        artifact_gate_decisions: list[ArtifactGateDecision],
        retrieval_chunks: list[RetrievalChunk],
        baseline_summary: dict[str, Any] | None,
    ) -> dict[str, Any]:
        status_counts = Counter(
            resource.parse_status for resource in external_resources
        )
        content_type_counts = Counter(
            resource.content_type_hint for resource in external_resources
        )
        multi_url_lines = Counter(
            (resource.source, resource.line_number) for resource in external_resources
        )
        duplicate_chunk_text = Counter(chunk.text for chunk in retrieval_chunks)
        tiny_chunks = [
            chunk for chunk in retrieval_chunks if len(chunk.text.strip()) <= 3
        ]
        placeholder_chunks = [
            chunk
            for chunk in retrieval_chunks
            if chunk.text.strip() == IMAGE_PLACEHOLDER
        ]
        code_fence_chunks = [
            chunk
            for chunk in retrieval_chunks
            if is_code_fence_marker(chunk.text.strip())
        ]
        code_fence_marker_lines = [
            line
            for chunk in retrieval_chunks
            for line in chunk.text.splitlines()
            if is_code_fence_marker(line.strip())
        ]
        header_only_chunks = [
            chunk
            for chunk in retrieval_chunks
            if chunk.chunk_type == "remnote_section"
            and all(
                is_header_text(line.strip())
                for line in chunk.text.splitlines()
                if line.strip()
            )
        ]
        missing_provenance = [
            chunk.id
            for chunk in retrieval_chunks
            if not chunk.source_block_ids
            or not chunk.source
            or not chunk.path
            or chunk.line_start is None
        ]
        block_by_id = {block.id: block for block in blocks}
        block_source_by_id = {block.id: block.source for block in blocks}
        chunk_ids_by_block_id: dict[str, list[set[str]]] = {}
        for chunk in retrieval_chunks:
            chunk_block_ids = set(chunk.source_block_ids)
            for block_id in chunk.source_block_ids:
                chunk_ids_by_block_id.setdefault(block_id, []).append(chunk_block_ids)

        def subtree_block_ids(block: RemNoteBlock) -> set[str]:
            ids = {block.id}
            for child_id in block.child_ids:
                child = block_by_id.get(child_id)
                if child:
                    ids.update(subtree_block_ids(child))
            return ids

        mixed_source_chunks = []
        for chunk in retrieval_chunks:
            chunk_block_sources = {
                block_source_by_id[block_id]
                for block_id in chunk.source_block_ids
                if block_id in block_source_by_id
            }
            if len(chunk_block_sources) > 1 or (
                chunk_block_sources and chunk.source not in chunk_block_sources
            ):
                mixed_source_chunks.append(chunk.id)
        orphan_list_parent_chunks = []
        split_list_item_blocks = []
        for block in blocks:
            if not is_ordered_list_item(block.text) or not block.child_ids:
                continue
            subtree_ids = subtree_block_ids(block)
            containing_chunks = chunk_ids_by_block_id.get(block.id, [])
            if containing_chunks and not any(
                subtree_ids.issubset(chunk_ids) for chunk_ids in containing_chunks
            ):
                split_list_item_blocks.append(block.id)
            direct_child_ids = {
                child_id for child_id in block.child_ids if child_id in block_by_id
            }
            if (
                containing_chunks
                and direct_child_ids
                and not any(
                    direct_child_ids & chunk_ids for chunk_ids in containing_chunks
                )
            ):
                orphan_list_parent_chunks.append(block.id)

        resource_only_chunks = [
            chunk
            for chunk in retrieval_chunks
            if chunk.chunk_type == "remnote_section"
            and chunk.external_resource_ids
            and not any(
                block_id in block_by_id
                and not is_noise_text(block_by_id[block_id].text)
                for block_id in chunk.source_block_ids
            )
        ]
        bad_resource_paths = [
            resource.id
            for resource in external_resources
            if is_bad_artifact_path(resource.artifact_path)
        ]
        image_binary_selected_despite_md_sibling = [
            resource.id
            for resource in external_resources
            if resource.content_type_hint == "image"
            and resource.artifact_path
            and Path(resource.artifact_path).suffix.casefold() in IMAGE_EXTENSIONS
            and Path(resource.artifact_path).with_suffix(".md").exists()
        ]
        artifact_policy_counts = Counter(
            decision.policy for decision in artifact_gate_decisions
        )
        dataset_metadata_only = [
            decision
            for decision in artifact_gate_decisions
            if decision.policy == "metadata_only"
            and "dataset_artifact" in decision.reason_codes
        ]
        duplicate_metadata_only = [
            decision
            for decision in artifact_gate_decisions
            if decision.policy == "metadata_only"
            and "duplicate_content_hash" in decision.reason_codes
        ]
        url_mismatch_quarantine = [
            decision
            for decision in artifact_gate_decisions
            if decision.policy == "quarantine"
            and "url_mismatch" in decision.reason_codes
        ]
        low_quality_ocr_quarantine = [
            decision
            for decision in artifact_gate_decisions
            if decision.policy == "quarantine"
            and "low_quality_ocr" in decision.reason_codes
        ]
        external_artifact_chunks = [
            chunk
            for chunk in retrieval_chunks
            if chunk.chunk_type == "external_artifact"
        ]
        external_artifact_chunks_with_context = [
            chunk
            for chunk in external_artifact_chunks
            if chunk.context_block_ids and chunk.context_text
        ]
        external_artifact_embedding_support_label_chunks = [
            chunk
            for chunk in external_artifact_chunks
            if chunk.embedding_text
            and (
                "external:" in chunk.embedding_text
                or "RemNote context:" in chunk.embedding_text
                or "Parsed external content:" in chunk.embedding_text
            )
        ]

        def embedded_chunk_count_for_reason(reason_code: str) -> int:
            return sum(
                decision.emitted_chunk_count
                for decision in artifact_gate_decisions
                if reason_code in decision.reason_codes
            )

        raw_url_count = sum(document.url_count for document in source_documents)
        parser_visible_url_count = len(external_resources)
        baseline = baseline_summary or {}
        baseline_external = (
            baseline.get("external_parsing", {}) if isinstance(baseline, dict) else {}
        )
        baseline_docstore = (
            baseline.get("docstore", {}) if isinstance(baseline, dict) else {}
        )
        baseline_chunking = (
            baseline.get("chunking_quality", {}) if isinstance(baseline, dict) else {}
        )

        return {
            "source_document_count": len(source_documents),
            "raw_block_count": len(blocks),
            "raw_url_occurrences": raw_url_count,
            "parser_visible_url_resources": parser_visible_url_count,
            "url_count_match": raw_url_count == parser_visible_url_count,
            "multi_url_line_count": sum(
                1 for count in multi_url_lines.values() if count > 1
            ),
            "external_resource_status_counts": dict(status_counts),
            "external_resource_content_type_counts": dict(content_type_counts),
            "parsed_artifact_count": len(parsed_artifacts),
            "retrieval_chunk_count": len(retrieval_chunks),
            "standalone_tiny_chunk_count": len(tiny_chunks),
            "placeholder_only_chunk_count": len(placeholder_chunks),
            "code_fence_only_chunk_count": len(code_fence_chunks),
            "code_fence_marker_line_count": len(code_fence_marker_lines),
            "header_only_chunk_count": len(header_only_chunks),
            "orphan_list_parent_chunk_count": len(orphan_list_parent_chunks),
            "split_list_item_subtree_count": len(split_list_item_blocks),
            "resource_only_chunk_count": len(resource_only_chunks),
            "duplicate_retrieval_chunk_text_keys": sum(
                1 for count in duplicate_chunk_text.values() if count > 1
            ),
            "chunks_missing_provenance_count": len(missing_provenance),
            "mixed_source_retrieval_chunk_count": len(mixed_source_chunks),
            "failed_path_current_dir_count": len(bad_resource_paths),
            "image_binary_selected_despite_md_sibling_count": len(
                image_binary_selected_despite_md_sibling
            ),
            "artifact_gate_policy_counts": dict(artifact_policy_counts),
            "dataset_artifact_metadata_only_count": len(dataset_metadata_only),
            "url_mismatch_quarantine_count": len(url_mismatch_quarantine),
            "duplicate_artifact_metadata_only_count": len(duplicate_metadata_only),
            "low_quality_ocr_quarantine_count": len(low_quality_ocr_quarantine),
            "external_artifact_chunk_count": len(external_artifact_chunks),
            "external_artifact_chunks_with_context_count": len(
                external_artifact_chunks_with_context
            ),
            "external_artifact_chunks_without_context_count": (
                len(external_artifact_chunks)
                - len(external_artifact_chunks_with_context)
            ),
            "external_artifact_embedding_support_label_count": len(
                external_artifact_embedding_support_label_chunks
            ),
            "embedded_dataset_dump_chunk_count": embedded_chunk_count_for_reason(
                "dataset_artifact"
            ),
            "embedded_url_mismatch_chunk_count": embedded_chunk_count_for_reason(
                "url_mismatch"
            ),
            "embedded_duplicate_artifact_chunk_count": embedded_chunk_count_for_reason(
                "duplicate_content_hash"
            ),
            "embedded_low_quality_ocr_chunk_count": embedded_chunk_count_for_reason(
                "low_quality_ocr"
            ),
            "success_criteria": {
                "raw_url_count_equals_parser_visible": raw_url_count
                == parser_visible_url_count,
                "no_current_dir_failed_paths": len(bad_resource_paths) == 0,
                "no_image_binary_selected_when_md_sibling_exists": len(
                    image_binary_selected_despite_md_sibling
                )
                == 0,
                "tiny_chunks_near_zero": len(tiny_chunks) <= 3,
                "no_placeholder_or_code_fence_chunks": not placeholder_chunks
                and not code_fence_chunks,
                "no_code_fence_marker_lines": len(code_fence_marker_lines) == 0,
                "no_header_only_chunks": not header_only_chunks,
                "no_orphan_list_parent_chunks": not orphan_list_parent_chunks,
                "no_split_list_item_subtrees": not split_list_item_blocks,
                "no_resource_only_chunks": not resource_only_chunks,
                "all_chunks_have_provenance": not missing_provenance,
                "no_mixed_source_chunks": not mixed_source_chunks,
                "no_embedded_dataset_dumps": embedded_chunk_count_for_reason(
                    "dataset_artifact"
                )
                == 0,
                "no_embedded_url_mismatch_artifacts": embedded_chunk_count_for_reason(
                    "url_mismatch"
                )
                == 0,
                "no_embedded_duplicate_artifacts": embedded_chunk_count_for_reason(
                    "duplicate_content_hash"
                )
                == 0,
                "no_embedded_low_quality_ocr": embedded_chunk_count_for_reason(
                    "low_quality_ocr"
                )
                == 0,
                "no_external_artifact_embedding_support_labels": not external_artifact_embedding_support_label_chunks,
            },
            "baseline_comparison": {
                "baseline_raw_url_total_in_selected_files": baseline_external.get(
                    "raw_url_total_in_selected_files"
                ),
                "baseline_parser_visible_url_candidate_nodes": baseline_external.get(
                    "parser_visible_url_candidate_nodes"
                ),
                "baseline_multi_url_line_gap_count": baseline_external.get(
                    "multi_url_line_gap_count"
                ),
                "baseline_tiny_node_count_len_1_to_3": baseline_docstore.get(
                    "tiny_node_count_len_1_to_3"
                ),
                "baseline_duplicate_source_text_keys": baseline_chunking.get(
                    "duplicate_source_text_keys"
                ),
                "optimized_multi_url_gap_count": raw_url_count
                - parser_visible_url_count,
                "optimized_tiny_retrieval_chunk_count": len(tiny_chunks),
                "optimized_duplicate_retrieval_chunk_text_keys": sum(
                    1 for count in duplicate_chunk_text.values() if count > 1
                ),
            },
        }


class RemNoteParserOptimized:
    """Drop-in shadow parser that writes optimized retrieval chunks to a LlamaIndex docstore.

    This class intentionally composes the optimized IR parser instead of replacing
    production parsing code. Raw RemNote blocks, external resources, parsed
    artifacts, and gate decisions remain provenance records; only accepted
    RetrievalChunk objects become TextNodes.
    """

    def __init__(
        self,
        path_settings: Any,
        storage_settings: Any,
        *,
        prepare_external_artifacts: bool = True,
        copy_existing_artifacts: bool = False,
        existing_artifacts_dir: Path | None = None,
        force_rebuild: bool = False,
        write_ir: bool = True,
    ) -> None:
        self.path_settings = path_settings
        self.storage_settings = storage_settings
        self.prepare_external_artifacts_enabled = prepare_external_artifacts
        self.copy_existing_artifacts_enabled = copy_existing_artifacts
        self.existing_artifacts_dir = existing_artifacts_dir
        self.force_rebuild = force_rebuild
        self.write_ir = write_ir
        self.kg_storage: Any = None
        self.document_storage_type = storage_settings.document_storage.storage_type
        self.ocr_pipeline: Any = None
        self.last_result: OptimizedParseResult | None = None
        self.last_copied_artifact_count = 0
        self.last_prepared_artifact_count = 0

    def _parsed_roots(self) -> list[Path]:
        from backend.data_processing.artifact_preparation import (
            parsed_roots_from_settings,
        )

        return parsed_roots_from_settings(self.path_settings)

    def _ensure_parsed_dirs(self) -> None:
        from backend.data_processing.artifact_preparation import ensure_parsed_dirs

        ensure_parsed_dirs(self.path_settings)

    def _ensure_storage(self) -> Any:
        if self.kg_storage is None:
            from backend.knowledge_graph.storage import KnowledgeGraphStorage

            self.kg_storage = KnowledgeGraphStorage(
                self.path_settings,
                self.storage_settings,
                local_storage=self.path_settings.local_storage_dir,
            )
        return self.kg_storage

    @property
    def optimized_ir_dir(self) -> Path:
        """Directory where optimized parser IR JSONL files are written."""

        return Path(self.path_settings.local_storage_dir).parent / "optimized_parser_ir"

    def _docstore_docs(self) -> dict[str, Any]:
        kg_storage = self._ensure_storage()
        return kg_storage.storage_context.docstore.docs

    def _persist_if_local(self) -> None:
        storage_type = getattr(
            self.document_storage_type, "value", self.document_storage_type
        )
        if storage_type == "local":
            self.kg_storage.storage_context.persist(
                persist_dir=str(self.path_settings.local_storage_dir)
            )

    def _clear_docstore(self) -> None:
        docstore = self._ensure_storage().storage_context.docstore
        docs = getattr(docstore, "docs", {})
        for node_id in list(docs.keys()):
            try:
                docstore.delete_document(node_id)
            except TypeError:
                docstore.delete_document(node_id, raise_error=False)
        if getattr(docstore, "docs", {}):
            docstore.docs.clear()
        self._persist_if_local()

    def copy_existing_artifacts(self, source_dir: Path | None = None) -> int:
        """Copy cached OCR Markdown artifacts into the isolated parsed-images cache.

        The testing flow reuses reviewed ``.md`` OCR outputs and deliberately avoids
        OCR/network work. Only Markdown files are copied, filenames are preserved,
        and existing non-empty target files are left intact.
        """

        from backend.data_processing.artifact_preparation import copy_existing_artifacts

        configured_source = (
            source_dir if source_dir is not None else self.existing_artifacts_dir
        )
        copied = copy_existing_artifacts(
            configured_source, Path(self.path_settings.parsed_images_dir)
        )
        self.last_copied_artifact_count = copied
        return copied

    def _get_ocr_pipeline(self) -> Any:
        if self.ocr_pipeline is None:
            from backend.data_processing.ocr import PaddleOCRPipeline

            self.ocr_pipeline = PaddleOCRPipeline()
        return self.ocr_pipeline

    def prepare_external_artifacts(self) -> int:
        """Download/OCR missing external artifacts when explicitly enabled.

        This method is intentionally best-effort. Final source/resource/artifact
        records are produced by a subsequent optimized parse after cached artifacts
        have been created.
        """

        from backend.data_processing.artifact_preparation import (
            prepare_external_artifacts,
        )

        prepared_count = prepare_external_artifacts(
            self.path_settings,
            get_ocr_pipeline=self._get_ocr_pipeline,
        )
        self.last_prepared_artifact_count = prepared_count
        return prepared_count

    def _parse_optimized_result(self) -> OptimizedParseResult:
        return OptimizedRemNoteParser(
            Path(self.path_settings.raw_data_dir),
            parsed_roots=self._parsed_roots(),
        ).run()

    def _write_ir_outputs(self, result: OptimizedParseResult) -> Path:
        return write_optimized_parser_ir(self.optimized_ir_dir, result)

    def write_ir_outputs(self, result: OptimizedParseResult) -> Path:
        """Write optimized parser IR files and return the output directory."""

        return self._write_ir_outputs(result)

    @staticmethod
    def _llama_index_schema() -> tuple[Any, Any, Any]:
        from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

        return TextNode, NodeRelationship, RelatedNodeInfo

    @staticmethod
    def _json_safe_metadata(value: Any) -> Any:
        if isinstance(value, Path):
            return normalize_nfc(str(value))
        if isinstance(value, tuple):
            return [RemNoteParserOptimized._json_safe_metadata(item) for item in value]
        if isinstance(value, list):
            return [RemNoteParserOptimized._json_safe_metadata(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): RemNoteParserOptimized._json_safe_metadata(item)
                for key, item in value.items()
            }
        return value

    def _chunk_metadata(
        self,
        chunk: RetrievalChunk,
        resources_by_id: dict[str, ExternalResource],
        gates_by_resource_id: dict[str, ArtifactGateDecision],
    ) -> dict[str, Any]:
        gate_decisions = {
            resource_id: gates_by_resource_id[resource_id]
            for resource_id in chunk.external_resource_ids
            if resource_id in gates_by_resource_id
        }
        metadata: dict[str, Any] = {
            "docstore_node_kind": "retrieval_chunk",
            "chunk_id": chunk.id,
            "chunk_type": chunk.chunk_type,
            "chunk_role": chunk.chunk_role,
            "source": chunk.source,
            "path": chunk.path,
            "heading_path": chunk.heading_path,
            "line_number": chunk.line_start,
            "line_start": chunk.line_start,
            "line_end": chunk.line_end,
            "depth_level": max(len(chunk.path) - 1, 0),
            "original_text": chunk.display_text or chunk.text,
            "display_text": chunk.display_text or chunk.text,
            "source_block_ids": chunk.source_block_ids,
            "external_resource_ids": chunk.external_resource_ids,
            "parent_block_id": chunk.parent_block_id,
            "context_block_ids": chunk.context_block_ids,
            "context_text": chunk.context_text,
            "source_relation": chunk.source_relation,
            "artifact_path": chunk.artifact_path,
            "artifact_line_start": chunk.artifact_line_start,
            "artifact_line_end": chunk.artifact_line_end,
            "chunk_quality_flags": chunk.chunk_quality_flags,
            "child_ids": [],
        }

        if chunk.external_resource_ids:
            metadata.update(
                {
                    "external_resource_urls": [
                        resources_by_id[resource_id].url
                        for resource_id in chunk.external_resource_ids
                        if resource_id in resources_by_id
                    ],
                    "external_resource_content_type_hints": [
                        resources_by_id[resource_id].content_type_hint
                        for resource_id in chunk.external_resource_ids
                        if resource_id in resources_by_id
                    ],
                    "external_resource_parse_statuses": [
                        resources_by_id[resource_id].parse_status
                        for resource_id in chunk.external_resource_ids
                        if resource_id in resources_by_id
                    ],
                    "external_resource_artifact_paths": [
                        resources_by_id[resource_id].artifact_path
                        for resource_id in chunk.external_resource_ids
                        if resource_id in resources_by_id
                    ],
                }
            )

        if gate_decisions:
            metadata["artifact_gate_policy_by_resource_id"] = {
                resource_id: decision.policy
                for resource_id, decision in gate_decisions.items()
            }
            metadata["artifact_gate_reason_codes_by_resource_id"] = {
                resource_id: decision.reason_codes
                for resource_id, decision in gate_decisions.items()
            }
            metadata["artifact_gate_content_hash_by_resource_id"] = {
                resource_id: decision.content_hash
                for resource_id, decision in gate_decisions.items()
            }
            metadata["artifact_gate_stats_by_resource_id"] = {
                resource_id: decision.stats
                for resource_id, decision in gate_decisions.items()
            }
            if len(gate_decisions) == 1:
                decision = next(iter(gate_decisions.values()))
                metadata["artifact_gate_policy"] = decision.policy
                metadata["artifact_gate_reason_codes"] = decision.reason_codes
                metadata["artifact_gate_content_hash"] = decision.content_hash

        return self._json_safe_metadata(metadata)

    @staticmethod
    def _excluded_llm_metadata_keys(metadata: dict[str, Any]) -> list[str]:
        technical_prefixes = ("artifact_gate_", "external_resource_")
        technical_keys = {
            "docstore_node_kind",
            "chunk_id",
            "display_text",
            "source_block_ids",
            "external_resource_ids",
            "parent_block_id",
            "context_block_ids",
            "artifact_path",
            "artifact_line_start",
            "artifact_line_end",
            "line_number",
            "line_start",
            "line_end",
            "depth_level",
            "original_text",
            "child_ids",
        }
        return [
            key
            for key in metadata
            if key in technical_keys
            or any(key.startswith(prefix) for prefix in technical_prefixes)
        ]

    def _assign_relationships(
        self,
        nodes_by_id: dict[str, Any],
        chunks: list[RetrievalChunk],
        result: OptimizedParseResult,
        node_relationship: Any,
        related_node_info: Any,
    ) -> None:
        blocks_by_id = {block.id: block for block in result.blocks}
        block_to_chunk_id: dict[str, str] = {}
        for chunk in chunks:
            for block_id in chunk.source_block_ids:
                block_to_chunk_id.setdefault(block_id, chunk.id)

        def add_child(parent_id: str, child_id: str) -> None:
            parent_node = nodes_by_id[parent_id]
            child_node = nodes_by_id[child_id]
            parent_node.metadata.setdefault("child_ids", [])
            if child_id not in parent_node.metadata["child_ids"]:
                parent_node.metadata["child_ids"].append(child_id)
            parent_node.relationships.setdefault(node_relationship.CHILD, [])
            if not any(
                info.node_id == child_id
                for info in parent_node.relationships[node_relationship.CHILD]
            ):
                parent_node.relationships[node_relationship.CHILD].append(
                    related_node_info(
                        node_id=child_id, metadata={"title": child_node.text[:160]}
                    )
                )

        def parent_chunk_id_for(chunk: RetrievalChunk) -> str | None:
            candidate_block_ids = list(reversed(chunk.context_block_ids))
            if chunk.parent_block_id:
                candidate_block_ids.append(chunk.parent_block_id)

            seen_blocks: set[str] = set()
            for block_id in candidate_block_ids:
                current_id = block_id
                while (
                    current_id
                    and current_id in blocks_by_id
                    and current_id not in seen_blocks
                ):
                    seen_blocks.add(current_id)
                    mapped_chunk_id = block_to_chunk_id.get(current_id)
                    if mapped_chunk_id and mapped_chunk_id != chunk.id:
                        return mapped_chunk_id
                    current_id = blocks_by_id[current_id].parent_id
            return None

        for chunk in chunks:
            parent_id = parent_chunk_id_for(chunk)
            if not parent_id or parent_id not in nodes_by_id:
                continue
            node = nodes_by_id[chunk.id]
            parent_node = nodes_by_id[parent_id]
            node.relationships[node_relationship.PARENT] = related_node_info(
                node_id=parent_id,
                metadata={"title": parent_node.text[:160]},
            )
            add_child(parent_id, chunk.id)

        previous_relationship = getattr(node_relationship, "PREVIOUS", None)
        next_relationship = getattr(node_relationship, "NEXT", None)
        if previous_relationship is None or next_relationship is None:
            return

        groups: dict[tuple[str, tuple[str, ...]], list[RetrievalChunk]] = {}
        for chunk in chunks:
            groups.setdefault(
                (chunk.source, tuple(chunk.heading_path or chunk.path)), []
            ).append(chunk)

        for group_chunks in groups.values():
            ordered = sorted(
                group_chunks, key=lambda item: (item.line_start, item.line_end, item.id)
            )
            for previous_chunk, next_chunk in zip(ordered, ordered[1:]):
                previous_node = nodes_by_id[previous_chunk.id]
                next_node = nodes_by_id[next_chunk.id]
                previous_node.relationships[next_relationship] = related_node_info(
                    node_id=next_chunk.id,
                    metadata={"title": next_node.text[:160]},
                )
                next_node.relationships[previous_relationship] = related_node_info(
                    node_id=previous_chunk.id,
                    metadata={"title": previous_node.text[:160]},
                )

    def to_text_nodes(self, result: OptimizedParseResult) -> list[Any]:
        """Converts accepted RetrievalChunk records to LlamaIndex TextNodes."""

        text_node_cls, node_relationship, related_node_info = self._llama_index_schema()
        resources_by_id = {
            resource.id: resource for resource in result.external_resources
        }
        gates_by_resource_id = {
            decision.external_resource_id: decision
            for decision in result.artifact_gate_decisions
        }
        nodes: list[Any] = []
        nodes_by_id: dict[str, Any] = {}

        for chunk in result.retrieval_chunks:
            node_text = chunk.embedding_text or chunk.text
            metadata = self._chunk_metadata(
                chunk, resources_by_id, gates_by_resource_id
            )
            node = text_node_cls(
                text=node_text,
                metadata=metadata,
                excluded_embed_metadata_keys=list(metadata.keys()),
                excluded_llm_metadata_keys=self._excluded_llm_metadata_keys(metadata),
            )
            node.id_ = chunk.id
            nodes.append(node)
            nodes_by_id[chunk.id] = node

        self._assign_relationships(
            nodes_by_id,
            result.retrieval_chunks,
            result,
            node_relationship,
            related_node_info,
        )
        return nodes

    def get_text_nodes(self) -> list[Any]:
        """Builds optimized retrieval TextNodes from raw RemNote Markdown and cached artifacts."""

        self._ensure_parsed_dirs()
        if self.copy_existing_artifacts_enabled:
            self.copy_existing_artifacts()
        if self.prepare_external_artifacts_enabled:
            self.prepare_external_artifacts()

        result = self._parse_optimized_result()
        self.last_result = result
        if self.write_ir:
            self._write_ir_outputs(result)
        return self.to_text_nodes(result)

    def add_text_nodes(self, text_nodes: list[Any], allow_update: bool = True) -> None:
        """Adds optimized retrieval TextNodes to the configured LlamaIndex docstore."""

        kg_storage = self._ensure_storage()
        if not text_nodes:
            return
        kg_storage.storage_context.docstore.add_documents(
            text_nodes, allow_update=allow_update
        )
        self._persist_if_local()

    def run(self) -> OptimizedParseResult | None:
        """Runs the optimized shadow parsing pipeline with production-parser idempotency."""

        if self._docstore_docs():
            if not self.force_rebuild:
                return self.last_result
            self._clear_docstore()

        text_nodes = self.get_text_nodes()
        self.add_text_nodes(text_nodes)
        assert self._docstore_docs(), "No docs found in the docstore"
        return self.last_result
