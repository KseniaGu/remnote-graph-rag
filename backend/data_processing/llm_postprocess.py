"""LLM sidecar post-processing for optimized RemNote retrieval chunks.

This module is intentionally additive: it reads optimized parser outputs,
validates LLM enrichment decisions, and writes sidecar artifacts. It does not
mutate parser IR, docstores, vector stores, or property graph stores.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from html import unescape
from html.parser import HTMLParser
from difflib import SequenceMatcher
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.data_processing.concept_registry import (
    CANONICAL_CONCEPT_TYPE_HINT,
    ConceptAdjudicationFailure,
    ConceptAdjudicationResponse,
    ConceptPairScore,
    ConceptRegistryEntry,
    ConceptResolution,
    UncertainConceptCluster,
    mention_id_for,
)
from backend.data_processing.parser_optimized import (
    ArtifactGateDecision,
    ExternalResource,
    OptimizedParseResult,
    RetrievalChunk,
    normalize_nfc,
)


SCHEMA_VERSION = "1.1"
DEFAULT_PROMPT_VERSION = "v1"
DEFAULT_MODEL_NAME = "nemotron-3-super:cloud"
DEFAULT_SMOKE_LIMIT = 15
DEFAULT_MAX_BATCH_CHUNKS = 1
DEFAULT_MAX_BATCH_CHARS = 9000
DEFAULT_LLM_BASE_URL = "https://ollama.com"
DEFAULT_LLM_TEMPERATURE = 0.0
DEFAULT_LLM_TOP_K = 10
DEFAULT_LLM_TOP_P = 0.1
DEFAULT_LLM_NUM_CTX = 10240
DEFAULT_LLM_NUM_PREDICT = 4096
DEFAULT_QUALITY_NUM_PREDICT = 2048
DEFAULT_GRAPH_NUM_PREDICT = 6144
DEFAULT_CONCEPT_RESOLUTION_NUM_PREDICT = 1536

DECISIONS_FILENAME = "llm_postprocess_decisions.jsonl"
FAILURES_FILENAME = "llm_postprocess_failures.jsonl"
RELATION_REGISTRY_FILENAME = "llm_relation_registry.jsonl"
CONCEPT_REGISTRY_FILENAME = "llm_concept_registry.jsonl"
CONCEPT_MERGE_REVIEW_FILENAME = "llm_concept_merge_review.jsonl"
CONCEPT_PAIR_SCORES_FILENAME = "llm_concept_pair_scores.jsonl"
CONCEPT_ADJUDICATIONS_FILENAME = "llm_concept_adjudications.jsonl"
CONCEPT_ADJUDICATION_FAILURES_FILENAME = "llm_concept_adjudication_failures.jsonl"
GRAPH_PREVIEW_FILENAME = "llm_graph_projection_preview.jsonl"
REPORT_JSON_FILENAME = "llm_postprocess_report.json"
REPORT_MD_FILENAME = "llm_postprocess_report.md"
INPUTS_FILENAME = "llm_postprocess_inputs.jsonl"
CACHE_DIRNAME = "llm_response_cache"

EXISTING_RELATION_LABELS = {
    "IS_A",
    "PART_OF",
    "HAS_COMPONENT",
    "CALCULATES",
    "SOLVES",
    "PROPOSES",
    "USES",
    "REQUIRES",
    "PRODUCES",
    "DERIVES_FROM",
    "CITES",
    "TRAINS",
    "TRAINED_ON",
    "OPTIMIZES",
    "EVALUATED_ON",
    "COMPARES_TO",
    "RELATED_TO",
}
PREDICATE_CANONICAL_ALIASES = {
    "IS_PART_OF": "PART_OF",
}
UNKNOWN_EXISTING_PREDICATE_WARNING = "unknown_existing_predicate_downgraded"
PREDICATE_ALIAS_WARNING = "predicate_alias_normalized"
DEBUG_LABEL_PATTERNS = (
    r"external:[A-Za-z0-9_-]+",
    r"Parsed external content:",
    r"RemNote context:",
    r"artifact_path",
    r"artifact line",
)
HTML_TAG_RE = re.compile(r"<[^>]+>")
SOURCE_CARD_FIELD_RE = re.compile(
    r"^(?:title|url|hostname|description|sitename|date|author|tags|categories):\s*",
    re.IGNORECASE | re.MULTILINE,
)
CAPTION_RE = re.compile(r"\b(?:fig(?:ure)?|chart|plot|diagram|visualization|snippet|image|table)\b", re.IGNORECASE)
PREDICATE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9_+-]{1,}")
MARKUP_TAG_NAMES_PATTERN = r"(?:div|img|html|body|table|thead|tbody|tr|td|th|center|span|br|p)"
MARKUP_ATTRS_PATTERN = (
    r"(?:\s+[A-Za-z_:][-A-Za-z0-9_:.]*(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+))?)*"
)
PARTIAL_MARKUP_TAG_RE = re.compile(
    rf"</?{MARKUP_TAG_NAMES_PATTERN}\b{MARKUP_ATTRS_PATTERN}\s*/?>?",
    re.IGNORECASE,
)
INCOMPLETE_MARKUP_TAG_RE = re.compile(
    rf"</?(?P<tag>{MARKUP_TAG_NAMES_PATTERN})\b"
    r"(?:\s+(?:style|src|alt|width|height|class|id|border)\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+))*\s*",
    re.IGNORECASE,
)
GENERIC_IMAGE_ALT_TEXTS = {
    "",
    "image",
    "img",
    "picture",
    "photo",
    "figure",
    "diagram",
    "chart",
    "plot",
    "visualization",
    "graphic",
    "screenshot",
}


class ChunkAction(StrEnum):
    KEEP = "keep"
    KEEP_WITH_CLEANED_TEXT = "keep_with_cleaned_text"
    METADATA_ONLY = "metadata_only"
    NEEDS_VISUAL_REPARSE = "needs_visual_reparse"
    EXCLUDE_FROM_EMBEDDING = "exclude_from_embedding"
    GRAPH_ONLY = "graph_only"


class PredicateStatus(StrEnum):
    EXISTING = "existing"
    PROPOSED = "proposed"


class ProcessingMode(StrEnum):
    ENRICHMENT = "enrichment"
    QUALITY_REVIEW = "quality_review"
    BOTH = "both"


class ChunkPreflags(BaseModel):
    model_config = ConfigDict(extra="forbid")

    html_fragment_detected: bool = False
    caption_only_candidate: bool = False
    source_card_candidate: bool = False
    boilerplate_candidate: bool = False
    visual_content_missing_candidate: bool = False
    small_but_kept: bool = False

    @property
    def suspicious_count(self) -> int:
        return sum(
            int(value)
            for value in (
                self.html_fragment_detected,
                self.caption_only_candidate,
                self.source_card_candidate,
                self.boilerplate_candidate,
                self.visual_content_missing_candidate,
                self.small_but_kept,
            )
        )


class ChunkPostprocessInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    chunk_type: str
    chunk_role: str
    source: str
    path: list[str] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)
    line_start: int
    line_end: int
    source_block_ids: list[str] = Field(default_factory=list)
    external_resource_ids: list[str] = Field(default_factory=list)
    text: str
    embedding_text: Optional[str] = None
    display_text: Optional[str] = None
    context_text: Optional[str] = None
    artifact_path: Optional[str] = None
    artifact_line_start: Optional[int] = None
    artifact_line_end: Optional[int] = None
    chunk_quality_flags: list[str] = Field(default_factory=list)
    external_resource_urls: list[str] = Field(default_factory=list)
    external_resource_content_type_hints: list[str] = Field(default_factory=list)
    external_resource_parse_statuses: list[str] = Field(default_factory=list)
    artifact_gate_policy_by_resource_id: dict[str, str] = Field(default_factory=dict)
    artifact_gate_reason_codes_by_resource_id: dict[str, list[str]] = Field(default_factory=dict)
    artifact_gate_stats_by_resource_id: dict[str, dict[str, Any]] = Field(default_factory=dict)
    preflags: ChunkPreflags = Field(default_factory=ChunkPreflags)
    processing_mode: ProcessingMode = ProcessingMode.ENRICHMENT
    input_hash: str = ""

    @model_validator(mode="after")
    def set_hash(self) -> "ChunkPostprocessInput":
        if not self.input_hash:
            payload = self.model_dump(exclude={"input_hash"}, mode="json")
            self.input_hash = stable_hash(payload, length=24)
        return self

    def evidence_text(self) -> str:
        parts = [
            "\n".join(self.heading_path or self.path),
            self.text,
            self.embedding_text or "",
            self.display_text or "",
            self.context_text or "",
        ]
        return "\n".join(part for part in parts if part)


class ChunkPostprocessBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str
    chunks: list[ChunkPostprocessInput]

    @model_validator(mode="after")
    def require_chunks(self) -> "ChunkPostprocessBatch":
        if not self.chunks:
            raise ValueError("batch must contain at least one chunk")
        return self

    @property
    def input_hash(self) -> str:
        return stable_hash([chunk.input_hash for chunk in self.chunks], length=24)


class ConceptCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local_id: str
    canonical_name: str
    display_name: Optional[str] = None
    type: str = "CONCEPT"
    aliases: list[str] = Field(default_factory=list)
    salience: float = Field(ge=0.0, le=1.0)
    description: Optional[str] = None
    evidence_spans: list[str] = Field(default_factory=list)

    @field_validator("local_id", "canonical_name", "type")
    @classmethod
    def non_empty(cls, value: str) -> str:
        value = normalize_whitespace(value)
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("aliases", "evidence_spans")
    @classmethod
    def clean_lists(cls, values: list[str]) -> list[str]:
        return unique_preserving_order(normalize_whitespace(value) for value in values if normalize_whitespace(value))

    @model_validator(mode="after")
    def defaults(self) -> "ConceptCandidate":
        if self.display_name is None:
            self.display_name = self.canonical_name
        else:
            self.display_name = normalize_whitespace(self.display_name)
        self.type = normalize_whitespace(self.type).upper()
        return self


class RelationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_concept_id: str
    target_concept_id: str
    raw_predicate: str
    canonical_predicate: str
    predicate_status: PredicateStatus
    predicate_family: str
    predicate_definition: Optional[str] = None
    relation_phrase: Optional[str] = None
    generality_score: float = Field(default=0.5, ge=0.0, le=1.0)
    retrieval_usefulness: float = Field(default=0.5, ge=0.0, le=1.0)
    visualization_usefulness: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    evidence_spans: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator(
        "source_concept_id",
        "target_concept_id",
        "raw_predicate",
        "canonical_predicate",
        "predicate_family",
    )
    @classmethod
    def clean_non_empty(cls, value: str) -> str:
        value = normalize_whitespace(value)
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("raw_predicate", "canonical_predicate")
    @classmethod
    def predicate_shape(cls, value: str) -> str:
        value = normalize_predicate(value)
        if not PREDICATE_RE.match(value):
            raise ValueError("predicate must be uppercase snake case")
        return value

    @field_validator("evidence_chunk_ids", "evidence_spans")
    @classmethod
    def clean_lists(cls, values: list[str]) -> list[str]:
        return unique_preserving_order(normalize_whitespace(value) for value in values if normalize_whitespace(value))

    @field_validator("relation_phrase", "predicate_definition")
    @classmethod
    def clean_optional_short_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = normalize_whitespace(value)
        return cleaned or None


class LLMChunkDecision(BaseModel):
    """Schema expected from the LLM for a single chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    action: ChunkAction
    issue_types: list[str] = Field(default_factory=list)
    educational_usefulness: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)
    cleaned_embedding_text: Optional[str] = None
    cleaned_display_text: Optional[str] = None
    chunk_summary: Optional[str] = None
    concepts: list[ConceptCandidate] = Field(default_factory=list)
    relations: list[RelationCandidate] = Field(default_factory=list)
    reason: Optional[str] = None

    @field_validator("chunk_id")
    @classmethod
    def chunk_id_not_empty(cls, value: str) -> str:
        value = normalize_whitespace(value)
        if not value:
            raise ValueError("chunk_id must not be empty")
        return value

    @field_validator("issue_types", "warnings")
    @classmethod
    def clean_string_lists(cls, values: list[str]) -> list[str]:
        return unique_preserving_order(normalize_issue_label(value) for value in values if normalize_issue_label(value))

    @field_validator("cleaned_embedding_text", "cleaned_display_text", "chunk_summary", "reason")
    @classmethod
    def clean_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class LLMPostprocessBatchResponse(BaseModel):
    """Top-level LLM response for one prompt call."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[LLMChunkDecision]

    @model_validator(mode="after")
    def require_decisions(self) -> "LLMPostprocessBatchResponse":
        if not self.decisions:
            raise ValueError("LLM response must contain at least one decision")
        return self


class ChunkEnrichmentDecision(LLMChunkDecision):
    """Validated, auditable sidecar decision stored on disk."""

    decision_id: str
    schema_version: str
    prompt_version: str
    model_name: str
    input_hash: str


class PostprocessFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure_id: str
    chunk_id: Optional[str] = None
    batch_id: Optional[str] = None
    schema_version: str = SCHEMA_VERSION
    prompt_version: str = DEFAULT_PROMPT_VERSION
    model_name: str = DEFAULT_MODEL_NAME
    input_hash: Optional[str] = None
    error_type: str
    message: str
    raw_response: Optional[str] = None


class ConceptGraphProjection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    evidence_links: list[dict[str, Any]]


class LLMResponseCache:
    """Tiny JSON-file cache for raw LLM responses."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Optional[str]:
        path = self._path(key)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("raw_response")

    def set(self, key: str, raw_response: str, metadata: Optional[dict[str, Any]] = None) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {"raw_response": raw_response, "metadata": metadata or {}}
        self._path(key).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def stable_hash(payload: Any, *, length: int = 16) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(normalize_nfc(text).encode("utf-8")).hexdigest()[:length]


class MarkupSanitizationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    changed: bool
    removed_tag_count: int = 0
    removed_image_count: int = 0
    preserved_alt_texts: list[str] = Field(default_factory=list)


class _EmbeddingMarkupParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.removed_tag_count = 0
        self.removed_image_count = 0
        self.preserved_alt_texts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        self._record_tag(tag, attrs, is_end=False)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        self._record_tag(tag, attrs, is_end=False)

    def handle_endtag(self, tag: str) -> None:
        self._record_tag(tag, [], is_end=True)

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(data)

    def _record_tag(self, tag: str, attrs: list[tuple[str, Optional[str]]], *, is_end: bool) -> None:
        tag = tag.casefold()
        self.removed_tag_count += 1
        if tag == "img" and not is_end:
            self.removed_image_count += 1
            alt_text = meaningful_image_alt(dict(attrs).get("alt"))
            if alt_text:
                self.preserved_alt_texts.append(alt_text)
                self.parts.extend([" ", alt_text, " "])
            return
        if tag in {"br", "p", "div", "center", "tr", "table", "thead", "tbody", "html", "body"}:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" | ")
        elif tag == "li" and not is_end:
            self.parts.append("\n")


def meaningful_image_alt(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = normalize_whitespace(unescape(value))
    normalized = re.sub(r"[^a-zа-яё0-9]+", " ", text.casefold()).strip()
    if normalized in GENERIC_IMAGE_ALT_TEXTS or re.fullmatch(r"(?:image|img|figure)\s*\d*", normalized):
        return None
    return text[:180] if text else None


def sanitize_markup_for_embedding(text: str) -> MarkupSanitizationResult:
    raw_text = str(text or "")
    incomplete_tag_count = 0
    incomplete_image_count = 0

    def remove_incomplete_tag(match: re.Match[str]) -> str:
        nonlocal incomplete_tag_count, incomplete_image_count
        incomplete_tag_count += 1
        if match.group("tag").casefold() == "img":
            incomplete_image_count += 1
        return " "

    parser_input = raw_text
    if "<" in raw_text and ">" not in raw_text:
        parser_input = INCOMPLETE_MARKUP_TAG_RE.sub(remove_incomplete_tag, raw_text)
    parser = _EmbeddingMarkupParser()
    parser.feed(parser_input)
    parser.close()
    parsed_text = unescape("".join(parser.parts))

    partial_tag_count = len(PARTIAL_MARKUP_TAG_RE.findall(parsed_text))
    if partial_tag_count:
        parsed_text = PARTIAL_MARKUP_TAG_RE.sub(" ", parsed_text)

    sanitized = normalize_whitespace(parsed_text)
    original_normalized = normalize_whitespace(unescape(raw_text))
    return MarkupSanitizationResult(
        text=sanitized,
        changed=sanitized != original_normalized,
        removed_tag_count=parser.removed_tag_count + partial_tag_count + incomplete_tag_count,
        removed_image_count=parser.removed_image_count + incomplete_image_count,
        preserved_alt_texts=unique_preserving_order(parser.preserved_alt_texts),
    )


def normalize_issue_label(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_")


def normalize_predicate(value: str) -> str:
    value = normalize_whitespace(value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return value.upper()


def concept_key(name: str, concept_type: str) -> str:
    normalized_name = normalize_whitespace(name).casefold()
    normalized_name = re.sub(r"[^a-zа-яё0-9+._ -]+", "", normalized_name)
    normalized_name = re.sub(r"\s+", " ", normalized_name).strip()
    return f"{normalize_whitespace(concept_type).upper()}::{normalized_name}"


def concept_id_from_key(key: str) -> str:
    return f"concept_{stable_hash(key, length=20)}"


def relation_id_from_parts(*parts: Any) -> str:
    return f"rel_{stable_hash(parts, length=20)}"


def strip_html(text: str) -> str:
    return sanitize_markup_for_embedding(text).text


def alpha_tokens(text: str) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(text)}


def token_overlap_ratio(candidate: str, evidence_text: str) -> float:
    candidate_tokens = alpha_tokens(candidate)
    if not candidate_tokens:
        return 1.0
    evidence_tokens = alpha_tokens(evidence_text)
    return len(candidate_tokens & evidence_tokens) / len(candidate_tokens)


def span_supported(span: str, evidence_text: str) -> bool:
    span = normalize_whitespace(span)
    if not span:
        return False
    if span in evidence_text:
        return True
    return normalize_whitespace(span) in normalize_whitespace(evidence_text)


def cleaned_text_is_safe(text: Optional[str], chunk_input: ChunkPostprocessInput) -> tuple[bool, Optional[str]]:
    if not text:
        return True, None
    evidence_text = chunk_input.evidence_text()
    for pattern in DEBUG_LABEL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return False, f"cleaned text contains debug label matching {pattern!r}"
    if chunk_input.artifact_path and chunk_input.artifact_path in text:
        return False, "cleaned text contains artifact path"
    if token_overlap_ratio(text, evidence_text) < 0.55:
        return False, "cleaned text is not sufficiently grounded in source text"
    return True, None


def detect_preflags(chunk: dict[str, Any]) -> ChunkPreflags:
    text_parts = [
        chunk.get("embedding_text") or "",
        chunk.get("text") or "",
        chunk.get("display_text") or "",
        chunk.get("context_text") or "",
    ]
    combined = "\n".join(part for part in text_parts if part)
    combined_no_html = normalize_whitespace(strip_html(combined))
    lower = combined.casefold()
    html_fragment_detected = bool(HTML_TAG_RE.search(combined))
    source_card_candidate = bool(SOURCE_CARD_FIELD_RE.search(combined))
    boilerplate_markers = (
        "© 2026 google llc",
        "terms of service",
        "privacy policy",
        "we read every piece of feedback",
        "о сервисе",
        "авторам",
        "рекламодателям",
        "условия использования",
        "конфиденциальность",
        "как работает youtube",
    )
    boilerplate_candidate = any(marker in lower for marker in boilerplate_markers)
    has_visual_marker = "<img" in lower or "img src" in lower or "image_box" in lower or "chart_box" in lower
    caption_like = bool(CAPTION_RE.search(combined))
    short_after_html = len(combined_no_html) <= 260
    caption_only_candidate = caption_like and (short_after_html or combined_no_html.casefold().startswith("visualization:"))
    visual_content_missing_candidate = has_visual_marker or (
        caption_only_candidate and not re.search(r"\b(?:explains|shows that|means|because|therefore)\b", lower)
    )
    small_but_kept = "small_but_kept" in set(chunk.get("chunk_quality_flags") or [])
    return ChunkPreflags(
        html_fragment_detected=html_fragment_detected,
        caption_only_candidate=caption_only_candidate,
        source_card_candidate=source_card_candidate,
        boilerplate_candidate=boilerplate_candidate,
        visual_content_missing_candidate=visual_content_missing_candidate,
        small_but_kept=small_but_kept,
    )


def processing_mode_for(chunk: dict[str, Any], preflags: ChunkPreflags) -> ProcessingMode:
    if preflags.suspicious_count:
        if len(normalize_whitespace(chunk.get("text") or "")) >= 220:
            return ProcessingMode.BOTH
        return ProcessingMode.QUALITY_REVIEW
    return ProcessingMode.ENRICHMENT


def chunk_to_input(
    chunk: dict[str, Any],
    resources_by_id: dict[str, dict[str, Any]],
    gates_by_resource_id: dict[str, dict[str, Any]],
) -> ChunkPostprocessInput:
    resource_ids = list(chunk.get("external_resource_ids") or [])
    gate_decisions = {
        resource_id: gates_by_resource_id[resource_id]
        for resource_id in resource_ids
        if resource_id in gates_by_resource_id
    }
    preflags = detect_preflags(chunk)
    return ChunkPostprocessInput(
        chunk_id=chunk["id"],
        chunk_type=chunk["chunk_type"],
        chunk_role=chunk.get("chunk_role") or "paragraph_group",
        source=chunk["source"],
        path=list(chunk.get("path") or []),
        heading_path=list(chunk.get("heading_path") or []),
        line_start=chunk["line_start"],
        line_end=chunk["line_end"],
        source_block_ids=list(chunk.get("source_block_ids") or []),
        external_resource_ids=resource_ids,
        text=chunk.get("text") or "",
        embedding_text=chunk.get("embedding_text"),
        display_text=chunk.get("display_text"),
        context_text=chunk.get("context_text"),
        artifact_path=chunk.get("artifact_path"),
        artifact_line_start=chunk.get("artifact_line_start"),
        artifact_line_end=chunk.get("artifact_line_end"),
        chunk_quality_flags=list(chunk.get("chunk_quality_flags") or []),
        external_resource_urls=[
            resources_by_id[resource_id]["url"] for resource_id in resource_ids if resource_id in resources_by_id
        ],
        external_resource_content_type_hints=[
            resources_by_id[resource_id].get("content_type_hint", "")
            for resource_id in resource_ids
            if resource_id in resources_by_id
        ],
        external_resource_parse_statuses=[
            resources_by_id[resource_id].get("parse_status", "")
            for resource_id in resource_ids
            if resource_id in resources_by_id
        ],
        artifact_gate_policy_by_resource_id={
            resource_id: decision.get("policy", "") for resource_id, decision in gate_decisions.items()
        },
        artifact_gate_reason_codes_by_resource_id={
            resource_id: list(decision.get("reason_codes") or [])
            for resource_id, decision in gate_decisions.items()
        },
        artifact_gate_stats_by_resource_id={
            resource_id: dict(decision.get("stats") or {}) for resource_id, decision in gate_decisions.items()
        },
        preflags=preflags,
        processing_mode=processing_mode_for(chunk, preflags),
    )


def inputs_from_jsonl_dir(input_dir: Path) -> list[ChunkPostprocessInput]:
    input_dir = Path(input_dir)
    chunks = read_jsonl(input_dir / "retrieval_chunks.jsonl")
    resources = read_jsonl_if_exists(input_dir / "external_resources.jsonl")
    gates = read_jsonl_if_exists(input_dir / "artifact_gate_decisions.jsonl")
    resources_by_id = {row["id"]: row for row in resources}
    gates_by_resource_id = {row["external_resource_id"]: row for row in gates}
    return [chunk_to_input(chunk, resources_by_id, gates_by_resource_id) for chunk in chunks]


def inputs_from_optimized_result(result: OptimizedParseResult) -> list[ChunkPostprocessInput]:
    chunks = [_model_or_dataclass_to_dict(chunk) for chunk in result.retrieval_chunks]
    resources = [_model_or_dataclass_to_dict(resource) for resource in result.external_resources]
    gates = [_model_or_dataclass_to_dict(gate) for gate in result.artifact_gate_decisions]
    resources_by_id = {row["id"]: row for row in resources}
    gates_by_resource_id = {row["external_resource_id"]: row for row in gates}
    return [chunk_to_input(chunk, resources_by_id, gates_by_resource_id) for chunk in chunks]


def select_candidate_inputs(inputs: Iterable[ChunkPostprocessInput]) -> list[ChunkPostprocessInput]:
    candidates = [
        item
        for item in inputs
        if item.processing_mode != ProcessingMode.ENRICHMENT
        or len(normalize_whitespace(item.text)) >= 120
        or item.chunk_type == "external_artifact"
    ]
    return sorted(
        candidates,
        key=lambda item: (
            -item.preflags.suspicious_count,
            item.source,
            tuple(item.heading_path),
            item.line_start,
            item.chunk_id,
        ),
    )


def build_batches(
    inputs: Iterable[ChunkPostprocessInput],
    *,
    max_batch_chunks: int = 1,
    max_batch_chars: int = 9000,
) -> list[ChunkPostprocessBatch]:
    batches: list[ChunkPostprocessBatch] = []
    current: list[ChunkPostprocessInput] = []
    current_chars = 0
    current_key: Optional[tuple[str, tuple[str, ...]]] = None

    def flush() -> None:
        nonlocal current, current_chars, current_key
        if not current:
            return
        batch_id = f"batch_{stable_hash([item.input_hash for item in current], length=20)}"
        batches.append(ChunkPostprocessBatch(batch_id=batch_id, chunks=current))
        current = []
        current_chars = 0
        current_key = None

    for item in inputs:
        key = (item.source, tuple(item.heading_path or item.path))
        item_chars = len(item.evidence_text())
        should_flush = bool(current) and (
            key != current_key
            or len(current) >= max_batch_chunks
            or current_chars + item_chars > max_batch_chars
        )
        if should_flush:
            flush()
        current.append(item)
        current_chars += item_chars
        current_key = key
    flush()
    return batches


def batch_prompt_payload(batch: ChunkPostprocessBatch) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "batch_id": batch.batch_id,
        "chunks": [chunk.model_dump(mode="json") for chunk in batch.chunks],
    }


def response_schema_hint(*, pass_name: str = "single") -> dict[str, Any]:
    base_decision = {
        "chunk_id": "copy the exact input chunk_id",
        "action": (
            "keep | keep_with_cleaned_text | metadata_only | "
            "needs_visual_reparse | exclude_from_embedding | graph_only"
        ),
        "issue_types": ["short_snake_case_issue"],
        "educational_usefulness": "0.0-1.0",
        "confidence": "0.0-1.0",
        "warnings": ["short_snake_case_warning"],
        "cleaned_embedding_text": "null or <=350 chars of source-grounded cleaned text",
        "cleaned_display_text": "null or <=350 chars of source-grounded cleaned text",
        "chunk_summary": "null or <=25 words",
    }
    if pass_name == "quality":
        return {
            "decisions": [
                {
                    **base_decision,
                    "concepts": [],
                    "relations": [],
                    "reason": "<=20 words",
                }
            ]
        }
    return {
        "decisions": [
            {
                **base_decision,
                "concepts": [
                    {
                        "local_id": "c1",
                        "canonical_name": "concise English academic term",
                        "display_name": "concise visible label",
                        "type": CANONICAL_CONCEPT_TYPE_HINT,
                        "aliases": ["exact source aliases"],
                        "salience": "0.0-1.0",
                        "description": "null or <=15 words",
                        "evidence_spans": ["exact substring copied from provided input"],
                    }
                ],
                "relations": [
                    {
                        "source_concept_id": "c1",
                        "target_concept_id": "c2",
                        "raw_predicate": "UPPERCASE_SNAKE_CASE",
                        "canonical_predicate": "UPPERCASE_SNAKE_CASE",
                        "predicate_status": "existing for listed existing labels only | proposed for other labels",
                        "predicate_family": "hierarchy | composition | computation | causality | dependency | method | resource | data | training | evaluation | example | comparison | citation | other",
                        "predicate_definition": "null or <=12 words",
                        "relation_phrase": "null or <=20 words describing the grounded relation",
                        "generality_score": "0.0-1.0",
                        "retrieval_usefulness": "0.0-1.0",
                        "visualization_usefulness": "0.0-1.0",
                        "confidence": "0.0-1.0",
                        "evidence_chunk_ids": ["current chunk_id"],
                        "evidence_spans": ["exact substring copied from provided input"],
                    }
                ],
                "reason": "<=20 words",
            }
        ]
    }


def cache_key_for_batch(
    batch: ChunkPostprocessBatch,
    *,
    model_name: str,
    prompt_version: str,
    prompt_content_hash: Optional[str] = None,
    generation_settings: Optional[dict[str, Any]] = None,
) -> str:
    return stable_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "prompt_version": prompt_version,
            "prompt_content_hash": prompt_content_hash,
            "model_name": model_name,
            "generation_settings": generation_settings or {},
            "batch_input_hash": batch.input_hash,
        },
        length=32,
    )


def parse_llm_response(raw_response: str) -> LLMPostprocessBatchResponse:
    cleaned = clean_json_markdown(raw_response)
    payload = load_llm_json_payload(cleaned)
    if isinstance(payload, list):
        payload = {"decisions": payload}
    if "decisions" not in payload and "chunk_id" in payload:
        payload = {"decisions": [payload]}
    return LLMPostprocessBatchResponse.model_validate(payload)


def load_llm_json_payload(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as original_error:
        repaired = repair_llm_json_text(text)
        if repaired and repaired != text:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        raise original_error


def repair_llm_json_text(text: str) -> Optional[str]:
    candidate = extract_json_candidate(text)
    if not candidate:
        return None
    if candidate != text:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    decision_object = extract_first_decision_object(candidate)
    if decision_object:
        return '{"decisions":[' + decision_object + ']}'
    if candidate.startswith("{"):
        single_object = extract_balanced_json_object(candidate, 0)
        if single_object:
            return single_object
    return None


def extract_json_candidate(text: str) -> Optional[str]:
    stripped = text.strip()
    if not stripped:
        return None
    starts = [index for index in (stripped.find("{"), stripped.find("[")) if index >= 0]
    if not starts:
        return None
    start = min(starts)
    return stripped[start:].strip()


def extract_first_decision_object(text: str) -> Optional[str]:
    decisions_index = text.find('"decisions"')
    if decisions_index < 0:
        return None
    array_index = text.find("[", decisions_index)
    if array_index < 0:
        return None
    object_index = text.find("{", array_index)
    if object_index < 0:
        return None
    return extract_balanced_json_object(text, object_index)


def extract_balanced_json_object(text: str, start_index: int) -> Optional[str]:
    if start_index < 0 or start_index >= len(text) or text[start_index] != "{":
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_index:index + 1]
    return None


def validate_and_enrich_response(
    response: LLMPostprocessBatchResponse,
    batch: ChunkPostprocessBatch,
    *,
    model_name: str,
    prompt_version: str,
    raw_response: Optional[str] = None,
) -> tuple[list[ChunkEnrichmentDecision], list[PostprocessFailure]]:
    by_chunk_id = {chunk.chunk_id: chunk for chunk in batch.chunks}
    seen_decisions: set[str] = set()
    decisions: list[ChunkEnrichmentDecision] = []
    failures: list[PostprocessFailure] = []

    for llm_decision in response.decisions:
        chunk_input = by_chunk_id.get(llm_decision.chunk_id)
        if chunk_input is None:
            failures.append(
                make_failure(
                    "unknown_chunk_id",
                    f"LLM returned decision for unknown chunk_id={llm_decision.chunk_id}",
                    batch=batch,
                    chunk_id=llm_decision.chunk_id,
                    raw_response=raw_response,
                    model_name=model_name,
                    prompt_version=prompt_version,
                )
            )
            continue
        if chunk_input.chunk_id in seen_decisions:
            failures.append(
                make_failure(
                    "duplicate_decision",
                    f"LLM response included more than one decision for chunk_id={chunk_input.chunk_id}",
                    batch=batch,
                    chunk_id=chunk_input.chunk_id,
                    input_hash=chunk_input.input_hash,
                    raw_response=raw_response,
                    model_name=model_name,
                    prompt_version=prompt_version,
                )
            )
            continue
        seen_decisions.add(chunk_input.chunk_id)
        llm_decision = normalize_decision_against_input(llm_decision, chunk_input)
        errors = validate_decision_against_input(llm_decision, chunk_input)
        if errors:
            failures.append(
                make_failure(
                    "validation_error",
                    "; ".join(errors),
                    batch=batch,
                    chunk_id=chunk_input.chunk_id,
                    raw_response=raw_response,
                    model_name=model_name,
                    prompt_version=prompt_version,
                )
            )
            continue

        decision_payload = llm_decision.model_dump(mode="json")
        decision_payload.update(
            {
                "decision_id": f"decision_{stable_hash([chunk_input.input_hash, decision_payload], length=24)}",
                "schema_version": SCHEMA_VERSION,
                "prompt_version": prompt_version,
                "model_name": model_name,
                "input_hash": chunk_input.input_hash,
            }
        )
        decisions.append(ChunkEnrichmentDecision.model_validate(decision_payload))

    for chunk_id, chunk_input in by_chunk_id.items():
        if chunk_id in seen_decisions:
            continue
        failures.append(
            make_failure(
                "missing_decision",
                f"LLM response did not include a decision for chunk_id={chunk_id}",
                batch=batch,
                chunk_id=chunk_id,
                input_hash=chunk_input.input_hash,
                raw_response=raw_response,
                model_name=model_name,
                prompt_version=prompt_version,
            )
        )
    return decisions, failures


def normalize_decision_against_input(
    decision: LLMChunkDecision,
    chunk_input: ChunkPostprocessInput,
) -> LLMChunkDecision:
    payload = decision.model_dump(mode="json")
    evidence_text = chunk_input.evidence_text()
    warnings = list(payload.get("warnings") or [])
    changed = False

    if payload.get("action") == ChunkAction.NEEDS_VISUAL_REPARSE.value:
        issue_types = list(payload.get("issue_types") or [])
        if "visual_content_missing" not in issue_types:
            issue_types.append("visual_content_missing")
            payload["issue_types"] = issue_types
            changed = True

    for concept in payload.get("concepts") or []:
        spans, span_changed = normalize_evidence_spans(concept.get("evidence_spans") or [], evidence_text)
        concept["evidence_spans"] = spans
        changed = changed or span_changed

    for relation in payload.get("relations") or []:
        canonical_predicate = normalize_predicate(relation.get("canonical_predicate", ""))
        canonical_alias = PREDICATE_CANONICAL_ALIASES.get(canonical_predicate)
        if canonical_alias:
            relation["canonical_predicate"] = canonical_alias
            warnings.append(PREDICATE_ALIAS_WARNING)
            changed = True
            canonical_predicate = canonical_alias
        if (
            relation.get("predicate_status") == PredicateStatus.EXISTING.value
            and canonical_predicate
            and canonical_predicate not in EXISTING_RELATION_LABELS
        ):
            relation["predicate_status"] = PredicateStatus.PROPOSED.value
            relation["predicate_definition"] = relation.get("predicate_definition") or "Proposed grounded relation."
            warnings.append(UNKNOWN_EXISTING_PREDICATE_WARNING)
            changed = True
        spans, span_changed = normalize_evidence_spans(relation.get("evidence_spans") or [], evidence_text)
        relation["evidence_spans"] = spans
        changed = changed or span_changed

    if changed:
        warnings = unique_preserving_order([*warnings, "postprocess_normalized_llm_output"])
        payload["warnings"] = warnings
    return LLMChunkDecision.model_validate(payload)


def normalize_evidence_spans(spans: list[str], evidence_text: str) -> tuple[list[str], bool]:
    normalized: list[str] = []
    changed = False
    for span in spans:
        if span_supported(span, evidence_text):
            normalized.append(span)
            continue
        supported = find_supported_evidence_variant(span, evidence_text)
        if supported:
            normalized.append(supported)
            changed = True
        else:
            normalized.append(span)
    return unique_preserving_order(normalized), changed


def find_supported_evidence_variant(span: str, evidence_text: str) -> Optional[str]:
    span = normalize_whitespace(span)
    if not span or not evidence_text:
        return None

    casefold_index = evidence_text.casefold().find(span.casefold())
    if casefold_index >= 0:
        return evidence_text[casefold_index:casefold_index + len(span)]

    target_key = fuzzy_evidence_key(span)
    if not target_key:
        return None
    target_token_count = max(1, len(TOKEN_RE.findall(span)))
    source_tokens = list(TOKEN_RE.finditer(evidence_text))
    if not source_tokens:
        return None

    best_score = 0.0
    best_candidate: Optional[str] = None
    min_size = max(1, target_token_count - 2)
    max_size = min(len(source_tokens), target_token_count + 2, 12)
    for size in range(min_size, max_size + 1):
        for start in range(0, len(source_tokens) - size + 1):
            end = start + size - 1
            candidate = evidence_text[source_tokens[start].start():source_tokens[end].end()]
            if len(candidate) > max(240, len(span) * 4):
                continue
            candidate_key = fuzzy_evidence_key(candidate)
            if candidate_key == target_key:
                return candidate
            score = SequenceMatcher(None, target_key, candidate_key).ratio()
            if score > best_score:
                best_score = score
                best_candidate = candidate

    threshold = 0.94 if len(target_key) <= 8 else 0.86
    if best_candidate and best_score >= threshold:
        return best_candidate
    return None


def fuzzy_evidence_key(text: str) -> str:
    replacements = {
        "l": "i",
        "1": "i",
        "|": "i",
        "і": "i",
        "І": "i",
        "а": "a",
        "А": "a",
        "е": "e",
        "Е": "e",
        "ё": "e",
        "Ё": "e",
        "о": "o",
        "О": "o",
        "р": "p",
        "Р": "p",
        "с": "c",
        "С": "c",
        "у": "y",
        "У": "y",
        "х": "x",
        "Х": "x",
        "в": "v",
        "В": "v",
        "м": "m",
        "М": "m",
        "т": "t",
        "Т": "t",
        "н": "h",
        "Н": "h",
        "к": "k",
        "К": "k",
    }
    translated = "".join(replacements.get(char, char) for char in text.casefold())
    return " ".join(TOKEN_RE.findall(translated))


def validate_decision_against_input(decision: LLMChunkDecision, chunk_input: ChunkPostprocessInput) -> list[str]:
    errors: list[str] = []
    evidence_text = chunk_input.evidence_text()
    concept_ids = {concept.local_id for concept in decision.concepts}

    if decision.chunk_id != chunk_input.chunk_id:
        errors.append("decision chunk_id does not match input chunk_id")

    for field_name, text in (
        ("cleaned_embedding_text", decision.cleaned_embedding_text),
        ("cleaned_display_text", decision.cleaned_display_text),
    ):
        ok, reason = cleaned_text_is_safe(text, chunk_input)
        if not ok:
            errors.append(f"{field_name}: {reason}")

    for concept in decision.concepts:
        if not concept.evidence_spans:
            errors.append(f"concept {concept.local_id} has no evidence_spans")
        for span in concept.evidence_spans:
            if not span_supported(span, evidence_text):
                errors.append(f"concept {concept.local_id} evidence span is not source-grounded: {span!r}")

    for relation in decision.relations:
        if (
            relation.predicate_status == PredicateStatus.EXISTING
            and relation.canonical_predicate not in EXISTING_RELATION_LABELS
        ):
            errors.append(
                f"relation {relation.canonical_predicate} is marked existing but is not in existing relation labels"
            )
        if relation.source_concept_id not in concept_ids:
            errors.append(f"relation source_concept_id is unknown: {relation.source_concept_id}")
        if relation.target_concept_id not in concept_ids:
            errors.append(f"relation target_concept_id is unknown: {relation.target_concept_id}")
        if chunk_input.chunk_id not in relation.evidence_chunk_ids:
            errors.append(f"relation {relation.canonical_predicate} lacks current chunk evidence id")
        if not relation.evidence_spans:
            errors.append(f"relation {relation.canonical_predicate} has no evidence_spans")
        for span in relation.evidence_spans:
            if not span_supported(span, evidence_text):
                errors.append(f"relation {relation.canonical_predicate} evidence span is not source-grounded: {span!r}")
        if relation.canonical_predicate == "RELATED_TO" and specific_relation_cue_present(relation.evidence_spans):
            errors.append("RELATED_TO is too generic for evidence with a specific relation cue")
        if relation.relation_phrase and len(relation.relation_phrase.split()) > 20:
            errors.append(f"relation {relation.canonical_predicate} relation_phrase exceeds 20 words")

    return errors


def specific_relation_cue_present(spans: Iterable[str]) -> bool:
    text = " ".join(spans).casefold()
    cues = (
        "uses",
        "produce",
        "generates",
        "computes",
        "calculates",
        "part of",
        "component",
        "solves",
        "requires",
        "derives",
        "is a",
        "example of",
    )
    return any(cue in text for cue in cues)


def make_failure(
    error_type: str,
    message: str,
    *,
    batch: Optional[ChunkPostprocessBatch] = None,
    chunk_id: Optional[str] = None,
    input_hash: Optional[str] = None,
    raw_response: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> PostprocessFailure:
    payload = {
        "error_type": error_type,
        "message": message,
        "batch_id": batch.batch_id if batch else None,
        "chunk_id": chunk_id,
        "input_hash": input_hash or (batch.input_hash if batch else None),
        "raw_response_hash": stable_hash(raw_response or "", length=16),
    }
    return PostprocessFailure(
        failure_id=f"failure_{stable_hash(payload, length=24)}",
        chunk_id=chunk_id,
        batch_id=batch.batch_id if batch else None,
        prompt_version=prompt_version,
        model_name=model_name,
        input_hash=input_hash or (batch.input_hash if batch else None),
        error_type=error_type,
        message=message,
        raw_response=raw_response,
    )


def build_graph_projection(
    decisions: Iterable[ChunkEnrichmentDecision],
    concept_resolution: Optional[ConceptResolution] = None,
) -> ConceptGraphProjection:
    decision_list = list(decisions)
    node_records: dict[str, dict[str, Any]] = {}
    local_to_global: dict[tuple[str, str], str] = {}
    edge_records: dict[str, dict[str, Any]] = {}
    evidence_links: list[dict[str, Any]] = []

    if concept_resolution is not None:
        for entry in concept_resolution.registry_entries:
            node_records[entry.concept_id] = {
                "id": entry.concept_id,
                "canonical_name": entry.canonical_name,
                "display_name": entry.display_name,
                "type": entry.type,
                "aliases": list(entry.aliases),
                "descriptions": list(entry.descriptions),
                "source_names": list(entry.source_names),
                "source_types": list(entry.source_types),
                "source_chunk_ids": list(entry.source_chunk_ids),
                "evidence_spans": list(entry.evidence_spans),
                "mention_ids": list(entry.mention_ids),
                "resolution_source": entry.resolution_source,
                "resolution_sources": list(entry.resolution_sources),
                "merge_status": entry.merge_status,
                "merge_statuses": list(entry.merge_statuses),
                "merge_score": entry.merge_score,
                "merge_flags": list(entry.merge_flags),
                "adjudication_cluster_ids": list(entry.adjudication_cluster_ids),
                "adjudication_actions": list(entry.adjudication_actions),
                "adjudication_rationales": list(entry.adjudication_rationales),
                "max_salience": entry.max_salience,
            }

    for decision in decision_list:
        for concept in decision.concepts:
            global_id = None
            if concept_resolution is not None:
                mention_id = mention_id_for(decision.decision_id, decision.chunk_id, concept.local_id)
                global_id = concept_resolution.mention_to_concept_id.get(mention_id)
            if global_id is None:
                key = concept_key(concept.canonical_name, concept.type)
                global_id = concept_id_from_key(key)
            local_to_global[(decision.decision_id, concept.local_id)] = global_id
            record = node_records.setdefault(
                global_id,
                {
                    "id": global_id,
                    "canonical_name": concept.canonical_name,
                    "display_name": concept.display_name or concept.canonical_name,
                    "type": concept.type,
                    "aliases": [],
                    "descriptions": [],
                    "source_chunk_ids": [],
                    "evidence_spans": [],
                    "max_salience": concept.salience,
                },
            )
            record["aliases"] = unique_preserving_order([*record.get("aliases", []), *concept.aliases])
            if concept.description:
                record["descriptions"] = unique_preserving_order([*record.get("descriptions", []), concept.description])
            record["source_chunk_ids"] = unique_preserving_order([*record.get("source_chunk_ids", []), decision.chunk_id])
            record["evidence_spans"] = unique_preserving_order(
                [*record.get("evidence_spans", []), *concept.evidence_spans]
            )
            record["max_salience"] = max(record.get("max_salience", 0.0), concept.salience)
            evidence_links.append(
                {
                    "concept_id": global_id,
                    "chunk_id": decision.chunk_id,
                    "evidence_spans": concept.evidence_spans,
                    "decision_id": decision.decision_id,
                }
            )

    for decision in decision_list:
        for relation in decision.relations:
            source_id = local_to_global.get((decision.decision_id, relation.source_concept_id))
            target_id = local_to_global.get((decision.decision_id, relation.target_concept_id))
            if not source_id or not target_id:
                continue
            if source_id == target_id:
                continue
            relation_id = relation_id_from_parts(source_id, relation.canonical_predicate, target_id)
            record = edge_records.setdefault(
                relation_id,
                {
                    "id": relation_id,
                    "source_concept_id": source_id,
                    "target_concept_id": target_id,
                    "canonical_predicate": relation.canonical_predicate,
                    "raw_predicates": [],
                    "predicate_statuses": [],
                    "predicate_family": relation.predicate_family,
                    "predicate_definitions": [],
                    "relation_phrases": [],
                    "evidence_chunk_ids": [],
                    "evidence_spans": [],
                    "max_confidence": relation.confidence,
                    "max_generality_score": relation.generality_score,
                    "max_retrieval_usefulness": relation.retrieval_usefulness,
                    "max_visualization_usefulness": relation.visualization_usefulness,
                    "decision_ids": [],
                },
            )
            record["raw_predicates"] = unique_preserving_order([*record["raw_predicates"], relation.raw_predicate])
            record["predicate_statuses"] = unique_preserving_order(
                [*record["predicate_statuses"], relation.predicate_status.value]
            )
            if relation.predicate_definition:
                record["predicate_definitions"] = unique_preserving_order(
                    [*record["predicate_definitions"], relation.predicate_definition]
                )
            if relation.relation_phrase:
                record["relation_phrases"] = unique_preserving_order(
                    [*record["relation_phrases"], relation.relation_phrase]
                )
            record["evidence_chunk_ids"] = unique_preserving_order(
                [*record["evidence_chunk_ids"], *relation.evidence_chunk_ids]
            )
            record["evidence_spans"] = unique_preserving_order([*record["evidence_spans"], *relation.evidence_spans])
            record["max_confidence"] = max(record["max_confidence"], relation.confidence)
            record["max_generality_score"] = max(record["max_generality_score"], relation.generality_score)
            record["max_retrieval_usefulness"] = max(
                record["max_retrieval_usefulness"], relation.retrieval_usefulness
            )
            record["max_visualization_usefulness"] = max(
                record["max_visualization_usefulness"], relation.visualization_usefulness
            )
            record["decision_ids"] = unique_preserving_order([*record["decision_ids"], decision.decision_id])

    return ConceptGraphProjection(
        nodes=sorted(node_records.values(), key=lambda item: (item["type"], item["canonical_name"])),
        edges=sorted(edge_records.values(), key=lambda item: item["id"]),
        evidence_links=evidence_links,
    )


def build_relation_registry(decisions: Iterable[ChunkEnrichmentDecision]) -> list[dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        for relation in decision.relations:
            record = registry.setdefault(
                relation.canonical_predicate,
                {
                    "canonical_predicate": relation.canonical_predicate,
                    "raw_predicates": [],
                    "predicate_statuses": [],
                    "predicate_families": [],
                    "predicate_definitions": [],
                    "relation_phrases": [],
                    "count": 0,
                    "max_generality_score": 0.0,
                    "max_retrieval_usefulness": 0.0,
                    "max_visualization_usefulness": 0.0,
                    "example_chunk_ids": [],
                },
            )
            record["count"] += 1
            record["raw_predicates"] = unique_preserving_order([*record["raw_predicates"], relation.raw_predicate])
            record["predicate_statuses"] = unique_preserving_order(
                [*record["predicate_statuses"], relation.predicate_status.value]
            )
            record["predicate_families"] = unique_preserving_order(
                [*record["predicate_families"], relation.predicate_family]
            )
            if relation.predicate_definition:
                record["predicate_definitions"] = unique_preserving_order(
                    [*record["predicate_definitions"], relation.predicate_definition]
                )
            if relation.relation_phrase:
                record["relation_phrases"] = unique_preserving_order(
                    [*record["relation_phrases"], relation.relation_phrase]
                )[:5]
            record["max_generality_score"] = max(record["max_generality_score"], relation.generality_score)
            record["max_retrieval_usefulness"] = max(
                record["max_retrieval_usefulness"], relation.retrieval_usefulness
            )
            record["max_visualization_usefulness"] = max(
                record["max_visualization_usefulness"], relation.visualization_usefulness
            )
            record["example_chunk_ids"] = unique_preserving_order(
                [*record["example_chunk_ids"], *relation.evidence_chunk_ids]
            )[:5]
    return sorted(registry.values(), key=lambda item: (-item["count"], item["canonical_predicate"]))


def build_report_payload(
    inputs: Iterable[ChunkPostprocessInput],
    decisions: Iterable[ChunkEnrichmentDecision],
    failures: Iterable[PostprocessFailure],
    projection: ConceptGraphProjection,
    relation_registry: list[dict[str, Any]],
    concept_resolution: Optional[ConceptResolution] = None,
    *,
    cache_hits: int = 0,
    cache_misses: int = 0,
    concept_cache_hits: int = 0,
    concept_cache_misses: int = 0,
) -> dict[str, Any]:
    input_list = list(inputs)
    decision_list = list(decisions)
    failure_list = list(failures)
    action_counts = Counter(decision.action.value for decision in decision_list)
    issue_counts = Counter(issue for decision in decision_list for issue in decision.issue_types)
    mode_counts = Counter(item.processing_mode.value for item in input_list)
    concept_type_counts = Counter(entry.type for entry in concept_resolution.registry_entries) if concept_resolution else Counter()
    concept_source_type_counts = (
        Counter(source_type for entry in concept_resolution.registry_entries for source_type in entry.source_types)
        if concept_resolution
        else Counter()
    )
    review_clusters = concept_resolution.review_clusters if concept_resolution else []
    adjudication_failures = concept_resolution.adjudication_failures if concept_resolution else []
    preflag_counts = Counter()
    for item in input_list:
        for key, value in item.preflags.model_dump(mode="json").items():
            if value:
                preflag_counts[key] += 1
    failed_chunk_ids = sorted({failure.chunk_id for failure in failure_list if failure.chunk_id})
    return {
        "schema_version": SCHEMA_VERSION,
        "input_count": len(input_list),
        "decision_count": len(decision_list),
        "failure_count": len(failure_list),
        "failure_record_count": len(failure_list),
        "failed_chunk_count": len(failed_chunk_ids),
        "failed_chunk_ids": failed_chunk_ids,
        "action_counts": dict(action_counts),
        "issue_counts": dict(issue_counts),
        "processing_mode_counts": dict(mode_counts),
        "preflag_counts": dict(preflag_counts),
        "concept_node_count": len(projection.nodes),
        "relation_edge_count": len(projection.edges),
        "relation_registry_count": len(relation_registry),
        "concept_registry_count": len(concept_resolution.registry_entries) if concept_resolution else 0,
        "concept_review_cluster_count": len(concept_resolution.review_clusters) if concept_resolution else 0,
        "concept_pair_score_count": len(concept_resolution.pair_scores) if concept_resolution else 0,
        "concept_adjudication_count": len(concept_resolution.adjudications) if concept_resolution else 0,
        "concept_adjudication_failure_count": len(concept_resolution.adjudication_failures) if concept_resolution else 0,
        "concept_type_counts": dict(concept_type_counts),
        "concept_source_type_counts": dict(concept_source_type_counts),
        "max_concept_review_cluster_mentions": max((len(cluster.mention_ids) for cluster in review_clusters), default=0),
        "max_concept_review_cluster_pair_scores": max((len(cluster.pair_scores) for cluster in review_clusters), default=0),
        "skipped_over_budget_concept_cluster_count": sum(
            1 for failure in adjudication_failures if failure.error_type == "prompt_budget_exceeded"
        ),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "concept_cache_hits": concept_cache_hits,
        "concept_cache_misses": concept_cache_misses,
        "failures_by_type": dict(Counter(failure.error_type for failure in failure_list)),
    }


def build_report_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# LLM Postprocess Report",
        "",
        "## Summary",
        "",
        f"- Inputs: {payload['input_count']}",
        f"- Valid decisions: {payload['decision_count']}",
        f"- Failures: {payload['failure_count']}",
        f"- Failed chunks: {payload.get('failed_chunk_count', 0)}",
        f"- Concept graph nodes: {payload['concept_node_count']}",
        f"- Concept graph edges: {payload['relation_edge_count']}",
        f"- Relation registry entries: {payload['relation_registry_count']}",
        f"- Concept registry entries: {payload['concept_registry_count']}",
        f"- Concept review clusters: {payload['concept_review_cluster_count']}",
        f"- Max concept review cluster size: {payload.get('max_concept_review_cluster_mentions', 0)} mentions / {payload.get('max_concept_review_cluster_pair_scores', 0)} pairs",
        f"- Skipped over-budget concept clusters: {payload.get('skipped_over_budget_concept_cluster_count', 0)}",
        f"- Concept adjudications/failures: {payload['concept_adjudication_count']} / {payload['concept_adjudication_failure_count']}",
        f"- Cache hits/misses: {payload['cache_hits']} / {payload['cache_misses']}",
        f"- Concept cache hits/misses: {payload['concept_cache_hits']} / {payload['concept_cache_misses']}",
        "",
        "## Actions",
        "",
    ]
    for key, value in sorted(payload["action_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Preflags", ""])
    for key, value in sorted(payload["preflag_counts"].items()):
        lines.append(f"- `{key}`: {value}")
    if payload.get("concept_type_counts"):
        lines.extend(["", "## Concept Types", ""])
        for key, value in sorted(payload["concept_type_counts"].items()):
            lines.append(f"- `{key}`: {value}")
    if payload["failures_by_type"]:
        lines.extend(["", "## Failures", ""])
        for key, value in sorted(payload["failures_by_type"].items()):
            lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def write_sidecar_outputs(
    output_dir: Path,
    *,
    inputs: Iterable[ChunkPostprocessInput],
    decisions: Iterable[ChunkEnrichmentDecision],
    failures: Iterable[PostprocessFailure],
    concept_resolution: Optional[ConceptResolution] = None,
    cache_hits: int = 0,
    cache_misses: int = 0,
    concept_cache_hits: int = 0,
    concept_cache_misses: int = 0,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_list = list(inputs)
    decisions_list = list(decisions)
    failures_list = list(failures)
    projection = build_graph_projection(decisions_list, concept_resolution=concept_resolution)
    relation_registry = build_relation_registry(decisions_list)
    report_payload = build_report_payload(
        inputs_list,
        decisions_list,
        failures_list,
        projection,
        relation_registry,
        concept_resolution,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        concept_cache_hits=concept_cache_hits,
        concept_cache_misses=concept_cache_misses,
    )

    write_jsonl(output_dir / INPUTS_FILENAME, [item.model_dump(mode="json") for item in inputs_list])
    write_jsonl(output_dir / DECISIONS_FILENAME, [item.model_dump(mode="json") for item in decisions_list])
    write_jsonl(output_dir / FAILURES_FILENAME, [item.model_dump(mode="json") for item in failures_list])
    write_jsonl(output_dir / RELATION_REGISTRY_FILENAME, relation_registry)
    if concept_resolution is not None:
        write_jsonl(
            output_dir / CONCEPT_REGISTRY_FILENAME,
            [item.model_dump(mode="json") for item in concept_resolution.registry_entries],
        )
        write_jsonl(
            output_dir / CONCEPT_MERGE_REVIEW_FILENAME,
            [item.model_dump(mode="json") for item in concept_resolution.review_clusters],
        )
        write_jsonl(
            output_dir / CONCEPT_PAIR_SCORES_FILENAME,
            [item.model_dump(mode="json") for item in concept_resolution.pair_scores],
        )
        write_jsonl(
            output_dir / CONCEPT_ADJUDICATIONS_FILENAME,
            [item.model_dump(mode="json") for item in concept_resolution.adjudications],
        )
        write_jsonl(
            output_dir / CONCEPT_ADJUDICATION_FAILURES_FILENAME,
            [item.model_dump(mode="json") for item in concept_resolution.adjudication_failures],
        )
    write_jsonl(output_dir / GRAPH_PREVIEW_FILENAME, graph_projection_jsonl_rows(projection))
    write_json(output_dir / REPORT_JSON_FILENAME, report_payload)
    (output_dir / REPORT_MD_FILENAME).write_text(build_report_markdown(report_payload), encoding="utf-8")
    return report_payload


def graph_projection_jsonl_rows(projection: ConceptGraphProjection) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend({"record_type": "concept_node", **node} for node in projection.nodes)
    rows.extend({"record_type": "relation_edge", **edge} for edge in projection.edges)
    rows.extend({"record_type": "evidence_link", **link} for link in projection.evidence_links)
    return rows


def fake_llm_response_for_batch(batch: ChunkPostprocessBatch) -> str:
    decisions: list[dict[str, Any]] = []
    for chunk in batch.chunks:
        evidence = first_supported_evidence(chunk)
        concepts = []
        relations = []
        if chunk.preflags.visual_content_missing_candidate:
            action = ChunkAction.NEEDS_VISUAL_REPARSE.value
            issue_types = ["visual_content_missing"]
            usefulness = 0.2
            confidence = 0.75
        else:
            action = ChunkAction.KEEP.value
            issue_types = []
            usefulness = 0.8
            confidence = 0.7
            concept_name = default_concept_name(chunk)
            concepts.append(
                {
                    "local_id": "c1",
                    "canonical_name": concept_name,
                    "display_name": concept_name,
                    "type": "CONCEPT",
                    "aliases": [],
                    "salience": 0.7,
                    "description": None,
                    "evidence_spans": [evidence],
                }
            )
            if "produces gradients" in evidence.casefold():
                concepts.append(
                    {
                        "local_id": "c2",
                        "canonical_name": "Gradient",
                        "display_name": "Gradient",
                        "type": "CONCEPT",
                        "aliases": ["gradients"],
                        "salience": 0.75,
                        "description": None,
                        "evidence_spans": ["gradients"],
                    }
                )
                relations.append(
                    {
                        "source_concept_id": "c1",
                        "target_concept_id": "c2",
                        "raw_predicate": "PRODUCES",
                        "canonical_predicate": "PRODUCES",
                        "predicate_status": "existing",
                        "predicate_family": "computation",
                        "predicate_definition": None,
                        "relation_phrase": "produces gradients",
                        "generality_score": 0.8,
                        "retrieval_usefulness": 0.85,
                        "visualization_usefulness": 0.75,
                        "confidence": 0.75,
                        "evidence_chunk_ids": [chunk.chunk_id],
                        "evidence_spans": ["produces gradients"],
                    }
                )
        decisions.append(
            {
                "chunk_id": chunk.chunk_id,
                "action": action,
                "issue_types": issue_types,
                "educational_usefulness": usefulness,
                "confidence": confidence,
                "warnings": [],
                "cleaned_embedding_text": None,
                "cleaned_display_text": None,
                "chunk_summary": evidence,
                "concepts": concepts,
                "relations": relations,
                "reason": "deterministic fake response for tests and dry runs",
            }
        )
    return json.dumps({"decisions": decisions}, ensure_ascii=False)


def first_supported_evidence(chunk: ChunkPostprocessInput, max_chars: int = 180) -> str:
    text = normalize_whitespace(strip_html(chunk.text or chunk.embedding_text or chunk.source))
    if not text:
        text = normalize_whitespace(" > ".join(chunk.heading_path or chunk.path) or chunk.source)
    sentence = re.split(r"(?<=[.!?])\s+", text)[0]
    return sentence[:max_chars].strip() or chunk.source


def default_concept_name(chunk: ChunkPostprocessInput) -> str:
    for item in reversed(chunk.heading_path or chunk.path):
        cleaned = re.sub(r"^#{1,6}\s+", "", normalize_whitespace(item)).strip()
        if cleaned and not cleaned.startswith("external:"):
            return cleaned[:120]
    return normalize_whitespace(chunk.source)[:120] or "RemNote Chunk"


def clean_json_markdown(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSONL file does not exist: {path}")
    return read_jsonl_if_exists(path)


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_postprocess_inputs(output_dir: Path) -> list[ChunkPostprocessInput]:
    return [ChunkPostprocessInput.model_validate(row) for row in read_jsonl(Path(output_dir) / INPUTS_FILENAME)]


def load_postprocess_decisions(output_dir: Path) -> list[ChunkEnrichmentDecision]:
    return [ChunkEnrichmentDecision.model_validate(row) for row in read_jsonl(Path(output_dir) / DECISIONS_FILENAME)]


def load_postprocess_failures(output_dir: Path) -> list[PostprocessFailure]:
    return [PostprocessFailure.model_validate(row) for row in read_jsonl_if_exists(Path(output_dir) / FAILURES_FILENAME)]


def load_concept_resolution_sidecars(output_dir: Path) -> Optional[ConceptResolution]:
    output_dir = Path(output_dir)
    registry_path = output_dir / CONCEPT_REGISTRY_FILENAME
    if not registry_path.exists():
        return None
    registry_entries = [
        ConceptRegistryEntry.model_validate(row) for row in read_jsonl(registry_path)
    ]
    mention_to_concept_id = {
        mention_id: entry.concept_id
        for entry in registry_entries
        for mention_id in entry.mention_ids
    }
    return ConceptResolution(
        registry_entries=registry_entries,
        mention_to_concept_id=mention_to_concept_id,
        review_clusters=[
            UncertainConceptCluster.model_validate(row)
            for row in read_jsonl_if_exists(output_dir / CONCEPT_MERGE_REVIEW_FILENAME)
        ],
        pair_scores=[
            ConceptPairScore.model_validate(row)
            for row in read_jsonl_if_exists(output_dir / CONCEPT_PAIR_SCORES_FILENAME)
        ],
        adjudications=[
            ConceptAdjudicationResponse.model_validate(row)
            for row in read_jsonl_if_exists(output_dir / CONCEPT_ADJUDICATIONS_FILENAME)
        ],
        adjudication_failures=[
            ConceptAdjudicationFailure.model_validate(row)
            for row in read_jsonl_if_exists(output_dir / CONCEPT_ADJUDICATION_FAILURES_FILENAME)
        ],
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _model_or_dataclass_to_dict(value: RetrievalChunk | ExternalResource | ArtifactGateDecision) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return dict(value.__dict__)
