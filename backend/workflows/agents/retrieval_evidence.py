import ast
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.configs.constants import MAX_SOURCE_CHARS, RELATION_DROP_SCORE

FACT_BLOCK_HEADER = "Here are some facts extracted from the provided text:"
PROPERTY_PATTERN = re.compile(r"\(\{.*?\}\)", re.DOTALL)


class NormalizedMetadata(BaseModel):
    node_id: str | None = None
    chunk_id: str | None = None
    source: str | None = None
    path: list[str] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)
    line_start: int | None = None
    line_end: int | None = None
    source_block_ids: list[str] = Field(default_factory=list)
    external_resource_ids: list[str] = Field(default_factory=list)
    retrieval_enabled: bool | None = None
    graph_enabled: bool | None = None
    quarantined: bool | None = None
    postprocess_decision_id: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class SourceEvidence(BaseModel):
    kind: Literal["source"] = "source"
    rank: int
    score: float | None = None
    text: str
    metadata: NormalizedMetadata
    derived_from_relation_node: bool = False
    source_paths: list[str] = Field(default_factory=list)
    parse_warnings: list[str] = Field(default_factory=list)


class RelationEvidence(BaseModel):
    kind: Literal["relation"] = "relation"
    rank: int
    score: float | None = None
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    raw_relation: str
    relation_category: str = "triplet"
    metadata: NormalizedMetadata
    relation_properties: list[dict[str, Any]] = Field(default_factory=list)
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    evidence_spans: list[str] = Field(default_factory=list)
    confidence: float | None = None
    source_paths: list[str] = Field(default_factory=list)
    parse_warnings: list[str] = Field(default_factory=list)

    def as_triplet(self) -> tuple[str, str, str] | None:
        if self.subject and self.predicate and self.object:
            return self.subject, self.predicate, self.object

        parts = [p.strip() for p in self.raw_relation.split(" -> ")]
        if len(parts) == 3 and all(parts):
            return parts[0], parts[1], parts[2]
        return None


class QueryEvidenceResult(BaseModel):
    query: str
    items: list[SourceEvidence | RelationEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def evidence_from_retrieved_node(
    item: Any, *, query: str, rank: int
) -> list[SourceEvidence | RelationEvidence]:
    node, score = _unwrap_retrieved_item(item)
    text = _node_text(node)
    metadata = normalize_metadata(node)

    if "->" not in text:
        if not text:
            return []
        return [
            SourceEvidence(
                rank=rank,
                score=score,
                text=text,
                metadata=metadata,
                source_paths=_metadata_paths(metadata),
            )
        ]

    if FACT_BLOCK_HEADER in text:
        return _evidence_from_fact_block(
            text, score=score, rank=rank, metadata=metadata
        )

    return _evidence_from_relation_text(text, score=score, rank=rank, metadata=metadata)


def normalize_metadata(node: Any) -> NormalizedMetadata:
    raw_metadata = getattr(node, "metadata", None) or {}
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    node_id = _node_id(node)

    return NormalizedMetadata(
        node_id=node_id,
        chunk_id=_string_or_none(metadata.get("chunk_id") or node_id),
        source=_string_or_none(metadata.get("source")),
        path=_string_list(metadata.get("path")),
        heading_path=_string_list(metadata.get("heading_path")),
        line_start=_int_or_none(
            metadata.get("line_start")
            if metadata.get("line_start") is not None
            else metadata.get("line_number")
        ),
        line_end=_int_or_none(metadata.get("line_end")),
        source_block_ids=_string_list(metadata.get("source_block_ids")),
        external_resource_ids=_string_list(metadata.get("external_resource_ids")),
        retrieval_enabled=_bool_or_none(metadata.get("retrieval_enabled")),
        graph_enabled=_bool_or_none(metadata.get("graph_enabled")),
        quarantined=_bool_or_none(metadata.get("quarantined")),
        postprocess_decision_id=_string_or_none(
            metadata.get("postprocess_decision_id")
        ),
        raw=metadata,
    )


def format_search_results(results: list[QueryEvidenceResult]) -> str:
    evidence_lines: list[str] = []
    seen_sources_global: set[str] = set()
    seen_paths_global: set[str] = set()

    for result in results:
        query_lines: list[str] = []

        for evidence in result.items:
            if isinstance(evidence, RelationEvidence):
                query_lines.append(
                    _clean(
                        f"[RELATION] {evidence.raw_relation} (Score: {_score_for_display(evidence.score):.2f})"
                    )
                )

            if isinstance(evidence, SourceEvidence) and evidence.text:
                clipped = _smart_truncate(evidence.text.strip(), MAX_SOURCE_CHARS)
                dedup_key = clipped[:120]
                if dedup_key not in seen_sources_global:
                    seen_sources_global.add(dedup_key)
                    score_tag = (
                        ""
                        if evidence.derived_from_relation_node
                        else f" (Score: {_score_for_display(evidence.score):.2f})"
                    )
                    query_lines.append(f"[SOURCE]{score_tag} {_clean(clipped)}")

            for path in _evidence_paths(evidence):
                if path and path not in seen_paths_global:
                    seen_paths_global.add(path)
                    query_lines.append(f"[SOURCE PATH] {_clean(path)}")

        if query_lines:
            evidence_lines.append(f"QUERY: {result.query}\n" + "\n".join(query_lines))

    if not evidence_lines:
        return "No relevant information found."

    return "RETRIEVER RESULTS:\n\n" + "\n\n".join(evidence_lines)


def format_visualization_results(
    results: list[QueryEvidenceResult],
) -> tuple[list[str], list[tuple[str, str, str]], list[str]]:
    all_nodes: list[str] = []
    all_triplets: list[tuple[str, str, str]] = []
    queries = [result.query for result in results]

    for result in results:
        for evidence in result.items:
            if isinstance(evidence, RelationEvidence):
                triplet = evidence.as_triplet()
                if triplet:
                    all_triplets.append(triplet)
            elif evidence.metadata.node_id and not evidence.derived_from_relation_node:
                all_nodes.append(evidence.metadata.node_id)

    return _ordered_unique(all_nodes), _ordered_unique(all_triplets), queries


def _evidence_from_fact_block(
    text: str,
    *,
    score: float | None,
    rank: int,
    metadata: NormalizedMetadata,
) -> list[SourceEvidence | RelationEvidence]:
    warnings: list[str] = []
    parts = text.split("\n\n", 2)
    if len(parts) < 3:
        warnings.append(
            "fact block did not contain expected relation and source sections"
        )
        relations_text = text.replace(FACT_BLOCK_HEADER, "").strip()
        source_text = ""
    else:
        _, relations_text, source_text = parts

    evidence: list[SourceEvidence | RelationEvidence] = []
    for relation_line in _ordered_unique(
        line.strip() for line in relations_text.splitlines() if line.strip()
    ):
        parsed = _parse_relation_line(relation_line)
        evidence.append(
            RelationEvidence(
                rank=rank,
                score=score,
                subject=parsed["subject"],
                predicate=parsed["predicate"],
                object=parsed["object"],
                raw_relation=parsed["raw_relation"],
                relation_category="fact_block",
                metadata=metadata,
                relation_properties=parsed["properties"],
                evidence_chunk_ids=parsed["evidence_chunk_ids"],
                evidence_spans=parsed["evidence_spans"],
                confidence=parsed["confidence"],
                source_paths=_metadata_paths(metadata),
                parse_warnings=[*warnings, *parsed["warnings"]],
            )
        )

    if source_text.strip():
        evidence.append(
            SourceEvidence(
                rank=rank,
                score=score,
                text=source_text.strip(),
                metadata=metadata,
                derived_from_relation_node=True,
                source_paths=_metadata_paths(metadata),
                parse_warnings=warnings,
            )
        )

    return evidence


def _evidence_from_relation_text(
    text: str,
    *,
    score: float | None,
    rank: int,
    metadata: NormalizedMetadata,
) -> list[SourceEvidence | RelationEvidence]:
    if "PARENT" in text:
        return []

    score_value = _score_for_display(score)
    if "CHILD" in text and score_value < RELATION_DROP_SCORE:
        return []

    parsed = _parse_relation_line(text)
    source_paths = _paths_from_properties(parsed["properties"]) or _metadata_paths(
        metadata
    )
    relation_text = parsed["raw_relation"]

    if "CHILD" in text:
        child_relation = _child_relation_from_properties(parsed["properties"])
        if child_relation:
            relation_text, source_paths = child_relation

    if not relation_text:
        return []

    return [
        RelationEvidence(
            rank=rank,
            score=score,
            subject=parsed["subject"],
            predicate=parsed["predicate"],
            object=parsed["object"],
            raw_relation=relation_text,
            relation_category="child" if "CHILD" in text else "triplet",
            metadata=metadata,
            relation_properties=parsed["properties"],
            evidence_chunk_ids=parsed["evidence_chunk_ids"],
            evidence_spans=parsed["evidence_spans"],
            confidence=parsed["confidence"],
            source_paths=source_paths,
            parse_warnings=parsed["warnings"],
        )
    ]


def _parse_relation_line(line: str) -> dict[str, Any]:
    warnings: list[str] = []
    properties: list[dict[str, Any]] = []
    relation = line.strip()
    id_to_name: dict[str, str] = {}

    for raw_property in _ordered_unique(PROPERTY_PATTERN.findall(line)):
        parsed = _parse_property(raw_property)
        if parsed is None:
            warnings.append("failed to parse relation property")
            relation = relation.replace(raw_property, "")
            continue

        properties.append(parsed)
        relation = _replace_property_in_relation(relation, raw_property, parsed)

        internal_name = parsed.get("name")
        display_name = (
            parsed.get("entity_name")
            or parsed.get("display_name")
            or parsed.get("text")
        )
        if internal_name and display_name:
            id_to_name[str(internal_name)] = str(display_name)

    for internal_name, display_name in id_to_name.items():
        relation = relation.replace(internal_name, display_name)

    relation = _normalize_relation_text(relation)
    subject, predicate, object_ = _split_triplet(relation)

    return {
        "subject": subject,
        "predicate": predicate,
        "object": object_,
        "raw_relation": relation,
        "properties": properties,
        "evidence_chunk_ids": _ordered_unique(
            value
            for prop in properties
            for value in _string_list(
                prop.get("evidence_chunk_ids") or prop.get("source_chunk_ids")
            )
        ),
        "evidence_spans": _ordered_unique(
            value
            for prop in properties
            for value in _string_list(prop.get("evidence_spans"))
        ),
        "confidence": _max_number(
            prop.get("max_confidence") or prop.get("confidence") for prop in properties
        ),
        "warnings": warnings,
    }


def _parse_property(raw_property: str) -> dict[str, Any] | None:
    try:
        value = ast.literal_eval(raw_property)
    except Exception:
        return None

    if isinstance(value, dict):
        return value
    return None


def _replace_property_in_relation(
    relation: str, raw_property: str, prop: dict[str, Any]
) -> str:
    label = _label_from_property(prop)
    internal_name = _string_or_none(prop.get("name"))

    if internal_name:
        pattern = rf"\b{re.escape(internal_name)}\s*{re.escape(raw_property)}"
        replaced, count = re.subn(pattern, label, relation, count=1)
        if count:
            return replaced

    return relation.replace(raw_property, label)


def _child_relation_from_properties(
    properties: list[dict[str, Any]],
) -> tuple[str, list[str]] | None:
    node_texts: list[str] = []
    parsed_paths: list[str] = []

    for prop in properties[::2]:
        text = _string_or_none(prop.get("text"))
        if text:
            node_texts.append(text)

        path = _path_to_display(prop.get("path"))
        if path:
            parsed_paths.append(path)

    if len(node_texts) != 2:
        return None

    return f"{node_texts[0]} -> CHILD -> {node_texts[1]}", parsed_paths[:2]


def _paths_from_properties(properties: list[dict[str, Any]]) -> list[str]:
    return [
        path
        for path in (_path_to_display(prop.get("path")) for prop in properties)
        if path
    ]


def _metadata_paths(metadata: NormalizedMetadata) -> list[str]:
    path = _path_to_display(metadata.path)
    return [path] if path else []


def _evidence_paths(evidence: SourceEvidence | RelationEvidence) -> list[str]:
    if evidence.source_paths:
        return evidence.source_paths
    return _metadata_paths(evidence.metadata)


def _unwrap_retrieved_item(item: Any) -> tuple[Any, float | None]:
    node = getattr(item, "node", None)
    if node is not None:
        return node, _float_or_none(getattr(item, "score", None))
    return item, _float_or_none(getattr(item, "score", None))


def _node_text(node: Any) -> str:
    text = getattr(node, "text", None)
    if text is not None:
        return str(text)

    get_content = getattr(node, "get_content", None)
    if callable(get_content):
        try:
            return str(get_content())
        except Exception:
            return ""
    return ""


def _node_id(node: Any) -> str | None:
    for attr in ("node_id", "id_", "id"):
        value = getattr(node, attr, None)
        if value is not None:
            return str(value)
    return None


def _label_from_property(prop: dict[str, Any]) -> str:
    for key in ("entity_name", "display_name", "text", "name"):
        value = prop.get(key)
        if value:
            return str(value)
    return ""


def _split_triplet(relation: str) -> tuple[str | None, str | None, str | None]:
    parts = [part.strip() for part in relation.split(" -> ")]
    if len(parts) == 3 and all(parts):
        return parts[0], parts[1], parts[2]
    return None, None, None


def _normalize_relation_text(text: str) -> str:
    text = text.replace("  ", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*->\s*", " -> ", text)
    return text.strip()


def _path_to_display(value: Any) -> str:
    parts = _string_list(value)
    return " > ".join(parts)


def _string_list(value: Any) -> list[str]:
    parsed = _parse_structured_value(value)
    if parsed is None:
        return []
    if isinstance(parsed, list | tuple | set):
        return [str(item) for item in parsed if item is not None]
    return [str(parsed)] if parsed != "" else []


def _parse_structured_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return None
    if stripped in {"None", "null"}:
        return None

    if stripped[0] in "[{\"'" or stripped in {"true", "false"}:
        try:
            return json.loads(stripped)
        except Exception:
            try:
                return ast.literal_eval(stripped)
            except Exception:
                return value
    return value


def _string_or_none(value: Any) -> str | None:
    parsed = _parse_structured_value(value)
    if parsed is None:
        return None
    return str(parsed)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    parsed = _parse_structured_value(value)
    if isinstance(parsed, bool):
        return parsed
    if isinstance(parsed, str):
        lowered = parsed.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    if parsed is None:
        return None
    return bool(parsed)


def _max_number(values: Any) -> float | None:
    parsed_values = [_float_or_none(value) for value in values]
    numbers = [value for value in parsed_values if value is not None]
    return max(numbers) if numbers else None


def _score_for_display(score: float | None) -> float:
    return score if score is not None else 0.0


def _smart_truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in (". ", "! ", "? ", "\n"):
        idx = cut.rfind(sep)
        if idx >= int(limit * 0.6):
            return cut[: idx + len(sep)].rstrip() + " ...[truncated]"
    return cut.rstrip() + " ...[truncated]"


def _clean(text: str) -> str:
    return text.encode("utf-8", "ignore").decode("utf-8")


def _ordered_unique(values: Any) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
