from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "retrieval-benchmark-v0"


class BenchmarkValidationError(ValueError):
    """Raised when benchmark case files are malformed."""


@dataclass
class EvidenceListSpec:
    required_all: list[str] = field(default_factory=list)
    required_any: list[str] = field(default_factory=list)

    @property
    def expected(self) -> list[str]:
        return _ordered_unique([*self.required_all, *self.required_any])

    @classmethod
    def from_raw(cls, value: Any, *, field_name: str) -> EvidenceListSpec:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise BenchmarkValidationError(f"{field_name} must be an object")
        _reject_unknown_keys(value, {"required_all", "required_any"}, field_name)
        return cls(
            required_all=_string_list(value.get("required_all")),
            required_any=_string_list(value.get("required_any")),
        )


@dataclass
class RelationSpec:
    subject_id: str | None = None
    subject_label: str | None = None
    predicate: str | None = None
    object_id: str | None = None
    object_label: str | None = None
    evidence_chunk_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_raw(cls, value: Any, *, field_name: str) -> RelationSpec:
        if not isinstance(value, dict):
            raise BenchmarkValidationError(f"{field_name} relation must be an object")
        _reject_unknown_keys(
            value,
            {
                "subject_id",
                "subject_label",
                "predicate",
                "object_id",
                "object_label",
                "evidence_chunk_ids",
            },
            field_name,
        )
        return cls(
            subject_id=_string_or_none(value.get("subject_id")),
            subject_label=_string_or_none(value.get("subject_label")),
            predicate=_string_or_none(value.get("predicate")),
            object_id=_string_or_none(value.get("object_id")),
            object_label=_string_or_none(value.get("object_label")),
            evidence_chunk_ids=_string_list(value.get("evidence_chunk_ids")),
        )

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        return _dump_dataclass(self, exclude_none=exclude_none)


@dataclass
class RelationListSpec:
    required_all: list[RelationSpec] = field(default_factory=list)
    required_any: list[RelationSpec] = field(default_factory=list)

    @property
    def expected(self) -> list[RelationSpec]:
        return [*self.required_all, *self.required_any]

    @classmethod
    def from_raw(cls, value: Any, *, field_name: str) -> RelationListSpec:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise BenchmarkValidationError(f"{field_name} must be an object")
        _reject_unknown_keys(value, {"required_all", "required_any"}, field_name)
        return cls(
            required_all=[
                RelationSpec.from_raw(item, field_name=f"{field_name}.required_all")
                for item in value.get("required_all", []) or []
            ],
            required_any=[
                RelationSpec.from_raw(item, field_name=f"{field_name}.required_any")
                for item in value.get("required_any", []) or []
            ],
        )


@dataclass
class ExpectedEvidence:
    source_chunk_ids: EvidenceListSpec = field(default_factory=EvidenceListSpec)
    source_paths: EvidenceListSpec = field(default_factory=EvidenceListSpec)
    concept_ids: EvidenceListSpec = field(default_factory=EvidenceListSpec)
    concept_labels: EvidenceListSpec = field(default_factory=EvidenceListSpec)
    relation_ids: EvidenceListSpec = field(default_factory=EvidenceListSpec)
    relations: RelationListSpec = field(default_factory=RelationListSpec)
    required_answer_points: list[str] = field(default_factory=list)

    @classmethod
    def from_raw(cls, value: Any) -> ExpectedEvidence:
        if not isinstance(value, dict):
            raise BenchmarkValidationError("expected_evidence must be an object")
        _reject_unknown_keys(
            value,
            {
                "source_chunk_ids",
                "source_paths",
                "concept_ids",
                "concept_labels",
                "relation_ids",
                "relations",
                "required_answer_points",
            },
            "expected_evidence",
        )
        return cls(
            source_chunk_ids=EvidenceListSpec.from_raw(
                value.get("source_chunk_ids"), field_name="source_chunk_ids"
            ),
            source_paths=EvidenceListSpec.from_raw(
                value.get("source_paths"), field_name="source_paths"
            ),
            concept_ids=EvidenceListSpec.from_raw(
                value.get("concept_ids"), field_name="concept_ids"
            ),
            concept_labels=EvidenceListSpec.from_raw(
                value.get("concept_labels"), field_name="concept_labels"
            ),
            relation_ids=EvidenceListSpec.from_raw(
                value.get("relation_ids"), field_name="relation_ids"
            ),
            relations=RelationListSpec.from_raw(
                value.get("relations"), field_name="relations"
            ),
            required_answer_points=_string_list(value.get("required_answer_points")),
        )


@dataclass
class ForbiddenEvidence:
    source_chunk_ids: list[str] = field(default_factory=list)
    concept_ids: list[str] = field(default_factory=list)
    concept_labels: list[str] = field(default_factory=list)
    relations: list[RelationSpec] = field(default_factory=list)

    @classmethod
    def from_raw(cls, value: Any) -> ForbiddenEvidence:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise BenchmarkValidationError("forbidden_evidence must be an object")
        _reject_unknown_keys(
            value,
            {"source_chunk_ids", "concept_ids", "concept_labels", "relations"},
            "forbidden_evidence",
        )
        return cls(
            source_chunk_ids=_string_list(value.get("source_chunk_ids")),
            concept_ids=_string_list(value.get("concept_ids")),
            concept_labels=_string_list(value.get("concept_labels")),
            relations=[
                RelationSpec.from_raw(item, field_name="forbidden_evidence.relations")
                for item in value.get("relations", []) or []
            ],
        )


@dataclass
class Thresholds:
    source_recall_at_6: float | None = None
    source_mrr_at_6: float | None = None
    concept_recall: float | None = None
    relation_recall: float | None = None
    node_recall: float | None = None
    edge_recall: float | None = None
    max_dangling_edges: int | None = None
    max_chunk_nodes: int | None = None
    forbidden_evidence_count: int | None = None

    @classmethod
    def from_raw(cls, value: Any) -> Thresholds:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise BenchmarkValidationError("thresholds must be an object")
        _reject_unknown_keys(
            value,
            {
                "source_recall_at_6",
                "source_mrr_at_6",
                "concept_recall",
                "relation_recall",
                "node_recall",
                "edge_recall",
                "max_dangling_edges",
                "max_chunk_nodes",
                "forbidden_evidence_count",
            },
            "thresholds",
        )
        return cls(
            source_recall_at_6=_float_or_none(value.get("source_recall_at_6")),
            source_mrr_at_6=_float_or_none(value.get("source_mrr_at_6")),
            concept_recall=_float_or_none(value.get("concept_recall")),
            relation_recall=_float_or_none(value.get("relation_recall")),
            node_recall=_float_or_none(value.get("node_recall")),
            edge_recall=_float_or_none(value.get("edge_recall")),
            max_dangling_edges=_int_or_none(value.get("max_dangling_edges")),
            max_chunk_nodes=_int_or_none(value.get("max_chunk_nodes")),
            forbidden_evidence_count=_int_or_none(
                value.get("forbidden_evidence_count")
            ),
        )


@dataclass
class BenchmarkCase:
    schema_version: str
    id: str
    review_status: str
    mode: str
    queries: list[str]
    expected_evidence: ExpectedEvidence
    seeded_from: list[str] = field(default_factory=list)
    why_selected: str | None = None
    forbidden_evidence: ForbiddenEvidence = field(default_factory=ForbiddenEvidence)
    thresholds: Thresholds = field(default_factory=Thresholds)
    review_notes: str | None = None

    @classmethod
    def from_raw(cls, value: Any) -> BenchmarkCase:
        if not isinstance(value, dict):
            raise BenchmarkValidationError("benchmark row must be an object")
        _reject_unknown_keys(
            value,
            {
                "schema_version",
                "id",
                "review_status",
                "mode",
                "queries",
                "seeded_from",
                "why_selected",
                "expected_evidence",
                "forbidden_evidence",
                "thresholds",
                "review_notes",
            },
            "benchmark case",
        )
        schema_version = _required_string(value, "schema_version")
        if schema_version != SCHEMA_VERSION:
            raise BenchmarkValidationError(
                f"unsupported schema_version {schema_version!r}"
            )
        mode = _required_string(value, "mode")
        if mode not in {"analyst", "visualizer"}:
            raise BenchmarkValidationError(f"unsupported mode {mode!r}")
        queries = _string_list(value.get("queries"))
        if not queries:
            raise BenchmarkValidationError("queries must contain at least one query")
        return cls(
            schema_version=schema_version,
            id=_required_string(value, "id"),
            review_status=_required_string(value, "review_status"),
            mode=mode,
            queries=queries,
            seeded_from=_string_list(value.get("seeded_from")),
            why_selected=_string_or_none(value.get("why_selected")),
            expected_evidence=ExpectedEvidence.from_raw(value.get("expected_evidence")),
            forbidden_evidence=ForbiddenEvidence.from_raw(
                value.get("forbidden_evidence")
            ),
            thresholds=Thresholds.from_raw(value.get("thresholds")),
            review_notes=_string_or_none(value.get("review_notes")),
        )


@dataclass
class RelationEvidence:
    predicate: str
    relation_id: str | None = None
    subject_id: str | None = None
    subject_label: str | None = None
    object_id: str | None = None
    object_label: str | None = None
    evidence_chunk_ids: list[str] = field(default_factory=list)

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        return _dump_dataclass(self, exclude_none=exclude_none)


@dataclass
class ActualEvidence:
    source_chunk_ids_ranked: list[str] = field(default_factory=list)
    source_paths: list[str] = field(default_factory=list)
    concept_ids: list[str] = field(default_factory=list)
    concept_labels: list[str] = field(default_factory=list)
    relation_ids: list[str] = field(default_factory=list)
    relations: list[RelationEvidence] = field(default_factory=list)
    graph_metrics: dict[str, Any] = field(default_factory=dict)

    def compact(self) -> dict[str, Any]:
        return _dump_dataclass(self, exclude_none=True)


@dataclass
class RelationCatalogEntry:
    relation_id: str
    subject_id: str
    predicate: str
    object_id: str
    evidence_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class EvidenceCatalog:
    concept_labels_by_id: dict[str, set[str]] = field(default_factory=dict)
    relation_by_id: dict[str, RelationCatalogEntry] = field(default_factory=dict)

    def labels_for_concept(self, concept_id: str | None) -> set[str]:
        if not concept_id:
            return set()
        return self.concept_labels_by_id.get(concept_id, set())

    def relation_for_id(self, relation_id: str) -> RelationCatalogEntry | None:
        return self.relation_by_id.get(relation_id)

    @classmethod
    def from_storage_dir(
        cls, storage_dir: Path, *, root_dir: Path | None = None
    ) -> EvidenceCatalog:
        root = root_dir or Path.cwd()
        manifest_path = storage_dir / "postprocessed_graph_storage_manifest.json"
        if not manifest_path.exists():
            return cls()
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return cls()
        postprocess_dir_raw = manifest.get("postprocess_dir")
        if not postprocess_dir_raw:
            return cls()
        postprocess_dir = Path(str(postprocess_dir_raw))
        if not postprocess_dir.is_absolute():
            postprocess_dir = root / postprocess_dir
        catalog = cls()
        catalog._load_concepts(postprocess_dir / "llm_concept_registry.jsonl")
        catalog._load_projection(postprocess_dir / "llm_graph_projection_preview.jsonl")
        return catalog

    def _load_concepts(self, path: Path) -> None:
        if not path.exists():
            return
        for row in _read_jsonl(path):
            concept_id = _string_or_none(row.get("concept_id") or row.get("id"))
            if not concept_id:
                continue
            labels = _labels_from_row(row)
            if labels:
                self.concept_labels_by_id.setdefault(concept_id, set()).update(labels)

    def _load_projection(self, path: Path) -> None:
        if not path.exists():
            return
        for row in _read_jsonl(path):
            record_type = row.get("record_type")
            if record_type == "concept_node":
                concept_id = _string_or_none(row.get("id"))
                if concept_id:
                    self.concept_labels_by_id.setdefault(concept_id, set()).update(
                        _labels_from_row(row)
                    )
            elif record_type == "relation_edge":
                relation_id = _string_or_none(row.get("id"))
                subject_id = _string_or_none(row.get("source_concept_id"))
                object_id = _string_or_none(row.get("target_concept_id"))
                predicate = _string_or_none(row.get("canonical_predicate"))
                if relation_id and subject_id and object_id and predicate:
                    self.relation_by_id[relation_id] = RelationCatalogEntry(
                        relation_id=relation_id,
                        subject_id=subject_id,
                        predicate=predicate,
                        object_id=object_id,
                        evidence_chunk_ids=_string_list(row.get("evidence_chunk_ids")),
                    )


@dataclass
class CaseResult:
    case_id: str
    mode: str
    passed: bool
    scores: dict[str, float | int | None]
    failures: list[str] = field(default_factory=list)
    missing_required: dict[str, list[Any]] = field(default_factory=dict)
    forbidden_hits: dict[str, list[Any]] = field(default_factory=dict)
    actual_evidence: dict[str, Any] = field(default_factory=dict)

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        return _dump_dataclass(self, exclude_none=exclude_none)


@dataclass
class ReferenceValidationReport:
    case_count: int
    expected_source_chunk_ids: list[str] = field(default_factory=list)
    expected_concept_ids: list[str] = field(default_factory=list)
    expected_relation_ids: list[str] = field(default_factory=list)
    missing_source_chunk_ids: list[str] = field(default_factory=list)
    source_chunk_ids_not_embedded: list[str] = field(default_factory=list)
    disabled_source_chunk_ids: list[str] = field(default_factory=list)
    quarantined_source_chunk_ids: list[str] = field(default_factory=list)
    missing_concept_ids: list[str] = field(default_factory=list)
    missing_relation_ids: list[str] = field(default_factory=list)
    relation_ids_missing_graph_triplet: list[dict[str, str]] = field(
        default_factory=list
    )

    @property
    def passed(self) -> bool:
        return not any(
            [
                self.missing_source_chunk_ids,
                self.source_chunk_ids_not_embedded,
                self.disabled_source_chunk_ids,
                self.quarantined_source_chunk_ids,
                self.missing_concept_ids,
                self.missing_relation_ids,
                self.relation_ids_missing_graph_triplet,
            ]
        )

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        payload = _dump_dataclass(self, exclude_none=exclude_none)
        payload["passed"] = self.passed
        payload["counts"] = {
            "expected_source_chunk_ids": len(self.expected_source_chunk_ids),
            "expected_concept_ids": len(self.expected_concept_ids),
            "expected_relation_ids": len(self.expected_relation_ids),
        }
        return payload


def load_benchmark_cases(
    path: Path, *, mode: str = "both", case_ids: set[str] | None = None
) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    errors: list[str] = []
    if mode not in {"both", "analyst", "visualizer"}:
        raise BenchmarkValidationError(f"unsupported mode {mode!r}")
    if not path.exists():
        raise BenchmarkValidationError(f"Benchmark file does not exist: {path}")

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            case = BenchmarkCase.from_raw(payload)
        except (json.JSONDecodeError, BenchmarkValidationError) as exc:
            errors.append(f"line {line_number}: {exc}")
            continue
        if case.id in seen_ids:
            errors.append(
                f"line {line_number}: duplicate benchmark case id {case.id!r}"
            )
            continue
        seen_ids.add(case.id)
        if mode != "both" and case.mode != mode:
            continue
        if case_ids is not None and case.id not in case_ids:
            continue
        cases.append(case)

    if errors:
        raise BenchmarkValidationError("Invalid benchmark file:\n" + "\n".join(errors))
    if case_ids is not None:
        missing = sorted(case_ids - seen_ids)
        if missing:
            raise BenchmarkValidationError(
                f"Unknown benchmark case id(s): {', '.join(missing)}"
            )
    return cases


def validate_benchmark_references(
    cases: list[BenchmarkCase],
    catalog: EvidenceCatalog,
    *,
    source_metadata_by_id: dict[str, dict[str, Any]] | None = None,
    source_status_by_id: dict[str, dict[str, Any]] | None = None,
    embedded_source_ids: set[str] | None = None,
    graph_triplets: set[tuple[str, str, str]] | None = None,
) -> ReferenceValidationReport:
    """Checks benchmark evidence IDs against final storage/catalog references."""

    references = _collect_benchmark_references(cases)
    source_metadata_by_id = source_metadata_by_id or {}
    source_status_by_id = source_status_by_id or {}

    expected_sources = references["source_chunk_ids"]
    expected_concepts = references["concept_ids"]
    expected_relations = references["relation_ids"]
    graph_triplet_set = {
        (str(subject_id), _normalize_predicate(predicate), str(object_id))
        for subject_id, predicate, object_id in graph_triplets or set()
    }

    missing_sources = []
    not_embedded = []
    disabled_sources = []
    quarantined_sources = []
    for source_id in expected_sources:
        source_metadata = source_metadata_by_id.get(source_id)
        status_metadata = source_status_by_id.get(source_id, {})
        merged_metadata = {**(source_metadata or {}), **status_metadata}
        if source_metadata_by_id and source_metadata is None:
            missing_sources.append(source_id)
        if embedded_source_ids is not None and source_id not in embedded_source_ids:
            not_embedded.append(source_id)
        if _source_is_disabled(merged_metadata):
            disabled_sources.append(source_id)
        if _source_is_quarantined(merged_metadata):
            quarantined_sources.append(source_id)

    missing_concepts = [
        concept_id
        for concept_id in expected_concepts
        if concept_id not in catalog.concept_labels_by_id
    ]
    missing_relations = [
        relation_id
        for relation_id in expected_relations
        if relation_id not in catalog.relation_by_id
    ]
    missing_triplets = []
    if graph_triplets is not None:
        for relation_id in expected_relations:
            relation = catalog.relation_for_id(relation_id)
            if relation is None:
                continue
            triplet = (
                relation.subject_id,
                _normalize_predicate(relation.predicate),
                relation.object_id,
            )
            if triplet not in graph_triplet_set:
                missing_triplets.append(
                    {
                        "relation_id": relation_id,
                        "subject_id": relation.subject_id,
                        "predicate": relation.predicate,
                        "object_id": relation.object_id,
                    }
                )

    return ReferenceValidationReport(
        case_count=len(cases),
        expected_source_chunk_ids=expected_sources,
        expected_concept_ids=expected_concepts,
        expected_relation_ids=expected_relations,
        missing_source_chunk_ids=missing_sources,
        source_chunk_ids_not_embedded=not_embedded,
        disabled_source_chunk_ids=disabled_sources,
        quarantined_source_chunk_ids=quarantined_sources,
        missing_concept_ids=missing_concepts,
        missing_relation_ids=missing_relations,
        relation_ids_missing_graph_triplet=missing_triplets,
    )


def score_case(
    case: BenchmarkCase, actual: ActualEvidence, catalog: EvidenceCatalog | None = None
) -> CaseResult:
    catalog = catalog or EvidenceCatalog()
    scores: dict[str, float | int | None] = {}
    failures: list[str] = []
    missing_required: dict[str, list[Any]] = {}

    source_eval = _eval_string_spec(
        set(actual.source_chunk_ids_ranked),
        case.expected_evidence.source_chunk_ids,
        normalize=False,
    )
    _add_missing(missing_required, "source_chunk_ids", source_eval["missing_all"])
    scores["source_recall"] = source_eval["recall"]
    scores["source_recall_at_6"] = _recall_at_k(
        actual.source_chunk_ids_ranked,
        case.expected_evidence.source_chunk_ids.expected,
        6,
    )
    scores["source_mrr_at_6"] = _mrr_at_k(
        actual.source_chunk_ids_ranked,
        case.expected_evidence.source_chunk_ids.expected,
        6,
    )

    path_eval = _eval_string_spec(
        set(actual.source_paths), case.expected_evidence.source_paths, normalize=True
    )
    _add_missing(missing_required, "source_paths", path_eval["missing_all"])
    scores["source_path_recall"] = path_eval["recall"]

    concept_id_eval = _eval_string_spec(
        set(actual.concept_ids), case.expected_evidence.concept_ids, normalize=False
    )
    concept_label_eval = _eval_string_spec(
        set(actual.concept_labels),
        case.expected_evidence.concept_labels,
        normalize=True,
    )
    _add_missing(missing_required, "concept_ids", concept_id_eval["missing_all"])
    _add_missing(missing_required, "concept_labels", concept_label_eval["missing_all"])
    scores["concept_id_recall"] = concept_id_eval["recall"]
    scores["concept_label_recall"] = concept_label_eval["recall"]
    scores["concept_recall"] = _best_available_score(
        concept_id_eval["recall"], concept_label_eval["recall"]
    )
    scores["node_recall"] = scores["concept_recall"]

    relation_id_eval = _eval_relation_ids(
        case.expected_evidence.relation_ids, actual, catalog
    )
    relation_spec_eval = _eval_relation_specs(
        case.expected_evidence.relations, actual, catalog
    )
    _add_missing(missing_required, "relation_ids", relation_id_eval["missing_all"])
    _add_missing(missing_required, "relations", relation_spec_eval["missing_all"])
    scores["relation_id_recall"] = relation_id_eval["recall"]
    scores["relation_spec_recall"] = relation_spec_eval["recall"]
    scores["relation_recall"] = _best_available_score(
        relation_id_eval["recall"], relation_spec_eval["recall"]
    )
    scores["edge_recall"] = scores["relation_recall"]

    forbidden_hits = _forbidden_hits(case.forbidden_evidence, actual, catalog)
    scores["forbidden_evidence_count"] = sum(
        len(values) for values in forbidden_hits.values()
    )
    scores["dangling_edge_count"] = _int_metric(
        actual.graph_metrics, "dangling_edge_count"
    )
    scores["chunk_node_count"] = _chunk_node_count(actual)

    _apply_required_all_failures(missing_required, failures)
    _apply_thresholds(case.thresholds, scores, failures)

    return CaseResult(
        case_id=case.id,
        mode=case.mode,
        passed=not failures,
        scores=scores,
        failures=failures,
        missing_required=missing_required,
        forbidden_hits=forbidden_hits,
        actual_evidence=actual.compact(),
    )


def summarize_results(results: list[CaseResult]) -> dict[str, Any]:
    score_totals: dict[str, list[float]] = {}
    for result in results:
        for metric, value in result.scores.items():
            if isinstance(value, (int, float)):
                score_totals.setdefault(metric, []).append(float(value))
    return {
        "case_count": len(results),
        "passed_count": sum(1 for result in results if result.passed),
        "failed_count": sum(1 for result in results if not result.passed),
        "failed_case_ids": [result.case_id for result in results if not result.passed],
        "mean_scores": {
            metric: sum(values) / len(values)
            for metric, values in sorted(score_totals.items())
            if values
        },
    }


def actual_evidence_from_analyst_result(
    result: dict[str, Any], catalog: EvidenceCatalog | None = None
) -> ActualEvidence:
    catalog = catalog or EvidenceCatalog()
    source_ids: list[str] = []
    source_paths: list[str] = []
    concept_ids: list[str] = []
    concept_labels: list[str] = []
    relations: list[RelationEvidence] = []
    relation_ids: list[str] = []

    for source in result.get("sources", []) or []:
        source_id = _string_or_none(
            getattr(getattr(source, "metadata", None), "chunk_id", None)
        ) or _string_or_none(getattr(source, "node_id", None))
        if source_id:
            source_ids.append(source_id)
        metadata = getattr(source, "metadata", None)
        source_paths.extend(_metadata_paths(metadata))
        for concept in getattr(source, "mentioned_concepts", []) or []:
            concept_id = _node_id(concept)
            label = _node_label(concept)
            if concept_id:
                concept_ids.append(concept_id)
                concept_labels.extend(catalog.labels_for_concept(concept_id))
            if label:
                concept_labels.append(label)

    for relation in result.get("relations", []) or []:
        relation_id = _string_or_none(getattr(relation, "relation_id", None))
        if relation_id:
            relation_ids.append(relation_id)
        actual_relation = RelationEvidence(
            relation_id=relation_id,
            subject_label=_string_or_none(getattr(relation, "subject", None)),
            predicate=str(getattr(relation, "predicate", "")),
            object_label=_string_or_none(getattr(relation, "object", None)),
            evidence_chunk_ids=_string_list(
                getattr(relation, "evidence_chunk_ids", [])
            ),
        )
        relations.append(actual_relation)
        concept_labels.extend(
            [
                label
                for label in [
                    actual_relation.subject_label,
                    actual_relation.object_label,
                ]
                if label
            ]
        )

    return ActualEvidence(
        source_chunk_ids_ranked=_ordered_unique(source_ids),
        source_paths=_ordered_unique(source_paths),
        concept_ids=_ordered_unique(concept_ids),
        concept_labels=_ordered_unique(concept_labels),
        relation_ids=_ordered_unique(relation_ids),
        relations=relations,
    )


def actual_evidence_from_visualizer_result(
    *,
    nodes: list[str],
    triplets: list[tuple[str, str, str]],
    node_details: list[dict[str, Any]],
    metrics: dict[str, Any],
    catalog: EvidenceCatalog | None = None,
) -> ActualEvidence:
    catalog = catalog or EvidenceCatalog()
    labels_by_node = {
        str(detail.get("id")): str(detail.get("label"))
        for detail in node_details
        if detail.get("id") and detail.get("label")
    }
    concept_labels: list[str] = []
    source_chunk_ids: list[str] = []
    relation_ids: list[str] = []
    relations: list[RelationEvidence] = []

    for node_id in nodes:
        concept_labels.extend(catalog.labels_for_concept(node_id))
        if node_id in labels_by_node:
            concept_labels.append(labels_by_node[node_id])

    for subject_id, predicate, object_id in triplets:
        relation_entry = _catalog_relation_for_triplet(
            catalog, subject_id, predicate, object_id
        )
        if relation_entry:
            relation_ids.append(relation_entry.relation_id)
            source_chunk_ids.extend(relation_entry.evidence_chunk_ids)
        relations.append(
            RelationEvidence(
                relation_id=relation_entry.relation_id if relation_entry else None,
                subject_id=subject_id,
                subject_label=labels_by_node.get(subject_id),
                predicate=predicate,
                object_id=object_id,
                object_label=labels_by_node.get(object_id),
                evidence_chunk_ids=relation_entry.evidence_chunk_ids
                if relation_entry
                else [],
            )
        )

    return ActualEvidence(
        source_chunk_ids_ranked=_ordered_unique(source_chunk_ids),
        concept_ids=_ordered_unique(nodes),
        concept_labels=_ordered_unique(concept_labels),
        relation_ids=_ordered_unique(relation_ids),
        relations=relations,
        graph_metrics=metrics,
    )


def render_markdown_summary(summary: dict[str, Any], results: list[CaseResult]) -> str:
    lines = [
        "# Retrieval Benchmark Summary",
        "",
        f"- Cases: {summary['case_count']}",
        f"- Passed: {summary['passed_count']}",
        f"- Failed: {summary['failed_count']}",
    ]
    failed = [result for result in results if not result.passed]
    if failed:
        lines.extend(["", "## Failed Cases"])
        for result in failed:
            lines.append(f"- `{result.case_id}`: " + "; ".join(result.failures))
    lines.extend(["", "## Mean Scores"])
    for metric, value in summary.get("mean_scores", {}).items():
        lines.append(f"- `{metric}`: {value:.3f}")
    lines.append("")
    return "\n".join(lines)


def _eval_string_spec(
    actual_values: set[str], spec: EvidenceListSpec, *, normalize: bool
) -> dict[str, Any]:
    if normalize:
        actual = {
            _normalize_text(value) for value in actual_values if _normalize_text(value)
        }
        required_all = [
            _normalize_text(value)
            for value in spec.required_all
            if _normalize_text(value)
        ]
        required_any = [
            _normalize_text(value)
            for value in spec.required_any
            if _normalize_text(value)
        ]
    else:
        actual = set(actual_values)
        required_all = list(spec.required_all)
        required_any = list(spec.required_any)
    expected = _ordered_unique([*required_all, *required_any])
    hits = [value for value in expected if value in actual]
    return {
        "hits": hits,
        "missing_all": [value for value in required_all if value not in actual],
        "recall": (len(hits) / len(expected)) if expected else None,
    }


def _eval_relation_ids(
    spec: EvidenceListSpec, actual: ActualEvidence, catalog: EvidenceCatalog
) -> dict[str, Any]:
    expected = spec.expected
    hits: list[str] = []
    for relation_id in expected:
        if relation_id in actual.relation_ids:
            hits.append(relation_id)
            continue
        relation_entry = catalog.relation_for_id(relation_id)
        if relation_entry and any(
            _relation_matches_entry(item, relation_entry, catalog)
            for item in actual.relations
        ):
            hits.append(relation_id)
    return {
        "hits": hits,
        "missing_all": [
            relation_id for relation_id in spec.required_all if relation_id not in hits
        ],
        "recall": (len(hits) / len(expected)) if expected else None,
    }


def _eval_relation_specs(
    spec: RelationListSpec, actual: ActualEvidence, catalog: EvidenceCatalog
) -> dict[str, Any]:
    expected = spec.expected
    hits = [
        item
        for item in expected
        if any(
            _relation_matches_spec(relation, item, catalog)
            for relation in actual.relations
        )
    ]
    return {
        "hits": hits,
        "missing_all": [
            item.model_dump(exclude_none=True)
            for item in spec.required_all
            if not any(
                _relation_matches_spec(relation, item, catalog)
                for relation in actual.relations
            )
        ],
        "recall": (len(hits) / len(expected)) if expected else None,
    }


def _forbidden_hits(
    forbidden: ForbiddenEvidence, actual: ActualEvidence, catalog: EvidenceCatalog
) -> dict[str, list[Any]]:
    hits: dict[str, list[Any]] = {}
    source_hits = sorted(
        set(forbidden.source_chunk_ids).intersection(actual.source_chunk_ids_ranked)
    )
    if source_hits:
        hits["source_chunk_ids"] = source_hits
    concept_hits = sorted(set(forbidden.concept_ids).intersection(actual.concept_ids))
    if concept_hits:
        hits["concept_ids"] = concept_hits
    actual_labels = {_normalize_text(label) for label in actual.concept_labels}
    label_hits = [
        label
        for label in forbidden.concept_labels
        if _normalize_text(label) in actual_labels
    ]
    if label_hits:
        hits["concept_labels"] = label_hits
    relation_hits = [
        spec.model_dump(exclude_none=True)
        for spec in forbidden.relations
        if any(
            _relation_matches_spec(relation, spec, catalog)
            for relation in actual.relations
        )
    ]
    if relation_hits:
        hits["relations"] = relation_hits
    return hits


def _relation_matches_entry(
    relation: RelationEvidence, entry: RelationCatalogEntry, catalog: EvidenceCatalog
) -> bool:
    return _relation_matches_spec(
        relation,
        RelationSpec(
            subject_id=entry.subject_id,
            predicate=entry.predicate,
            object_id=entry.object_id,
            evidence_chunk_ids=entry.evidence_chunk_ids,
        ),
        catalog,
    )


def _relation_matches_spec(
    relation: RelationEvidence, spec: RelationSpec, catalog: EvidenceCatalog
) -> bool:
    if spec.predicate and _normalize_predicate(
        relation.predicate
    ) != _normalize_predicate(spec.predicate):
        return False
    if spec.subject_id and not _endpoint_matches(
        actual_id=relation.subject_id,
        actual_label=relation.subject_label,
        expected_id=spec.subject_id,
        expected_label=None,
        catalog=catalog,
    ):
        return False
    if spec.object_id and not _endpoint_matches(
        actual_id=relation.object_id,
        actual_label=relation.object_label,
        expected_id=spec.object_id,
        expected_label=None,
        catalog=catalog,
    ):
        return False
    if spec.subject_label and _normalize_text(spec.subject_label) != _normalize_text(
        relation.subject_label or ""
    ):
        return False
    if spec.object_label and _normalize_text(spec.object_label) != _normalize_text(
        relation.object_label or ""
    ):
        return False
    if spec.evidence_chunk_ids:
        actual_chunks = set(relation.evidence_chunk_ids)
        if not actual_chunks.intersection(spec.evidence_chunk_ids):
            return False
    return True


def _endpoint_matches(
    *,
    actual_id: str | None,
    actual_label: str | None,
    expected_id: str,
    expected_label: str | None,
    catalog: EvidenceCatalog,
) -> bool:
    if actual_id == expected_id:
        return True
    if expected_label and _normalize_text(actual_label or "") == _normalize_text(
        expected_label
    ):
        return True
    expected_labels = {
        _normalize_text(label) for label in catalog.labels_for_concept(expected_id)
    }
    return bool(actual_label and _normalize_text(actual_label) in expected_labels)


def _apply_thresholds(
    thresholds: Thresholds, scores: dict[str, float | int | None], failures: list[str]
) -> None:
    minimum_metrics = {
        "source_recall_at_6": thresholds.source_recall_at_6,
        "source_mrr_at_6": thresholds.source_mrr_at_6,
        "concept_recall": thresholds.concept_recall,
        "relation_recall": thresholds.relation_recall,
        "node_recall": thresholds.node_recall,
        "edge_recall": thresholds.edge_recall,
    }
    for metric, threshold in minimum_metrics.items():
        if threshold is None:
            continue
        value = scores.get(metric)
        if value is None or float(value) < threshold:
            failures.append(f"{metric}={value} below threshold {threshold}")

    maximum_metrics = {
        "forbidden_evidence_count": thresholds.forbidden_evidence_count,
        "dangling_edge_count": thresholds.max_dangling_edges,
        "chunk_node_count": thresholds.max_chunk_nodes,
    }
    for metric, threshold in maximum_metrics.items():
        if threshold is None:
            continue
        value = scores.get(metric)
        if value is None or int(value) > threshold:
            failures.append(f"{metric}={value} above threshold {threshold}")


def _apply_required_all_failures(
    missing_required: dict[str, list[Any]], failures: list[str]
) -> None:
    for evidence_name, missing in missing_required.items():
        if missing:
            failures.append(f"missing required {evidence_name}: {missing}")


def _recall_at_k(
    ranked_values: list[str], expected_values: list[str], k: int
) -> float | None:
    if not expected_values:
        return None
    return len(set(ranked_values[:k]).intersection(expected_values)) / len(
        set(expected_values)
    )


def _mrr_at_k(
    ranked_values: list[str], expected_values: list[str], k: int
) -> float | None:
    expected = set(expected_values)
    if not expected:
        return None
    for rank, value in enumerate(ranked_values[:k], start=1):
        if value in expected:
            return 1.0 / rank
    return 0.0


def _best_available_score(*values: float | int | None) -> float | None:
    present = [float(value) for value in values if value is not None]
    return max(present) if present else None


def _catalog_relation_for_triplet(
    catalog: EvidenceCatalog, subject_id: str, predicate: str, object_id: str
) -> RelationCatalogEntry | None:
    normalized_predicate = _normalize_predicate(predicate)
    for relation in catalog.relation_by_id.values():
        if (
            relation.subject_id == subject_id
            and relation.object_id == object_id
            and _normalize_predicate(relation.predicate) == normalized_predicate
        ):
            return relation
    return None


def _collect_benchmark_references(cases: list[BenchmarkCase]) -> dict[str, list[str]]:
    source_chunk_ids: list[str] = []
    concept_ids: list[str] = []
    relation_ids: list[str] = []

    for case in cases:
        expected = case.expected_evidence
        forbidden = case.forbidden_evidence
        source_chunk_ids.extend(expected.source_chunk_ids.expected)
        source_chunk_ids.extend(forbidden.source_chunk_ids)
        concept_ids.extend(expected.concept_ids.expected)
        concept_ids.extend(forbidden.concept_ids)
        relation_ids.extend(expected.relation_ids.expected)

        for relation in [*expected.relations.expected, *forbidden.relations]:
            source_chunk_ids.extend(relation.evidence_chunk_ids)
            if relation.subject_id:
                concept_ids.append(relation.subject_id)
            if relation.object_id:
                concept_ids.append(relation.object_id)

    return {
        "source_chunk_ids": _ordered_unique(source_chunk_ids),
        "concept_ids": _ordered_unique(concept_ids),
        "relation_ids": _ordered_unique(relation_ids),
    }


def _source_is_disabled(metadata: dict[str, Any]) -> bool:
    if _bool_flag(metadata.get("retrieval_disabled")):
        return True
    if metadata.get("retrieval_enabled") is not None and not _bool_flag(
        metadata.get("retrieval_enabled")
    ):
        return True
    action = _string_or_none(
        metadata.get("postprocess_action") or metadata.get("action")
    )
    return bool(action and action not in {"keep", "keep_with_cleaned_text"})


def _source_is_quarantined(metadata: dict[str, Any]) -> bool:
    if _bool_flag(metadata.get("quarantined")) or _bool_flag(
        metadata.get("is_quarantined")
    ):
        return True
    action = _string_or_none(
        metadata.get("postprocess_action") or metadata.get("action")
    )
    if action in {"quarantine", "quarantined", "discard", "drop"}:
        return True
    issue_types = {
        _normalize_text(issue_type)
        for issue_type in _string_list(metadata.get("issue_types"))
    }
    issue_types.update(
        _normalize_text(issue_type)
        for issue_type in _string_list(metadata.get("postprocess_issue_types"))
    )
    return any("quarantine" in issue_type for issue_type in issue_types)


def _bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _normalize_text(value) in {"1", "true", "yes", "y", "enabled"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _chunk_node_count(actual: ActualEvidence) -> int:
    return sum(1 for node_id in actual.concept_ids if node_id.startswith("chunk_"))


def _int_metric(metrics: dict[str, Any], name: str) -> int:
    value = metrics.get(name, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _metadata_paths(metadata: Any) -> list[str]:
    paths: list[list[str]] = []
    for attr_name in ("heading_path", "path"):
        value = getattr(metadata, attr_name, None)
        if value:
            paths.append(_string_list(value))
    rendered = []
    for path in paths:
        cleaned = [
            part.strip().lstrip("#").strip()
            for part in path
            if part and not str(part).startswith("external:")
        ]
        if cleaned:
            rendered.append(" > ".join(cleaned))
    return _ordered_unique(rendered)


def _node_id(node: Any) -> str | None:
    return _string_or_none(
        getattr(node, "id", None)
        or getattr(node, "node_id", None)
        or getattr(node, "id_", None)
    )


def _node_label(node: Any) -> str | None:
    properties = getattr(node, "properties", {}) or {}
    return _string_or_none(
        properties.get("display_name")
        or properties.get("entity_name")
        or properties.get("canonical_name")
        or getattr(node, "label", None)
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _labels_from_row(row: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for key in ("display_name", "canonical_name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            labels.add(value.strip())
    for key in ("aliases", "source_names"):
        for value in row.get(key) or []:
            if isinstance(value, str) and value.strip():
                labels.add(value.strip())
    return labels


def _add_missing(
    missing_required: dict[str, list[Any]], key: str, missing: list[Any]
) -> None:
    if missing:
        missing_required[key] = missing


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return []


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_string(value: dict[str, Any], key: str) -> str:
    text = _string_or_none(value.get(key))
    if not text:
        raise BenchmarkValidationError(f"{key} is required")
    return text


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkValidationError(
            f"expected numeric threshold, got {value!r}"
        ) from exc


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkValidationError(
            f"expected integer threshold, got {value!r}"
        ) from exc


def _ordered_unique(values: list[str] | Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _normalize_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip().lower())
    return value.lstrip("#").strip()


def _normalize_predicate(value: str) -> str:
    return _normalize_text(value).replace(" ", "_")


def _reject_unknown_keys(
    value: dict[str, Any], allowed: set[str], field_name: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise BenchmarkValidationError(
            f"{field_name} has unknown key(s): {', '.join(unknown)}"
        )


def _dump_dataclass(value: Any, *, exclude_none: bool) -> dict[str, Any]:
    dumped = asdict(value)
    if not exclude_none:
        return dumped
    return _drop_none(dumped)


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_none(item) for key, item in value.items() if item is not None
        }
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value
