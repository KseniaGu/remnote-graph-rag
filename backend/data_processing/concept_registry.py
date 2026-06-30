"""Concept mention resolution for RemNote LLM post-processing sidecars.

The chunk-level LLM extractor emits local concept mentions. This module turns
those mentions into stable global concept registry entries with conservative
deterministic merging and optional LLM adjudication for uncertain clusters.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from enum import StrEnum
from typing import Any, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "1.0"
DEFAULT_CONCEPT_RESOLUTION_PROMPT_VERSION = "v1"
AUTO_MERGE_THRESHOLD = 0.90
AUTO_SPLIT_THRESHOLD = 0.55
AUTO_GROUP_MAX_DISTINCT_CANONICAL_NAMES = 8
AUTO_GROUP_MAX_SOURCE_NAMES = 25

NAME_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9+#]+")
PUNCT_RE = re.compile(r"[^a-zа-яё0-9+#]+")
FORMULA_SYMBOL_RE = re.compile(r"[$\\{}^=<>|~]|[∂∇∑∫√∞≈≠≤≥±×÷→←↔∆ΔΑ-ω]")
SYMBOL_WITH_PARENS_RE = re.compile(r"^[A-Za-zА-Яа-яЁё]\s*\([^)]{1,12}\)$")
SHORT_CODE_LIKE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+$")

HIGH_RISK_FLAGS = {
    "acronym_ambiguity",
    "auto_group_safety_guard",
    "formula_or_symbol_alias",
    "modifier_scope_difference",
    "ocr_like_short_variant",
    "parent_child_candidate",
    "short_symbolic_alias",
    "type_broadening",
    "type_conflict",
}
GENERIC_TYPES = {"CONCEPT"}
TOKEN_STOPWORDS = {
    "about",
    "after",
    "algorithm",
    "and",
    "data",
    "from",
    "into",
    "method",
    "model",
    "module",
    "note",
    "part",
    "system",
    "task",
    "that",
    "the",
    "this",
    "with",
}
GENERIC_NAMES = {
    "agent",
    "algorithm",
    "component",
    "data",
    "function",
    "method",
    "model",
    "module",
    "process",
    "system",
    "task",
    "tool",
    "view",
}
INVARIANT_SINGULAR_TOKENS = {
    "analysis",
    "bayes",
    "bias",
    "series",
    "softplus",
    "species",
}
INVARIANT_DISPLAY_NAMES = {
    "bag of words",
}


class ConceptPairAction(StrEnum):
    AUTO_MERGE = "auto_merge"
    NO_MERGE = "no_merge"
    LLM_ADJUDICATE = "llm_adjudicate"


class ConceptMergeStatus(StrEnum):
    SINGLE = "single"
    AUTO_MERGED = "auto_merged"
    LLM_MERGED = "llm_merged"
    LLM_SPLIT = "llm_split"


class ConceptAdjudicationAction(StrEnum):
    MERGE_ALL = "merge_all"
    SPLIT_ALL = "split_all"
    MERGE_GROUPS = "merge_groups"


class ConceptMention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mention_id: str
    decision_id: str
    chunk_id: str
    local_id: str
    canonical_name: str
    display_name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    salience: float = Field(ge=0.0, le=1.0)
    description: Optional[str] = None
    evidence_spans: list[str] = Field(default_factory=list)

    @field_validator("mention_id", "decision_id", "chunk_id", "local_id", "canonical_name", "display_name", "type")
    @classmethod
    def clean_required_text(cls, value: str) -> str:
        value = normalize_whitespace(value)
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("aliases", "evidence_spans")
    @classmethod
    def clean_string_list(cls, values: list[str]) -> list[str]:
        return unique_preserving_order(normalize_whitespace(value) for value in values if normalize_whitespace(value))

    @model_validator(mode="after")
    def normalize_type_and_description(self) -> "ConceptMention":
        self.type = normalize_whitespace(self.type).upper()
        if self.description is not None:
            self.description = normalize_whitespace(self.description) or None
        return self

    def source_names(self) -> list[str]:
        return unique_preserving_order(
            [
                self.canonical_name,
                self.display_name,
                *self.aliases,
            ]
        )


class ConceptPairScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_mention_id: str
    target_mention_id: str
    score: float = Field(ge=0.0, le=1.0)
    signals: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    recommended_action: ConceptPairAction


class UncertainConceptCluster(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    mention_ids: list[str]
    concept_names: list[str]
    types: list[str]
    source_chunk_ids: list[str]
    score: float = Field(ge=0.0, le=1.0)
    risk_flags: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)
    pair_scores: list[ConceptPairScore] = Field(default_factory=list)


class ConceptRegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concept_id: str
    canonical_name: str
    display_name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    descriptions: list[str] = Field(default_factory=list)
    source_names: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    evidence_spans: list[str] = Field(default_factory=list)
    mention_ids: list[str] = Field(default_factory=list)
    source_decision_ids: list[str] = Field(default_factory=list)
    resolution_source: str
    resolution_sources: list[str] = Field(default_factory=list)
    merge_status: str
    merge_statuses: list[str] = Field(default_factory=list)
    merge_score: Optional[float] = None
    merge_flags: list[str] = Field(default_factory=list)
    adjudication_cluster_ids: list[str] = Field(default_factory=list)
    adjudication_actions: list[str] = Field(default_factory=list)
    adjudication_rationales: list[str] = Field(default_factory=list)
    max_salience: float = Field(ge=0.0, le=1.0)


class ConceptAdjudicationGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mention_ids: list[str]
    canonical_name: str
    display_name: Optional[str] = None
    type: str
    aliases: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("mention_ids")
    @classmethod
    def clean_mention_ids(cls, values: list[str]) -> list[str]:
        return [normalize_whitespace(value) for value in values if normalize_whitespace(value)]

    @field_validator("aliases")
    @classmethod
    def clean_aliases(cls, values: list[str]) -> list[str]:
        return unique_preserving_order(normalize_whitespace(value) for value in values if normalize_whitespace(value))

    @field_validator("canonical_name", "type")
    @classmethod
    def clean_required(cls, value: str) -> str:
        value = normalize_whitespace(value)
        if not value:
            raise ValueError("value must not be empty")
        return value

    @model_validator(mode="after")
    def fill_display_name(self) -> "ConceptAdjudicationGroup":
        self.type = normalize_whitespace(self.type).upper()
        self.display_name = normalize_whitespace(self.display_name or self.canonical_name)
        return self


class ConceptAdjudicationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    action: ConceptAdjudicationAction
    groups: list[ConceptAdjudicationGroup]
    rationale: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("cluster_id")
    @classmethod
    def clean_cluster_id(cls, value: str) -> str:
        value = normalize_whitespace(value)
        if not value:
            raise ValueError("cluster_id must not be empty")
        return value

    @field_validator("warnings")
    @classmethod
    def clean_warnings(cls, values: list[str]) -> list[str]:
        return unique_preserving_order(normalize_warning(value) for value in values if normalize_warning(value))

    @model_validator(mode="after")
    def require_groups(self) -> "ConceptAdjudicationResponse":
        if not self.groups:
            raise ValueError("adjudication must include at least one group")
        if self.rationale is not None:
            self.rationale = normalize_whitespace(self.rationale) or None
        return self


class ConceptAdjudicationFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure_id: str
    cluster_id: str
    schema_version: str = SCHEMA_VERSION
    prompt_version: str = DEFAULT_CONCEPT_RESOLUTION_PROMPT_VERSION
    model_name: str
    error_type: str
    message: str
    raw_response: Optional[str] = None


class ConceptResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    mentions: list[ConceptMention] = Field(default_factory=list)
    registry_entries: list[ConceptRegistryEntry] = Field(default_factory=list)
    mention_to_concept_id: dict[str, str] = Field(default_factory=dict)
    review_clusters: list[UncertainConceptCluster] = Field(default_factory=list)
    pair_scores: list[ConceptPairScore] = Field(default_factory=list)
    adjudications: list[ConceptAdjudicationResponse] = Field(default_factory=list)
    adjudication_failures: list[ConceptAdjudicationFailure] = Field(default_factory=list)

    def concept_id_for(self, decision_id: str, chunk_id: str, local_id: str) -> Optional[str]:
        return self.mention_to_concept_id.get(mention_id_for(decision_id, chunk_id, local_id))


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root

    def groups(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for value in self.parent:
            grouped[self.find(value)].append(value)
        return {root: sorted(values) for root, values in grouped.items()}


def build_concept_resolution(decisions: Iterable[Any]) -> ConceptResolution:
    mentions = extract_concept_mentions(decisions)
    return build_concept_resolution_from_mentions(mentions)


def build_concept_resolution_from_mentions(mentions: Iterable[ConceptMention]) -> ConceptResolution:
    mention_list = sorted(list(mentions), key=lambda item: item.mention_id)
    mention_ids = [mention.mention_id for mention in mention_list]
    if not mention_list:
        return ConceptResolution()

    pair_scores = score_candidate_pairs(mention_list)
    auto_dsu = _DisjointSet(mention_ids)
    for pair in pair_scores:
        if pair.recommended_action == ConceptPairAction.AUTO_MERGE:
            auto_dsu.union(pair.source_mention_id, pair.target_mention_id)

    auto_groups = auto_dsu.groups()
    guarded_groups = split_unsafe_auto_groups(auto_groups, mention_list)
    auto_dsu = _DisjointSet(mention_ids)
    for ids in guarded_groups.values():
        first, *rest = ids
        for mention_id in rest:
            auto_dsu.union(first, mention_id)
    auto_groups = auto_dsu.groups()
    pair_scores = downgrade_guarded_auto_merge_pairs(pair_scores, auto_groups)
    review_clusters = build_uncertain_clusters(pair_scores, auto_dsu, auto_groups, mention_list)
    group_specs = []
    for root, ids in auto_groups.items():
        status = ConceptMergeStatus.AUTO_MERGED if len(ids) > 1 else ConceptMergeStatus.SINGLE
        group_pairs = [
            pair
            for pair in pair_scores
            if pair.recommended_action == ConceptPairAction.AUTO_MERGE
            and pair.source_mention_id in ids
            and pair.target_mention_id in ids
        ]
        group_specs.append(
            {
                "group_id": root,
                "mention_ids": ids,
                "status": status.value,
                "pairs": group_pairs,
                "override": None,
            }
        )

    registry_entries, mention_to_concept_id = build_registry_from_group_specs(mention_list, group_specs)
    return ConceptResolution(
        mentions=mention_list,
        registry_entries=registry_entries,
        mention_to_concept_id=mention_to_concept_id,
        review_clusters=review_clusters,
        pair_scores=pair_scores,
    )


def extract_concept_mentions(decisions: Iterable[Any]) -> list[ConceptMention]:
    mentions: list[ConceptMention] = []
    for decision in decisions:
        for concept in getattr(decision, "concepts", []):
            local_id = str(getattr(concept, "local_id"))
            chunk_id = str(getattr(decision, "chunk_id"))
            decision_id = str(getattr(decision, "decision_id"))
            canonical_name = str(getattr(concept, "canonical_name"))
            display_name = str(getattr(concept, "display_name") or canonical_name)
            mentions.append(
                ConceptMention(
                    mention_id=mention_id_for(decision_id, chunk_id, local_id),
                    decision_id=decision_id,
                    chunk_id=chunk_id,
                    local_id=local_id,
                    canonical_name=canonical_name,
                    display_name=display_name,
                    type=str(getattr(concept, "type")),
                    aliases=list(getattr(concept, "aliases") or []),
                    salience=float(getattr(concept, "salience")),
                    description=getattr(concept, "description", None),
                    evidence_spans=list(getattr(concept, "evidence_spans") or []),
                )
            )
    return mentions


def score_candidate_pairs(mentions: list[ConceptMention]) -> list[ConceptPairScore]:
    by_id = {mention.mention_id: mention for mention in mentions}
    pair_ids: set[tuple[str, str]] = set()
    blocks: dict[str, list[str]] = defaultdict(list)
    for mention in mentions:
        for key in blocking_keys(mention):
            blocks[key].append(mention.mention_id)

    for ids in blocks.values():
        ids = sorted(set(ids))
        if len(ids) < 2 or len(ids) > 50:
            continue
        for index, left in enumerate(ids):
            for right in ids[index + 1:]:
                pair_ids.add((left, right))

    scores = [score_concept_pair(by_id[left], by_id[right]) for left, right in sorted(pair_ids)]
    return [score for score in scores if score.recommended_action != ConceptPairAction.NO_MERGE]


def split_unsafe_auto_groups(
    auto_groups: dict[str, list[str]],
    mentions: list[ConceptMention],
) -> dict[str, list[str]]:
    """Splits deterministic groups that look like transitive over-merges."""

    mentions_by_id = {mention.mention_id: mention for mention in mentions}
    guarded_groups: dict[str, list[str]] = {}
    for root, ids in auto_groups.items():
        group_mentions = [mentions_by_id[mention_id] for mention_id in ids]
        if not auto_group_needs_safety_split(group_mentions):
            guarded_groups[root] = ids
            continue
        by_safe_key: dict[str, list[str]] = defaultdict(list)
        for mention in group_mentions:
            by_safe_key[auto_group_safety_key(mention)].append(mention.mention_id)
        for index, split_ids in enumerate(by_safe_key.values()):
            guarded_groups[f"{root}:guarded:{index}"] = sorted(split_ids)
    return guarded_groups


def downgrade_guarded_auto_merge_pairs(
    pair_scores: list[ConceptPairScore],
    auto_groups: dict[str, list[str]],
) -> list[ConceptPairScore]:
    root_by_mention_id = {
        mention_id: root
        for root, mention_ids in auto_groups.items()
        for mention_id in mention_ids
    }
    guarded_scores: list[ConceptPairScore] = []
    for pair in pair_scores:
        if (
            pair.recommended_action == ConceptPairAction.AUTO_MERGE
            and root_by_mention_id.get(pair.source_mention_id) != root_by_mention_id.get(pair.target_mention_id)
        ):
            guarded_scores.append(
                pair.model_copy(
                    update={
                        "risk_flags": unique_preserving_order(
                            [*pair.risk_flags, "auto_group_safety_guard"]
                        ),
                        "recommended_action": ConceptPairAction.LLM_ADJUDICATE,
                    }
                )
            )
            continue
        guarded_scores.append(pair)
    return guarded_scores


def auto_group_needs_safety_split(mentions: list[ConceptMention]) -> bool:
    if len(mentions) < 2:
        return False
    concrete_types = {mention.type for mention in mentions if mention.type not in GENERIC_TYPES}
    if len(concrete_types) > 1:
        return True
    canonical_keys = {
        singularize_normalized_name(normalize_name(mention.canonical_name))
        for mention in mentions
        if normalize_name(mention.canonical_name)
    }
    if len(canonical_keys) > AUTO_GROUP_MAX_DISTINCT_CANONICAL_NAMES:
        return True
    source_name_keys = {
        normalize_name(name)
        for mention in mentions
        for name in mention.source_names()
        if normalize_name(name)
    }
    return len(source_name_keys) > AUTO_GROUP_MAX_SOURCE_NAMES


def auto_group_safety_key(mention: ConceptMention) -> str:
    primary = singularize_normalized_name(normalize_name(mention.canonical_name))
    if not primary:
        primary = mention.mention_id
    return f"{mention.type}:{primary}"


def score_concept_pair(left: ConceptMention, right: ConceptMention) -> ConceptPairScore:
    signals: list[str] = []
    risk_flags: list[str] = []
    score = 0.0

    left_primary = normalize_name(left.canonical_name)
    right_primary = normalize_name(right.canonical_name)
    left_forms = concept_name_forms(left)
    right_forms = concept_name_forms(right)
    exact_primary_match = bool(left_primary and left_primary == right_primary)
    exact_alias_match = bool(left_forms["normalized"] & right_forms["normalized"])
    singular_match = bool(left_forms["singular"] & right_forms["singular"])

    if exact_primary_match:
        score = max(score, 0.98)
        signals.append("exact_normalized_name")
        if not is_safe_primary_name(left.canonical_name) or not is_safe_primary_name(right.canonical_name):
            risk_flags.append("short_symbolic_alias")
    elif exact_alias_match:
        score = max(score, 0.95)
        signals.append("exact_alias_match")

    if singular_match:
        score = max(score, 0.93)
        signals.append("singular_plural_variant")

    acronym_match = (
        (left_forms["acronym"] & right_forms["acronym"])
        or (left_forms["acronym"] & right_forms["expansion_acronym"])
        or (left_forms["expansion_acronym"] & right_forms["acronym"])
    )
    if acronym_match:
        score = max(score, 0.92)
        signals.append("acronym_match")
        if not exact_primary_match and not exact_alias_match and not singular_match:
            risk_flags.append("acronym_ambiguity")

    if left_forms["ocr_short"] & right_forms["ocr_short"] and not (
        left_forms["acronym"] & right_forms["acronym"]
    ):
        score = max(score, 0.88)
        signals.append("ocr_like_short_match")
        risk_flags.append("ocr_like_short_variant")

    if left_forms["derivational"] & right_forms["derivational"] and score < 0.90:
        score = max(score, 0.78)
        signals.append("derivational_variant")
        risk_flags.append("derivational_variant")

    left_tokens = content_tokens(left_primary)
    right_tokens = content_tokens(right_primary)
    if left_tokens and right_tokens:
        smaller = left_tokens if len(left_tokens) <= len(right_tokens) else right_tokens
        larger = right_tokens if smaller is left_tokens else left_tokens
        overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        if smaller < larger and smaller.issubset(larger):
            score = max(score, 0.72)
            signals.append("modifier_containment")
            risk_flags.extend(["modifier_scope_difference", "parent_child_candidate"])
        elif overlap >= 0.66:
            score = max(score, 0.68)
            signals.append("token_overlap")

    ratio = SequenceMatcher(None, left_primary, right_primary).ratio() if left_primary and right_primary else 0.0
    if ratio >= 0.88:
        score = max(score, 0.82)
        signals.append("high_string_similarity")

    if left.type == right.type:
        score = min(1.0, score + 0.03) if score else 0.0
        if score:
            signals.append("same_type")
    elif left.type in GENERIC_TYPES or right.type in GENERIC_TYPES:
        if score:
            risk_flags.append("type_broadening")
    elif score:
        score = max(0.0, score - 0.10)
        risk_flags.append("type_conflict")

    if score and (left_primary in GENERIC_NAMES or right_primary in GENERIC_NAMES) and left_primary != right_primary:
        risk_flags.append("generic_name")
        score = min(score, 0.80)

    signals = unique_preserving_order(signals)
    risk_flags = unique_preserving_order(risk_flags)
    high_risk = bool(set(risk_flags) & HIGH_RISK_FLAGS)
    if score >= AUTO_MERGE_THRESHOLD and not high_risk:
        action = ConceptPairAction.AUTO_MERGE
    elif score <= AUTO_SPLIT_THRESHOLD:
        action = ConceptPairAction.NO_MERGE
    else:
        action = ConceptPairAction.LLM_ADJUDICATE

    return ConceptPairScore(
        source_mention_id=left.mention_id,
        target_mention_id=right.mention_id,
        score=round(score, 4),
        signals=signals,
        risk_flags=risk_flags,
        recommended_action=action,
    )


def build_uncertain_clusters(
    pair_scores: list[ConceptPairScore],
    auto_dsu: _DisjointSet,
    auto_groups: dict[str, list[str]],
    mentions: list[ConceptMention],
) -> list[UncertainConceptCluster]:
    uncertain_roots = _DisjointSet(auto_groups.keys())
    has_uncertain_edge = False
    uncertain_pairs: list[ConceptPairScore] = []
    for pair in pair_scores:
        if pair.recommended_action != ConceptPairAction.LLM_ADJUDICATE:
            continue
        left_root = auto_dsu.find(pair.source_mention_id)
        right_root = auto_dsu.find(pair.target_mention_id)
        if left_root == right_root:
            continue
        uncertain_roots.union(left_root, right_root)
        uncertain_pairs.append(pair)
        has_uncertain_edge = True

    if not has_uncertain_edge:
        return []

    mentions_by_id = {mention.mention_id: mention for mention in mentions}
    root_components = uncertain_roots.groups()
    clusters: list[UncertainConceptCluster] = []
    for root_ids in root_components.values():
        component_pairs = [
            pair
            for pair in uncertain_pairs
            if auto_dsu.find(pair.source_mention_id) in root_ids or auto_dsu.find(pair.target_mention_id) in root_ids
        ]
        if not component_pairs:
            continue
        mention_ids = sorted({mention_id for root_id in root_ids for mention_id in auto_groups[root_id]})
        cluster_mentions = [mentions_by_id[mention_id] for mention_id in mention_ids]
        cluster_payload = {
            "mention_ids": mention_ids,
            "pair_ids": [(pair.source_mention_id, pair.target_mention_id) for pair in component_pairs],
        }
        clusters.append(
            UncertainConceptCluster(
                cluster_id=f"concept_cluster_{stable_hash(cluster_payload, length=20)}",
                mention_ids=mention_ids,
                concept_names=unique_preserving_order(mention.canonical_name for mention in cluster_mentions),
                types=unique_preserving_order(mention.type for mention in cluster_mentions),
                source_chunk_ids=unique_preserving_order(mention.chunk_id for mention in cluster_mentions),
                score=max(pair.score for pair in component_pairs),
                risk_flags=unique_preserving_order(flag for pair in component_pairs for flag in pair.risk_flags),
                signals=unique_preserving_order(signal for pair in component_pairs for signal in pair.signals),
                pair_scores=sorted(component_pairs, key=lambda item: (-item.score, item.source_mention_id)),
            )
        )
    return sorted(clusters, key=lambda item: (-item.score, item.cluster_id))


def resolution_source_for_status(status: str) -> str:
    if status == ConceptMergeStatus.SINGLE.value:
        return "single"
    if status == ConceptMergeStatus.AUTO_MERGED.value:
        return "deterministic"
    if status in {ConceptMergeStatus.LLM_MERGED.value, ConceptMergeStatus.LLM_SPLIT.value}:
        return "llm_adjudicated"
    return "unknown"



def build_registry_from_group_specs(
    mentions: list[ConceptMention],
    group_specs: list[dict[str, Any]],
) -> tuple[list[ConceptRegistryEntry], dict[str, str]]:
    mentions_by_id = {mention.mention_id: mention for mention in mentions}
    records: dict[str, dict[str, Any]] = {}
    mention_to_concept_id: dict[str, str] = {}

    for spec in group_specs:
        group_mentions = [mentions_by_id[mention_id] for mention_id in spec["mention_ids"]]
        override = spec.get("override")
        if override:
            canonical_name = canonicalize_display_name(override["canonical_name"])
            display_name = normalize_whitespace(override.get("display_name") or canonical_name)
            concept_type = normalize_whitespace(override["type"]).upper()
            extra_aliases = list(override.get("aliases") or [])
        else:
            canonical_name = choose_canonical_name(group_mentions)
            display_name = canonical_name
            concept_type = choose_concept_type(group_mentions)
            extra_aliases = []

        concept_id = concept_id_for(canonical_name, concept_type)
        group_pairs = list(spec.get("pairs") or [])
        group_score = max((pair.score for pair in group_pairs), default=None)
        group_flags = unique_preserving_order(flag for pair in group_pairs for flag in pair.risk_flags)
        status = str(spec["status"])
        resolution_source = resolution_source_for_status(status)
        adjudication = spec.get("adjudication") or {}
        record = records.setdefault(
            concept_id,
            {
                "concept_id": concept_id,
                "canonical_name": canonical_name,
                "display_name": display_name,
                "type": concept_type,
                "aliases": [],
                "descriptions": [],
                "source_names": [],
                "source_types": [],
                "source_chunk_ids": [],
                "evidence_spans": [],
                "mention_ids": [],
                "source_decision_ids": [],
                "resolution_source": resolution_source,
                "resolution_sources": [],
                "merge_status": status,
                "merge_statuses": [],
                "merge_score": group_score,
                "merge_flags": [],
                "adjudication_cluster_ids": [],
                "adjudication_actions": [],
                "adjudication_rationales": [],
                "max_salience": 0.0,
            },
        )
        record["merge_statuses"] = unique_preserving_order([*record["merge_statuses"], status])
        record["merge_status"] = record["merge_statuses"][0] if len(record["merge_statuses"]) == 1 else "mixed"
        record["resolution_sources"] = unique_preserving_order([*record["resolution_sources"], resolution_source])
        record["resolution_source"] = (
            record["resolution_sources"][0] if len(record["resolution_sources"]) == 1 else "mixed"
        )
        if group_score is not None:
            record["merge_score"] = max(record["merge_score"] or 0.0, group_score)
        record["merge_flags"] = unique_preserving_order([*record["merge_flags"], *group_flags])
        if adjudication:
            record["adjudication_cluster_ids"] = unique_preserving_order(
                [*record["adjudication_cluster_ids"], adjudication.get("cluster_id", "")]
            )
            record["adjudication_actions"] = unique_preserving_order(
                [*record["adjudication_actions"], adjudication.get("action", "")]
            )
            if adjudication.get("rationale"):
                record["adjudication_rationales"] = unique_preserving_order(
                    [*record["adjudication_rationales"], adjudication["rationale"]]
                )

        aliases = []
        for mention in group_mentions:
            source_names = mention.source_names()
            aliases.extend(source_names)
            record["source_names"] = unique_preserving_order([*record["source_names"], *source_names])
            record["source_types"] = unique_preserving_order([*record["source_types"], mention.type])
            if mention.description:
                record["descriptions"] = unique_preserving_order([*record["descriptions"], mention.description])
            record["source_chunk_ids"] = unique_preserving_order([*record["source_chunk_ids"], mention.chunk_id])
            record["evidence_spans"] = unique_preserving_order([*record["evidence_spans"], *mention.evidence_spans])
            record["mention_ids"] = unique_preserving_order([*record["mention_ids"], mention.mention_id])
            record["source_decision_ids"] = unique_preserving_order([*record["source_decision_ids"], mention.decision_id])
            record["max_salience"] = max(record["max_salience"], mention.salience)
            mention_to_concept_id[mention.mention_id] = concept_id

        aliases.extend(extra_aliases)
        record["aliases"] = unique_preserving_order(
            alias
            for alias in aliases
            if normalize_name(alias) and normalize_name(alias) != normalize_name(canonical_name)
        )

    entries = [ConceptRegistryEntry.model_validate(record) for record in records.values()]
    return sorted(entries, key=lambda item: (item.type, item.canonical_name, item.concept_id)), mention_to_concept_id


def apply_concept_adjudications(
    resolution: ConceptResolution,
    adjudications: Iterable[ConceptAdjudicationResponse],
) -> ConceptResolution:
    adjudication_list = list(adjudications)
    if not adjudication_list:
        return resolution

    mentions = resolution.mentions
    mentions_by_id = {mention.mention_id: mention for mention in mentions}
    mention_ids = set(mentions_by_id)
    adjudicated_ids = {
        mention_id for adjudication in adjudication_list for group in adjudication.groups for mention_id in group.mention_ids
    }
    if not adjudicated_ids <= mention_ids:
        unknown = sorted(adjudicated_ids - mention_ids)
        raise ValueError(f"adjudications include unknown mention IDs: {unknown}")

    entries_by_id = {entry.concept_id: entry for entry in resolution.registry_entries}
    previous_entry_by_mention_id = {
        mention_id: entries_by_id.get(concept_id)
        for mention_id, concept_id in resolution.mention_to_concept_id.items()
    }
    existing_groups: dict[str, list[str]] = defaultdict(list)
    for mention_id, concept_id in resolution.mention_to_concept_id.items():
        if mention_id not in adjudicated_ids:
            existing_groups[concept_id].append(mention_id)

    group_specs: list[dict[str, Any]] = []
    for concept_id, ids in sorted(existing_groups.items()):
        entry = entries_by_id.get(concept_id)
        status = entry.merge_status if entry else ConceptMergeStatus.SINGLE.value
        group_specs.append(
            {
                "group_id": concept_id,
                "mention_ids": sorted(ids),
                "status": status,
                "pairs": [],
                "override": (
                    {
                        "canonical_name": entry.canonical_name,
                        "display_name": entry.display_name,
                        "type": entry.type,
                        "aliases": entry.aliases,
                    }
                    if entry
                    else None
                ),
            }
        )

    for adjudication in adjudication_list:
        for index, group in enumerate(adjudication.groups):
            status = (
                ConceptMergeStatus.LLM_MERGED.value
                if len(group.mention_ids) > 1
                else ConceptMergeStatus.LLM_SPLIT.value
            )
            previous_entry = previous_entry_by_mention_id.get(group.mention_ids[0]) if len(group.mention_ids) == 1 else None
            previous_entry_matches_group = previous_entry is not None and set(previous_entry.mention_ids) == set(group.mention_ids)
            if previous_entry_matches_group:
                override = {
                    "canonical_name": previous_entry.canonical_name,
                    "display_name": previous_entry.display_name,
                    "type": previous_entry.type,
                    "aliases": unique_preserving_order(
                        [
                            *previous_entry.aliases,
                            *(group.aliases or []),
                            group.canonical_name,
                            group.display_name or group.canonical_name,
                        ]
                    ),
                }
            elif len(group.mention_ids) == 1:
                mention = mentions_by_id[group.mention_ids[0]]
                override = {
                    "canonical_name": canonicalize_display_name(mention.canonical_name),
                    "display_name": mention.display_name or mention.canonical_name,
                    "type": mention.type,
                    "aliases": unique_preserving_order(
                        [
                            *mention.source_names(),
                            *(group.aliases or []),
                            group.canonical_name,
                            group.display_name or group.canonical_name,
                        ]
                    ),
                }
            else:
                override = {
                    "canonical_name": group.canonical_name,
                    "display_name": group.display_name or group.canonical_name,
                    "type": group.type,
                    "aliases": group.aliases,
                }
            group_specs.append(
                {
                    "group_id": f"{adjudication.cluster_id}:{index}",
                    "mention_ids": group.mention_ids,
                    "status": status,
                    "pairs": [],
                    "override": override,
                    "adjudication": {
                        "cluster_id": adjudication.cluster_id,
                        "action": adjudication.action.value,
                        "rationale": adjudication.rationale,
                    },
                }
            )

    registry_entries, mention_to_concept_id = build_registry_from_group_specs(mentions, group_specs)
    remaining_review_clusters = [
        cluster for cluster in resolution.review_clusters if not set(cluster.mention_ids) <= adjudicated_ids
    ]
    return ConceptResolution(
        mentions=mentions,
        registry_entries=registry_entries,
        mention_to_concept_id=mention_to_concept_id,
        review_clusters=remaining_review_clusters,
        pair_scores=resolution.pair_scores,
        adjudications=[*resolution.adjudications, *adjudication_list],
        adjudication_failures=resolution.adjudication_failures,
    )


def concept_adjudication_prompt_payload(
    cluster: UncertainConceptCluster,
    mentions: Iterable[ConceptMention],
) -> dict[str, Any]:
    mentions_by_id = {mention.mention_id: mention for mention in mentions}
    return {
        "schema_version": SCHEMA_VERSION,
        "cluster_id": cluster.cluster_id,
        "deterministic_score": cluster.score,
        "risk_flags": cluster.risk_flags,
        "signals": cluster.signals,
        "pair_scores": [pair.model_dump(mode="json") for pair in cluster.pair_scores],
        "mentions": [
            mentions_by_id[mention_id].model_dump(mode="json")
            for mention_id in cluster.mention_ids
            if mention_id in mentions_by_id
        ],
    }


def concept_adjudication_schema_hint() -> dict[str, Any]:
    return {
        "cluster_id": "copy exact cluster_id",
        "action": "merge_all | split_all | merge_groups",
        "groups": [
            {
                "mention_ids": ["each input mention_id exactly once"],
                "canonical_name": "concise English graph concept name",
                "display_name": "concise visible label",
                "type": "CONCEPT | MODEL | METHOD | FORMULA | TOOL | TASK | PAPER | PROBLEM | COMPONENT",
                "aliases": ["source names/aliases only"],
                "confidence": "0.0-1.0",
            }
        ],
        "rationale": "<=20 words",
        "warnings": ["short_snake_case_warning"],
    }


def concept_adjudication_cache_key(
    cluster: UncertainConceptCluster,
    *,
    model_name: str,
    prompt_version: str,
    generation_settings: Optional[dict[str, Any]] = None,
) -> str:
    return stable_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "prompt_version": prompt_version,
            "model_name": model_name,
            "generation_settings": generation_settings or {},
            "cluster": cluster.model_dump(mode="json"),
        },
        length=32,
    )


def parse_concept_adjudication_response(raw_response: str) -> ConceptAdjudicationResponse:
    payload = load_json_payload(clean_json_markdown(raw_response))
    if isinstance(payload, list):
        payload = {"groups": payload}
    return ConceptAdjudicationResponse.model_validate(payload)


def validate_concept_adjudication_response(
    response: ConceptAdjudicationResponse,
    cluster: UncertainConceptCluster,
) -> list[str]:
    errors: list[str] = []
    if response.cluster_id != cluster.cluster_id:
        errors.append("response cluster_id does not match input cluster_id")

    expected = set(cluster.mention_ids)
    seen: list[str] = []
    for group in response.groups:
        seen.extend(group.mention_ids)
        if not group.mention_ids:
            errors.append("adjudication group has no mention_ids")
    seen_set = set(seen)
    if len(seen) != len(seen_set):
        errors.append("adjudication groups contain duplicate mention_ids")
    if seen_set != expected:
        missing = sorted(expected - seen_set)
        unknown = sorted(seen_set - expected)
        if missing:
            errors.append(f"adjudication missing mention_ids: {missing}")
        if unknown:
            errors.append(f"adjudication has unknown mention_ids: {unknown}")

    if response.action == ConceptAdjudicationAction.MERGE_ALL and len(response.groups) != 1:
        errors.append("merge_all must return exactly one group")
    if response.action == ConceptAdjudicationAction.SPLIT_ALL:
        if any(len(group.mention_ids) != 1 for group in response.groups):
            errors.append("split_all groups must each contain exactly one mention_id")
        if len(response.groups) != len(expected):
            errors.append("split_all must return one group per mention")
    return errors


def make_concept_adjudication_failure(
    error_type: str,
    message: str,
    *,
    cluster: UncertainConceptCluster,
    model_name: str,
    prompt_version: str,
    raw_response: Optional[str] = None,
) -> ConceptAdjudicationFailure:
    payload = {
        "cluster_id": cluster.cluster_id,
        "error_type": error_type,
        "message": message,
        "raw_response_hash": stable_hash(raw_response or "", length=16),
    }
    return ConceptAdjudicationFailure(
        failure_id=f"concept_failure_{stable_hash(payload, length=24)}",
        cluster_id=cluster.cluster_id,
        prompt_version=prompt_version,
        model_name=model_name,
        error_type=error_type,
        message=message,
        raw_response=raw_response,
    )


def fake_concept_adjudication_response(cluster: UncertainConceptCluster) -> str:
    if "derivational_variant" in cluster.risk_flags:
        group = {
            "mention_ids": cluster.mention_ids,
            "canonical_name": canonicalize_display_name(cluster.concept_names[0]),
            "display_name": canonicalize_display_name(cluster.concept_names[0]),
            "type": cluster.types[0],
            "aliases": cluster.concept_names[1:],
            "confidence": 0.82,
        }
        action = ConceptAdjudicationAction.MERGE_ALL.value
        groups = [group]
    else:
        action = ConceptAdjudicationAction.SPLIT_ALL.value
        groups = []
        for index, mention_id in enumerate(cluster.mention_ids):
            name = cluster.concept_names[min(index, len(cluster.concept_names) - 1)]
            groups.append(
                {
                    "mention_ids": [mention_id],
                    "canonical_name": canonicalize_display_name(name),
                    "display_name": canonicalize_display_name(name),
                    "type": cluster.types[min(index, len(cluster.types) - 1)],
                    "aliases": [],
                    "confidence": 0.75,
                }
            )
    return json.dumps(
        {
            "cluster_id": cluster.cluster_id,
            "action": action,
            "groups": groups,
            "rationale": "Deterministic fake adjudication.",
            "warnings": [],
        },
        ensure_ascii=False,
    )


def normalize_whitespace(text: Any) -> str:
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
    text = unicodedata.normalize("NFC", text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def mention_id_for(decision_id: str, chunk_id: str, local_id: str) -> str:
    return f"mention_{stable_hash([decision_id, chunk_id, local_id], length=20)}"


def concept_id_for(canonical_name: str, concept_type: str) -> str:
    key = f"{normalize_whitespace(concept_type).upper()}::{normalize_name(canonical_name)}"
    return f"concept_{stable_hash(key, length=20)}"


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", normalize_whitespace(value)).casefold()
    value = value.replace("_", " ").replace("-", " ").replace("–", " ").replace("—", " ")
    value = PUNCT_RE.sub(" ", value)
    return normalize_whitespace(value)


def singularize_normalized_name(value: str) -> str:
    tokens = value.split()
    if not tokens:
        return value
    tokens[-1] = singularize_token(tokens[-1])
    return " ".join(tokens)


def singularize_token(token: str) -> str:
    if len(token) <= 3:
        return token
    if token in INVARIANT_SINGULAR_TOKENS:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ices") and len(token) > 5:
        return token[:-4] + "ex"
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("xes") or token.endswith("ches") or token.endswith("shes"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def derivational_normalize(value: str) -> str:
    tokens = value.split()
    if not tokens:
        return value
    last = tokens[-1]
    if last.endswith("ing") and len(last) > 5:
        root = last[:-3]
        if len(root) >= 2 and root[-1] == root[-2] and root[-1] not in {"l", "s", "z"}:
            root = root[:-1]
        tokens[-1] = root
    else:
        tokens[-1] = singularize_token(last)
    return " ".join(tokens)


def content_tokens(value: str) -> set[str]:
    return {token for token in NAME_TOKEN_RE.findall(value) if token not in TOKEN_STOPWORDS}


def concept_name_forms(mention: ConceptMention) -> dict[str, set[str]]:
    source_names = mention.source_names()
    merge_names = [name for name in source_names if is_merge_safe_name(name)]
    normalized = {normalize_name(name) for name in merge_names}
    normalized = {name for name in normalized if name}
    singular = {singularize_normalized_name(name) for name in normalized}
    derivational = {derivational_normalize(name) for name in normalized}
    acronym = set()
    expansion_acronym = set()
    ocr_short = set()
    for original_name in source_names:
        maybe_acronym = acronym_key(original_name)
        if maybe_acronym:
            acronym.add(maybe_acronym)
    for original_name, normalized_name in zip(merge_names, [normalize_name(name) for name in merge_names]):
        expansion = acronym_from_expansion(normalized_name)
        if expansion:
            expansion_acronym.add(expansion)
        maybe_ocr = ocr_short_key(original_name)
        if maybe_ocr:
            ocr_short.add(maybe_ocr)
    return {
        "normalized": normalized,
        "singular": singular,
        "derivational": derivational,
        "acronym": acronym,
        "expansion_acronym": expansion_acronym,
        "ocr_short": ocr_short,
    }


def blocking_keys(mention: ConceptMention) -> set[str]:
    forms = concept_name_forms(mention)
    keys: set[str] = set()
    primary = normalize_name(mention.canonical_name)
    if primary and is_safe_primary_name(mention.canonical_name):
        keys.add(f"primary:{primary}")
    for family in ("normalized", "singular", "derivational", "acronym", "expansion_acronym", "ocr_short"):
        for value in forms[family]:
            if value:
                keys.add(f"{family}:{value}")
    for token in content_tokens(normalize_name(mention.canonical_name)):
        if len(token) >= 4 and token not in TOKEN_STOPWORDS:
            keys.add(f"token:{token}")
    return keys


def acronym_key(value: str) -> Optional[str]:
    if is_formula_like_name(value):
        return None
    compact = re.sub(r"[^A-Za-z0-9]", "", value)
    if not (2 <= len(compact) <= 8):
        return None
    if compact.upper() == compact and any(char.isalpha() for char in compact):
        return compact.upper()
    return None


def is_safe_primary_name(value: str) -> bool:
    if is_formula_like_name(value):
        return False
    compact = re.sub(r"[^a-zа-яё0-9+#]+", "", normalize_name(value))
    if len(compact) >= 4:
        return True
    return acronym_key(value) is not None


def is_merge_safe_name(value: str) -> bool:
    if is_formula_like_name(value):
        return False
    normalized = normalize_name(value)
    if not normalized:
        return False
    compact = re.sub(r"[^a-zа-яё0-9+#]+", "", normalized)
    if len(compact) < 4:
        return False
    return True


def is_formula_like_name(value: str) -> bool:
    raw = normalize_whitespace(value)
    if not raw:
        return True
    if FORMULA_SYMBOL_RE.search(raw):
        return True
    if SYMBOL_WITH_PARENS_RE.fullmatch(raw):
        return True
    if SHORT_CODE_LIKE_RE.fullmatch(raw) and len(raw) <= 16:
        return True
    tokens = NAME_TOKEN_RE.findall(raw)
    if len(tokens) > 8:
        return True
    if len(tokens) > 3 and re.search(r"[.!?;:]", raw):
        return True
    return False


def acronym_from_expansion(value: str) -> Optional[str]:
    tokens = [token for token in NAME_TOKEN_RE.findall(value) if token not in TOKEN_STOPWORDS]
    if len(tokens) < 2 or len(tokens) > 8:
        return None
    acronym = "".join(token[0] for token in tokens).upper()
    return acronym if len(acronym) >= 2 else None


def ocr_short_key(value: str) -> Optional[str]:
    compact = re.sub(r"[^a-z0-9]", "", value.casefold())
    if not (2 <= len(compact) <= 8):
        return None
    replacements = str.maketrans({"l": "i", "1": "i", "|": "i", "0": "o"})
    return compact.translate(replacements).upper()


def choose_canonical_name(mentions: list[ConceptMention]) -> str:
    weighted: dict[str, float] = defaultdict(float)
    original_by_key: dict[str, list[str]] = defaultdict(list)
    for mention in mentions:
        for name in [mention.canonical_name, mention.display_name]:
            key = singularize_normalized_name(normalize_name(name))
            if not key:
                continue
            weighted[key] += 1.0 + mention.salience
            original_by_key[key].append(name)
    if not weighted:
        return canonicalize_display_name(mentions[0].canonical_name)
    key = max(weighted, key=lambda item: (weighted[item], -len(item), item))
    originals = original_by_key[key]
    original = min(originals, key=lambda item: (len(item), item.casefold()))
    return canonicalize_display_name(original)


def canonicalize_display_name(value: str) -> str:
    value = normalize_whitespace(value)
    if not value:
        return value
    if normalize_name(value) in INVARIANT_DISPLAY_NAMES:
        return value
    tokens = value.split()
    if not tokens:
        return value
    last_norm = normalize_name(tokens[-1])
    singular = singularize_token(last_norm)
    if singular != last_norm and tokens[-1].casefold().endswith("s"):
        tokens[-1] = tokens[-1][:-1]
    return " ".join(tokens)


def choose_concept_type(mentions: list[ConceptMention]) -> str:
    weights: dict[str, float] = defaultdict(float)
    for mention in mentions:
        weights[mention.type] += 1.0 + mention.salience
    if len(weights) > 1 and "CONCEPT" in weights:
        non_generic = {key: value for key, value in weights.items() if key != "CONCEPT"}
        if non_generic:
            return max(non_generic, key=lambda item: (non_generic[item], item))
    return max(weights, key=lambda item: (weights[item], item))


def normalize_warning(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_")


def clean_json_markdown(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def load_json_payload(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        candidate = extract_json_candidate(text)
        if candidate and candidate != text:
            return json.loads(candidate)
        raise


def extract_json_candidate(text: str) -> Optional[str]:
    stripped = text.strip()
    starts = [index for index in (stripped.find("{"), stripped.find("[")) if index >= 0]
    if not starts:
        return None
    return stripped[min(starts):].strip()
