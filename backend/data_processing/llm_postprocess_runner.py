"""Shared execution helpers for RemNote LLM post-processing runners.

This module owns the LLM execution, cache, repair, and concept-resolution loops
used by both the standalone postprocess script and the optimized two-pass
pipeline. The script entry points keep CLI orchestration local, while this file
keeps cache keys, prompt resolution, and failure handling consistent across
those workflows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from backend.configs.enums import ModelRoleType, PromptType
from backend.configs.models import OllamaSettings
from backend.data_processing.concept_registry import (
    ConceptAdjudicationFailure,
    ConceptAdjudicationResponse,
    ConceptResolution,
    MAX_CONCEPT_ADJUDICATION_PROMPT_CHARS,
    apply_concept_adjudications,
    build_concept_resolution,
    concept_adjudication_cache_key,
    concept_adjudication_prompt_payload,
    concept_adjudication_schema_hint,
    fake_concept_adjudication_response,
    make_concept_adjudication_failure,
    parse_concept_adjudication_response,
    validate_concept_adjudication_response,
)
from backend.data_processing.llm_postprocess import (
    CACHE_DIRNAME,
    DECISIONS_FILENAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_SMOKE_LIMIT,
    FAILURES_FILENAME,
    INPUTS_FILENAME,
    ChunkAction,
    ChunkEnrichmentDecision,
    ChunkPostprocessBatch,
    ChunkPostprocessInput,
    LLMResponseCache,
    PostprocessFailure,
    build_batches,
    batch_prompt_payload,
    cache_key_for_batch,
    fake_llm_response_for_batch,
    make_failure,
    parse_llm_response,
    read_jsonl,
    read_jsonl_if_exists,
    response_schema_hint,
    stable_hash,
    unique_preserving_order,
    validate_and_enrich_response,
)
from backend.utils.prompt_engine import PromptEngine


GRAPH_ACTIONS = {
    ChunkAction.KEEP,
    ChunkAction.KEEP_WITH_CLEANED_TEXT,
    ChunkAction.METADATA_ONLY,
    ChunkAction.GRAPH_ONLY,
}
PASS_PROMPT_NAMES = {
    "quality": "remnote_postprocess_quality",
    "graph": "remnote_postprocess_graph",
}
DEFAULT_POSTPROCESS_PROMPT_NAME = "remnote_postprocess"


@dataclass(frozen=True)
class PostprocessPassSpec:
    """Resolved runtime identity for one post-processing pass.

    The standalone runner historically used a single prompt and a flat cache
    directory. The optimized pipeline uses separate quality and graph passes, so
    this value object keeps the prompt label, cache namespace, and prompt-hash
    behavior explicit for each mode.
    """

    pass_name: str
    prompt_name: str
    prompt_version: str
    cache_namespace: str | None = None
    prompt_hash_includes_pass: bool = False


@dataclass(frozen=True)
class PostprocessPassResult:
    """Result of a chunk post-processing pass.

    The iterator is retained for compatibility with older call sites and tests
    that unpacked the helper return value as a tuple.
    """

    decisions: list[ChunkEnrichmentDecision]
    failures: list[PostprocessFailure]
    cache_hits: int
    cache_misses: int
    aborted: bool

    def __iter__(self):
        yield self.decisions
        yield self.failures
        yield self.cache_hits
        yield self.cache_misses
        yield self.aborted


@dataclass(frozen=True)
class ConceptResolutionRunResult:
    """Result of deterministic and optional LLM concept resolution.

    The iterator is retained for compatibility with older call sites and tests
    that unpacked the helper return value as a tuple.
    """

    resolution: ConceptResolution | None
    cache_hits: int
    cache_misses: int

    def __iter__(self):
        yield self.resolution
        yield self.cache_hits
        yield self.cache_misses


def effective_num_predict(args: argparse.Namespace, pass_name: str) -> int:
    if pass_name == "quality":
        override = getattr(args, "quality_num_predict", None)
        if override is not None:
            return override
    if pass_name == "graph":
        override = getattr(args, "graph_num_predict", None)
        if override is not None:
            return override
    if pass_name == "concept_resolution":
        override = getattr(args, "concept_resolution_num_predict", None)
        if override is not None:
            return override
    return args.num_predict


def generation_settings_for_pass(args: argparse.Namespace, pass_name: str) -> dict[str, Any]:
    """Builds the generation settings that participate in response cache keys."""

    return {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "num_ctx": args.num_ctx,
        "num_predict": effective_num_predict(args, pass_name),
    }


def legacy_generation_settings_for_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
    }


def can_use_legacy_cache_key(args: argparse.Namespace, pass_name: str) -> bool:
    """Returns whether current settings can read caches created before generation settings were keyed."""

    return generation_settings_for_pass(args, pass_name) == legacy_generation_settings_for_args(args)


def is_usage_limit_error(value: object) -> bool:
    text = str(value).casefold()
    return (
        "status code: 429" in text
        or "status 429" in text
        or "weekly usage limit" in text
        or "rate limit" in text
        or "too many requests" in text
    )


def is_empty_llm_response(raw_response: Optional[str]) -> bool:
    return raw_response is None or not raw_response.strip()


def resolve_run_limit(limit: Optional[int], *, allow_full_run: bool) -> int:
    if limit is None:
        return 0 if allow_full_run else DEFAULT_SMOKE_LIMIT
    return limit


def select_run_inputs(
    inputs: list[Any],
    *,
    limit: Optional[int],
    offset: int = 0,
    allow_full_run: bool,
) -> list[Any]:
    """Applies the shared smoke/full-run limit semantics used by CLI runners."""

    limit = resolve_run_limit(limit, allow_full_run=allow_full_run)
    sliced = inputs[offset:]
    if limit == 0 and allow_full_run:
        return sliced
    return sliced[:limit]


def load_excluded_chunk_ids(output_dirs: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for output_dir in output_dirs:
        path = Path(output_dir) / INPUTS_FILENAME
        if not path.exists():
            raise SystemExit(f"--exclude-output-dir is missing {INPUTS_FILENAME}: {output_dir}")
        for row in read_jsonl(path):
            chunk_id = row.get("chunk_id")
            if chunk_id:
                excluded.add(chunk_id)
    return excluded


def prompt_file_for_name(prompt_dir: Path, prompt_name: str, prompt_version: str) -> Path:
    return prompt_dir / "learner_workflow" / "orchestrator" / prompt_name / f"{prompt_version}.yaml"


def resolved_prompt_name_for_pass(args: argparse.Namespace, pass_name: str) -> str:
    prompt_name = PASS_PROMPT_NAMES.get(pass_name, DEFAULT_POSTPROCESS_PROMPT_NAME)
    if prompt_name == DEFAULT_POSTPROCESS_PROMPT_NAME:
        return prompt_name
    if prompt_file_for_name(args.prompt_dir, prompt_name, args.prompt_version).exists():
        return prompt_name
    return DEFAULT_POSTPROCESS_PROMPT_NAME


def postprocess_pass_spec(args: argparse.Namespace, pass_name: str) -> PostprocessPassSpec:
    """Resolves prompt and cache identity for a standalone, quality, or graph pass."""

    prompt_name = resolved_prompt_name_for_pass(args, pass_name)
    if pass_name == "single":
        return PostprocessPassSpec(
            pass_name=pass_name,
            prompt_name=prompt_name,
            prompt_version=args.prompt_version,
        )
    return PostprocessPassSpec(
        pass_name=pass_name,
        prompt_name=prompt_name,
        prompt_version=f"{args.prompt_version}:{pass_name}",
        cache_namespace=pass_name,
        prompt_hash_includes_pass=True,
    )


def load_prompt(prompt_dir: Path, prompt_version: str, *, prompt_name: str = DEFAULT_POSTPROCESS_PROMPT_NAME) -> tuple[str, str]:
    user_template, system_payload = PromptEngine(prompt_dir).render(
        PromptType.learner_workflow,
        ModelRoleType.orchestrator,
        prompt_version,
        prompt_name,
    )
    if not isinstance(system_payload, dict):
        return user_template, str(system_payload)
    return user_template, str(system_payload.get("system_instruction", ""))


def load_prompt_for_pass(args: argparse.Namespace, pass_name: str) -> tuple[tuple[str, str], str]:
    prompt_name = resolved_prompt_name_for_pass(args, pass_name)
    return load_prompt(args.prompt_dir, args.prompt_version, prompt_name=prompt_name), prompt_name


def build_ollama_llm(
    args: argparse.Namespace,
    *,
    model_name: Optional[str] = None,
    prompt_version: Optional[str] = None,
    num_predict: Optional[int] = None,
) -> Any:
    """Creates the JSON-bound Ollama chat model used by post-processing calls."""

    from backend.workflows.agents.factory import AgentsFactory

    settings = OllamaSettings(
        role=ModelRoleType.orchestrator,
        model_name=model_name or args.model_name,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_ctx=args.num_ctx,
        num_predict=num_predict if num_predict is not None else args.num_predict,
        base_url=args.base_url,
        prompt_version=prompt_version or args.prompt_version,
    )
    return AgentsFactory.add_retry(AgentsFactory.get_llm_by_role(settings), settings.provider).bind(format="json")


def invoke_llm_batch(
    llm: Any,
    prompt: tuple[str, str],
    batch: ChunkPostprocessBatch,
    *,
    pass_name: str = "single",
) -> tuple[str, dict[str, Any]]:
    user_template, system_prompt = prompt
    input_json = json.dumps(batch_prompt_payload(batch), ensure_ascii=False, indent=2)
    schema_json = json.dumps(response_schema_hint(pass_name=pass_name), ensure_ascii=False, indent=2)
    user_prompt = (
        user_template.replace("{input_json}", input_json)
        .replace("{response_schema}", schema_json)
        .replace("{pass_name}", pass_name)
    )
    system_prompt = system_prompt.replace("{pass_name}", pass_name)
    return invoke_messages(llm, system_prompt, user_prompt)


def invoke_llm_repair(
    llm: Any,
    prompt: tuple[str, str],
    batch: ChunkPostprocessBatch,
    raw_response: str,
    failures: list[PostprocessFailure],
    *,
    pass_name: str = "single",
) -> tuple[str, dict[str, Any]]:
    _, system_prompt = prompt
    input_json = json.dumps(batch_prompt_payload(batch), ensure_ascii=False, indent=2)
    schema_json = json.dumps(response_schema_hint(pass_name=pass_name), ensure_ascii=False, indent=2)
    failure_summary = "; ".join(f"{failure.error_type}: {failure.message[:300]}" for failure in failures)
    if pass_name == "single":
        repair_prompt = (
            "Generate a fresh compact JSON response for the same input. "
            "Do not repair by continuing the previous answer. Return JSON only. "
            "If exact evidence is uncertain, drop the concept or relation. "
            "Use null for cleaned text unless a short safe cleanup is required.\n\n"
            f"Validation errors to avoid:\n{failure_summary}\n\n"
            f"Input batch JSON:\n{input_json}\n\n"
            f"Field contract:\n{schema_json}\n\n"
            f"Previous response prefix for context only:\n{raw_response[:1200]}"
        )
    else:
        repair_prompt = (
            "Generate a fresh compact JSON response for the same input batch. "
            "Do not continue the previous answer. Return JSON only.\n\n"
            f"Validation errors to avoid:\n{failure_summary}\n\n"
            f"Input batch JSON:\n{input_json}\n\n"
            f"Field contract:\n{schema_json}\n\n"
            f"Previous response prefix for context only:\n{raw_response[:1200]}"
        )
    return invoke_messages(llm, system_prompt.replace("{pass_name}", pass_name), repair_prompt)


def invoke_messages(llm: Any, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    raw_text = content_to_text(getattr(response, "content", response))
    metadata = {
        "response_metadata": getattr(response, "response_metadata", {}),
        "usage_metadata": getattr(response, "usage_metadata", {}),
    }
    return raw_text, metadata


def content_to_text(content: Any) -> str:
    """Normalizes LangChain response content into text without assuming one provider shape."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content") or ""
                if value:
                    parts.append(str(value))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def validate_raw_response(
    raw_response: str,
    batch: ChunkPostprocessBatch,
    *,
    model_name: str,
    prompt_version: str,
) -> tuple[list[ChunkEnrichmentDecision], list[PostprocessFailure]]:
    """Parses and validates one raw LLM batch response into decisions or failures."""

    try:
        parsed = parse_llm_response(raw_response)
    except Exception as exc:
        return [], [
            make_failure(
                "json_parse_error",
                f"{type(exc).__name__}: {exc}",
                batch=batch,
                raw_response=raw_response,
                model_name=model_name,
                prompt_version=prompt_version,
            )
        ]
    return validate_and_enrich_response(
        parsed,
        batch,
        model_name=model_name,
        prompt_version=prompt_version,
        raw_response=raw_response,
    )


def _pass_cache_dir(output_dir: Path, spec: PostprocessPassSpec) -> Path:
    cache_dir = output_dir / CACHE_DIRNAME
    if spec.cache_namespace:
        return cache_dir / spec.cache_namespace
    return cache_dir


def _prompt_content_hash(spec: PostprocessPassSpec, prompt: tuple[str, str]) -> str:
    if spec.prompt_hash_includes_pass:
        return stable_hash([spec.pass_name, prompt], length=16)
    return stable_hash(prompt, length=16)


def _print_prefix(spec: PostprocessPassSpec, batch_number: int, batch_count: int, batch: ChunkPostprocessBatch) -> str:
    if spec.pass_name == "single":
        return f"[{batch_number}/{batch_count}] {batch.batch_id}:"
    return f"[{spec.pass_name} {batch_number}/{batch_count}]"


def _pass_metadata(spec: PostprocessPassSpec, prompt_name: str) -> dict[str, Any]:
    if spec.pass_name == "single":
        return {}
    return {"pass_name": spec.pass_name, "prompt_name": prompt_name}


def run_postprocess_pass(
    args: argparse.Namespace,
    *,
    pass_name: str,
    inputs: list[ChunkPostprocessInput],
    output_dir: Path,
) -> PostprocessPassResult:
    """Runs one post-processing pass over selected chunks.

    The pass can be the legacy single-pass runner or one phase of the optimized
    quality/graph pipeline. Successful model responses are cached only after
    schema validation succeeds, which keeps failed or partial generations from
    becoming future replay baselines.
    """

    batches = build_batches(inputs, max_batch_chunks=args.max_batch_chunks, max_batch_chars=args.max_batch_chars)
    spec = postprocess_pass_spec(args, pass_name)
    prompt = load_prompt(args.prompt_dir, args.prompt_version, prompt_name=spec.prompt_name)
    prompt_content_hash = _prompt_content_hash(spec, prompt)
    cache = LLMResponseCache(_pass_cache_dir(output_dir, spec))
    generation_settings = generation_settings_for_pass(args, pass_name)
    llm = None if args.fake_llm else build_ollama_llm(
        args,
        num_predict=effective_num_predict(args, pass_name),
    )
    decisions: list[ChunkEnrichmentDecision] = []
    failures: list[PostprocessFailure] = []
    cache_hits = 0
    cache_misses = 0
    aborted = False
    for batch_number, batch in enumerate(batches, start=1):
        cache_key = cache_key_for_batch(
            batch,
            model_name=args.model_name,
            prompt_version=spec.prompt_version,
            prompt_content_hash=prompt_content_hash,
            generation_settings=generation_settings,
        )
        raw_response: Optional[str] = None
        response_metadata: dict[str, Any] = {}
        if not args.fake_llm and not args.force_refresh_cache:
            raw_response = cache.get(cache_key)
            if raw_response is not None:
                cache_hits += 1
            elif can_use_legacy_cache_key(args, pass_name):
                legacy_cache_key = cache_key_for_batch(
                    batch,
                    model_name=args.model_name,
                    prompt_version=spec.prompt_version,
                    prompt_content_hash=prompt_content_hash,
                )
                raw_response = cache.get(legacy_cache_key)
                if raw_response is not None:
                    cache_hits += 1
                    response_metadata = {
                        "migrated_from_legacy_cache": True,
                        "generation_settings": generation_settings,
                        **_pass_metadata(spec, spec.prompt_name),
                    }

        if raw_response is None:
            cache_misses += 1
            start = time.perf_counter()
            if args.fake_llm:
                raw_response = fake_llm_response_for_batch(batch)
                response_metadata = {
                    "provider": "fake_llm",
                    "latency_seconds": 0.0,
                    **_pass_metadata(spec, spec.prompt_name),
                }
                if spec.pass_name != "single":
                    response_metadata["generation_settings"] = generation_settings
            else:
                try:
                    raw_response, response_metadata = invoke_llm_batch(
                        llm,
                        prompt,
                        batch,
                        pass_name=pass_name,
                    )
                except Exception as exc:
                    error_type = "llm_usage_limit_error" if is_usage_limit_error(exc) else "llm_call_error"
                    failures.append(
                        make_failure(
                            error_type,
                            f"{type(exc).__name__}: {exc}",
                            batch=batch,
                            model_name=args.model_name,
                            prompt_version=spec.prompt_version,
                        )
                    )
                    print(f"{_print_prefix(spec, batch_number, len(batches), batch)} LLM call failed: {exc}")
                    if error_type == "llm_usage_limit_error":
                        if spec.pass_name == "single":
                            print("Usage limit reached; stopping postprocess loop so it can resume from cache later.")
                        else:
                            print(f"[{spec.pass_name}] Usage limit reached; stopping pass so it can resume from cache later.")
                        aborted = True
                        break
                    continue
            response_metadata["latency_seconds"] = round(time.perf_counter() - start, 3)
            response_metadata["generation_settings"] = generation_settings
            response_metadata.update(_pass_metadata(spec, spec.prompt_name))

        if is_empty_llm_response(raw_response):
            failures.append(
                make_failure(
                    "empty_llm_response",
                    "LLM returned an empty response; skipping repair to avoid a likely wasted retry.",
                    batch=batch,
                    raw_response=raw_response,
                    model_name=args.model_name,
                    prompt_version=spec.prompt_version,
                )
            )
            print(f"{_print_prefix(spec, batch_number, len(batches), batch)} empty LLM response")
            continue

        batch_decisions, batch_failures = validate_raw_response(
            raw_response,
            batch,
            model_name=args.model_name,
            prompt_version=spec.prompt_version,
        )

        if batch_failures and not args.fake_llm:
            try:
                repair_raw, repair_metadata = invoke_llm_repair(
                    llm,
                    prompt,
                    batch,
                    raw_response,
                    batch_failures,
                    pass_name=pass_name,
                )
                repair_decisions, repair_failures = validate_raw_response(
                    repair_raw,
                    batch,
                    model_name=args.model_name,
                    prompt_version=spec.prompt_version,
                )
                if repair_decisions and not repair_failures:
                    raw_response = repair_raw
                    response_metadata = {
                        "repair": True,
                        "generation_settings": generation_settings,
                        **_pass_metadata(spec, spec.prompt_name),
                        **repair_metadata,
                    }
                    batch_decisions = repair_decisions
                    batch_failures = []
                else:
                    batch_failures.extend(repair_failures)
            except Exception as exc:
                error_type = "llm_usage_limit_error" if is_usage_limit_error(exc) else "json_repair_error"
                batch_failures.append(
                    make_failure(
                        error_type,
                        f"{type(exc).__name__}: {exc}",
                        batch=batch,
                        raw_response=raw_response,
                        model_name=args.model_name,
                        prompt_version=spec.prompt_version,
                    )
                )
                if error_type == "llm_usage_limit_error":
                    if spec.pass_name == "single":
                        print("Usage limit reached during repair; stopping postprocess loop.")
                    else:
                        print(f"[{spec.pass_name}] Usage limit reached during repair; stopping pass.")
                    aborted = True

        if batch_decisions and not batch_failures and not args.fake_llm:
            cache.set(cache_key, raw_response, metadata=response_metadata)
        decisions.extend(batch_decisions)
        failures.extend(batch_failures)
        print(f"{_print_prefix(spec, batch_number, len(batches), batch)} {len(batch_decisions)} valid, {len(batch_failures)} failures")
        if aborted:
            break
    return PostprocessPassResult(decisions, failures, cache_hits, cache_misses, aborted)


def graph_worthy_inputs(
    quality_inputs: list[ChunkPostprocessInput],
    quality_decisions: list[ChunkEnrichmentDecision],
    quality_failures: list[PostprocessFailure],
) -> list[ChunkPostprocessInput]:
    """Select chunks whose quality decision allows a second graph-enrichment pass."""

    failed_chunk_ids = {failure.chunk_id for failure in quality_failures if failure.chunk_id}
    decisions_by_chunk = {decision.chunk_id: decision for decision in quality_decisions}
    selected: list[ChunkPostprocessInput] = []
    for item in quality_inputs:
        decision = decisions_by_chunk.get(item.chunk_id)
        if not decision or item.chunk_id in failed_chunk_ids:
            continue
        if decision.action in GRAPH_ACTIONS:
            selected.append(item)
    return selected


def merge_quality_and_graph_decisions(
    quality_decisions: list[ChunkEnrichmentDecision],
    graph_decisions: list[ChunkEnrichmentDecision],
) -> list[ChunkEnrichmentDecision]:
    """Overlays graph-pass concepts and relations onto quality-pass decisions.

    Quality remains the source of truth for filtering and cleaned text. The graph
    pass contributes semantic graph fields and may refine the summary, while the
    merged confidence preserves the more conservative of the two pass scores.
    """

    graph_by_chunk = {decision.chunk_id: decision for decision in graph_decisions}
    merged: list[ChunkEnrichmentDecision] = []
    for quality in quality_decisions:
        graph = graph_by_chunk.get(quality.chunk_id)
        if graph is None:
            merged.append(quality)
            continue
        payload = quality.model_dump(mode="json")
        payload.update(
            {
                "concepts": [concept.model_dump(mode="json") for concept in graph.concepts],
                "relations": [relation.model_dump(mode="json") for relation in graph.relations],
                "chunk_summary": graph.chunk_summary or quality.chunk_summary,
                "warnings": unique_preserving_order([*quality.warnings, *graph.warnings]),
                "confidence": min(quality.confidence, graph.confidence),
                "decision_id": f"decision_{stable_hash(['two_pass', quality.decision_id, graph.decision_id], length=24)}",
                "prompt_version": f"{quality.prompt_version}+graph:{graph.prompt_version}",
                "model_name": graph.model_name,
            }
        )
        merged.append(ChunkEnrichmentDecision.model_validate(payload))
    return merged


def validate_raw_concept_adjudication(
    raw_response: str,
    cluster: Any,
    *,
    model_name: str,
    prompt_version: str,
) -> tuple[Optional[ConceptAdjudicationResponse], list[ConceptAdjudicationFailure]]:
    try:
        parsed = parse_concept_adjudication_response(raw_response)
    except Exception as exc:
        return None, [
            make_concept_adjudication_failure(
                "json_parse_error",
                f"{type(exc).__name__}: {exc}",
                cluster=cluster,
                raw_response=raw_response,
                model_name=model_name,
                prompt_version=prompt_version,
            )
        ]
    errors = validate_concept_adjudication_response(parsed, cluster)
    if errors:
        return None, [
            make_concept_adjudication_failure(
                "validation_error",
                "; ".join(errors),
                cluster=cluster,
                raw_response=raw_response,
                model_name=model_name,
                prompt_version=prompt_version,
            )
        ]
    return parsed, []


def invoke_concept_adjudication(
    llm: Any,
    prompt: tuple[str, str],
    cluster: Any,
    resolution: ConceptResolution,
) -> tuple[str, dict[str, Any]]:
    _, system_prompt = prompt
    user_prompt = render_concept_adjudication_user_prompt(prompt, cluster, resolution)
    return invoke_messages(llm, system_prompt, user_prompt)


def render_concept_adjudication_user_prompt(
    prompt: tuple[str, str],
    cluster: Any,
    resolution: ConceptResolution,
) -> str:
    user_template, _ = prompt
    input_json = json.dumps(concept_adjudication_prompt_payload(cluster, resolution.mentions), ensure_ascii=False, indent=2)
    schema_json = json.dumps(concept_adjudication_schema_hint(), ensure_ascii=False, indent=2)
    return user_template.replace("{input_json}", input_json).replace("{response_schema}", schema_json)


def invoke_concept_adjudication_repair(
    llm: Any,
    prompt: tuple[str, str],
    cluster: Any,
    resolution: ConceptResolution,
    raw_response: str,
    failures: list[ConceptAdjudicationFailure],
) -> tuple[str, dict[str, Any]]:
    _, system_prompt = prompt
    input_json = json.dumps(concept_adjudication_prompt_payload(cluster, resolution.mentions), ensure_ascii=False, indent=2)
    schema_json = json.dumps(concept_adjudication_schema_hint(), ensure_ascii=False, indent=2)
    failure_summary = "; ".join(f"{failure.error_type}: {failure.message[:300]}" for failure in failures)
    repair_prompt = (
        "Generate a fresh compact JSON response for the same concept cluster. "
        "Do not continue the previous answer. Return JSON only. "
        "Every input mention_id must appear exactly once. "
        "If uncertain whether concepts are identical, split them.\n\n"
        f"Validation errors to avoid:\n{failure_summary}\n\n"
        f"Input cluster JSON:\n{input_json}\n\n"
        f"Field contract:\n{schema_json}\n\n"
        f"Previous response prefix for context only:\n{raw_response[:1200]}"
    )
    return invoke_messages(llm, system_prompt, repair_prompt)


def run_concept_adjudications(
    args: argparse.Namespace,
    resolution: ConceptResolution,
    *,
    llm: Any,
    prompt: tuple[str, str],
    cache: LLMResponseCache,
    model_name: str,
    prompt_version: str,
) -> tuple[list[ConceptAdjudicationResponse], list[ConceptAdjudicationFailure], int, int]:
    """Runs optional LLM adjudication for uncertain concept clusters.

    Deterministic concept resolution builds the full candidate registry first.
    This loop only asks the LLM to split or merge clusters that remain uncertain,
    and caches validated adjudication responses independently from chunk
    post-processing responses.
    """

    clusters = select_run_inputs(
        resolution.review_clusters,
        limit=args.concept_resolution_limit,
        allow_full_run=args.allow_full_run,
    )
    if len(clusters) < len(resolution.review_clusters):
        print(
            f"Concept adjudication limited to {len(clusters)} of "
            f"{len(resolution.review_clusters)} uncertain clusters."
        )

    adjudications: list[ConceptAdjudicationResponse] = []
    failures: list[ConceptAdjudicationFailure] = []
    cache_hits = 0
    cache_misses = 0
    generation_settings = generation_settings_for_pass(args, "concept_resolution")
    stopped_for_usage_limit = False
    for cluster_number, cluster in enumerate(clusters, start=1):
        rendered_prompt_chars = len(render_concept_adjudication_user_prompt(prompt, cluster, resolution))
        if rendered_prompt_chars > MAX_CONCEPT_ADJUDICATION_PROMPT_CHARS:
            failure = make_concept_adjudication_failure(
                "prompt_budget_exceeded",
                (
                    f"Rendered concept adjudication prompt is {rendered_prompt_chars} characters; "
                    f"limit is {MAX_CONCEPT_ADJUDICATION_PROMPT_CHARS}."
                ),
                cluster=cluster,
                model_name=model_name,
                prompt_version=prompt_version,
            )
            failures.append(failure)
            print(
                f"[concept {cluster_number}/{len(clusters)}] "
                f"{cluster.cluster_id}: skipped over-budget prompt ({rendered_prompt_chars} chars)"
            )
            continue

        cache_key = concept_adjudication_cache_key(
            cluster,
            model_name=model_name,
            prompt_version=prompt_version,
            generation_settings=generation_settings,
        )
        raw_response: Optional[str] = None
        response_metadata: dict[str, Any] = {}
        if not args.fake_llm and not args.force_refresh_cache:
            raw_response = cache.get(cache_key)
            if raw_response is not None:
                cache_hits += 1
            elif can_use_legacy_cache_key(args, "concept_resolution"):
                legacy_cache_key = concept_adjudication_cache_key(
                    cluster,
                    model_name=model_name,
                    prompt_version=prompt_version,
                )
                raw_response = cache.get(legacy_cache_key)
                if raw_response is not None:
                    cache_hits += 1
                    response_metadata = {
                        "migrated_from_legacy_cache": True,
                        "generation_settings": generation_settings,
                    }

        if raw_response is None:
            cache_misses += 1
            start = time.perf_counter()
            if args.fake_llm:
                raw_response = fake_concept_adjudication_response(cluster)
                response_metadata = {"provider": "fake_llm", "latency_seconds": 0.0}
            else:
                try:
                    raw_response, response_metadata = invoke_concept_adjudication(
                        llm,
                        prompt,
                        cluster,
                        resolution,
                    )
                except Exception as exc:
                    error_type = "llm_usage_limit_error" if is_usage_limit_error(exc) else "llm_call_error"
                    failure = make_concept_adjudication_failure(
                        error_type,
                        f"{type(exc).__name__}: {exc}",
                        cluster=cluster,
                        model_name=model_name,
                        prompt_version=prompt_version,
                    )
                    failures.append(failure)
                    print(
                        f"[concept {cluster_number}/{len(clusters)}] "
                        f"LLM call failed for {cluster.cluster_id}: {exc}"
                    )
                    if error_type == "llm_usage_limit_error":
                        print("Usage limit reached; stopping concept adjudication so it can resume from cache later.")
                        stopped_for_usage_limit = True
                        break
                    continue
            response_metadata["latency_seconds"] = round(time.perf_counter() - start, 3)
            response_metadata["generation_settings"] = generation_settings

        if is_empty_llm_response(raw_response):
            failures.append(
                make_concept_adjudication_failure(
                    "empty_llm_response",
                    "LLM returned an empty response; skipping repair to avoid a likely wasted retry.",
                    cluster=cluster,
                    raw_response=raw_response,
                    model_name=model_name,
                    prompt_version=prompt_version,
                )
            )
            print(f"[concept {cluster_number}/{len(clusters)}] {cluster.cluster_id}: empty LLM response")
            continue

        adjudication, cluster_failures = validate_raw_concept_adjudication(
            raw_response,
            cluster,
            model_name=model_name,
            prompt_version=prompt_version,
        )
        if cluster_failures and not args.fake_llm:
            try:
                repair_raw, repair_metadata = invoke_concept_adjudication_repair(
                    llm,
                    prompt,
                    cluster,
                    resolution,
                    raw_response,
                    cluster_failures,
                )
                repair_adjudication, repair_failures = validate_raw_concept_adjudication(
                    repair_raw,
                    cluster,
                    model_name=model_name,
                    prompt_version=prompt_version,
                )
                if repair_adjudication and not repair_failures:
                    raw_response = repair_raw
                    response_metadata = {
                        "repair": True,
                        "generation_settings": generation_settings,
                        **repair_metadata,
                    }
                    adjudication = repair_adjudication
                    cluster_failures = []
                else:
                    cluster_failures.extend(repair_failures)
            except Exception as exc:
                error_type = "llm_usage_limit_error" if is_usage_limit_error(exc) else "json_repair_error"
                cluster_failures.append(
                    make_concept_adjudication_failure(
                        error_type,
                        f"{type(exc).__name__}: {exc}",
                        cluster=cluster,
                        raw_response=raw_response,
                        model_name=model_name,
                        prompt_version=prompt_version,
                    )
                )
                if error_type == "llm_usage_limit_error":
                    print("Usage limit reached during concept repair; stopping concept adjudication.")
                    stopped_for_usage_limit = True

        if adjudication and not cluster_failures:
            adjudications.append(adjudication)
            if not args.fake_llm:
                cache.set(cache_key, raw_response, metadata=response_metadata)
        failures.extend(cluster_failures)
        print(
            f"[concept {cluster_number}/{len(clusters)}] {cluster.cluster_id}: "
            f"{1 if adjudication and not cluster_failures else 0} valid, {len(cluster_failures)} failures"
        )
        if stopped_for_usage_limit:
            break
    return adjudications, failures, cache_hits, cache_misses


def resolve_concepts(
    args: argparse.Namespace,
    output_dir: Path,
    decisions: list[ChunkEnrichmentDecision],
    *,
    concept_model_name: str | None = None,
) -> ConceptResolutionRunResult:
    """Build the concept registry and optionally enrich uncertain merges with LLM adjudication."""

    if args.concept_resolution_mode == "off":
        return ConceptResolutionRunResult(None, 0, 0)
    resolution = build_concept_resolution(decisions)
    concept_cache_hits = 0
    concept_cache_misses = 0
    print(
        "Concept resolution: "
        f"{len(resolution.registry_entries)} registry entries, "
        f"{len(resolution.review_clusters)} uncertain clusters."
    )
    if args.concept_resolution_mode == "llm" and resolution.review_clusters:
        resolved_model_name = concept_model_name or args.concept_resolution_model_name or args.model_name
        concept_prompt = load_prompt(
            args.prompt_dir,
            args.concept_resolution_prompt_version,
            prompt_name="remnote_concept_resolution",
        )
        concept_cache = LLMResponseCache(output_dir / CACHE_DIRNAME / "concept_resolution")
        concept_llm = None if args.fake_llm else build_ollama_llm(
            args,
            model_name=resolved_model_name,
            prompt_version=args.concept_resolution_prompt_version,
            num_predict=effective_num_predict(args, "concept_resolution"),
        )
        adjudications, adjudication_failures, concept_cache_hits, concept_cache_misses = run_concept_adjudications(
            args,
            resolution,
            llm=concept_llm,
            prompt=concept_prompt,
            cache=concept_cache,
            model_name=resolved_model_name,
            prompt_version=args.concept_resolution_prompt_version,
        )
        if adjudications:
            resolution = apply_concept_adjudications(resolution, adjudications)
        if adjudication_failures:
            resolution = resolution.model_copy(
                update={"adjudication_failures": [*resolution.adjudication_failures, *adjudication_failures]}
            )
    return ConceptResolutionRunResult(resolution, concept_cache_hits, concept_cache_misses)


def load_existing_postprocess_sidecars(
    input_dir: Path,
) -> tuple[list[ChunkPostprocessInput], list[ChunkEnrichmentDecision], list[PostprocessFailure]]:
    """Load prior sidecars for concept-resolution-only reruns."""

    input_dir = Path(input_dir)
    inputs = [ChunkPostprocessInput.model_validate(row) for row in read_jsonl(input_dir / INPUTS_FILENAME)]
    decisions = [ChunkEnrichmentDecision.model_validate(row) for row in read_jsonl(input_dir / DECISIONS_FILENAME)]
    failures = [PostprocessFailure.model_validate(row) for row in read_jsonl_if_exists(input_dir / FAILURES_FILENAME)]
    return inputs, decisions, failures


def infer_concept_resolution_model_name(args: argparse.Namespace, decisions: list[ChunkEnrichmentDecision]) -> str:
    """Chooses the model name recorded for concept-resolution-only sidecars.

    When the caller does not explicitly pass a model, reuse the most common
    model from existing decisions so regenerated concept sidecars do not look as
    if they came from a different run.
    """

    if args.concept_resolution_model_name:
        return args.concept_resolution_model_name
    if "--model-name" in sys.argv:
        return args.model_name
    counts = Counter(decision.model_name for decision in decisions if decision.model_name)
    if counts:
        return counts.most_common(1)[0][0]
    return args.model_name or DEFAULT_MODEL_NAME


def validate_run_bounds(args: argparse.Namespace) -> None:
    """Validate and normalize standalone runner limits before work begins."""

    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    if args.offset < 0:
        raise SystemExit("--offset must be >= 0")
    args.limit = resolve_run_limit(args.limit, allow_full_run=args.allow_full_run)
    if not args.allow_full_run and args.limit > DEFAULT_SMOKE_LIMIT:
        raise SystemExit(
            f"--limit defaults to a smoke-test ceiling of {DEFAULT_SMOKE_LIMIT}. "
            "Use --allow-full-run to process more candidates."
        )
    if args.limit == 0 and not args.allow_full_run:
        raise SystemExit("--limit 0 means all candidates and requires --allow-full-run")
    _validate_concept_resolution_limits(args)
    if args.dry_run and args.fake_llm:
        raise SystemExit("Choose either --dry-run or --fake-llm, not both.")


def validate_concept_resolution_only_args(args: argparse.Namespace) -> None:
    """Validates arguments for rerunning concept resolution against existing sidecars."""

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if input_dir == output_dir and not args.allow_in_place:
        raise SystemExit("--output-dir equals --input-dir. Use --allow-in-place to overwrite sidecars in place.")
    if not args.input_dir.exists():
        raise SystemExit(f"--input-dir does not exist: {args.input_dir}")
    if args.concept_resolution_mode == "off":
        raise SystemExit("--concept-resolution-only requires --concept-resolution-mode deterministic or llm")
    _validate_concept_resolution_limits(args)
    if args.dry_run:
        raise SystemExit("--dry-run is not supported with --concept-resolution-only")


def _validate_concept_resolution_limits(args: argparse.Namespace) -> None:
    if args.concept_resolution_limit is not None and args.concept_resolution_limit < 0:
        raise SystemExit("--concept-resolution-limit must be >= 0")
    if args.num_predict <= 0:
        raise SystemExit("--num-predict must be > 0")
    if args.concept_resolution_num_predict is not None and args.concept_resolution_num_predict <= 0:
        raise SystemExit("--concept-resolution-num-predict must be > 0")
    args.concept_resolution_limit = resolve_run_limit(
        args.concept_resolution_limit,
        allow_full_run=args.allow_full_run,
    )
    if args.concept_resolution_mode == "llm":
        if not args.allow_full_run and args.concept_resolution_limit > DEFAULT_SMOKE_LIMIT:
            raise SystemExit(
                f"--concept-resolution-limit defaults to a smoke-test ceiling of {DEFAULT_SMOKE_LIMIT}. "
                "Use --allow-full-run to adjudicate more clusters."
            )
        if args.concept_resolution_limit == 0 and not args.allow_full_run:
            raise SystemExit("--concept-resolution-limit 0 means all clusters and requires --allow-full-run")
