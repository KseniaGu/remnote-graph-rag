#!/usr/bin/env python
"""Run LLM sidecar post-processing for optimized RemNote parser outputs.

This script reads optimized parser IR JSONL files and writes reviewable sidecar
artifacts. It does not mutate Redis, Memgraph, vector stores, docstores, or the
parser IR.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, SystemMessage

from backend.configs.enums import ModelRoleType, PromptType
from backend.configs.models import OllamaSettings
from backend.data_processing.concept_registry import (
    DEFAULT_CONCEPT_RESOLUTION_PROMPT_VERSION,
    ConceptAdjudicationFailure,
    ConceptAdjudicationResponse,
    ConceptResolution,
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
    DEFAULT_CONCEPT_RESOLUTION_NUM_PREDICT,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_NUM_CTX,
    DEFAULT_LLM_NUM_PREDICT,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TOP_K,
    DEFAULT_LLM_TOP_P,
    DEFAULT_MAX_BATCH_CHARS,
    DEFAULT_MAX_BATCH_CHUNKS,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_SMOKE_LIMIT,
    FAILURES_FILENAME,
    INPUTS_FILENAME,
    ChunkEnrichmentDecision,
    ChunkPostprocessBatch,
    ChunkPostprocessInput,
    LLMResponseCache,
    PostprocessFailure,
    build_batches,
    batch_prompt_payload,
    cache_key_for_batch,
    fake_llm_response_for_batch,
    inputs_from_jsonl_dir,
    make_failure,
    parse_llm_response,
    read_jsonl,
    read_jsonl_if_exists,
    response_schema_hint,
    select_candidate_inputs,
    stable_hash,
    validate_and_enrich_response,
    write_sidecar_outputs,
)
from backend.utils.prompt_engine import PromptEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM sidecar post-processing for optimized RemNote parser chunks."
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory with optimized parser IR JSONL files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for sidecar postprocess outputs.")
    parser.add_argument(
        "--concept-resolution-only",
        action="store_true",
        help=(
            "Read existing LLM postprocess sidecars from --input-dir and rerun only concept resolution. "
            "In this mode --input-dir must contain llm_postprocess_decisions.jsonl."
        ),
    )
    parser.add_argument(
        "--allow-in-place",
        action="store_true",
        help="With --concept-resolution-only, allow --output-dir to equal --input-dir and overwrite sidecars.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of candidate chunks to process. Defaults to all candidates with "
            "--allow-full-run, otherwise the smoke-test limit."
        ),
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many candidates after applying exclusions. Useful for evaluating the next deterministic slice.",
    )
    parser.add_argument(
        "--exclude-output-dir",
        action="append",
        default=[],
        type=Path,
        help="Previous postprocess output dir whose llm_postprocess_inputs.jsonl chunk IDs should be skipped. Repeatable.",
    )
    parser.add_argument(
        "--allow-full-run",
        action="store_true",
        help="Allow processing more than the default smoke-test limit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build inputs/report only. Does not call the LLM or fake LLM.",
    )
    parser.add_argument(
        "--fake-llm",
        action="store_true",
        help="Use deterministic fake LLM responses for tests and local review.",
    )
    parser.add_argument(
        "--force-refresh-cache",
        action="store_true",
        help="Ignore cached LLM responses and call the model again.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Ollama model name.")
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION, help="Prompt version under remnote_postprocess.")
    parser.add_argument(
        "--concept-resolution-mode",
        choices=("off", "deterministic", "llm"),
        default="deterministic",
        help="Resolve local concept mentions into global concept registry IDs.",
    )
    parser.add_argument(
        "--concept-resolution-prompt-version",
        default=DEFAULT_CONCEPT_RESOLUTION_PROMPT_VERSION,
        help="Prompt version under remnote_concept_resolution.",
    )
    parser.add_argument(
        "--concept-resolution-model-name",
        default=None,
        help="Ollama model for uncertain concept adjudication. Defaults to --model-name.",
    )
    parser.add_argument(
        "--concept-resolution-limit",
        type=int,
        default=None,
        help=(
            "Maximum uncertain concept clusters to adjudicate with the LLM. Defaults to all clusters with "
            "--allow-full-run, otherwise the smoke-test limit."
        ),
    )
    parser.add_argument("--prompt-dir", type=Path, default=ROOT / "backend" / "llm" / "prompts")
    parser.add_argument("--max-batch-chunks", type=int, default=DEFAULT_MAX_BATCH_CHUNKS)
    parser.add_argument("--max-batch-chars", type=int, default=DEFAULT_MAX_BATCH_CHARS)
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_LLM_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_LLM_TOP_K)
    parser.add_argument("--top-p", type=float, default=DEFAULT_LLM_TOP_P)
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_LLM_NUM_CTX)
    parser.add_argument("--num-predict", type=int, default=DEFAULT_LLM_NUM_PREDICT)
    parser.add_argument(
        "--concept-resolution-num-predict",
        type=int,
        default=DEFAULT_CONCEPT_RESOLUTION_NUM_PREDICT,
        help="Override --num-predict for LLM concept adjudication calls.",
    )
    return parser.parse_args()


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
    return generation_settings_for_pass(args, pass_name) == legacy_generation_settings_for_args(args)


def is_usage_limit_error(value: object) -> bool:
    text = str(value).casefold()
    return (
        "status code: 429" in text
        or "status 429" in text
        or "weekly usage limit" in text
        or "rate limit" in text
    )


def is_empty_llm_response(raw_response: Optional[str]) -> bool:
    return raw_response is None or not raw_response.strip()


def main() -> int:
    args = parse_args()
    if args.concept_resolution_only:
        return run_concept_resolution_only(args)

    validate_run_bounds(args)

    all_inputs = inputs_from_jsonl_dir(args.input_dir)
    candidates = select_candidate_inputs(all_inputs)
    excluded_chunk_ids = load_excluded_chunk_ids(args.exclude_output_dir)
    candidate_pool = [item for item in candidates if item.chunk_id not in excluded_chunk_ids]
    selected_inputs = select_run_inputs(
        candidate_pool,
        limit=args.limit,
        offset=args.offset,
        allow_full_run=args.allow_full_run,
    )
    batches = build_batches(
        selected_inputs,
        max_batch_chunks=args.max_batch_chunks,
        max_batch_chars=args.max_batch_chars,
    )

    print(
        f"Loaded {len(all_inputs)} chunks; {len(candidates)} candidates before exclusions; "
        f"excluded {len(excluded_chunk_ids)} chunk IDs; selected {len(selected_inputs)} "
        f"candidates in {len(batches)} batches."
    )

    if args.dry_run:
        report = write_sidecar_outputs(
            args.output_dir,
            inputs=selected_inputs,
            decisions=[],
            failures=[],
        )
        print(f"Dry run complete. Report: {args.output_dir / 'llm_postprocess_report.md'}")
        print(json.dumps(report, indent=2))
        return 0

    prompt = load_prompt(args.prompt_dir, args.prompt_version)
    prompt_content_hash = stable_hash(prompt, length=16)
    cache = LLMResponseCache(args.output_dir / CACHE_DIRNAME)
    llm = None if args.fake_llm else build_ollama_llm(args)

    decisions: list[ChunkEnrichmentDecision] = []
    failures: list[PostprocessFailure] = []
    cache_hits = 0
    cache_misses = 0

    stopped_for_usage_limit = False
    for batch_number, batch in enumerate(batches, start=1):
        generation_settings = generation_settings_for_pass(args, "single")
        cache_key = cache_key_for_batch(
            batch,
            model_name=args.model_name,
            prompt_version=args.prompt_version,
            prompt_content_hash=prompt_content_hash,
            generation_settings=generation_settings,
        )
        raw_response: Optional[str] = None
        response_metadata: dict[str, Any] = {}
        if not args.fake_llm and not args.force_refresh_cache:
            raw_response = cache.get(cache_key)
            if raw_response is not None:
                cache_hits += 1
            elif can_use_legacy_cache_key(args, "single"):
                legacy_cache_key = cache_key_for_batch(
                    batch,
                    model_name=args.model_name,
                    prompt_version=args.prompt_version,
                    prompt_content_hash=prompt_content_hash,
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
                raw_response = fake_llm_response_for_batch(batch)
                response_metadata = {"provider": "fake_llm", "latency_seconds": 0.0}
            else:
                try:
                    raw_response, response_metadata = invoke_llm_batch(llm, prompt, batch, pass_name="single")
                except Exception as exc:
                    error_type = "llm_usage_limit_error" if is_usage_limit_error(exc) else "llm_call_error"
                    failures.append(
                        make_failure(
                            error_type,
                            f"{type(exc).__name__}: {exc}",
                            batch=batch,
                            model_name=args.model_name,
                            prompt_version=args.prompt_version,
                        )
                    )
                    print(f"[{batch_number}/{len(batches)}] LLM call failed for {batch.batch_id}: {exc}")
                    if error_type == "llm_usage_limit_error":
                        print("Usage limit reached; stopping postprocess loop so it can resume from cache later.")
                        stopped_for_usage_limit = True
                        break
                    continue
            response_metadata["latency_seconds"] = round(time.perf_counter() - start, 3)
            response_metadata["generation_settings"] = generation_settings

        if is_empty_llm_response(raw_response):
            failures.append(
                make_failure(
                    "empty_llm_response",
                    "LLM returned an empty response; skipping repair to avoid a likely wasted retry.",
                    batch=batch,
                    raw_response=raw_response,
                    model_name=args.model_name,
                    prompt_version=args.prompt_version,
                )
            )
            print(f"[{batch_number}/{len(batches)}] {batch.batch_id}: empty LLM response")
            continue

        batch_decisions, batch_failures = validate_raw_response(
            raw_response,
            batch,
            model_name=args.model_name,
            prompt_version=args.prompt_version,
        )

        if batch_failures and not args.fake_llm:
            try:
                repair_raw, repair_metadata = invoke_llm_repair(
                    llm,
                    prompt,
                    batch,
                    raw_response,
                    batch_failures,
                    pass_name="single",
                )
                repair_decisions, repair_failures = validate_raw_response(
                    repair_raw,
                    batch,
                    model_name=args.model_name,
                    prompt_version=args.prompt_version,
                )
                if repair_decisions and not repair_failures:
                    raw_response = repair_raw
                    response_metadata = {
                        "repair": True,
                        "generation_settings": generation_settings,
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
                        prompt_version=args.prompt_version,
                    )
                )
                if error_type == "llm_usage_limit_error":
                    print("Usage limit reached during repair; stopping postprocess loop.")
                    stopped_for_usage_limit = True

        if batch_decisions and not batch_failures and not args.fake_llm:
            cache.set(cache_key, raw_response, metadata=response_metadata)

        decisions.extend(batch_decisions)
        failures.extend(batch_failures)
        print(
            f"[{batch_number}/{len(batches)}] {batch.batch_id}: "
            f"{len(batch_decisions)} valid, {len(batch_failures)} failures"
        )
        if stopped_for_usage_limit:
            break

    concept_resolution: ConceptResolution | None = None
    concept_cache_hits = 0
    concept_cache_misses = 0
    if args.concept_resolution_mode != "off":
        concept_resolution = build_concept_resolution(decisions)
        print(
            "Concept resolution: "
            f"{len(concept_resolution.registry_entries)} registry entries, "
            f"{len(concept_resolution.review_clusters)} uncertain clusters."
        )
        if args.concept_resolution_mode == "llm" and concept_resolution.review_clusters:
            concept_model_name = args.concept_resolution_model_name or args.model_name
            concept_prompt = load_prompt(
                args.prompt_dir,
                args.concept_resolution_prompt_version,
                prompt_name="remnote_concept_resolution",
            )
            concept_cache = LLMResponseCache(args.output_dir / CACHE_DIRNAME / "concept_resolution")
            concept_llm = None if args.fake_llm else build_ollama_llm(
                args,
                model_name=concept_model_name,
                prompt_version=args.concept_resolution_prompt_version,
                num_predict=effective_num_predict(args, "concept_resolution"),
            )
            adjudications, adjudication_failures, concept_cache_hits, concept_cache_misses = run_concept_adjudications(
                args,
                concept_resolution,
                llm=concept_llm,
                prompt=concept_prompt,
                cache=concept_cache,
                model_name=concept_model_name,
                prompt_version=args.concept_resolution_prompt_version,
            )
            if adjudications:
                concept_resolution = apply_concept_adjudications(concept_resolution, adjudications)
            if adjudication_failures:
                concept_resolution = concept_resolution.model_copy(
                    update={
                        "adjudication_failures": [
                            *concept_resolution.adjudication_failures,
                            *adjudication_failures,
                        ]
                    }
                )

    report = write_sidecar_outputs(
        args.output_dir,
        inputs=selected_inputs,
        decisions=decisions,
        failures=failures,
        concept_resolution=concept_resolution,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        concept_cache_hits=concept_cache_hits,
        concept_cache_misses=concept_cache_misses,
    )
    print(f"Postprocess complete. Report: {args.output_dir / 'llm_postprocess_report.md'}")
    print(json.dumps(report, indent=2))
    concept_failures = concept_resolution.adjudication_failures if concept_resolution else []
    return 1 if failures or concept_failures else 0


def run_concept_resolution_only(args: argparse.Namespace) -> int:
    validate_concept_resolution_only_args(args)
    inputs, decisions, failures = load_existing_postprocess_sidecars(args.input_dir)
    concept_resolution = build_concept_resolution(decisions)
    print(
        f"Loaded {len(inputs)} inputs, {len(decisions)} decisions, {len(failures)} existing failures. "
        f"Deterministic concept resolution produced {len(concept_resolution.registry_entries)} registry entries "
        f"and {len(concept_resolution.review_clusters)} uncertain clusters."
    )

    concept_cache_hits = 0
    concept_cache_misses = 0
    if args.concept_resolution_mode == "llm" and concept_resolution.review_clusters:
        concept_model_name = infer_concept_resolution_model_name(args, decisions)
        concept_prompt = load_prompt(
            args.prompt_dir,
            args.concept_resolution_prompt_version,
            prompt_name="remnote_concept_resolution",
        )
        concept_cache = LLMResponseCache(args.output_dir / CACHE_DIRNAME / "concept_resolution")
        concept_llm = None if args.fake_llm else build_ollama_llm(
            args,
            model_name=concept_model_name,
            prompt_version=args.concept_resolution_prompt_version,
            num_predict=effective_num_predict(args, "concept_resolution"),
        )
        adjudications, adjudication_failures, concept_cache_hits, concept_cache_misses = run_concept_adjudications(
            args,
            concept_resolution,
            llm=concept_llm,
            prompt=concept_prompt,
            cache=concept_cache,
            model_name=concept_model_name,
            prompt_version=args.concept_resolution_prompt_version,
        )
        if adjudications:
            concept_resolution = apply_concept_adjudications(concept_resolution, adjudications)
        if adjudication_failures:
            concept_resolution = concept_resolution.model_copy(
                update={
                    "adjudication_failures": [
                        *concept_resolution.adjudication_failures,
                        *adjudication_failures,
                    ]
                }
            )

    report = write_sidecar_outputs(
        args.output_dir,
        inputs=inputs,
        decisions=decisions,
        failures=failures,
        concept_resolution=concept_resolution,
        concept_cache_hits=concept_cache_hits,
        concept_cache_misses=concept_cache_misses,
    )
    print(f"Concept resolution complete. Report: {args.output_dir / 'llm_postprocess_report.md'}")
    print(json.dumps(report, indent=2))
    return 1 if concept_resolution.adjudication_failures else 0


def validate_concept_resolution_only_args(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if input_dir == output_dir and not args.allow_in_place:
        raise SystemExit("--output-dir equals --input-dir. Use --allow-in-place to overwrite sidecars in place.")
    if not args.input_dir.exists():
        raise SystemExit(f"--input-dir does not exist: {args.input_dir}")
    if args.concept_resolution_mode == "off":
        raise SystemExit("--concept-resolution-only requires --concept-resolution-mode deterministic or llm")
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
    if args.dry_run:
        raise SystemExit("--dry-run is not supported with --concept-resolution-only")


def load_existing_postprocess_sidecars(
    input_dir: Path,
) -> tuple[list[ChunkPostprocessInput], list[ChunkEnrichmentDecision], list[PostprocessFailure]]:
    input_dir = Path(input_dir)
    inputs = [ChunkPostprocessInput.model_validate(row) for row in read_jsonl(input_dir / INPUTS_FILENAME)]
    decisions = [ChunkEnrichmentDecision.model_validate(row) for row in read_jsonl(input_dir / DECISIONS_FILENAME)]
    failures = [PostprocessFailure.model_validate(row) for row in read_jsonl_if_exists(input_dir / FAILURES_FILENAME)]
    return inputs, decisions, failures


def infer_concept_resolution_model_name(args: argparse.Namespace, decisions: list[ChunkEnrichmentDecision]) -> str:
    if args.concept_resolution_model_name:
        return args.concept_resolution_model_name
    if "--model-name" in sys.argv:
        return args.model_name
    counts = Counter(decision.model_name for decision in decisions if decision.model_name)
    if counts:
        return counts.most_common(1)[0][0]
    return args.model_name or DEFAULT_MODEL_NAME


def validate_run_bounds(args: argparse.Namespace) -> None:
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
    if args.dry_run and args.fake_llm:
        raise SystemExit("Choose either --dry-run or --fake-llm, not both.")


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
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                chunk_id = row.get("chunk_id")
                if chunk_id:
                    excluded.add(chunk_id)
    return excluded


def load_prompt(prompt_dir: Path, prompt_version: str, *, prompt_name: str = "remnote_postprocess") -> tuple[str, str]:
    user_template, system_payload = PromptEngine(prompt_dir).render(
        PromptType.learner_workflow,
        ModelRoleType.orchestrator,
        prompt_version,
        prompt_name,
    )
    if not isinstance(system_payload, dict):
        return user_template, str(system_payload)
    return user_template, str(system_payload.get("system_instruction", ""))


def build_ollama_llm(
    args: argparse.Namespace,
    *,
    model_name: Optional[str] = None,
    prompt_version: Optional[str] = None,
    num_predict: Optional[int] = None,
) -> Any:
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
        user_template
        .replace("{input_json}", input_json)
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
    system_prompt = system_prompt.replace("{pass_name}", pass_name)
    input_json = json.dumps(batch_prompt_payload(batch), ensure_ascii=False, indent=2)
    schema_json = json.dumps(response_schema_hint(pass_name=pass_name), ensure_ascii=False, indent=2)
    failure_summary = "; ".join(f"{failure.error_type}: {failure.message[:300]}" for failure in failures)
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
    return invoke_messages(llm, system_prompt, repair_prompt)


def invoke_messages(llm: Any, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    content = getattr(response, "content", response)
    raw_text = content_to_text(content)
    metadata = {
        "response_metadata": getattr(response, "response_metadata", {}),
        "usage_metadata": getattr(response, "usage_metadata", {}),
    }
    return raw_text, metadata


def content_to_text(content: Any) -> str:
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


def invoke_concept_adjudication(
    llm: Any,
    prompt: tuple[str, str],
    cluster: Any,
    resolution: ConceptResolution,
) -> tuple[str, dict[str, Any]]:
    user_template, system_prompt = prompt
    input_json = json.dumps(concept_adjudication_prompt_payload(cluster, resolution.mentions), ensure_ascii=False, indent=2)
    schema_json = json.dumps(concept_adjudication_schema_hint(), ensure_ascii=False, indent=2)
    user_prompt = user_template.replace("{input_json}", input_json).replace("{response_schema}", schema_json)
    return invoke_messages(llm, system_prompt, user_prompt)


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


def validate_raw_response(
    raw_response: str,
    batch: ChunkPostprocessBatch,
    *,
    model_name: str,
    prompt_version: str,
) -> tuple[list[ChunkEnrichmentDecision], list[PostprocessFailure]]:
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


if __name__ == "__main__":
    raise SystemExit(main())
