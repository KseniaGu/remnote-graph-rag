#!/usr/bin/env python
"""Run the optimized parser plus two-pass LLM postprocess sidecar pipeline.

This is an experimental production-candidate pipeline. It writes staging parser
storage/IR and LLM sidecars, but does not mutate the legacy production graph path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from backend.configs.paths import PathSettings
from backend.configs.storage import LocalStorageSettings, StorageSettings
from backend.data_processing.concept_registry import (
    DEFAULT_CONCEPT_RESOLUTION_PROMPT_VERSION,
    ConceptResolution,
    apply_concept_adjudications,
    build_concept_resolution,
)
from backend.data_processing.llm_postprocess import (
    CACHE_DIRNAME,
    DEFAULT_CONCEPT_RESOLUTION_NUM_PREDICT,
    DEFAULT_GRAPH_NUM_PREDICT,
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
    DEFAULT_QUALITY_NUM_PREDICT,
    DEFAULT_SMOKE_LIMIT,
    ChunkAction,
    ChunkEnrichmentDecision,
    ChunkPostprocessBatch,
    ChunkPostprocessInput,
    LLMResponseCache,
    PostprocessFailure,
    build_batches,
    cache_key_for_batch,
    fake_llm_response_for_batch,
    inputs_from_jsonl_dir,
    make_failure,
    select_candidate_inputs,
    stable_hash,
    unique_preserving_order,
    write_json,
    write_sidecar_outputs,
)
from backend.data_processing.parser_optimized import RemNoteParserOptimized
from run_llm_postprocess import (
    build_ollama_llm,
    can_use_legacy_cache_key,
    effective_num_predict,
    generation_settings_for_pass,
    invoke_llm_batch,
    invoke_llm_repair,
    is_empty_llm_response,
    is_usage_limit_error,
    load_prompt,
    run_concept_adjudications,
    resolve_run_limit,
    select_run_inputs,
    validate_raw_response,
)

GRAPH_ACTIONS = {
    ChunkAction.KEEP,
    ChunkAction.KEEP_WITH_CLEANED_TEXT,
    ChunkAction.METADATA_ONLY,
    ChunkAction.GRAPH_ONLY,
}
MANIFEST_FILENAME = "optimized_postprocess_pipeline_manifest.json"
PASS_PROMPT_NAMES = {
    "quality": "remnote_postprocess_quality",
    "graph": "remnote_postprocess_graph",
}
DEFAULT_POSTPROCESS_PROMPT_NAME = "remnote_postprocess"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optimized parser and LLM postprocess pipeline.")
    parser.add_argument("--raw-data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parsed-pdfs-dir", type=Path, default=None)
    parser.add_argument("--parsed-images-dir", type=Path, default=None)
    parser.add_argument("--parsed-texts-dir", type=Path, default=None)
    parser.add_argument("--staging-storage-dir", type=Path, default=None)
    parser.add_argument("--optimized-ir-dir", type=Path, default=None)
    parser.add_argument("--postprocess-dir", type=Path, default=None)
    parser.add_argument("--skip-parse", action="store_true")
    parser.add_argument("--copy-existing-artifacts", action="store_true")
    parser.add_argument(
        "--prepare-external-artifacts",
        dest="prepare_external_artifacts",
        action="store_true",
        default=True,
        help="Download and parse missing external artifacts before building optimized IR (default).",
    )
    parser.add_argument(
        "--skip-external-artifacts",
        dest="prepare_external_artifacts",
        action="store_false",
        help="Do not download/OCR missing external artifacts; unresolved resources remain metadata only.",
    )
    parser.add_argument("--force-rebuild-staging", action="store_true")
    parser.add_argument("--coverage", choices=("two-pass", "all", "candidates"), default="two-pass")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum chunks to process. Defaults to all selected chunks with --allow-full-run, "
            "otherwise the smoke-test limit."
        ),
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--allow-full-run", action="store_true")
    parser.add_argument("--fake-llm", action="store_true")
    parser.add_argument("--force-refresh-cache", action="store_true")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--concept-resolution-mode", choices=("off", "deterministic", "llm"), default="llm")
    parser.add_argument("--concept-resolution-prompt-version", default=DEFAULT_CONCEPT_RESOLUTION_PROMPT_VERSION)
    parser.add_argument("--concept-resolution-model-name", default=None)
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
        "--quality-num-predict",
        type=int,
        default=DEFAULT_QUALITY_NUM_PREDICT,
        help="Override --num-predict for the quality pass.",
    )
    parser.add_argument(
        "--graph-num-predict",
        type=int,
        default=DEFAULT_GRAPH_NUM_PREDICT,
        help="Override --num-predict for the graph pass.",
    )
    parser.add_argument(
        "--concept-resolution-num-predict",
        type=int,
        default=DEFAULT_CONCEPT_RESOLUTION_NUM_PREDICT,
        help="Override --num-predict for LLM concept adjudication calls.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    if args.offset < 0:
        raise SystemExit("--offset must be >= 0")
    args.limit = resolve_run_limit(args.limit, allow_full_run=args.allow_full_run)
    if args.limit == 0 and not args.allow_full_run:
        raise SystemExit("--limit 0 means all selected chunks and requires --allow-full-run")
    if not args.allow_full_run and args.limit > DEFAULT_SMOKE_LIMIT:
        raise SystemExit(f"Use --allow-full-run to process more than {DEFAULT_SMOKE_LIMIT} chunks")
    if args.concept_resolution_limit is not None and args.concept_resolution_limit < 0:
        raise SystemExit("--concept-resolution-limit must be >= 0")
    args.concept_resolution_limit = resolve_run_limit(
        args.concept_resolution_limit,
        allow_full_run=args.allow_full_run,
    )
    if args.concept_resolution_mode == "llm" and args.concept_resolution_limit == 0 and not args.allow_full_run:
        raise SystemExit("--concept-resolution-limit 0 means all clusters and requires --allow-full-run")
    if args.num_predict <= 0:
        raise SystemExit("--num-predict must be > 0")
    for name in ("quality_num_predict", "graph_num_predict", "concept_resolution_num_predict"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be > 0")


def make_staging_settings(args: argparse.Namespace) -> tuple[PathSettings, StorageSettings]:
    output_root = args.output_root.expanduser().resolve()
    staging_storage_dir = (args.staging_storage_dir or output_root / "staging_storage").expanduser().resolve()
    defaults = PathSettings()
    path_settings = PathSettings(
        raw_data_dir=args.raw_data_dir.expanduser().resolve(),
        parsed_pdfs_dir=args.parsed_pdfs_dir or output_root / "parsed_pdfs",
        parsed_images_dir=args.parsed_images_dir or output_root / "parsed_images",
        parsed_texts_dir=args.parsed_texts_dir or output_root / "parsed_texts",
        local_storage_dir=staging_storage_dir,
        prompts_dir=defaults.prompts_dir,
    )
    local = LocalStorageSettings(storage_path=staging_storage_dir)
    storage_settings = StorageSettings(
        document_storage=local,
        index_storage=local,
        vector_storage=local,
        property_graph_storage=local,
    )
    return path_settings, storage_settings


def run_optimized_parse(args: argparse.Namespace) -> Path:
    path_settings, storage_settings = make_staging_settings(args)
    parser = RemNoteParserOptimized(
        path_settings,
        storage_settings,
        prepare_external_artifacts=args.prepare_external_artifacts,
        copy_existing_artifacts=args.copy_existing_artifacts,
        force_rebuild=args.force_rebuild_staging,
        write_ir=True,
    )
    if not args.skip_parse:
        parser.run()
    ir_dir = args.optimized_ir_dir or parser.optimized_ir_dir
    if not (ir_dir / "retrieval_chunks.jsonl").exists():
        raise FileNotFoundError(f"Optimized IR retrieval chunks not found: {ir_dir}")
    return ir_dir


def select_inputs(args: argparse.Namespace, all_inputs: list[ChunkPostprocessInput]) -> list[ChunkPostprocessInput]:
    pool = select_candidate_inputs(all_inputs) if args.coverage == "candidates" else list(all_inputs)
    return select_run_inputs(pool, limit=args.limit, offset=args.offset, allow_full_run=args.allow_full_run)


def prompt_file_for_name(prompt_dir: Path, prompt_name: str, prompt_version: str) -> Path:
    return prompt_dir / "learner_workflow" / "orchestrator" / prompt_name / f"{prompt_version}.yaml"


def resolved_prompt_name_for_pass(args: argparse.Namespace, pass_name: str) -> str:
    prompt_name = PASS_PROMPT_NAMES.get(pass_name, DEFAULT_POSTPROCESS_PROMPT_NAME)
    if prompt_name == DEFAULT_POSTPROCESS_PROMPT_NAME:
        return prompt_name
    if prompt_file_for_name(args.prompt_dir, prompt_name, args.prompt_version).exists():
        return prompt_name
    return DEFAULT_POSTPROCESS_PROMPT_NAME


def load_prompt_for_pass(args: argparse.Namespace, pass_name: str) -> tuple[tuple[str, str], str]:
    prompt_name = resolved_prompt_name_for_pass(args, pass_name)
    return load_prompt(args.prompt_dir, args.prompt_version, prompt_name=prompt_name), prompt_name


def run_postprocess_pass(
    args: argparse.Namespace,
    *,
    pass_name: str,
    inputs: list[ChunkPostprocessInput],
    output_dir: Path,
) -> tuple[list[ChunkEnrichmentDecision], list[PostprocessFailure], int, int, bool]:
    batches = build_batches(inputs, max_batch_chunks=args.max_batch_chunks, max_batch_chars=args.max_batch_chars)
    prompt, prompt_name = load_prompt_for_pass(args, pass_name)
    prompt_content_hash = stable_hash([pass_name, prompt], length=16)
    pass_prompt_version = f"{args.prompt_version}:{pass_name}"
    cache = LLMResponseCache(output_dir / CACHE_DIRNAME / pass_name)
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
            prompt_version=pass_prompt_version,
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
                    prompt_version=pass_prompt_version,
                    prompt_content_hash=prompt_content_hash,
                )
                raw_response = cache.get(legacy_cache_key)
                if raw_response is not None:
                    cache_hits += 1
                    response_metadata = {
                        "migrated_from_legacy_cache": True,
                        "prompt_name": prompt_name,
                        "generation_settings": generation_settings,
                    }
        if raw_response is None:
            cache_misses += 1
            start = time.perf_counter()
            if args.fake_llm:
                raw_response = fake_llm_response_for_batch(batch)
                response_metadata = {
                    "provider": "fake_llm",
                    "latency_seconds": 0.0,
                    "pass_name": pass_name,
                    "prompt_name": prompt_name,
                    "generation_settings": generation_settings,
                }
            else:
                try:
                    raw_response, response_metadata = invoke_llm_batch(llm, prompt, batch, pass_name=pass_name)
                except Exception as exc:
                    error_type = "llm_usage_limit_error" if is_usage_limit_error(exc) else "llm_call_error"
                    failures.append(make_failure(
                        error_type,
                        f"{type(exc).__name__}: {exc}",
                        batch=batch,
                        model_name=args.model_name,
                        prompt_version=pass_prompt_version,
                    ))
                    print(f"[{pass_name} {batch_number}/{len(batches)}] LLM call failed: {exc}")
                    if error_type == "llm_usage_limit_error":
                        print(f"[{pass_name}] Usage limit reached; stopping pass so it can resume from cache later.")
                        aborted = True
                        break
                    continue
            response_metadata["latency_seconds"] = round(time.perf_counter() - start, 3)
            response_metadata["generation_settings"] = generation_settings
            response_metadata["prompt_name"] = prompt_name
        if is_empty_llm_response(raw_response):
            failures.append(make_failure(
                "empty_llm_response",
                "LLM returned an empty response; skipping repair to avoid a likely wasted retry.",
                batch=batch,
                raw_response=raw_response,
                model_name=args.model_name,
                prompt_version=pass_prompt_version,
            ))
            print(f"[{pass_name} {batch_number}/{len(batches)}] empty LLM response")
            continue
        batch_decisions, batch_failures = validate_raw_response(
            raw_response,
            batch,
            model_name=args.model_name,
            prompt_version=pass_prompt_version,
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
                    prompt_version=pass_prompt_version,
                )
                if repair_decisions and not repair_failures:
                    raw_response = repair_raw
                    response_metadata = {
                        "repair": True,
                        "pass_name": pass_name,
                        "prompt_name": prompt_name,
                        "generation_settings": generation_settings,
                        **repair_metadata,
                    }
                    batch_decisions = repair_decisions
                    batch_failures = []
                else:
                    batch_failures.extend(repair_failures)
            except Exception as exc:
                error_type = "llm_usage_limit_error" if is_usage_limit_error(exc) else "json_repair_error"
                batch_failures.append(make_failure(
                    error_type,
                    f"{type(exc).__name__}: {exc}",
                    batch=batch,
                    raw_response=raw_response,
                    model_name=args.model_name,
                    prompt_version=pass_prompt_version,
                ))
                if error_type == "llm_usage_limit_error":
                    print(f"[{pass_name}] Usage limit reached during repair; stopping pass.")
                    aborted = True
        if batch_decisions and not batch_failures and not args.fake_llm:
            cache.set(cache_key, raw_response, metadata=response_metadata)
        decisions.extend(batch_decisions)
        failures.extend(batch_failures)
        print(f"[{pass_name} {batch_number}/{len(batches)}] {len(batch_decisions)} valid, {len(batch_failures)} failures")
        if aborted:
            break
    return decisions, failures, cache_hits, cache_misses, aborted


def graph_worthy_inputs(
    quality_inputs: list[ChunkPostprocessInput],
    quality_decisions: list[ChunkEnrichmentDecision],
    quality_failures: list[PostprocessFailure],
) -> list[ChunkPostprocessInput]:
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


def resolve_concepts(
    args: argparse.Namespace,
    output_dir: Path,
    decisions: list[ChunkEnrichmentDecision],
) -> tuple[Optional[ConceptResolution], int, int]:
    if args.concept_resolution_mode == "off":
        return None, 0, 0
    resolution = build_concept_resolution(decisions)
    concept_cache_hits = 0
    concept_cache_misses = 0
    print(
        "Concept resolution: "
        f"{len(resolution.registry_entries)} registry entries, "
        f"{len(resolution.review_clusters)} uncertain clusters."
    )
    if args.concept_resolution_mode == "llm" and resolution.review_clusters:
        concept_model_name = args.concept_resolution_model_name or args.model_name
        concept_prompt = load_prompt(
            args.prompt_dir,
            args.concept_resolution_prompt_version,
            prompt_name="remnote_concept_resolution",
        )
        concept_cache = LLMResponseCache(output_dir / CACHE_DIRNAME / "concept_resolution")
        concept_llm = None if args.fake_llm else build_ollama_llm(
            args,
            model_name=concept_model_name,
            prompt_version=args.concept_resolution_prompt_version,
            num_predict=effective_num_predict(args, "concept_resolution"),
        )
        adjudications, adjudication_failures, concept_cache_hits, concept_cache_misses = run_concept_adjudications(
            args,
            resolution,
            llm=concept_llm,
            prompt=concept_prompt,
            cache=concept_cache,
            model_name=concept_model_name,
            prompt_version=args.concept_resolution_prompt_version,
        )
        if adjudications:
            resolution = apply_concept_adjudications(resolution, adjudications)
        if adjudication_failures:
            resolution = resolution.model_copy(
                update={"adjudication_failures": [*resolution.adjudication_failures, *adjudication_failures]}
            )
    return resolution, concept_cache_hits, concept_cache_misses


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output_root = args.output_root.expanduser().resolve()
    output_dir = (args.postprocess_dir or args.output_root / "llm_postprocess").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ir_dir = run_optimized_parse(args)
    all_inputs = inputs_from_jsonl_dir(ir_dir)
    selected_inputs = select_inputs(args, all_inputs)
    print(f"Loaded {len(all_inputs)} chunks from {ir_dir}; selected {len(selected_inputs)} for {args.coverage} coverage.")

    quality_decisions, quality_failures, quality_hits, quality_misses, quality_aborted = run_postprocess_pass(
        args,
        pass_name="quality",
        inputs=selected_inputs,
        output_dir=output_dir,
    )
    graph_decisions: list[ChunkEnrichmentDecision] = []
    graph_failures: list[PostprocessFailure] = []
    graph_hits = 0
    graph_misses = 0
    graph_aborted = False
    if args.coverage == "two-pass" and not quality_aborted:
        graph_inputs = graph_worthy_inputs(selected_inputs, quality_decisions, quality_failures)
        print(f"Graph pass selected {len(graph_inputs)} graph-worthy chunks.")
        graph_decisions, graph_failures, graph_hits, graph_misses, graph_aborted = run_postprocess_pass(
            args,
            pass_name="graph",
            inputs=graph_inputs,
            output_dir=output_dir,
        )
        final_decisions = merge_quality_and_graph_decisions(quality_decisions, graph_decisions)
    else:
        if quality_aborted and args.coverage == "two-pass":
            print("Skipping graph pass because quality pass stopped early.")
        final_decisions = quality_decisions

    failures = [*quality_failures, *graph_failures]
    original_concept_resolution_mode = args.concept_resolution_mode
    if (quality_aborted or graph_aborted) and args.concept_resolution_mode == "llm":
        print("Skipping LLM concept adjudication because an earlier LLM pass stopped early.")
        args.concept_resolution_mode = "deterministic"
    concept_resolution, concept_hits, concept_misses = resolve_concepts(args, output_dir, final_decisions)
    args.concept_resolution_mode = original_concept_resolution_mode
    report = write_sidecar_outputs(
        output_dir,
        inputs=selected_inputs,
        decisions=final_decisions,
        failures=failures,
        concept_resolution=concept_resolution,
        cache_hits=quality_hits + graph_hits,
        cache_misses=quality_misses + graph_misses,
        concept_cache_hits=concept_hits,
        concept_cache_misses=concept_misses,
    )
    manifest = {
        "schema_version": "1.0",
        "coverage": args.coverage,
        "optimized_ir_dir": str(ir_dir),
        "postprocess_dir": str(output_dir),
        "selected_input_count": len(selected_inputs),
        "quality_decision_count": len(quality_decisions),
        "graph_decision_count": len(graph_decisions),
        "final_decision_count": len(final_decisions),
        "failure_record_count": len(failures),
        "aborted": quality_aborted or graph_aborted,
        "aborted_passes": [
            name
            for name, aborted in (("quality", quality_aborted), ("graph", graph_aborted))
            if aborted
        ],
        "prompt_names": {
            "quality": resolved_prompt_name_for_pass(args, "quality"),
            "graph": resolved_prompt_name_for_pass(args, "graph"),
            "concept_resolution": "remnote_concept_resolution",
        },
        "generation_settings": {
            "quality": generation_settings_for_pass(args, "quality"),
            "graph": generation_settings_for_pass(args, "graph"),
            "concept_resolution": generation_settings_for_pass(args, "concept_resolution"),
        },
        "report": report,
    }
    write_json(output_dir / MANIFEST_FILENAME, manifest)
    print(f"Postprocess pipeline complete. Report: {output_dir / 'llm_postprocess_report.md'}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    concept_failures = concept_resolution.adjudication_failures if concept_resolution else []
    return 1 if concept_failures or quality_aborted or graph_aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
