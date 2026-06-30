#!/usr/bin/env python
"""Run the optimized parser plus two-pass LLM postprocess sidecar pipeline.

This is an experimental production-candidate pipeline. It writes staging parser
storage/IR and LLM sidecars, but does not mutate the legacy production graph path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.configs.paths import PathSettings
from backend.configs.storage import LocalStorageSettings, StorageSettings
from backend.data_processing.concept_registry import DEFAULT_CONCEPT_RESOLUTION_PROMPT_VERSION
from backend.data_processing.llm_postprocess import (
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
    inputs_from_jsonl_dir,
    select_candidate_inputs,
    write_json,
    write_sidecar_outputs,
)
from backend.data_processing.llm_postprocess_runner import (
    generation_settings_for_pass,
    graph_worthy_inputs,
    merge_quality_and_graph_decisions,
    resolve_concepts,
    resolve_run_limit,
    resolved_prompt_name_for_pass,
    run_postprocess_pass,
    select_run_inputs,
)
from backend.data_processing.parser_optimized import RemNoteParserOptimized


MANIFEST_FILENAME = "optimized_postprocess_pipeline_manifest.json"


def parse_args() -> argparse.Namespace:
    """Parses CLI options for the optimized parser plus postprocess pipeline."""

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
    """Validates and normalizes run bounds before parser or LLM work begins."""

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
    """Builds isolated local settings for optimized parser staging outputs."""

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
    """Runs the optimized parser unless an existing IR directory is requested.

    With --skip-parse and --optimized-ir-dir, this function only validates that
    retrieval chunk IR exists. Parsed PDF/image/text directories are used only
    when the parser is actually rebuilding artifacts.
    """

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


def select_inputs(args: argparse.Namespace, all_inputs: list) -> list:
    """Applies coverage mode and shared smoke/full-run limits to loaded chunks."""

    pool = select_candidate_inputs(all_inputs) if args.coverage == "candidates" else list(all_inputs)
    return select_run_inputs(pool, limit=args.limit, offset=args.offset, allow_full_run=args.allow_full_run)


def main() -> int:
    """Runs optimized parsing, two-pass post-processing, and sidecar writing."""

    args = parse_args()
    validate_args(args)
    args.output_root = args.output_root.expanduser().resolve()
    output_dir = (args.postprocess_dir or args.output_root / "llm_postprocess").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ir_dir = run_optimized_parse(args)
    all_inputs = inputs_from_jsonl_dir(ir_dir)
    selected_inputs = select_inputs(args, all_inputs)
    print(f"Loaded {len(all_inputs)} chunks from {ir_dir}; selected {len(selected_inputs)} for {args.coverage} coverage.")

    quality = run_postprocess_pass(
        args,
        pass_name="quality",
        inputs=selected_inputs,
        output_dir=output_dir,
    )
    graph_decisions = []
    graph_failures = []
    graph_hits = 0
    graph_misses = 0
    graph_aborted = False
    if args.coverage == "two-pass" and not quality.aborted:
        graph_inputs = graph_worthy_inputs(selected_inputs, quality.decisions, quality.failures)
        print(f"Graph pass selected {len(graph_inputs)} graph-worthy chunks.")
        graph = run_postprocess_pass(
            args,
            pass_name="graph",
            inputs=graph_inputs,
            output_dir=output_dir,
        )
        graph_decisions = graph.decisions
        graph_failures = graph.failures
        graph_hits = graph.cache_hits
        graph_misses = graph.cache_misses
        graph_aborted = graph.aborted
        final_decisions = merge_quality_and_graph_decisions(quality.decisions, graph.decisions)
    else:
        if quality.aborted and args.coverage == "two-pass":
            print("Skipping graph pass because quality pass stopped early.")
        final_decisions = quality.decisions

    failures = [*quality.failures, *graph_failures]
    original_concept_resolution_mode = args.concept_resolution_mode
    if (quality.aborted or graph_aborted) and args.concept_resolution_mode == "llm":
        print("Skipping LLM concept adjudication because an earlier LLM pass stopped early.")
        args.concept_resolution_mode = "deterministic"
    concept = resolve_concepts(args, output_dir, final_decisions)
    args.concept_resolution_mode = original_concept_resolution_mode

    report = write_sidecar_outputs(
        output_dir,
        inputs=selected_inputs,
        decisions=final_decisions,
        failures=failures,
        concept_resolution=concept.resolution,
        cache_hits=quality.cache_hits + graph_hits,
        cache_misses=quality.cache_misses + graph_misses,
        concept_cache_hits=concept.cache_hits,
        concept_cache_misses=concept.cache_misses,
    )
    manifest = {
        "schema_version": "1.0",
        "coverage": args.coverage,
        "optimized_ir_dir": str(ir_dir),
        "postprocess_dir": str(output_dir),
        "selected_input_count": len(selected_inputs),
        "quality_decision_count": len(quality.decisions),
        "graph_decision_count": len(graph_decisions),
        "final_decision_count": len(final_decisions),
        "failure_record_count": len(failures),
        "aborted": quality.aborted or graph_aborted,
        "aborted_passes": [
            name
            for name, aborted in (("quality", quality.aborted), ("graph", graph_aborted))
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
    concept_failures = concept.resolution.adjudication_failures if concept.resolution else []
    return 1 if concept_failures or quality.aborted or graph_aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
