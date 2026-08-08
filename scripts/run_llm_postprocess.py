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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.data_processing.concept_registry import (
    DEFAULT_CONCEPT_RESOLUTION_PROMPT_VERSION,
)
from backend.data_processing.llm_postprocess import (
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
    inputs_from_jsonl_dir,
    select_candidate_inputs,
    write_sidecar_outputs,
)
from backend.data_processing.llm_postprocess_runner import (
    graph_worthy_inputs,
    infer_concept_resolution_model_name,
    load_excluded_chunk_ids,
    load_existing_postprocess_sidecars,
    merge_quality_and_graph_decisions,
    resolve_concepts,
    run_postprocess_pass,
    select_run_inputs,
    validate_concept_resolution_only_args,
    validate_run_bounds,
)


def parse_args() -> argparse.Namespace:
    """Parses CLI options for the standalone sidecar post-processing runner."""

    parser = argparse.ArgumentParser(
        description="LLM sidecar post-processing for optimized RemNote parser chunks."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory with optimized parser IR JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for sidecar postprocess outputs.",
    )
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
    parser.add_argument(
        "--model-name", default=DEFAULT_MODEL_NAME, help="Ollama model name."
    )
    parser.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
        help="Prompt version under remnote_postprocess.",
    )
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
    parser.add_argument(
        "--prompt-dir", type=Path, default=ROOT / "backend" / "llm" / "prompts"
    )
    parser.add_argument(
        "--max-batch-chunks", type=int, default=DEFAULT_MAX_BATCH_CHUNKS
    )
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


def main() -> int:
    """Runs standalone chunk post-processing, or delegate to concept-resolution-only mode."""

    args = parse_args()
    if args.concept_resolution_only:
        return run_concept_resolution_only(args)

    validate_run_bounds(args)

    all_inputs = inputs_from_jsonl_dir(args.input_dir)
    candidates = select_candidate_inputs(all_inputs)
    excluded_chunk_ids = load_excluded_chunk_ids(args.exclude_output_dir)
    candidate_pool = [
        item for item in candidates if item.chunk_id not in excluded_chunk_ids
    ]
    selected_inputs = select_run_inputs(
        candidate_pool,
        limit=args.limit,
        offset=args.offset,
        allow_full_run=args.allow_full_run,
    )

    print(
        f"Loaded {len(all_inputs)} chunks; {len(candidates)} candidates before exclusions; "
        f"excluded {len(excluded_chunk_ids)} chunk IDs; selected {len(selected_inputs)} candidates."
    )

    if args.dry_run:
        report = write_sidecar_outputs(
            args.output_dir,
            inputs=selected_inputs,
            decisions=[],
            failures=[],
        )
        print(
            f"Dry run complete. Report: {args.output_dir / 'llm_postprocess_report.md'}"
        )
        print(json.dumps(report, indent=2))
        return 0

    quality_result = run_postprocess_pass(
        args,
        pass_name="quality",
        inputs=selected_inputs,
        output_dir=args.output_dir,
    )
    graph_inputs = (
        []
        if quality_result.aborted
        else graph_worthy_inputs(
            selected_inputs,
            quality_result.decisions,
            quality_result.failures,
        )
    )
    print(f"Graph pass selected {len(graph_inputs)} graph-worthy chunks.")

    graph_result = None
    if graph_inputs:
        graph_result = run_postprocess_pass(
            args,
            pass_name="graph",
            inputs=graph_inputs,
            output_dir=args.output_dir,
        )

    decisions = (
        merge_quality_and_graph_decisions(
            quality_result.decisions,
            graph_result.decisions,
        )
        if graph_result
        else quality_result.decisions
    )
    failures = [
        *quality_result.failures,
        *(graph_result.failures if graph_result else []),
    ]
    concept_result = resolve_concepts(args, args.output_dir, decisions)

    report = write_sidecar_outputs(
        args.output_dir,
        inputs=selected_inputs,
        decisions=decisions,
        failures=failures,
        concept_resolution=concept_result.resolution,
        cache_hits=quality_result.cache_hits
        + (graph_result.cache_hits if graph_result else 0),
        cache_misses=quality_result.cache_misses
        + (graph_result.cache_misses if graph_result else 0),
        concept_cache_hits=concept_result.cache_hits,
        concept_cache_misses=concept_result.cache_misses,
    )
    print(
        f"Postprocess complete. Report: {args.output_dir / 'llm_postprocess_report.md'}"
    )
    print(json.dumps(report, indent=2))
    concept_failures = (
        concept_result.resolution.adjudication_failures
        if concept_result.resolution
        else []
    )
    aborted = quality_result.aborted or bool(graph_result and graph_result.aborted)
    return 1 if aborted or failures or concept_failures else 0


def run_concept_resolution_only(args: argparse.Namespace) -> int:
    """Regenerates concept sidecars from existing postprocess decisions.

    This mode is useful when the concept-resolution prompt, model, or adjudication
    settings change but the chunk-level LLM decisions should remain fixed.
    """

    validate_concept_resolution_only_args(args)
    inputs, decisions, failures = load_existing_postprocess_sidecars(args.input_dir)
    print(
        f"Loaded {len(inputs)} inputs, {len(decisions)} decisions, {len(failures)} existing failures."
    )
    concept_result = resolve_concepts(
        args,
        args.output_dir,
        decisions,
        concept_model_name=infer_concept_resolution_model_name(args, decisions),
    )

    report = write_sidecar_outputs(
        args.output_dir,
        inputs=inputs,
        decisions=decisions,
        failures=failures,
        concept_resolution=concept_result.resolution,
        concept_cache_hits=concept_result.cache_hits,
        concept_cache_misses=concept_result.cache_misses,
    )
    print(
        f"Concept resolution complete. Report: {args.output_dir / 'llm_postprocess_report.md'}"
    )
    print(json.dumps(report, indent=2))
    concept_resolution = concept_result.resolution
    return 1 if concept_resolution and concept_resolution.adjudication_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
