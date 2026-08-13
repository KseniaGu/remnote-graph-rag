# Optimized Pipeline Runbook

This document describes how to run the optimized data-processing pipeline and
materialize the final local knowledge-graph storage used by the application.

## Default Paths

Project defaults from `PathSettings`:

| Purpose | Default path |
| --- | --- |
| Raw RemNote markdown | `data/raw/AI Research` |
| Default parsed PDFs | `data/raw/parsed_pdfs` |
| Default parsed images | `data/raw/parsed_images` |
| Default parsed text artifacts | `data/raw/parsed_texts` |
| Local application storage | `storage` |

Model paths are read from `.env` through `ModelSettings`:

| Purpose | Setting |
| --- | --- |
| Embedder used when building vector storage | `EMBEDDER_MODEL_PATH` |
| Default Analyst reranker used by optimized retrieval | `RERANKER_MODEL_PATH` |

The optimized production pipeline should also keep an auditable run directory,
for example:

```bash
RUN_ROOT="data/production/full_optimized_pipeline_run"
```

When `scripts/run_optimized_postprocess_pipeline.py` is given an `--output-root`,
it stores optimized pipeline artifacts under that run root by default:

| Artifact | Pipeline default |
| --- | --- |
| Staging LlamaIndex storage | `$RUN_ROOT/staging_storage` |
| Optimized parser IR | `$RUN_ROOT/optimized_parser_ir` |
| LLM postprocess sidecars | `$RUN_ROOT/llm_postprocess` |
| Parsed PDFs | `$RUN_ROOT/parsed_pdfs` |
| Parsed images | `$RUN_ROOT/parsed_images` |
| Parsed text artifacts | `$RUN_ROOT/parsed_texts` |

The final storage that the local app should load is `storage`, not
`$RUN_ROOT/final_storage`, unless you intentionally configure the app to use a
different local storage path.

## Full Local Run

Place the raw RemNote export in the default raw folder:

```bash
data/raw/AI Research
```

Then run optimized parsing plus LLM post-processing:

```bash
RUN_ROOT="data/production/full_optimized_pipeline_run"

uv run --locked --group scripts python scripts/run_optimized_postprocess_pipeline.py \
  --raw-data-dir "data/raw/AI Research" \
  --output-root "$RUN_ROOT" \
  --coverage two-pass \
  --allow-full-run
```

Notes:

- External artifact preparation is enabled by default. Use
  `--skip-external-artifacts` only for debugging or intentionally incomplete
  runs.
- To reuse reviewed cached artifact Markdown instead of relying on network/OCR
  work, pass both `--copy-existing-artifacts` and `--existing-artifacts-dir`.
  Cached artifact copying no longer uses hidden local test-data paths.
- The current LLM defaults include the production post-processing prompt version,
  split quality/graph passes, and LLM concept resolution. The prompt file
  version is `v1`; this is the renamed production prompt that replaced the old
  experimental `v9` naming.
- Important LLM defaults are `model-name=nemotron-3-super:cloud`,
  `base-url=https://ollama.com`, `max-batch-chunks=1`,
  `max-batch-chars=9000`, `num-ctx=10240`,
  `quality-num-predict=2048`, `graph-num-predict=6144`, and
  `concept-resolution-num-predict=1536`.
- Do not pass `--limit` for the full production run.
- Use `--force-rebuild-staging` only when intentionally rebuilding staging
  outputs from scratch.

After parsing and LLM post-processing finish, build final local application
storage:

```bash
RUN_ROOT="data/production/full_optimized_pipeline_run"

uv run --locked --group scripts python scripts/build_postprocessed_graph_storage.py \
  --optimized-ir-dir "$RUN_ROOT/optimized_parser_ir" \
  --postprocess-dir "$RUN_ROOT/llm_postprocess" \
  --final-storage-dir "storage" \
  --force-rebuild-final
```

The build script reads `EMBEDDER_MODEL_PATH` from `.env` through
`ModelSettings`. Pass `--embedder-model-path` only when you need to override
that value for a specific run.

`--force-rebuild-final` removes and rebuilds `storage`. Use it only when the
target storage can be replaced.

## Verification

Check the final storage manifest:

```bash
cat storage/postprocessed_graph_storage_manifest.json
```

Useful fields to inspect:

- `source_retrieval_chunk_count`
- `decision_count`
- `failure_record_count`
- `retrieval_enabled_count`
- `graph_enabled_count`
- `quarantined_count`
- `embedded_retrieval_chunks`
- `embedded_retrieval_passages`
- `embedded_concept_nodes`
- `property_graph_index_built`

`embedded_retrieval_chunks` counts parent chunks that are eligible for
retrieval. The current vector store embeds child passage nodes, reported as
`embedded_retrieval_passages`, plus concept nodes. Parent chunks stay in the
docstore/property graph as evidence units and are recovered from passage hits at
retrieval time.

For deterministic retrieval evaluation against the local final storage:

```bash
RUN_ROOT="data/production/full_optimized_pipeline_run"

uv run --locked --group scripts python scripts/evaluate_retrieval_pipeline.py \
  --storage-dir "storage" \
  --raw-data-dir "data/raw/AI Research" \
  --benchmark-file "evals/retrieval/benchmark_cases.jsonl" \
  --output-dir "$RUN_ROOT/retrieval_eval/local_final_storage" \
  --mode both
```

Use an explicit output directory name when comparing storage builds, for
example:

```bash
RUN_ROOT="data/production/full_optimized_pipeline_run"

uv run --locked --group scripts python scripts/evaluate_retrieval_pipeline.py \
  --storage-dir "storage" \
  --raw-data-dir "data/raw/AI Research" \
  --benchmark-file "evals/retrieval/benchmark_cases.jsonl" \
  --output-dir "$RUN_ROOT/retrieval_eval/local_final_storage_chunk_splitting_relation_updates" \
  --mode both
```

To check the benchmark IDs before loading embeddings or running retrieval:

```bash
RUN_ROOT="data/production/full_optimized_pipeline_run"

uv run --locked --group scripts python scripts/evaluate_retrieval_pipeline.py \
  --storage-dir "storage" \
  --raw-data-dir "data/raw/AI Research" \
  --benchmark-file "evals/retrieval/benchmark_cases.jsonl" \
  --output-dir "$RUN_ROOT/retrieval_eval/reference_validation" \
  --mode both \
  --validate-references-only
```

The benchmark file contains reviewed retrieval cases with expected source
chunks, concepts, relations, and forbidden evidence. The deterministic runner
does not require Ragas, LangSmith, or any LLM evaluator. It writes:

- `benchmark_reference_report.json` for source/chunk, concept, relation,
  embedding, and graph-triplet preflight checks.
- `summary.md` for a short human-readable pass/fail report.
- `summary.json` for aggregate counts and mean scores.
- `case_results.jsonl` for per-case scores, failures, and missing evidence.
- `actual_evidence.jsonl` for retrieved source, concept, and relation evidence
  used by the scorer.

Current benchmark scoring maps passage-vector hits back to parent source chunk
IDs before computing source recall and MRR. Reference validation also treats a
parent chunk as embedded when it has embedded passage children.

The current accepted local benchmark run is:

```text
data/production/full_optimized_pipeline_run/retrieval_eval/local_final_storage_chunk_splitting_relation_updates
```

At the time this document was updated, that run passed 8 of 12 reviewed cases.
The remaining misses are useful known gaps, not release blockers for the current
iteration: VAE relation evidence, diffusion/DDIM source recall, BERT/ModernBERT
relation selection, and attention/RoPE relation selection.

Run a smaller subset while reviewing or debugging a failure:

```bash
RUN_ROOT="data/production/full_optimized_pipeline_run"

uv run --locked --group scripts python scripts/evaluate_retrieval_pipeline.py \
  --storage-dir "storage" \
  --raw-data-dir "data/raw/AI Research" \
  --benchmark-file "evals/retrieval/benchmark_cases.jsonl" \
  --output-dir "$RUN_ROOT/retrieval_eval/naive_bayes_vs_lr" \
  --case-id visualizer_naive_bayes_vs_logistic_regression \
  --mode visualizer
```

The command exits with code `1` when any selected case misses its configured
thresholds. Use `--no-fail-on-threshold` when you want to collect artifacts from
a known-failing run without failing the shell command.

For unscored retrieval debugging with the default exploratory queries:

```bash
RUN_ROOT="data/production/full_optimized_pipeline_run"

uv run --locked --group scripts python scripts/evaluate_retrieval_pipeline.py \
  --storage-dir "storage" \
  --raw-data-dir "data/raw/AI Research" \
  --output-dir "$RUN_ROOT/retrieval_debug/local_final_storage" \
  --mode both \
  --debug-default-queries
```

For a broad concept-graph audit of the same final storage, render the full
postprocessed concept relation graph:

```bash
RUN_ROOT="data/production/full_optimized_pipeline_run"

uv run --locked --group scripts python scripts/plot_final_storage_concept_graph.py \
  --storage-dir "storage" \
  --output "$RUN_ROOT/retrieval_debug/local_final_storage/concept_relation_graph.html"
```

This is an optional review artifact. It is useful for inspecting concept-node
and semantic-relation quality, but it is not required for normal application
startup.

The optimized pipeline also writes
`$RUN_ROOT/llm_postprocess/optimized_postprocess_pipeline_manifest.json` and
`$RUN_ROOT/llm_postprocess/llm_postprocess_report.json`. Use those reports to
confirm the selected input count, quality/graph decision counts, cache
hits/misses, and concept-resolution counts before building or promoting storage.

## Concept-Resolution Repair

If the chunk-level LLM sidecars are usable but the global concept registry or
graph projection needs to be regenerated, rerun only concept resolution. This
does not rerun parsing, OCR, external artifact fetching, or chunk-level LLM
post-processing.

The hardened resolver uses a controlled production type set, filters unsafe
aliases out of merge keys, and caps every LLM adjudication cluster before a
provider call. The controlled type set is `CONCEPT`, `METHOD`, `MODEL`,
`COMPONENT`, `FORMULA`, `PARAMETER`, `METRIC`, `DATASET`, `TASK`, `PROBLEM`,
`TOOL`, and `PAPER`.

In `--concept-resolution-only` mode, `--input-dir` must contain existing
`llm_postprocess_inputs.jsonl` and `llm_postprocess_decisions.jsonl`. The script
copies those chunk-level sidecars to `--output-dir` and regenerates the concept
registry, concept merge review, pair scores, adjudications, graph projection
preview, and report. Write the repaired sidecars to a new directory first:

```bash
RUN_ROOT="data/production/full_optimized_pipeline_run"

uv run --locked --group scripts python scripts/run_llm_postprocess.py \
  --concept-resolution-only \
  --input-dir "$RUN_ROOT/llm_postprocess" \
  --output-dir "$RUN_ROOT/llm_postprocess_concept_hardened" \
  --concept-resolution-mode llm \
  --concept-resolution-model-name nemotron-3-super:cloud \
  --base-url https://ollama.com \
  --allow-full-run
```

`--output-dir` must differ from `--input-dir` unless you intentionally pass
`--allow-in-place`. `--concept-resolution-model-name` is optional; when omitted,
the runner reuses the most common model name recorded in the existing decisions.

Build review storage from the repaired sidecars without replacing the local app
storage:

```bash
uv run --locked --group scripts python scripts/build_postprocessed_graph_storage.py \
  --optimized-ir-dir "$RUN_ROOT/optimized_parser_ir" \
  --postprocess-dir "$RUN_ROOT/llm_postprocess_concept_hardened" \
  --final-storage-dir "$RUN_ROOT/final_storage_concept_hardened" \
  --force-rebuild-final
```

Evaluate that review storage before promoting it:

```bash
uv run --locked --group scripts python scripts/evaluate_retrieval_pipeline.py \
  --storage-dir "$RUN_ROOT/final_storage_concept_hardened" \
  --raw-data-dir "data/raw/AI Research" \
  --output-dir "$RUN_ROOT/retrieval_debug/concept_hardened" \
  --mode both
```

Before promoting the repaired sidecars, inspect
`$RUN_ROOT/llm_postprocess_concept_hardened/llm_postprocess_report.json`:

- `concept_type_counts` should contain only the production concept types.
- `concept_source_type_counts` should preserve raw LLM-emitted source types for
  audit only.
- `max_concept_review_cluster_mentions` should be at most `24`.
- `max_concept_review_cluster_pair_scores` should be at most `80`.
- `skipped_over_budget_concept_cluster_count` should be `0`.
- `llm_concept_pair_scores.jsonl` should exist in the repaired output
  directory; the storage loader expects that optional audit sidecar when present.
- The generated visualizer graphs should not contain known bad remappings such
  as `GRPO -> Logistic regression`, `Naive Bayes classifier -> DPO`, or
  `DALL-E -> Logistic regression`.

Once the repaired graph looks correct, rebuild the application storage from the
fixed postprocess directory:

```bash
uv run --locked --group scripts python scripts/build_postprocessed_graph_storage.py \
  --optimized-ir-dir "$RUN_ROOT/optimized_parser_ir" \
  --postprocess-dir "$RUN_ROOT/llm_postprocess_concept_hardened" \
  --final-storage-dir "storage" \
  --force-rebuild-final
```

## Remote Parse and LLM Processing, Local Graph Build

For long production runs, it is often better to run parsing and LLM
post-processing on a remote machine, then copy the processed artifacts locally
and build final storage on the local machine.

On the remote VM:

```bash
RUN_ROOT="data/production/full_optimized_pipeline_run"

uv run --locked --group scripts python scripts/run_optimized_postprocess_pipeline.py \
  --raw-data-dir "data/raw/AI Research" \
  --output-root "$RUN_ROOT" \
  --coverage two-pass \
  --allow-full-run
```

Copy the completed run directory back to the local repository. Example using a
Google Cloud VM:

```bash
gcloud compute scp \
  --recurse \
  VM_NAME:~/remnote-graph-rag/data/production/full_optimized_pipeline_run \
  data/production/ \
  --zone YOUR_VM_ZONE
```

The local copy should contain at least:

```text
data/production/full_optimized_pipeline_run/optimized_parser_ir
data/production/full_optimized_pipeline_run/llm_postprocess
```

Copying the parsed artifact folders as well is recommended for auditability and
debugging:

```text
data/production/full_optimized_pipeline_run/parsed_pdfs
data/production/full_optimized_pipeline_run/parsed_images
data/production/full_optimized_pipeline_run/parsed_texts
```

Then build final local application storage on the local machine:

```bash
RUN_ROOT="data/production/full_optimized_pipeline_run"

uv run --locked --group scripts python scripts/build_postprocessed_graph_storage.py \
  --optimized-ir-dir "$RUN_ROOT/optimized_parser_ir" \
  --postprocess-dir "$RUN_ROOT/llm_postprocess" \
  --final-storage-dir "storage" \
  --force-rebuild-final
```

This avoids rerunning parsing, OCR, network artifact fetching, or LLM
post-processing locally. The local step only materializes final docstore,
property graph, and vector storage from the optimized IR and sidecars.

## Running the App Locally

To run the app locally without changing storage path configuration, materialize
the final storage into:

```text
storage
```

That path matches `PathSettings.local_storage_dir`.

The current search defaults in `KnowledgeGraphSearchSettings` use
`legacy_vector_context` for Analyst retrieval and `optimized` for Visualizer
retrieval. The Analyst reranker is used only when the optimized Analyst mode is
selected explicitly. Future end-to-end runtime evaluation should use
`legacy_vector_context` as its base Analyst mode and treat optimized retrieval
as a comparison variant.

This only addresses the filesystem path. A default `StorageSettings()` uses
local document storage, Redis index storage, Pinecone vector storage, local
property-graph storage, and MongoDB checkpoints. Configure every selected backend
with prepared data before starting the application. Runtime startup is load-only
and fails when the index is absent or invalid; use the build and migration
commands in this runbook for intentional storage changes.
