# Full evaluation report runbook

This runbook describes how to produce one complete Graph RAG evaluation
scorecard for the default production configuration, then publish a sanitized,
commit-friendly copy. The steps are ordered from least expensive and most
deterministic to provider-consuming and optional.

A scorecard published with `--allow-incomplete` is an explicitly partial
verification snapshot, not a full report. It uses the same report structure so
missing requirements remain inspectable, but it must not be described as a
completed fresh campaign.

The default production baseline is:

- Analyst retrieval: `legacy_vector_context`;
- Visualizer retrieval: `optimized`;
- Analyst reranker: disabled for the baseline;
- prepared runtime storage: `storage`;
- local embedder: `models/all-MiniLM-L6-v2`;
- runtime scenarios: `evals/runtime/scenarios.json`;
- retrieval benchmark: `evals/retrieval/benchmark_cases.jsonl`.

The evaluation does not ingest, parse, embed, index, or mutate storage.

The active Retriever `v7` requires explicit `named` or `broad` knowledge-search
policy. Named searches require alias groups; broad searches require a validated
in-scope route and have no alias groups. This boundary is structural and does
not guarantee semantic relevance. A successful controlled-live out-of-scope
completion reconstructs the application fallback; a graph-only imported trace
does not prove the later application event. New controlled-live records use
`provenance.response_origin` values `application_fallback_reconstruction`,
`graph_message`, or `unavailable` to identify the captured response source.
Historical records can lack this field; preserve their original fingerprints and
scores. Keep routing, refusal delivery, relevance, and clarity separate when
interpreting results.

## What a full report contains

The scorecard has no overall score or overall status. It keeps independent
groups so a routing improvement cannot hide a grounding or latency regression.

| Scorecard block | Supplied by |
| --- | --- |
| Campaign coverage | Retrieval/offline/live/repetition/semantic/provenance accounting from all selected manifests |
| Retrieval quality | Retrieval benchmark; reviewed context labels add Context Precision@10 and Context Recall |
| Runtime task behavior | Offline trace evaluation and controlled live scenarios |
| Reliability | Repeated live scenarios plus offline/live failure observations |
| Efficiency | Runtime observations containing latency, model attempts, tool calls, retries, and provider token metadata |
| Optional semantic quality | Explicit judge runs over already captured runtime records |
| Diagnostics | All contributing executors, including optional AgentEvals trajectory rows |

No single command can create every block. Retrieval and runtime execution remain
separate because they exercise different contracts; the report builder combines
their compatible results.

## Token accounting and coverage

Use the versioned [token usage availability method](runtime_evaluation.md#token-usage-availability)
when interpreting efficiency data. New captures count distinct event/request
identities once and retain input, output, and total availability independently.
The report separates observed subtotals and field-specific observed/attempted
coverage from complete-run averages and their eligible-run denominators. Known
zero is included in a complete average; partial and unavailable runs are not.
Turn/root telemetry and runtime/judge usage remain separately attributed.
OpenRouter JSON wire usage is captured before adapter normalization where a
unique response ID is available. Streaming or normalized-only responses without
wire provenance remain unavailable; accounting does not buffer streams or infer
provider-reported totals from adapter-filled values.

An unavailable total does not become zero, a passing token budget, or a complete
cost-per-success calculation. Failed or uncertain reservations remain spent;
skips incur no judge request. Legacy zeros without availability evidence stay
unknown. A derived historical analysis must retain source hashes and limits;
never edit original runs, historical scorecards, or publication indexes to fill
missing fields. Token observations are not verified billing totals.

## Storage model: runs, fingerprints, and published reports

The ignored working tree is:

```text
data/evaluation/
├── runs/<run-id>/
├── fingerprints/<fingerprint>/
│   ├── scorecard.md
│   ├── scorecard.json
│   └── history.json
├── validation/<timestamp>/
├── latest.md
└── latest.json
```

The commit-friendly publication tree is:

```text
reports/evaluation/
├── README.md
├── latest.md
├── latest.json
├── latest_provenance.json
├── index.md
└── history/<date>_<fingerprint-prefix>/
    ├── scorecard.md
    ├── scorecard.json
    └── provenance.json
```

### A run is one real command execution

A directory under `runs/` is the immutable source record for one retrieval,
offline, live, or judge invocation. It contains the detailed artifacts produced
by that command and a `manifest.json`. Its run ID includes a timestamp, run kind,
and random suffix, so a later command does not overwrite it.

Examples:

- one retrieval baseline command is one run;
- one offline trace command is one run, even if the trace contains many cases;
- one live command with three cases is one run containing three observations;
- `--repetitions 3` is one command run containing three independent workflow
  executions;
- one judge command is a separate run that refers back to saved runtime records.

Runs are the audit trail. They answer: “What did this exact command observe?”
Failed and incomplete runs remain inspectable. They do not replace a successful
deterministic result.

### A fingerprint identifies a compatible application configuration

A fingerprint is a content hash of the application/evaluation state that must
remain compatible before results may be aggregated. It includes application and
evaluator code, prompts, machine-readable scenario/benchmark contracts, locked
dependencies, prepared-storage manifests, and non-secret model/search/storage
configuration. Git/source revisions are recorded as provenance, not identity.

Model configuration snapshots retain provider-specific generation settings,
including output limits, request timeouts, and reasoning options for every role
and both Researcher stages. Earlier snapshots could omit these fields through
base-class serialization. Capturing them now corrects effective fingerprint
input completeness without changing the hash formula. Recursive secret
redaction remains in place. Future fingerprints can therefore differ from older
ones; historical runs and published reports are not regenerated or rewritten.

The fingerprint is not a second copy of a run. The
`fingerprints/<fingerprint>/` directory is a generated aggregate view over all
runs with that identity. It answers: “What does the scorecard say for this exact
version and configuration?”

Invocation arguments are recorded separately. Their hash is the invocation key:

- retrieval and offline results are deterministic snapshots; only the newest
  successful run for the same invocation key contributes;
- older deterministic runs remain in `history.json`;
- live repetitions accumulate within one fingerprint;
- judge results accumulate only when dimension, provider, model, and rubric
  identity match;
- runs with different fingerprints are never blended.

Thus, `runs/` stores every command execution, while `fingerprints/` stores the
current computed view and history index for each compatible configuration.

### `latest` and `reports/evaluation` serve different purposes

`data/evaluation/latest.*` points to the most recently refreshed working
fingerprint. It is convenient locally but is ignored by Git.

`reports/evaluation` is an explicit publication layer. It contains the Markdown
scorecard, machine-readable JSON, and sanitized provenance—not raw traces,
retrieved excerpts, runtime records, or tool outputs. The Markdown scorecard groups
nested metrics and collapses verbose diagnostic detail for review; use the JSON
scorecard when exact keys or complete structured values are required. Publishing the
same fingerprint updates its stable history snapshot and the `latest` aliases.

Publish only after all intended commands for a campaign finish. Generated
evaluation/publication artifacts, `.DS_Store`, caches, bytecode, and temporary
outputs do not affect later fingerprints. Relevant content changes still do.

### Explicit local index for evaluation

Production `StorageSettings.index_storage` still defaults to Redis. Deployment
copies of `docstore.json` and `property_graph_store.json` establish local document
and graph data; they do not select local index storage. For the separately
approved evaluation configuration, the existing BaseSettings JSON override is:

```bash
export INDEX_STORAGE='{"storage_type":"local","storage_path":"/absolute/preserved/storage"}'
export DOCUMENT_STORAGE='{"storage_type":"local","storage_path":"/absolute/preserved/storage"}'
export PROPERTY_GRAPH_STORAGE='{"storage_type":"local","storage_path":"/absolute/preserved/storage"}'
export LOCAL_STORAGE_DIR=/absolute/preserved/storage
```

Replace the path with the actual preserved directory containing
`index_store.json`; `storage_path` belongs inside the `INDEX_STORAGE` JSON.
The preparation owner verified actual `StorageSettings()` selection and local
`KnowledgeGraphStorage.get_index_storage()` loading as `SimpleIndexStore` with
`IndexLPG`, under kernel network denial, without models or database connections.
This verifies local index loading only, not a complete live workflow.

For relocated preparation, bind `LOCAL_STORAGE_DIR` to that same directory and
record the effective `PathSettings` inputs: `RAW_DATA_DIR`, `PARSED_PDFS_DIR`,
`PARSED_IMAGES_DIR`, `PARSED_TEXTS_DIR`, and `PROMPTS_DIR`. They must point to the
preserved raw/parsed assets and selected source prompt tree, not stale original
checkout locations. `EMBEDDER_MODEL_PATH` selects the preserved
`models/all-MiniLM-L6-v2`; `RERANKER_MODEL_PATH` selects preserved
`models/Qwen3-Reranker-0.6B`. The preparation owner's private
`planned_collection_environment.json` records the exact verified paths; preserve
these values with the collection conditions. Do not copy a credential environment
file into the snapshot.

The explicit configuration uses local document/index/graph storage, Pinecone
vectors and MongoDB checkpoints. It removes Redis as a prerequisite for that
evaluation configuration; production remains unchanged. Preserve original
production settings and the effective evaluation override separately in
provenance, including actual document, graph, index and `PathSettings` paths.
Do not use `init_from_local` or vector overwrite to seed remote stores during
evaluation. Pinecone/MongoDB and remaining live prerequisites still apply.
OpenRouter remains the current runtime provider; this storage choice neither
substitutes Ollama nor authorizes collection.

Record downloaded document/graph file comparisons and exact hashes privately;
filesystem timestamps are not cloud object versions.

Local all-store retrieval, the configured headless hybrid workflow, and the
deployed demo are different measurement conditions. File-level document/graph
parity, when verified, does not establish Pinecone vector/namespace parity,
deployed credential identity, or cloud deployment equivalence. Preserve selected
remote index/namespace and data identities, mark unavailable parity explicitly,
and compare returned evidence with the preserved corpus only within authorized
collection. Without cloud/deployed-key verification, observations cannot establish
exact demo quality, production-traffic reliability/latency, or a fully frozen
remote-storage comparison.

Keep retrieval repeatability diagnostics bounded and private: record source,
settings, hash seeds, ordered evidence IDs/scores, missing labels, and per-case
metric availability. Different available-case denominators must remain visible;
seed-dependent pass/rank variation is a comparison limitation, not evidence of
restoration improvement or isolated cause. No ranking changes, full benchmark
reruns, or extra collection follow automatically from these diagnostics.

## Complexity 1 — validate retrieval references

This is local, deterministic, and does not load the model or index:

```bash
uv run --locked python scripts/evaluate_retrieval_pipeline.py \
  --validate-references-only \
  --benchmark-file evals/retrieval/benchmark_cases.jsonl \
  --storage-dir storage
```

It writes reference diagnostics under `data/evaluation/validation/<timestamp>/`.
It does not register a scored run or populate a scorecard block.

Use `--allow-invalid-benchmark-references` only for investigation. It is not
appropriate for a canonical report.

For the separate planned source-only, graph-assisted, reranked, and legacy
comparison, see [retrieval ablation](retrieval_ablation.md). Keep its run roots
and labels separate from the canonical report. Preparing those commands does
not itself authorize or run a retrieval campaign.

## Complexity 2 — create and review context relevance labels

Context Precision@10 and Context Recall require human-reviewed relevance data.
Embeddings, BERT classifiers, or an LLM judge do not create ground truth.

Generate a review sheet from an actual legacy Analyst retrieval run in a
separate working root:

```bash
uv run --locked python scripts/evaluate_retrieval_pipeline.py \
  --mode analyst \
  --analyst-variant legacy_vector_context \
  --analyst-reranker-mode disabled \
  --storage-dir storage \
  --embedder-model-path models/all-MiniLM-L6-v2 \
  --benchmark-file evals/retrieval/benchmark_cases.jsonl \
  --evaluation-root data/evaluation/relevance_work \
  --no-render-html \
  --write-relevance-review evals/retrieval/context_relevance_review.md
```

The generated worksheet resolves every returned legacy grounding chunk against
the same loaded prepared docstore. It includes a bounded excerpt, source path,
and storage-manifest hash. Existing machine-readable proposed labels and
answer-point mappings are prefilled as `AP1`, `AP2`, and so on. Verified
annotation chunks that were not returned in this run appear in a separate
additional-evidence table; they define acceptable support but receive no
retrieval credit unless a scored run returns them. Never insert such chunks as
ranked rows.

If a row still says `excerpt unavailable`, do not label it: the chunk ID could
not be resolved in the evaluated storage snapshot and the capture problem must
be investigated first. Reference validation covers both benchmark expectations
and context-annotation evidence IDs.

For each reviewed benchmark case:

1. Inspect every returned source item in the first ten ranks.
2. Label it `relevant`, `partially_relevant`, or `irrelevant`.
3. Define the factual answer points required for adequate coverage. Keep each
   point atomic and independently verifiable: avoid combining separable claims
   with “and” or “but”, align points with the query rather than incidental
   retrieved material, and keep query-disambiguation rules in relevance labels
   instead of treating them as answer facts.
4. Map each answer point to one or more supporting evidence IDs.
5. Confirm the generated prepared-storage manifest identity.
6. Synchronize approved values into the authoritative
   `evals/retrieval/benchmark_cases.jsonl`.
7. Set `context_relevance.review_status` to `reviewed` only when the labels and
   mappings are complete.

`review_status: reviewed` is necessary but not sufficient. Every source chunk
actually returned in the current Analyst top ten must have a label, and every
required answer point must have an approved evidence mapping. A retrieval rerun
can surface a previously unseen chunk; that case correctly receives `N/A` for
both context metrics until the new chunk is reviewed.

The Markdown sheet is a human review surface; it is never parsed as executable
truth. Missing or unreviewed labels produce visible `N/A` values rather than
false zeros. Precision weights relevant as 1, partial as 0.5, irrelevant or an
empty rank as 0, and divides the first-ten total by ten. Recall is the share of
required answer points supported by returned mapped evidence.

Label changes alter the fingerprint. Complete this step before canonical runs,
then repeat Complexity 1.

## Complexity 3 — run deterministic baselines

### Retrieval baseline

```bash
uv run --locked python scripts/evaluate_retrieval_pipeline.py \
  --mode both \
  --analyst-variant legacy_vector_context \
  --visualizer-variant optimized \
  --analyst-reranker-mode disabled \
  --storage-dir storage \
  --embedder-model-path models/all-MiniLM-L6-v2 \
  --benchmark-file evals/retrieval/benchmark_cases.jsonl \
  --no-render-html
```

This populates Retrieval quality and retrieval diagnostics. A known regression
can make the command exit with status 1 after the run is successfully written
and registered. `--no-fail-on-threshold` preserves the measurements but returns
status 0.

The omitted `--output-dir` is intentional: the evaluator allocates a unique
directory under `data/evaluation/runs`. Use an explicit output only for isolated
debugging; a canonical custom output must still be a unique child of
`data/evaluation/runs` to be discovered by the report builder.

The `DEFAULT_RUN_ROOT` parsed-artifact paths do not supply retrieval evidence.
Runtime retrieval loads the prepared index, vector stores, documents, and graph
from `--storage-dir`. Changing parsed-directory defaults should not change a
run. Different scores indicate some other changed input—most commonly storage
contents, code/benchmark state, environment configuration, or stochastic
ranking—and should be compared through the two manifests and case artifacts.

Copy the printed retrieval run ID and check context-label availability before
starting provider-consuming live or judge stages:

```bash
EVAL_RETRIEVAL_RUN_ID=<printed-retrieval-run-directory-name>

jq -c 'select(.mode == "analyst")
  | select(
      (.scores.context_precision_at_10 == null)
      or (.scores.context_recall == null)
    )
  | {
      case_id,
      review_status: .diagnostics.context_relevance_review_status,
      missing_labels: (.diagnostics.context_relevance_missing_labels // [])
    }' \
  "data/evaluation/runs/$EVAL_RETRIEVAL_RUN_ID/case_results.jsonl"
```

No output means every Analyst case produced both reviewed context metrics.
Output with `missing_labels` means the current retrieval returned at least one
unreviewed chunk even if the case itself is marked `reviewed`. To create a full
campaign, stop here, review that evidence, update the authoritative JSONL,
synchronize its Markdown review views, freeze the new fingerprint, revalidate
references, and restart the campaign. Do not relabel or rewrite the saved run.

If the purpose is instead to preserve a knowingly partial diagnostic snapshot,
continue only with that intent recorded and use the explicit incomplete
publication path in Complexity 6.

### Offline runtime baseline

Use one of these forms for a given trace export.

Native evaluator:

```bash
uv run --locked python scripts/evaluate_runtime_workflow.py offline \
  --trace-export exports/langsmith_traces_20260816T233652Z.json \
  --scenarios evals/runtime/scenarios.json
```

Native evaluator plus optional AgentEvals strict trajectory cross-check:

```bash
uv run --locked --group evaluation python scripts/evaluate_runtime_workflow.py offline \
  --trace-export exports/langsmith_traces_20260816T233652Z.json \
  --scenarios evals/runtime/scenarios.json \
  --framework agentevals
```

Do not run both forms for the same trace as a canonical baseline: the second
already includes all native checks. Offline evaluation reads saved LangSmith
exports and performs no model-provider or Tavily calls.

AgentEvals contributes only non-gating strict worker-trajectory matching in
Diagnostics. It does not score tools, arguments, retrieval, grounding, answers,
tokens, or latency; native checks remain authoritative for those dimensions.

To export newer traces:

```bash
uv run --locked python scripts/export_langsmith_logs.py \
  --frequency weekly \
  --output json \
  --output-path exports
```

Exporting accesses hosted LangSmith and requires its configured credentials. It
does not execute the learner workflow.

## Controlled live and judge evaluation

Follow the [evaluation request guard](runtime_evaluation.md#evaluation-request-guard)
and [live/judge examples](runtime_evaluation.md#commands). The operator supplies
a fresh quota observation and explicit attempt ceilings. Keep raw outputs and
optional diagnostics in ignored private directories; custom output paths are not
automatically ignored. Live and judge runs are optional and require provider
confirmation. A failed, interrupted or skipped run is not a quality score.

## Complexity 6 — rebuild, audit, and publish

Each scored command refreshes local scorecards automatically. Rebuild them
without executing an evaluator:

```bash
uv run --locked python scripts/build_evaluation_report.py
```

Record every printed canonical run-directory name. Verify that the complete
campaign ledger shares exactly one fingerprint before auditing:

```bash
uv run --locked python scripts/build_evaluation_report.py \
  --verify-run-fingerprints \
  --run-id "$EVAL_RETRIEVAL_RUN_ID" \
  --run-id "$EVAL_OFFLINE_RUN_ID" \
  --run-id "$EVAL_LIVE_LOCAL_RUN_ID" \
  --run-id "$EVAL_LIVE_MENTOR_RUN_ID" \
  --run-id "$EVAL_LIVE_CONTINUITY_RUN_ID" \
  --run-id "$EVAL_LIVE_WEB_RUN_ID" \
  --run-id "$EVAL_LIVE_REPETITION_RUN_ID" \
  --run-id "$EVAL_JUDGE_ANALYST_RUN_ID" \
  --run-id "$EVAL_JUDGE_MENTOR_RUN_ID" \
  --run-id "$EVAL_JUDGE_GRAPH_RUN_ID"
```

The command prints the per-run fingerprints and exits nonzero unless there is
exactly one. Set `EVAL_FINGERPRINT` to that single printed value. Do not copy
runs from either historical fingerprint into this ledger.

Audit `data/evaluation/latest.md` and the matching
`data/evaluation/fingerprints/<fingerprint>/history.json`. Confirm that
`contributing_run_ids` include the intended retrieval, runtime, repetition, and
judge runs. Follow their manifests and detailed artifacts under `runs/` when a
metric needs explanation.

Run the machine-readable completeness audit:

```bash
uv run --locked python scripts/build_evaluation_report.py \
  --fingerprint "$EVAL_FINGERPRINT" \
  --audit-completeness
```

The audit exits 0 only when every required contribution is accounted for. Judge
errors/skips may be accounted for without being successful. When the audit
passes, publish the exact fingerprint as a complete campaign:

```bash
uv run --locked python scripts/build_evaluation_report.py \
  --fingerprint "$EVAL_FINGERPRINT" \
  --confirm-complete \
  --publish
```

If the audit exits 1 and the missing requirement is deliberately accepted for a
partial verification snapshot—for example, `retrieval_relevance_labels` while
newly returned chunks await human review—publish with an exact fingerprint and
the explicit override:

```bash
uv run --locked python scripts/build_evaluation_report.py \
  --fingerprint "$EVAL_FINGERPRINT" \
  --allow-incomplete \
  --publish
```

Do not combine `--allow-incomplete` with `--confirm-complete`. The override does
not make the campaign complete, calculate missing metrics, or import compatible-
looking historical runs. It only permits publication of the selected scorecard
with its missing requirements still visible. In particular, fixing labels in
`benchmark_cases.jsonl` changes the fingerprint; a full report then requires a
fresh campaign under that new identity rather than rebuilding the old run.

Always pass `--fingerprint` in this runbook. The CLI accepts an exact
`--fingerprint ... --publish` selection without either policy flag because it is
not implicitly choosing the newest fingerprint. This runbook deliberately avoids
that less-expressive form: pair the exact fingerprint with `--confirm-complete`
or `--allow-incomplete` so publication intent is reviewable. Likewise, do not
rely on confirmation or an override without an exact fingerprint.

Publication writes latest aliases, a stable historical snapshot, sanitized
provenance, and `reports/evaluation/index.md`.

Review the generated files before committing them. Raw run artifacts remain
ignored in `data/evaluation`.

## Relevant optional argument changes

| Command | Option | Appropriate use |
| --- | --- | --- |
| Retrieval | `--mode analyst` or `--mode visualizer` | Targeted diagnostic or comparison; not the complete baseline |
| Retrieval | `--analyst-variant optimized` | Explicit post-baseline Analyst comparison |
| Retrieval | `--analyst-reranker-mode config` | Measure configured reranking rather than the locked disabled baseline |
| Retrieval | `--case-id ID` | Target a known regression; repeat for multiple cases |
| Retrieval | `--no-fail-on-threshold` | Preserve a zero exit code while still recording failed cases |
| Retrieval | `--render-html` | Produce local graph render artifacts; not needed for scoring |
| Any evaluator | `--evaluation-root PATH` | Keep an experiment completely separate from the canonical history |
| Any evaluator | `--output-dir PATH` | Debug-only explicit destination; use a unique directory and keep it under the selected root's `runs/` if it must aggregate |
| Offline | `--trace-export PATH` | Select an exact LangSmith export |
| Offline/live | `--framework agentevals` | Add non-gating strict trajectory comparison |
| Live/judge | `--provider-quota PATH --max-provider-attempts N` | Fresh quota observation and explicit per-run attempt ceiling |
| Live | `--max-web-attempts N` | Per-run web attempt ceiling, combined with `--allow-tavily` |
| Live | repeated `--case-id` | Select explicit workflow executions |
| Live | `--repetitions N` | Measure stochastic reliability |
| Live | `--allow-tavily` | Acknowledge expected web-provider use |
| Live | `--allow-expanded-live-suite` | Permit more than three executions after quota review |
| Judge | `--judge-provider openrouter --judge-model MODEL` | Explicit separately authorized judge identity; must match quota metadata |
| Judge | repeated `--dimension` | Select separate semantic rubrics |
| Judge | `--source-run RUN_ID` | Pin the judge to one immutable runtime run |
| Judge | `--structured-output-method METHOD` | Deliberate parser diagnostic; default `function_calling`, alternatives `json_schema` or `json_mode` |
| Judge | `--allow-expanded-judge-suite` | Permit more than three judge jobs after quota review |
| Report | `--fingerprint VALUE` | Select one exact configuration; selection alone is not a completeness claim |
| Report | `--audit-completeness` | Print coverage JSON and fail when required contributions are missing |
| Report | `--verify-run-fingerprints --run-id ID ...` | Prove an explicit campaign ledger has one fingerprint |
| Report | `--confirm-complete` | Publish the selected fingerprint only when completeness passes |
| Report | `--allow-incomplete` | Deliberately publish a visibly incomplete partial snapshot; never combine with `--confirm-complete` |
| Report | `--reports-root PATH` | Choose another commit-friendly publication destination |

For an optimized Analyst comparison, isolate it from the production baseline:

```bash
uv run --locked python scripts/evaluate_retrieval_pipeline.py \
  --mode analyst \
  --analyst-variant optimized \
  --storage-dir storage \
  --embedder-model-path models/all-MiniLM-L6-v2 \
  --benchmark-file evals/retrieval/benchmark_cases.jsonl \
  --evaluation-root data/evaluation_comparisons/optimized_analyst \
  --no-render-html
```

## Completion checklist

- [ ] The repository and effective configuration stayed frozen during the campaign.
- [ ] Retrieval references validate against the intended prepared storage.
- [ ] For a full campaign, every current Analyst result has numeric Context
      Precision@10 and Context Recall; otherwise the publication is explicitly
      labeled as an incomplete partial snapshot.
- [ ] The default legacy Analyst/optimized Visualizer retrieval run completed.
- [ ] Exactly one native or AgentEvals-enhanced offline baseline was registered
      for the selected export.
- [ ] Intended live cases and repetitions completed with reviewed provider quotas.
- [ ] Optional judge dimensions were run only where bounded observations exist.
- [ ] All contributing runs share the intended fingerprint.
- [ ] The completeness audit passed, or every missing requirement was explicitly
      accepted and remains visible in a partial publication.
- [ ] Diagnostics and failure reasons were inspected, not just aggregate rates.
- [ ] The selected fingerprint was published with `--confirm-complete` for a
      full campaign or `--allow-incomplete` for a partial snapshot.
- [ ] Only sanitized `reports/evaluation` artifacts are committed.

## Public evidence and offline verification

The [published scorecards](../reports/evaluation/README.md) are historical and
incomplete. They are not measurements of the current OpenRouter runtime or a
paired graph-versus-vector improvement. Semantic judge errors and unavailable
scores must remain visible. No fresh evaluation campaign is part of the M1
offline engineering checks.

Use [the setup and offline gates](project_overview.md#setup-and-offline-checks)
for a locked clean-checkout test run without `.env`, credentials, private
storage, downloaded models, or services. The commands in this evaluation
runbook have different requirements: retrieval needs prepared storage and a
local embedder; offline trace evaluation needs an existing sanitized export;
live workflow and judge modes contact configured providers, and web fallback
contacts Tavily. They are not substitutes for the offline test gate.

Configured OpenRouter `openrouter/free` identifies an automatic routing policy.
Response provenance separates `configured_provider` / `configured_model`
from response-observed `provider` / `model`. Missing response identities remain
unknown (empty strings); an echoed `openrouter/free` route does not identify
the resolved model. For OpenRouter, response metadata `provider` identifies the
serving provider; the adapter-injected `model_provider="openrouter"` label alone
does not. Supported legacy `model_provider` values remain accepted for other
adapters. Stop diagnostics accept `done_reason`, `finish_reason`, and
`stop_reason`; token observations accept canonical usage metadata and
OpenAI-style prompt/completion counts. Aggregate token-coverage analysis remains
future work. Output caps, reasoning requests, and request timeouts describe
configured behavior, not proof that every routed model has
identical capabilities. The default `LLM_PIPELINE=ollama` selector remains a
compatibility alias; active runtime roles use OpenRouter. Offline
post-processing and optional Ollama reranking have their own settings.
