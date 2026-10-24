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

## Complexity 0 — prepare and freeze the campaign

Before running measurements:

1. Complete intended code, prompt, scenario, benchmark, and documentation edits.
2. Confirm the prepared `storage` directory is the intended production snapshot.
3. Record the current revision and working-tree state.
4. Avoid editing tracked files until the campaign is complete.
5. Install locked normal dependencies:

```bash
uv sync --locked
```

Optional AgentEvals and judge dependencies are installed only when those stages
will be used:

```bash
uv sync --locked --group evaluation
```

Environment overrides are allowed, but they become part of run provenance and
may change the fingerprint. Never place API keys in commands or committed
reports.

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

## Complexity 4 — run controlled live scenarios

Live mode invokes the real compiled workflow with configured models and prepared
storage. It requires explicit provider confirmation and at least one `--case-id`.
There is deliberately no implicit “run everything” mode, which prevents an
accidental burst of model and Tavily calls.

The following batches cover the reviewed suite while respecting the normal
three-execution limit.

Local answer and graph behavior:

```bash
uv run --locked python scripts/evaluate_runtime_workflow.py live \
  --case-id local_transformer_answer_success \
  --case-id contextual_mamba_visualization \
  --case-id explicit_mamba_visualization_success \
  --confirm-provider-calls
```

Mentor and graph coverage:

```bash
uv run --locked python scripts/evaluate_runtime_workflow.py live \
  --case-id initial_attention_quiz_route \
  --case-id mentor_stuck_continuation \
  --case-id transformer_encoder_graph_coverage \
  --confirm-provider-calls
```

Continuity and out-of-scope control:

```bash
uv run --locked python scripts/evaluate_runtime_workflow.py live \
  --case-id contextual_medusa_web_query \
  --case-id colbert_guidance_mode_continuation \
  --case-id out_of_scope_biological_cats \
  --confirm-provider-calls \
  --allow-tavily
```

Web fallback regressions:

```bash
uv run --locked python scripts/evaluate_runtime_workflow.py live \
  --case-id ambiguous_eagle_without_history \
  --case-id explicit_medusa_web_fallback_success \
  --case-id in_scope_retnet_local_miss_fallback \
  --confirm-provider-calls \
  --allow-tavily
```

Researcher truncation reliability, as three independent executions:

```bash
uv run --locked python scripts/evaluate_runtime_workflow.py live \
  --case-id researcher_structured_output_truncation_reliability \
  --repetitions 3 \
  --confirm-provider-calls \
  --allow-tavily
```

`--allow-tavily` acknowledges cases expected to require web research; it is not
a tool sandbox. `--allow-expanded-live-suite` permits more than three total
case/repetition executions after quota review. `--framework agentevals` adds the
same optional non-gating trajectory cross-check to live runs.

Live results populate Runtime task behavior, Reliability, Efficiency, and
Diagnostics. Repetitions add Wilson intervals, route consistency, evidence-set
Jaccard stability, and truncation/failure frequencies.

## Complexity 5 — run optional semantic judges

Judge mode reads a completed runtime record and does not execute the learner
workflow or Tavily again. It does call the configured Analyst model, is
non-gating, and requires `--confirm-provider-calls`.

Current configuration:

- provider and model: the active Analyst settings from
  `backend/configs/models.py` and environment;
- repository default: OpenRouter `openrouter/free` automatic routing;
- temperature: forced to 0;
- maximum output: 4,096 tokens for claim faithfulness and 512 for Boolean dimensions;
- claim output: at most 32 atomic claims, five evidence IDs per claim, and bounded strings;
- structured-output transport: `function_calling`;
- evaluator retries: 0;
- normal limit: three case/dimension jobs per command;
- result rubric version: `runtime-semantic-v1`.

The command prints the effective provider and model before calling it. Boolean
dimensions use OpenEvals; claim faithfulness uses the repository's structured
claim-to-bounded-evidence assessment. `--structured-output-method json_schema`
and `--structured-output-method json_mode` are diagnostic alternatives; use
them only for a deliberate rerun because each judge job can call the provider.
JSON mode receives an explicit required-schema instruction.

Analyst grounding and usefulness:

```bash
uv run --locked --group evaluation python scripts/evaluate_runtime_workflow.py judge \
  --source-run latest \
  --case-id local_transformer_answer_success \
  --dimension claim_faithfulness \
  --dimension analyst_usefulness \
  --confirm-provider-calls
```

Mentor quality:

```bash
uv run --locked --group evaluation python scripts/evaluate_runtime_workflow.py judge \
  --source-run latest \
  --case-id mentor_stuck_continuation \
  --dimension mentor_pedagogy \
  --dimension conversational_continuity \
  --confirm-provider-calls
```

Graph usefulness:

```bash
uv run --locked --group evaluation python scripts/evaluate_runtime_workflow.py judge \
  --source-run latest \
  --case-id explicit_mamba_visualization_success \
  --dimension graph_usefulness \
  --confirm-provider-calls
```

`--source-run latest` means the newest completed offline/live run in the current
fingerprint containing every selected case. If no single run contains the
selected case, or tracked files/configuration changed since it ran, the command
reports “No completed runtime run contains all selected cases.” Use the
exact run ID to target an older compatible record, or rerun the needed live/offline
case after freezing the current fingerprint. `--records` supports direct legacy
record input but is not preferred for a canonical campaign.

Faithfulness is skipped/N/A without bounded evidence; graph usefulness is
skipped/N/A without bounded graph labels. A missing required final worker is
also skipped/N/A, while the deterministic routing failure remains visible.
Provider, parser, and schema failures are judge execution errors with unavailable
scores. All remain non-gating. `--allow-expanded-judge-suite` permits more than
three jobs after explicit quota review.

### Verify the judge corrections before continuing

A judge run manifest can be `completed` because its artifacts were written even
when individual judge jobs have `status=error`. The terminal now reports
`successful` and `error` job counts. HTTP 200 proves only that the provider
returned a response; success also requires schema parsing and validation.

Run the offline correction checks first; they make no provider or Tavily calls:

```bash
uv run --no-sync python scripts/check_offline.py tests/test_runtime_evaluation.py \
  tests/test_evaluation_reporting.py

uv run --no-sync ruff check \
  backend/evaluation/runtime_evaluation.py \
  backend/evaluation/runtime_live.py \
  backend/evaluation/evaluation_reporting.py \
  scripts/evaluate_runtime_workflow.py \
  tests/test_runtime_evaluation.py \
  tests/test_evaluation_reporting.py
```

These corrections change the evaluation fingerprint. Old runs remain immutable
history, but `--source-run latest` will not select an older fingerprint. Capture
fresh source records for all three judge examples:

```bash
uv run --locked python scripts/evaluate_runtime_workflow.py live \
  --case-id local_transformer_answer_success \
  --case-id mentor_stuck_continuation \
  --case-id explicit_mamba_visualization_success \
  --confirm-provider-calls
```

This command does not authorize Tavily. Copy the printed live run ID and inspect
non-content prerequisites:

```bash
EVAL_LIVE_RUN_ID=<printed-live-run-directory-name>

jq -c '{scenario_id, passed, failed_checks: [.checks[] | select(.status == "fail") | {name, gating, reason}]}' \
  "data/evaluation/runs/$EVAL_LIVE_RUN_ID/case_results.jsonl"

jq -c '{scenario_id, final_worker: (.worker_sequence[-1] // null), evidence_capture_status, bounded_evidence_count: (.bounded_evidence | length), graph_capture_status: .graph.capture_status, graph_label_count: (.graph.node_labels | length), labeled_edge_count: (.graph.labeled_edges | length)}' \
  "data/evaluation/runs/$EVAL_LIVE_RUN_ID/runtime_records.jsonl"
```

| Case | Required source observation |
| --- | --- |
| `local_transformer_answer_success` | Pass; final worker `analyst`; bounded evidence captured |
| `mentor_stuck_continuation` | Pass; final worker `mentor` |
| `explicit_mamba_visualization_success` | Pass; final worker `visualizer`; graph capture `captured` with labels and labeled edges |

Do not interpret a role-specific semantic score when its deterministic case
fails. The evaluator records the job as skipped before a provider call. If the Mentor
case routes to Retriever → Analyst, it retains that deterministic regression and
leaves Mentor pedagogy/continuity as `N/A`.

Run the Analyst judges against that exact fresh source run:

```bash
uv run --locked --group evaluation python scripts/evaluate_runtime_workflow.py judge \
  --source-run "$EVAL_LIVE_RUN_ID" \
  --case-id local_transformer_answer_success \
  --dimension claim_faithfulness \
  --dimension analyst_usefulness \
  --confirm-provider-calls
```

The expected terminal shape is:

```text
(2 successful, 0 skipped, 0 error; provider calls attempted: 2)
```

Copy the printed judge run ID and inspect its safe result metadata:

```bash
EVAL_JUDGE_RUN_ID=<printed-judge-run-directory-name>

jq -c '{scenario_id, dimension, status, score, provider_call_attempted, transport_status, http_status_code, stop_reason, configured_output_token_limit, output_present, output_size_chars, parser_classification, confirmed_truncation, error_type, error_cause_types, raw_output_omitted}' \
  "data/evaluation/runs/$EVAL_JUDGE_RUN_ID/judge_results.jsonl"
```

A parser failure now identifies its exception type/cause chain without
persisting raw model output. Try the default `function_calling` transport first.
If it still fails, review those fields before authorizing one deliberate
diagnostic rerun. Append exactly one of these alternatives to the same judge
command:

```bash
uv run --locked --group evaluation python scripts/evaluate_runtime_workflow.py judge \
  --source-run "$EVAL_LIVE_RUN_ID" \
  --case-id local_transformer_answer_success \
  --dimension claim_faithfulness \
  --dimension analyst_usefulness \
  --structured-output-method json_schema \
  --confirm-provider-calls
```

For the other alternative, replace `json_schema` with `json_mode`.

Do not run both automatically: every selected dimension is another provider
call, and a transport change is recorded in invocation/configuration identity.

Run Mentor quality only when the Mentor source prerequisite passed:

```bash
uv run --locked --group evaluation python scripts/evaluate_runtime_workflow.py judge \
  --source-run "$EVAL_LIVE_RUN_ID" \
  --case-id mentor_stuck_continuation \
  --dimension mentor_pedagogy \
  --dimension conversational_continuity \
  --confirm-provider-calls
```

Run graph usefulness only when the graph source prerequisite passed:

```bash
uv run --locked --group evaluation python scripts/evaluate_runtime_workflow.py judge \
  --source-run "$EVAL_LIVE_RUN_ID" \
  --case-id explicit_mamba_visualization_success \
  --dimension graph_usefulness \
  --confirm-provider-calls
```

Graph usefulness now reuses bounded node labels and labeled relations from the
captured Plotly artifact. It never queries current storage to repair an old run.
An old record with `graph.capture_status=unavailable` therefore needs a fresh
live source run; rerunning judge mode alone cannot supply its missing labels.

After intended judge jobs are successful or explicitly accepted as `N/A`,
continue with Complexity 6.

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
| Live | repeated `--case-id` | Select explicit workflow executions |
| Live | `--repetitions N` | Measure stochastic reliability |
| Live | `--allow-tavily` | Acknowledge expected web-provider use |
| Live | `--allow-expanded-live-suite` | Permit more than three executions after quota review |
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
