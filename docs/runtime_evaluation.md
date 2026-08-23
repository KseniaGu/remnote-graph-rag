# Graph RAG evaluation

Updated: 2026-08-23

## Purpose and boundary

The evaluation system measures the online Graph RAG workflow and its prepared
retrieval dependencies. It does not ingest, parse, post-process, embed, index, or
rebuild storage. It is independent of the Reflex UI.

The default baseline is intentionally asymmetric:

- Analyst: `legacy_vector_context`
- Visualizer: `optimized`
- optimized Analyst: explicit comparison only
- one tool per worker invocation: current workflow boundary
- live, Tavily, and semantic-judge calls: disabled without explicit confirmation

Retrieval and runtime remain separate executors because they answer different
diagnostic questions. Their completed runs are combined into one
per-fingerprint scorecard. The scorecard has no overall score and no overall
status. Its Markdown view uses human-readable labels, grouped subtables, and
collapsible diagnostic detail; its JSON companion remains the fully inspectable
machine-readable representation.

## Components

| Component | Responsibility |
| --- | --- |
| `backend/evaluation/retrieval_benchmark.py` | Reviewed prepared-storage contracts, evidence/graph scoring, optional context-relevance labels, and review rendering. |
| `backend/evaluation/runtime_evaluation.py` | Canonical runtime observations, trace normalization, deterministic checks, reliability metrics, and optional semantic judges. |
| `backend/evaluation/runtime_live.py` | Controlled headless invocation of the production LangGraph workflow. |
| `backend/evaluation/evaluation_reporting.py` | Immutable run registration, fingerprints, aggregation, scorecard rendering, and sanitized publication. |
| `scripts/evaluate_retrieval_pipeline.py` | Retrieval executor and relevance-review command. |
| `scripts/evaluate_runtime_workflow.py` | Offline, live, optional AgentEvals, and optional judge modes. |
| `scripts/build_evaluation_report.py` | Rebuild reports from evaluation history and optionally publish a commit-friendly snapshot. |

The authoritative contracts are
[`evals/retrieval/benchmark_cases.jsonl`](../evals/retrieval/benchmark_cases.jsonl)
and [`evals/runtime/scenarios.json`](../evals/runtime/scenarios.json).
Their Markdown companions are review views, not executable input.

## Commands

For an end-to-end sequence that produces every report block, including human
context-relevance annotation, use
[full_evaluation_report_runbook.md](full_evaluation_report_runbook.md). The
commands below describe individual evaluator modes.

### Output-directory behavior

Both executors allocate a unique immutable directory under
`data/evaluation/runs` when `--output-dir` is omitted. This is the canonical
behavior and makes every completed run discoverable by the report builder.
Explicit output directories are intended for isolated diagnostics; keep one
under the selected evaluation root's `runs/` if it must be aggregated.

Default retrieval baseline:

```bash
uv run --locked python scripts/evaluate_retrieval_pipeline.py \
  --mode both \
  --analyst-variant legacy_vector_context \
  --visualizer-variant optimized \
  --analyst-reranker-mode disabled \
  --no-render-html
```

Optional optimized Analyst comparison:

```bash
uv run --locked python scripts/evaluate_retrieval_pipeline.py \
  --mode analyst \
  --analyst-variant optimized \
  --no-render-html
```

Validate benchmark references without loading an index:

```bash
uv run --locked python scripts/evaluate_retrieval_pipeline.py \
  --validate-references-only
```

Offline trace evaluation, with zero workflow/provider/Tavily/judge calls:

```bash
uv run --locked python scripts/evaluate_runtime_workflow.py offline
```

Optional AgentEvals strict-trajectory cross-check:

```bash
uv sync --locked --group evaluation
uv run --locked --group evaluation python scripts/evaluate_runtime_workflow.py offline \
  --framework agentevals
```

The AgentEvals option still performs complete native offline scoring. It then
runs `graph_trajectory_strict_match` against each allowed worker sequence and
writes `framework_results.json`. It is deterministic, local, non-gating, and
appears only in Diagnostics. It does not score tools, retrieval, grounding,
answers, tokens, or latency. Run either native-only offline mode or the
AgentEvals form for one export, not both, to avoid duplicate observations.

Controlled live positive control:

```bash
uv run --locked python scripts/evaluate_runtime_workflow.py live \
  --case-id out_of_scope_biological_cats \
  --confirm-provider-calls
```

Controlled Tavily fallback regression:

```bash
uv run --locked python scripts/evaluate_runtime_workflow.py live \
  --case-id in_scope_retnet_local_miss_fallback \
  --confirm-provider-calls \
  --allow-tavily
```

Three independent Researcher truncation-reliability repetitions:

```bash
uv run --locked python scripts/evaluate_runtime_workflow.py live \
  --case-id researcher_structured_output_truncation_reliability \
  --repetitions 3 \
  --confirm-provider-calls \
  --allow-tavily
```

These are independent runs. A failed repetition followed by a successful
repetition is not an evaluator retry.

Optional claim-faithfulness judge over the newest current-fingerprint run that
contains the selected case:

```bash
uv run --locked --group evaluation python scripts/evaluate_runtime_workflow.py judge \
  --source-run latest \
  --case-id local_transformer_answer_success \
  --dimension claim_faithfulness \
  --confirm-provider-calls
```

The legacy `--records path/to/runtime_records.jsonl` input remains available.

Judge mode uses active Analyst model settings, not a separate judge role. Current
repository defaults are Ollama `qwen3.5:cloud` at `https://ollama.com`;
environment overrides remain effective. The evaluator forces temperature 0,
uses a 4,096-token output limit for bounded claim faithfulness and 512 tokens
for each Boolean dimension, uses result rubric `runtime-semantic-v1`, and
performs no evaluator retry.
Claim output is capped at 32 atomic claims, five evidence IDs per claim, and
bounded claim/reason strings. Boolean dimensions use OpenEvals; claim
faithfulness uses structured claim/evidence assessment. Structured output
defaults to `function_calling`. The `json_mode` prompt explicitly includes
the required JSON schema. All dimensions are optional and non-gating.

`--source-run latest` searches only completed `offline` or `live` manifests
under the current evaluation root and fingerprint, and one source run must
contain every selected case. The
error `no completed runtime run contains every selected case` means no source
satisfies all those conditions. Common causes are scratch output, a fingerprint
change, or cases saved in separate runs. Rerun the selected case into immutable
history, or specify an exact run ID/records file.

Generate a top-10 relevance review sheet from an actual Analyst retrieval run.
The sheet resolves legacy relation grounding IDs against the same loaded
docstore and records bounded excerpts, source paths, and the storage-manifest
hash. Existing proposed labels and answer-point mappings are displayed, while
verified annotation evidence absent from the current top ten appears in a
separate table and receives no retrieval credit for that run. Reference
validation also checks these annotated evidence IDs. Required answer points
must be atomic, independently verifiable, and aligned with the query; source
titles, disambiguation rules, and incidental retrieved details are not answer
facts. The sheet never writes labels back to the benchmark:

```bash
uv run --locked python scripts/evaluate_retrieval_pipeline.py \
  --mode analyst \
  --analyst-variant legacy_vector_context \
  --analyst-reranker-mode disabled \
  --no-render-html \
  --write-relevance-review evals/retrieval/context_relevance_review.md
```

Rebuild reports without running retrieval or the workflow:

```bash
uv run --locked python scripts/build_evaluation_report.py
```

Audit one exact scorecard after a campaign:

```bash
uv run --locked python scripts/build_evaluation_report.py \
  --fingerprint "$EVAL_FINGERPRINT" \
  --audit-completeness
```

When the audit exits 0, publish it as a complete campaign:

```bash
uv run --locked python scripts/build_evaluation_report.py \
  --fingerprint "$EVAL_FINGERPRINT" \
  --confirm-complete \
  --publish
```

When the audit exits 1 and the missing coverage is deliberately accepted for a
partial verification snapshot, publish it explicitly as incomplete. One example
is `retrieval_relevance_labels` while newly returned chunks await human review:

```bash
uv run --locked python scripts/build_evaluation_report.py \
  --fingerprint "$EVAL_FINGERPRINT" \
  --allow-incomplete \
  --publish
```

Do not combine `--confirm-complete` and `--allow-incomplete`. An incomplete
publication is not a full campaign: the override only permits publication while
the missing requirements remain visible. The CLI accepts an exact fingerprint
without either policy flag, but documented publication always pairs the exact
fingerprint with one flag so intent is explicit.

## Immutable history and fingerprints

Evaluation artifacts use this layout:

```text
data/evaluation/
├── runs/<run-id>/
├── fingerprints/<fingerprint>/
│   ├── scorecard.md
│   ├── scorecard.json
│   └── history.json
├── latest.md
└── latest.json
```

The root `latest.md` and `latest.json` are convenience copies of the most
recently refreshed fingerprint scorecard. The `validation/<timestamp>/`
directory instead contains disposable `benchmark_reference_report.json`
preflights from `--validate-references-only`. They add no metrics, are not
registered as runs, and do not contribute to history.

Each run directory is immutable. A manifest records run kind/status, timestamps,
invocation and invocation key, repository/source revision provenance, scoped
application/evaluator/contract hashes, effective non-secret model/search/storage
configuration, prompt selections, storage-manifest identity, provider/Tavily/
judge counts, and artifact names.

Fingerprints are content identities: application and evaluator code, prompts,
machine-readable scenario/benchmark contracts, locked dependencies, effective
application configuration, and prepared-storage manifests contribute. Generated
`data/evaluation` and `reports/evaluation` artifacts, `.DS_Store`, caches,
bytecode, and temporary files do not. Git revision and a trace's reported source
revision remain provenance and do not split identical content. Executor-only
arguments also do not split an otherwise identical application fingerprint.
Different fingerprints are never aggregated.

For the same deterministic retrieval or offline invocation, only the newest
completed run contributes to current metrics. Older and failed runs remain in
`history.json`. Live repetitions accumulate. Judge rows aggregate only under
the same dimension, provider, model, and rubric identity. Root `latest.*`
always copies one fingerprint's scorecard.

Archived evaluation output and legacy scratch directories are excluded from current history.
The report builder scans only `data/evaluation/runs/*/manifest.json`.

The tracked publication layer is `reports/evaluation`. It contains only the
selected scorecard, a generated index, and sanitized provenance. Raw traces,
runtime records, evidence text, and tool outputs remain in ignored working
history. See the full runbook for the detailed distinction between runs,
fingerprints, working `latest` aliases, and published snapshots.

## Canonical bounded observations

Runtime records capture evidence from tool/trace output at execution time.
Historical evidence is never reconstructed by querying mutable storage.

- at most 10 evidence items;
- at most 1,200 characters per excerpt and 10,000 characters total;
- evidence ID, kind, query, rank, score when available, source title/path, excerpt;
- at most 25 graph nodes and 35 graph edges, including labels when observable;
- configured provider output limit and observed stop reason;
- canonical worker/action signatures for the basic loop detector;
- explicit `captured`, `unavailable`, or `not_applicable` status.

System prompts, credentials, authorization values, and tool secrets are not
included. Missing evidence text or graph labels is `N/A`; it is never inferred
from an ID.

## Scorecard

The report keeps distinct failure modes visible. Each generated scorecard includes
machine-readable campaign coverage, every expected group and semantic dimension,
offline/live sample counts, contributing source run IDs, and a definition table
for every measurement family, so values can be traced to immutable detailed
artifacts. Missing, skipped, successful, and failed judge dimensions remain
visible; there is still no overall quality score or status.

| Group | Measurements |
| --- | --- |
| Retrieval quality | Case pass rate; chunk, concept, and relation recall; ID/label/spec diagnostics; forbidden evidence; adequacy/errors; dangling edges and chunk nodes; Context Precision@10 and Context Recall when reviewed. |
| Runtime task behavior | Task success; route and agent compliance; path efficiency; tool selection/arguments; one-tool compliance; local-to-web fallback; unnecessary web; source exhaustion; response, modality, termination, and graph contracts. |
| Reliability | Repetition/sample count; pass rate and Wilson interval; route consistency; evidence-set Jaccard stability; basic loop rate; failure and truncation frequencies. |
| Efficiency | Worker/model/provider/retry/Tavily counts; input/output/total tokens; tokens per successful run; time-to-resolution mean, median, and p95. |
| Optional semantic quality | Claim counts and grounded-claim rate; Analyst usefulness; Mentor pedagogy; conversational continuity; graph usefulness. |
| Diagnostics | Check reasons, trajectories, arguments, evidence and graph observations, stop reasons, failure classes, and run/configuration provenance. |

### Completeness contract

`coverage` is a machine-readable accounting contract, not an overall quality
status. A complete campaign must contain the prepared-storage retrieval baseline
with reviewed Context Precision@10/Recall, the trace-linked offline suite, every
reviewed live scenario, three independent
`researcher_structured_output_truncation_reliability` live observations, all
five semantic dimensions accounted for as success/skipped/error, AgentEvals
rows if that framework was requested, and invocation/configuration provenance.
An optional judge is not required to succeed, but it may not disappear.

Historical offline trace configuration is `verified` only when exported trace
metadata carries an evaluation fingerprint. A revision such as
`3cbef51-dirty` does not prove the exact dirty content; such records remain
usable observations but are labeled `needs verification`. Runtime metrics keep
combined values where useful while also exposing offline/live counts and failure
breakdowns.

Functional task success means every applicable gating contract passed. Routing,
tools, fallback, modality, termination, graph validity, loop, worker-step, and
Tavily bounds are gating. Model calls, provider attempts, retries, tokens, and
latency are report-only initially. A missing observation is `N/A`/
`not_observed`, not zero or success.

The basic loop detector flags an identical worker/action signature repeated
after orchestration, reports the repeated-action count, and contributes to loop
rate. Semantic no-progress detection is deferred.

Path efficiency is the shortest allowed completed worker path divided by
observed worker steps, capped at 1. Incomplete or short-circuited runs are
`N/A`.

## Context relevance

Context metrics apply only to ranked Analyst source evidence. Human reviewers
label each observed top-10 evidence item as `relevant` (1),
`partially_relevant` (0.5), or `irrelevant` (0), and map required answer
points to evidence IDs. Labels are tied to a storage snapshot.

Context Precision@10 is the weighted relevance sum divided by ten. Context
Recall is the fraction of reviewed required answer points supported by at least
one returned top-10 item. If labels are not reviewed, or a reviewed run returns
an unlabeled item, both metrics remain visibly `N/A`. Embeddings, BERT-like
classifiers, and LLMs do not create ground truth automatically.

`review_status: reviewed` alone is therefore insufficient for campaign
completeness. Check the saved retrieval run before provider-consuming stages:

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

No output means every current Analyst case produced both context metrics. If a
new top-10 chunk is missing a label, update the authoritative benchmark and its
Markdown review views, then start a fresh campaign because that contract change
changes the fingerprint. Use `--allow-incomplete` only to publish the old
scorecard as an explicitly partial snapshot; never rewrite its raw run.

Graph relevance stays separate: deterministic concept/relation coverage plus
the optional graph-usefulness judge. It is not folded into Analyst Context
Precision.

## Optional semantic judges

Semantic judges are non-gating and require explicit provider confirmation.

- `claim_faithfulness`: structured supported/partial/unsupported factual claims
  with evidence IDs; grounded-claim rate is
  `(supported + 0.5 × partial) / all claims`.
- `analyst_usefulness`
- `mentor_pedagogy`
- `conversational_continuity`
- `graph_usefulness`

The four non-faithfulness dimensions use separate Boolean OpenEvals rubrics.
The default structured-output transport is `function_calling`, which matches
the tool-capable Ollama model used by the workflow. Diagnostic alternatives are
`--structured-output-method json_schema` and
`--structured-output-method json_mode`; changing the method changes the
evaluation invocation and fingerprint.

Role-specific judges run only when the required agent is the final observed
worker: Analyst for faithfulness/usefulness, Mentor for pedagogy/continuity, and
Visualizer for graph usefulness. If that worker did not produce an applicable
output, or required bounded evidence/graph labels are unavailable, the semantic
job is `skipped` and its score is `N/A`; the deterministic routing or runtime
failure remains a real failure. Provider, parser, and schema failures are judge
execution `error` results with unavailable scores.

Results separately count successful, skipped, error, and provider-attempted
jobs. They record privacy-safe transport completion/status code, stop reason,
configured output limit, output presence/character size, parser classification,
and confirmed truncation. Raw judge output, prompts, evidence text, credentials,
and provider exception messages are not published. One selected
run/dimension makes at most one judge call, at temperature zero, with no
evaluator retry.

AgentEvals/OpenEvals are optional. AgentEvals strict graph-trajectory matching is
a non-gating cross-check; the native evaluator remains authoritative for this
repository's state, fallback, evidence, and budget contracts.

## Reviewed regression semantics

- EAGLE accepts an `in_scope` or `ambiguous` scope, must use an AI/ML-qualified
  research query, must not become `out_of_scope`, and must reach Analyst after
  successful web evidence. No clarification-question route is required.
- Contextual Medusa requires Retriever → Researcher → Analyst.
- RetNet accepts local `no_results`, `below_threshold`, or `topic_mismatch`,
  then requires a qualified RetNet/Retentive Network web query and forbids
  ResNet substitution. It is a reviewed known-failing regression.
- `researcher_structured_output_truncation_reliability` measures independent
  failure/success repetitions and output-limit stop reasons, not retry behavior.
- DDIM remains a specific retrieval capability.
- The text-classification visualization contract includes Naive Bayes, logistic
  regression, SVM, and FastText where grounded edges exist.
- Forbidden retrieval evidence is retained only with observed-run provenance.
  The current ColBERT negative records its two source runs in the JSONL review
  notes.

## Maintenance

When routes, tools, budgets, prompts, models, retrieval modes, storage defaults,
failure classes, evaluator formulas, or scenario contracts change:

1. Treat active workflow/configuration code as source of truth.
2. Update the machine-readable scenario or benchmark and increment its version
   or schema when its contract changes.
3. Regenerate/synchronize the Markdown review view and record human approval.
4. Update this document, `README.md`, `docs/project_overview.md`, and
   `AGENTS.md` when maintenance guidance changes.
5. Add/update deterministic tests for executable defaults and metric formulas.
6. Rebuild scorecards; do not edit generated scorecards manually.
7. Keep historical runs immutable and clearly label historical baseline claims.
8. Do not run live, Tavily, or judge modes without separate explicit approval.

## Deferred

No-progress semantic loop detection, monetary cost, safety/toxicity scoring,
automatic relevance-label generation, UI evaluation, ingestion/indexing
evaluation, and one aggregate quality score remain out of scope.
