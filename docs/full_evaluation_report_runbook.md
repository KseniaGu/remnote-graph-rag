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
- the opt-in fixed-judge pilot uses the separately versioned strict-schema and
  provider-pinning controls documented in [runtime evaluation](runtime_evaluation.md#optional-semantic-judges);
  it remains subject to its authorized execution timing and free-capacity checks.

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

## B0 preparation and resume

B0 is the preserved pre-fix subset, not the full campaign below. Its preparation
makes zero LLM, Tavily, or judge requests. Only the coordinator's separately
authorized read-only quota inspection is permitted before a readiness report
and explicit collection authorization. Confirmation flags in examples are
acknowledgments, not authorization. Do not run the full campaign or extra
positive controls to fill B0 gaps.

Before requesting collection authorization, record:

1. Exact preserved source revision, clean/dirty status, complete binary diff
   when needed, lockfile, code/prompt/contract hashes, and immutable restore
   instructions. Preserve the post-preparation/pre-behavior-fix state.
2. A content-identified private storage copy, embedding/reranker artifacts and
   revisions, access requirements, and measured restore/verification results.
   A manifest alone cannot restore data. Local files do not archive remote
   Redis/Pinecone/MongoDB state; disclose that gap and required service access.
3. Effective non-secret OpenRouter settings for every runtime role, separate
   judge configuration, configured alias versus observed resolved identities,
   search settings, temperature/output caps, hardware, scenario versions,
   exact commands, and prepared-storage identity. Keep credentials and private
   content out of public provenance.
4. Existing 12-case local retrieval results and representative deterministic
   runtime controls, including failures and unavailable metrics. Offline
   computation uses preserved local assets; prohibit network model downloads,
   production ingestion/reindexing, and external embedding/reranking calls.
5. Sanitized observed provider quota metadata with observation time, applicable
   reset period, remaining requests/credits when available, demo reserve, and
   unknown fields. Check Tavily access and judge prerequisites without sending
   service probes. A credit balance or API-key limit is not proof of a free-route
   request allowance or enforced monetary ceiling.
6. Verified aggregate enforcement and immutable manifests reconciled with one
   durable ledger across days/processes. Restrict evaluation credentials to the
   guarded sequential runner. Disable hidden SDK/transport retries; retain only
   counted workflow retries. If the dispatch envelope cannot be verified,
   disclose the blocker and collect no live observations.

See the [stage and whole-effort ceilings](runtime_evaluation.md#portfolio-collection-budgets-and-b0-checkpoint).
The readiness report records blockers and authorization still needed; it never
turns a command or a favorable quota observation into permission to collect.
Prepared private snapshots remain under ignored
`data/evaluation/portfolio_baseline/`; sanitized summaries may be published only
without personal-derived content. Local planning/execution notes are ignored
coordination records, not public prerequisites.

### Private snapshot restore

`scripts/preserve_portfolio_baseline.py capture` requires `--source-root`,
`--asset-root`, repeated explicit `--asset` relative paths, and a new `--output`
directory. Inventory the required storage/model/export inputs before selecting
assets. Keep the resulting source archive, commit object, complete binary diff,
source inventory, copied assets, manifest, and manifest checksum together under
the ignored baseline root. A private path and checksum without the files are
insufficient for restoration.

After the owner supplies exact private snapshot and new destination paths, verify
and restore using the repository's standard-library helper:

```bash
python scripts/preserve_portfolio_baseline.py verify --snapshot "$B0_SNAPSHOT"
python scripts/preserve_portfolio_baseline.py restore \
  --snapshot "$B0_SNAPSHOT" --destination "$B0_RESTORE"
python scripts/preserve_portfolio_baseline.py verify \
  --snapshot "$B0_SNAPSHOT" --restored-root "$B0_RESTORE"
```

Use a new destination. Record verification result, elapsed time and disk bytes;
this procedure is not a claim that a particular snapshot has passed. The helper
does not provision dependencies or copy credentials, `.env`, or remote database
contents. The tracked public root `.env.example` is included as source; real or
private dotenv files, nested templates, and private asset templates are excluded. Reproduction preparation uses Python 3.11.15, uv 0.12.3 and the preserved
`uv.lock`; record any environment difference. The helper's `offline-run` requires
Linux `libseccomp.so.2` and kernel support to deny Internet sockets before
application imports, including descendants. It refuses execution if that guard
cannot be installed. Check those prerequisites separately without sending
provider probes.

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

### Quota planning record

The coordinator's authorized read-only observation at 2026-10-24 19:11:34 UTC
reported OpenRouter free-model requests: 0 used, 50 daily limit, 50 remaining;
`is_free_tier=true`. Credit limit/remaining were null, usage was zero, and the
reported budgets list was empty. Null credit fields do not establish unlimited
request quota or a hard spend ceiling. The sanitized private record is
`data/evaluation/portfolio_baseline/preparation/openrouter_quota.json`.

[OpenRouter limits documentation](https://openrouter.ai/docs/api_reference/limits)
describes account-wide free-model limits and UTC-day resets. Do not infer RPM
from the deprecated `rate_limit` field. The inspected key's mapping to the
production demo account has not been independently verified. These are dated
observations, not a guarantee of availability when collection resumes.

The collection policy retains at least 20 requests of local demo headroom and
caps evaluation at 24 attempts per UTC day, including runtime, judges and retries,
subject to fresh remaining quota and tighter stage/effort limits. New days,
restarts, aliases, or refreshed metadata cannot loosen this policy. Stricter
metadata may reduce capacity or increase headroom. This policy does not authorize
collection or guarantee headroom against concurrent external traffic. Group scenarios by their
machine-readable `expectations.budgets`, retain retry/judge capacity, and reduce
or defer a batch when the observed remainder cannot cover it. Do not attempt a
large batch merely because fewer than three scenario IDs are selected.

### Predeclared B0 observations

The table below is the **historical pre-collection 12/day allocation**, retained
as planning provenance; it is not the current 24/day policy or a new run list.
The authorized completion amendment covers only the remaining original cases,
within the current stage/effort envelopes.

| Historical proposed UTC day | Scenario IDs, once each | Runtime attempt estimates | Judge jobs | Total |
| --- | --- | --- | --- | --- |
| 1 | `out_of_scope_biological_cats`, `local_transformer_answer_success`, `mentor_stuck_continuation` | 1 + 3 + 2 | Transformer claim/usefulness; Mentor pedagogy | 9 |
| 2 | `colbert_guidance_mode_continuation`, `contextual_mamba_visualization`, `ambiguous_eagle_without_history` | 2 + 3 + 4 | Mamba continuity/graph usefulness | 11 |
| 3 | `contextual_medusa_web_query`, `in_scope_retnet_local_miss_fallback` | 5 + 7 | None | 12 |

These batches respect the initial three-live-execution command limit. They are
planning groups; daily observed quota can require smaller batches. Enable Tavily
only for the relevant authorized commands. The estimate is 25 logical runtime
calls, 27 provider attempts, three logical web searches, and 97,000 runtime
tokens. This historical proposal totals 32 attempts including five judges, with
no extra retry allowance beyond those estimates. The current B0 ceiling is 33
under the explicit amendment; this does not authorize a replacement/extra job. Actual excess can stop
coverage before the final case or judge; it never raises the allocation. These
are relative days, not scheduled or authorized collection dates. No automatic
rerun or replacement case follows a failure.

RetNet's scenario estimate allows six logical calls/seven attempts, while the
production turn quota remains five logical calls/six attempts. Record any
resulting operational stop; this preparation does not weaken either the
scenario contract or runtime quota. Fresh quota metadata is not an atomic
reservation against production-demo traffic, so daily headroom is not guaranteed.

The five judge jobs are exactly:

| Source scenario | Dimensions |
| --- | --- |
| `local_transformer_answer_success` | `claim_faithfulness`, `analyst_usefulness` |
| `mentor_stuck_continuation` | `mentor_pedagogy` |
| `contextual_mamba_visualization` | `conversational_continuity`, `graph_usefulness` |

Use the exact immutable source run ID for each group. Keep jobs in groups of at
most three unless expanded confirmation is separately approved. Ineligible jobs
are recorded as skipped with their reason; retain negative scores and judge
errors. Do not substitute the full-campaign Mentor continuity or explicit Mamba
graph examples below. Those belong to a different campaign allocation.

### Guarded runner parameters

Use the [normalized quota schema and ledger contract](runtime_evaluation.md#portfolio-collection-budgets-and-b0-checkpoint).
For the **historical 12/day observation** above, normalization would use:

```json
{
  "observed_at": "2026-10-24T19:11:34+00:00",
  "provider": "openrouter",
  "model": "openrouter/free",
  "remaining_requests": 50,
  "demo_reserve": 20,
  "daily_batch_limit": 12
}
```

This dated example expires 15 minutes after its observation (19:26:34 UTC),
or at the UTC-day boundary if sooner, and is not collection authorization.
The runner checks freshness at entry and before every reservation, so expiration
can stop an ongoing batch. This conservative 15-minute policy does not guarantee
provider availability. Keep the same canonical
`data/evaluation/portfolio_budget/ledger.json` under the Git-common main checkout
for B0, final, runtime, and judges. Worktrees share this ledger;
`--evaluation-root` does not allocate a fresh budget. A standalone restored
repository must pass `--budget-ledger` with the original canonical existing
absolute ledger path before authorized collection. Do not initialize a new
allowance or restore an archived ledger over the latest mutable counts. For
example, after the coordinator supplies the original path:

```bash
PORTFOLIO_STAGE=B0
PORTFOLIO_QUOTA=data/evaluation/portfolio_baseline/preparation/quota_runner.json
PORTFOLIO_LEDGER=/workspaces/remnote-graph-rag/data/evaluation/portfolio_budget/ledger.json
uv run --locked python scripts/evaluate_runtime_workflow.py live \
  --budget-stage "$PORTFOLIO_STAGE" \
  --budget-ledger "$PORTFOLIO_LEDGER" \
  --provider-quota "$PORTFOLIO_QUOTA" \
  --case-id out_of_scope_biological_cats \
  --confirm-provider-calls
```

This illustrates the override for an already authorized planned observation; it
is not an extra positive control. The supplied ledger must already exist. Add
the same `--budget-ledger` argument to authorized judge commands in a standalone
restore. The helper writes the original canonical path and snapshot-manifest
identity into restored Git metadata, so the restored source cannot select a fresh
campaign. Original and restored admissions reject another existing ledger,
including copies and alternate hard-link paths; a symlink resolving to the same
original canonical file is permitted. The state binds its path and fixed daily
policy; missing or mismatched bindings stop without migration/reset. Older
snapshots lacking the binding need a corrected freeze, not a guessed ledger.
The private baseline excludes mutable campaign counts. Initialize a missing
canonical ledger once, only after explicit collection authorization from the
original/shared checkout, never from a restored source copy.

A locked ledger, exhausted
allowance, stale quota observation, provider/model mismatch, uncertain job, or
missing/modified completed artifact stops collection. Reconcile exact manifests
and reservations; do not bypass the stop by replacing ledger or job identities.

### Explicit completion policy amendment

The historical completion policy was **24 LLM attempts per UTC day, B0 33,
final 59, overall 92**. Judge sublimits remain 5/5/10, web limits remain B0 3/6,
final 7/14 and overall 10/20, demo headroom remains at least 20, and production
per-turn/free-only restrictions stay fixed. The remaining original Medusa and
RetNet cases are each admitted against the preserved six-attempt envelope;
this amendment does not authorize a cats rerun, failed-judge retry or new judge.

A normal live/judge command rejects old/mismatched ledger policy. After focused
verification/review and coordinator release, apply only the named one-time
amendment from the amended source, under the same reviewed effective environment.
Use the verified asset-free accepted source copy for the previous-source proof:

```bash
PORTFOLIO_PREVIOUS_SOURCE_ROOT=data/evaluation/portfolio_baseline/budget_amendment_20261026/accepted_source_53f629fc
uv run --no-sync python scripts/amend_portfolio_budget.py \
  --budget-ledger "$PORTFOLIO_LEDGER" \
  --expected-ledger-sha256 "$PORTFOLIO_PREVIOUS_LEDGER_SHA256" \
  --previous-source-root "$PORTFOLIO_PREVIOUS_SOURCE_ROOT" \
  --configuration "$PORTFOLIO_AMENDED_CONFIGURATION" \
  --confirm-b0-completion-amendment
```

The existing absolute ledger must resolve to the original canonical path.
`PORTFOLIO_PREVIOUS_LEDGER_SHA256` is the reviewed prior ledger's full hash;
`PORTFOLIO_PREVIOUS_SOURCE_ROOT` is the verified asset-free original source copy
shown above, reconstructed from the accepted source/patch identities; do not use
the asset-bearing restoration directory. `PORTFOLIO_AMENDED_CONFIGURATION` is a
reviewed JSON file containing the full
effective B0 configuration, not a partial override. Prepare those non-secret
proof inputs before release; flags alone do not authorize migration or calls.
The coordinator can obtain that full configuration from the existing runner's
`_evaluation_session(...).configuration` under the preserved effective
environment, before entering its context or dispatching anything; alternatively
copy the bound stage configuration and replace only the independently recomputed
fingerprint, then verify exact equality against the actual runner configuration.
Do not invent a partial settings file or relax path/chat-limit bindings. The
migration itself needs no model, search, judge or quota request. Fresh collection
quota metadata is a separate prerequisite: a supplied `daily_batch_limit` below
24 still tightens the amended policy, including a retained value of 12.

The amender accepts only the exact prior 12/32/60 policy and proves that
the allowed fingerprint-scoped changes are the budget module
`backend/evaluation/portfolio_budget.py` plus the exact help-only `12` → `24`
substitution in `scripts/evaluate_runtime_workflow.py`. Other CLI changes or
application changes are rejected. Application settings, actual chat limits/paths, prompt/storage hashes and
all other bound configuration remain identical; only the proven evaluator
fingerprint may change. Preserve the complete old state in the amendment
history, all reservations/jobs/day usage/observations and canonical binding.
Today's stored 12 allowance may become 24; past days and independently stricter
metadata limits remain unchanged. Identical replay is a no-op; changed proof,
configuration or hash fails closed. No generic/automatic migration, copied
ledger, reset, or uncertain-job redispatch is supported. Verify the resulting
state before freezing the amended evaluation identity and resuming collection.

### Prospective daily policy amendment

The authorized prospective policy raises the internal daily ceiling from 24 to
50 while retaining 20 minimum demo headroom and all stage, judge, web and turn
limits. Existing ledgers reject ordinary sessions until explicitly amended. The
amender requires the original canonical ledger, its reviewed pre-amendment SHA256,
the full bound final configuration with only its verified evaluator fingerprint
changed, independently reviewed complete before/after source inventories and an
exact reviewed patch with their SHA256 identities. The caller must verify and
retain these accepted identities before invoking the CLI; self-selected hashes
alone do not establish that a change was approved.

Use `scripts/amend_portfolio_budget.py` with `--confirm-daily-policy-amendment`,
`--preserve-current-day-limit`, `--previous-inventory`,
`--previous-inventory-sha256`, `--current-inventory`,
`--current-inventory-sha256`, `--approved-patch` and `--approved-patch-sha256`,
in addition to the canonical ledger, prior hash, previous source and full
configuration arguments. The conservative option preserves all existing daily limits, reservations, jobs
and history; it never infers that a stored 24 was only the internal policy.
Alternatively `--reviewed-day-proof` and `--reviewed-day-proof-sha256` require
independently reviewed contemporaneous origin evidence for every stored quota
observation, including hash-bound original quota and provider captures, and an
explicit independent ceiling. The reviewer must establish the evidence is
authentic and excludes other stricter limits; a caller-selected hash or later
assertion does not do so. Expansion is capped by the strictest proven limit.
Without either accepted proof, a 20-used/24-limit day cannot admit six attempts.

A separately authorized prospective basis can establish present capacity without
claiming the historical 24 limit was policy-derived. Use
`--prospective-authority-proof`, `--prospective-authority-proof-sha256`,
`--expected-account-id`, and `--expected-key-id` in place of the conservative
or historical-origin option. The proof must cite `daily-50-2026-10-30`, bind the
reviewed capture-time ledger snapshot and capture start, and include the same-day
quota plus separately hash-bound raw OpenRouter key and model-list GET captures.
A separately reviewed nonsecret credential-binding record must identify the
account/key, link both raw capture hashes, and attest that all six configured
runtime roles use the queried key. The raw account data must establish the
actual remaining requests and external daily cap; the raw free-route record must
show `openrouter/free` with zero prompt/completion prices. The identity mapping
is an attestation requiring independent review and fresh execution-time
verification; caller-selected IDs, JSON statements or hashes do not establish
it. Evidence must be no more than 15 minutes old at atomic amendment.

Before making the authorized account and free-route GETs, establish a new
capture directory using `scripts/amend_portfolio_budget.py --budget-ledger
<canonical-absolute-ledger> --expected-ledger-sha256 <reviewed-current-sha256>
--capture-prospective-boundary <new-absolute-directory>`. This takes the
canonical exclusive lock, validates the old policy, and records the read-only
ledger snapshot and capture boundary without entering a budget session. Only
after that command completes should the operator capture the account, route,
quota and local credential-equality evidence. Capture the actual GET envelopes,
not manually reconstructed provider claims. Preserve the command transcript and
all evidence for independent review. Hashes and timestamps establish identity
and consistency only; they do not prove that the command or GETs actually ran
in the asserted order. Acceptance requires a trusted operator/reviewer to
verify that chronology and the configured-key/account mapping. Never open the
snapshot as a ledger or use it to reset the campaign.

The captured ledger snapshot is read-only evidence, never a replacement budget
ledger. Under the canonical lock, the amendment verifies its reservation prefix,
bindings and history against the reviewed current ledger. It retains prior
observations and establishes the new baseline as the lesser of captured usage
and reservations strictly before capture started. Reservations since that
boundary remain charged against the newly reported remaining requests. A reused
observation, stale capture, mismatched account/key/route, insufficient capacity,
or stricter external ceiling that prevents the next six attempts stops the
amendment. No old attempt is refunded and no unsupported historical claim is
made. With a fresh 31 requests remaining and effective reserve 20, at most 11
requests remain available before subsequent consumption; the observation must
be refreshed before dispatch if it expires.

For the separately reviewed quota-input lifecycle, an already-migrated policy50
ledger needs an explicit final evaluator identity transition before using the new
fingerprint. `--confirm-final-evaluator-transition` requires the canonical ledger,
its expected current SHA256, the complete unchanged final configuration except
for the reviewed fingerprint, and reviewed previous/current source inventories
and exact patch. It changes only the final binding and appends transition evidence;
B0 and all saved jobs, contracts, reservations and history remain unchanged.

For each new quota observation, use `--capture-prospective-boundary` before
fetching current account/route metadata, then independently verify the raw
captures and all six runtime-key bindings as described above. Register the
reviewed proof explicitly with `--register-prospective-observation <proof>` and
`--prospective-authority-proof-sha256 <sha256>`, the canonical ledger and its
expected current hash. Registration preserves existing baselines, charges any
reservations after the protected boundary against fresh remaining capacity, and
rejects reused observations. New UTC days start with the existing policy defaults
without changing earlier day state. The witness and proof-review prerequisites
remain necessary: hashes alone do not prove capture chronology or key identity.

The successor accepts `--quota-snapshot <absolute-path>` after the reviewed
freeze SHA256. Pass the exact reviewed registered quota snapshot from a separate
path; it is copied to a unique per-attempt artifact, and the old continuation
`quota.json` remains untouched. The successor also requires the registered
baseline inside the locked live session; a stale, missing, changed or mismatched
observation fails before model dispatch.

For each single pending final run, pass `--require-final-runtime-admission` to
check the six-attempt bound inside the locked session before dispatch. This
checks runtime 54, final 59, overall 92, the stored day ceiling and spending
since the current quota observation. The successor also rehashes the captured
model/storage file inventory and verifies the accepted freeze and configuration
before each invocation. It does not authorize provider calls or
judge execution. The existing per-turn attempt and retry controls remain in force.

### Daily resumable collection after authorization

Apply the fixed daily ceiling of 50 combined runtime/retry/judge attempts and
minimum demo headroom of 20 on every UTC day; metadata may only tighten them.
Review current sanitized quota and spend conditions, reserve demo capacity, then
choose only remaining predeclared cases/jobs within both daily availability and
stage/whole-effort limits. Reserve retries and eligible judges before allocating
new runtime work. Carry the same ledger forward; a reset day, new shell, new run
ID, or changed evaluation root never resets effort usage. A quota pause leaves
remaining observations uncollected, not failed or successful.

Resume requires the same stage-bound effective application configuration,
actual `ChatLimitService` settings and `PathSettings`, selected prompt tree, and
storage-manifest hashes. Record these after applying the verified local-storage
overrides; changed values block resume rather than silently creating replacement
observations. Switching a model alias does not reset the shared observed quota.

Completed published jobs are skipped without another dispatch or report sample.
A fully repeated command creates no new completed contribution; a partial resume
publishes only new or previously unpublished jobs. Durable publication intent
and immutable manifest reconciliation prevent a crash/restart from counting a
previous observation twice. Preserve original run IDs; a saved job is not a new
independent live repetition. Never delete a manifest or ledger publication state
to force another attempt.

After each sequential batch, retain terminal output, raw records, evidence,
failures, immutable run manifests, configured/observed model identities, usage,
and ledger identity. Reconcile dispatched and uncertain attempts without
refunds; record unavailable billing/identity fields explicitly. Never edit a
completed run to fill a missing record. Resume only the uncollected planned
observations when authorization and verified quota allow it.

Submit B0 preservation, restore/preflight evidence, results and every uncollected
observation/blocker for checkpoint disposition before behavior/prompt changes.
An accepted archive-only live baseline permits local code/regression and retrieval
comparison; it cannot support improved live task success, latency, token usage,
or generated-answer quality. Historical traces and mocks cannot replace missing
OpenRouter observations. Later collection on the preserved checkout still needs
its own authorization within the same effort envelope.

## Complexity 0 — prepare and freeze the campaign

For a separately authorized full/final campaign (B0 first preserves the pre-fix
state described above), before running measurements:

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

## Complexity 4 — run controlled live scenarios

Provider-consuming examples below require the shared runner parameters. Set
these only after authorization and readiness review; file creation/flags do not
authorize calls. Full-campaign examples use `final`; B0 uses only its predeclared
subset and `B0` stage. The normalized quota file is distinct from the raw
sanitized provider response and must reflect a nonfuture observation from the
current UTC day, at most 15 minutes old at entry and every reservation:

```bash
PORTFOLIO_STAGE=final
PORTFOLIO_QUOTA=data/evaluation/portfolio_baseline/preparation/quota_runner.json
PORTFOLIO_JUDGE_MODEL=openrouter/free
```


These full-campaign examples exceed the B0 subset when combined. Apply the
approved stage allocation and shared ledger; do not execute them as a B0 batch.

Live mode invokes the real compiled workflow with configured models and prepared
storage. It requires explicit provider confirmation and at least one `--case-id`.
There is deliberately no implicit “run everything” mode, which prevents an
accidental burst of model and Tavily calls.

The following batches cover the reviewed suite while respecting the normal
three-execution limit.

Local answer and graph behavior:

```bash
uv run --locked python scripts/evaluate_runtime_workflow.py live \
  --budget-stage "$PORTFOLIO_STAGE" \
  --provider-quota "$PORTFOLIO_QUOTA" \
  --case-id local_transformer_answer_success \
  --case-id contextual_mamba_visualization \
  --case-id explicit_mamba_visualization_success \
  --confirm-provider-calls
```

Mentor and graph coverage:

```bash
uv run --locked python scripts/evaluate_runtime_workflow.py live \
  --budget-stage "$PORTFOLIO_STAGE" \
  --provider-quota "$PORTFOLIO_QUOTA" \
  --case-id initial_attention_quiz_route \
  --case-id mentor_stuck_continuation \
  --case-id transformer_encoder_graph_coverage \
  --confirm-provider-calls
```

Continuity and out-of-scope control:

```bash
uv run --locked python scripts/evaluate_runtime_workflow.py live \
  --budget-stage "$PORTFOLIO_STAGE" \
  --provider-quota "$PORTFOLIO_QUOTA" \
  --case-id contextual_medusa_web_query \
  --case-id colbert_guidance_mode_continuation \
  --case-id out_of_scope_biological_cats \
  --confirm-provider-calls \
  --allow-tavily
```

Web fallback regressions:

```bash
uv run --locked python scripts/evaluate_runtime_workflow.py live \
  --budget-stage "$PORTFOLIO_STAGE" \
  --provider-quota "$PORTFOLIO_QUOTA" \
  --case-id ambiguous_eagle_without_history \
  --case-id explicit_medusa_web_fallback_success \
  --case-id in_scope_retnet_local_miss_fallback \
  --confirm-provider-calls \
  --allow-tavily
```

Researcher truncation reliability, as three independent executions:

```bash
uv run --locked python scripts/evaluate_runtime_workflow.py live \
  --budget-stage "$PORTFOLIO_STAGE" \
  --provider-quota "$PORTFOLIO_QUOTA" \
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
workflow or Tavily again. It does call the explicitly selected judge model, is
non-gating, and requires `--confirm-provider-calls`.

Current configuration:

- provider and model: explicit `--judge-provider openrouter` and `--judge-model`;
  there is no automatic inheritance of Analyst model identity;
- OpenRouter `openrouter/free` may be selected separately as a variable router
  observation; its actual resolved identities remain unknown when unavailable;
- temperature: forced to 0;
- maximum output: 4,096 tokens for claim faithfulness and 512 for Boolean dimensions;
- claim output: at most 32 atomic claims, five evidence IDs per claim, and bounded strings;
- structured-output transport: `function_calling`;
- evaluator retries: 0;
- normal limit: three case/dimension jobs per command;
- result rubric version: `runtime-semantic-v2`.

The opt-in fixed pilot has a separate four-assessment limit and one shared,
durable retry for confirmed timeouts, connection failures, or HTTP 5xx. The
canonical ledger pins actual observed serving identity across jobs and saved
results. Available identity fields from failed attempts must also match later
attempts and saved history, even before the first complete identity is pinned;
an identity-free transient can still be retried. Each attempt is reserved and
reported, including unknown usage; failed
or uncertain jobs are not automatically redispatched. See the
[fixed-pilot contract](runtime_evaluation.md#optional-semantic-judges) for
non-retryable failures and the unchanged live-collection prerequisites.

For the approved fixed pilot, register a separate fresh judge-model quota proof using the same protected capture and ledger guards. Its route evidence must bind the exact free Xing model, zero-price Phala endpoint and supported structured-output parameters; the judge key/account binding is independent of runtime role evidence. The [fixed-pilot capacity and recovery rules](runtime_evaluation.md#fixed-pilot-capacity-registration-and-setup-recovery) describe the reviewed successor transition and the single no-dispatch recovery operation. They do not authorize a replay of any uncertain job.

The command prints the effective provider and model before calling it. Boolean
dimensions use OpenEvals; claim faithfulness uses the repository's structured
claim-to-bounded-evidence assessment. `--structured-output-method json_schema`
and `--structured-output-method json_mode` are diagnostic alternatives; use
them only for a deliberate rerun because each judge job can call the provider.
JSON mode receives an explicit required-schema instruction.

Analyst grounding and usefulness:

```bash
uv run --locked --group evaluation python scripts/evaluate_runtime_workflow.py judge \
  --budget-stage "$PORTFOLIO_STAGE" \
  --provider-quota "$PORTFOLIO_QUOTA" \
  --judge-provider openrouter \
  --judge-model "$PORTFOLIO_JUDGE_MODEL" \
  --source-run latest \
  --case-id local_transformer_answer_success \
  --dimension claim_faithfulness \
  --dimension analyst_usefulness \
  --confirm-provider-calls
```

Mentor quality:

```bash
uv run --locked --group evaluation python scripts/evaluate_runtime_workflow.py judge \
  --budget-stage "$PORTFOLIO_STAGE" \
  --provider-quota "$PORTFOLIO_QUOTA" \
  --judge-provider openrouter \
  --judge-model "$PORTFOLIO_JUDGE_MODEL" \
  --source-run latest \
  --case-id mentor_stuck_continuation \
  --dimension mentor_pedagogy \
  --dimension conversational_continuity \
  --confirm-provider-calls
```

Graph usefulness:

```bash
uv run --locked --group evaluation python scripts/evaluate_runtime_workflow.py judge \
  --budget-stage "$PORTFOLIO_STAGE" \
  --provider-quota "$PORTFOLIO_QUOTA" \
  --judge-provider openrouter \
  --judge-model "$PORTFOLIO_JUDGE_MODEL" \
  --source-run latest \
  --case-id explicit_mamba_visualization_success \
  --dimension graph_usefulness \
  --confirm-provider-calls
```

`--source-run latest` means the newest completed offline/live run in the current
fingerprint containing every selected case. If no single run contains the
selected case, or tracked files/configuration changed since it ran, the command
reports “No completed runtime run contains all selected cases.” Use the
exact run ID to target an older compatible record. New source collection needs
explicit authorization and remaining allocation after freezing the fingerprint. `--records` supports direct legacy
record input but is not preferred for a canonical campaign.

Faithfulness is skipped/N/A without bounded evidence; graph usefulness is
skipped/N/A without bounded graph labels. A missing required final worker is
also skipped/N/A, while the deterministic routing failure remains visible.
Provider, parser, and schema failures are judge execution errors with unavailable
scores. All remain non-gating. `--allow-expanded-judge-suite` permits more than
three jobs after explicit quota review.

### Verify the judge corrections before continuing

This is a full-campaign diagnostic procedure, not additional B0 collection.
Fresh source observations and diagnostic judge reruns below require explicit
authorization and available stage/effort/daily budget. Do not run them in addition
to the predeclared B0 jobs.

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
  --budget-stage "$PORTFOLIO_STAGE" \
  --provider-quota "$PORTFOLIO_QUOTA" \
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
  --budget-stage "$PORTFOLIO_STAGE" \
  --provider-quota "$PORTFOLIO_QUOTA" \
  --judge-provider openrouter \
  --judge-model "$PORTFOLIO_JUDGE_MODEL" \
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
  --budget-stage "$PORTFOLIO_STAGE" \
  --provider-quota "$PORTFOLIO_QUOTA" \
  --judge-provider openrouter \
  --judge-model "$PORTFOLIO_JUDGE_MODEL" \
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
  --budget-stage "$PORTFOLIO_STAGE" \
  --provider-quota "$PORTFOLIO_QUOTA" \
  --judge-provider openrouter \
  --judge-model "$PORTFOLIO_JUDGE_MODEL" \
  --source-run "$EVAL_LIVE_RUN_ID" \
  --case-id mentor_stuck_continuation \
  --dimension mentor_pedagogy \
  --dimension conversational_continuity \
  --confirm-provider-calls
```

Run graph usefulness only when the graph source prerequisite passed:

```bash
uv run --locked --group evaluation python scripts/evaluate_runtime_workflow.py judge \
  --budget-stage "$PORTFOLIO_STAGE" \
  --provider-quota "$PORTFOLIO_QUOTA" \
  --judge-provider openrouter \
  --judge-model "$PORTFOLIO_JUDGE_MODEL" \
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
| Live/judge | `--budget-stage B0` or `--budget-stage final` | Required shared effort allocation; never resets prior usage |
| Live/judge | `--provider-quota PATH` | Required normalized nonfuture same-UTC-day quota, at most 15 minutes old at entry/every reservation, with demo reserve and daily batch limit |
| Live/judge | `--budget-ledger PATH` | Existing absolute path resolving to the bound original canonical ledger; required for restored collection; other/copy ledgers rejected |
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
