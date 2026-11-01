# Portfolio evaluation: controlled final observations

This is an offline summary of saved evaluation artifacts, not a publication
scorecard or a production-traffic measurement. The controlled runtime completed
12 of 15 planned observations across 13 distinct scenarios. Ten of those 12 passed their deterministic
contracts; two failed, and three other observations were interrupted with
uncertain outcomes. The fixed-judge pilot stopped after one HTTP 429, so no judge
quality score is available.

| Area | Saved result and population | What it establishes |
| --- | --- | --- |
| Task / contract | 12/15 complete; 10/12 completed pass, 2/12 fail, 3/15 interrupted; 10/15 planned have observed passes | Contract compliance only, not factual correctness |
| Retrieval evidence | Score availability: Precision@10 176/176 observations; required-point Recall 145/176, with 31/99 development N/A and 77/77 frozen available | Reviewed source support in a separate local retrieval comparison, not answer quality |
| Answer quality | No successful judge assessment among four eligible inputs; final answer correctness/completeness unmeasured | Missing scores are N/A, not zero |
| Reliability | One judge request returned HTTP 429; three further assessments unstarted; no retry | A known request error; its rate-limit origin and cooldown are unknown |
| Latency | Fresh-process CPU retrieval-call timing only; matched C−B mean +274.266 s development (18 pairs), +204.258 s frozen (14 pairs) | Local reranking cost, not production latency; startup and scoring excluded |
| Resources | Final runtime 45/54 LLM attempts; judge 1/5; whole effort 79/92. Final web 5/7 searches and 5/14 attempts | Attempt reservations include errors/retries; complete token cost unavailable |

## Runtime outcomes and quality boundaries

A deterministic pass requires all applicable gating checks in the
[scenario contract](../../evals/runtime/scenarios.json) to pass; report-only
checks do not change that result (see [runtime methods](../../docs/runtime_evaluation.md#scorecard)).
The deterministic pass share is 83.3% of the 12 completed, scored records. The 10/15
planned observed-pass count does not assign a score to interrupted cases. The
Mentor continuation failed its routing/worker-step contract; RetNet failed its
contextual web-topic check although a response and fallback were present. The
three interruptions have no RuntimeRecord or quality score. These negatives and
missing outcomes are retained, with original fingerprints.

The pilot had four eligible inputs, one attempted error, zero successful/scored,
three unstarted, zero skipped and zero uncertain dispatches. The HTTP response
was 429; no response body or rate-limit headers were retained. Neither its origin
nor cooldown is known. The configured fixed model and Phala-only routing are
not observed serving identity: serving identity was unavailable for 1/1 attempt.
Input, output and total token availability were each 0/1; subtotals and averages
are N/A. The [sanitized pilot status](fixed_judge_pilot_status_20261031.json)
binds the saved stop evidence. Four judge attempts and nine runtime attempts
remain unspent; the shared judge retry was not used.

The user-endorsed Transformer reference verdicts concern support from evidence
captured for the saved baseline, not correctness labels for the final response.
Human findings on Mamba graph focus are retained; the graph-focus limitation is
unresolved. A contract pass, retrieved reference, and fluent answer are not
interchangeable measures of factual accuracy or completeness.

## Conditions and provenance

Runtime observation dates were October 30–31, 2026 UTC. The saved manifests
retain these original evaluator fingerprint groups; later evaluator changes do
not re-fingerprint the runtime records.

| Runtime observations (ordered inventory) | Original fingerprint | Disposition |
| --- | --- | --- |
| 1–7, October 30 | `1095d6342e69b04143ef971d` | Seven completed/scored |
| 8–9, October 30 | `2b1301ae85cf5a5a77713187` | Two completed/scored |
| 10–15, October 30–31 | `aacdf873baacf605652888e1` | Three completed/scored, three interrupted/unscored |

All three groups bind authoritative `evals/runtime/scenarios.json` SHA256
`b0dfd1af372437e8c52257c70613e76130efe9721907300252b81d1bc4619f31`
and selected prompt-tree SHA256
`f2f9faf450f38376fee1340064b8ce0a6f19bcaaa1afb519cae28ebdd51e1f4b`.
Role selections were Orchestrator routing v6 / graph-index v2, Retriever v7,
Researcher shared v6 (both stage settings v1), Analyst v6 and Mentor v4.
All generative roles configured variable `openrouter/free`; temperatures were
0 for Orchestrator/Retriever/both Researcher stages, 0.1 Analyst and 0.7 Mentor.
Analyst used `legacy_vector_context`, Visualizer `optimized`; document/index/graph
stores were local, vector storage Pinecone, and checkpoints MongoDB.
The selected prepared-storage manifest SHA256 was
`48c0d76bad4b1f8563199c8e398c0b2ce6d80543be30071138ad67d9ea5d6db7`,
also identified in the [sanitized retrieval provenance](retrieval_ablation_20261030.json).
This identifies local prepared assets, not cloud-vector or deployed-demo parity.
Some captured runtime events report free-model IDs that vary; missing
serving-provider fields and uncaptured/failed-attempt identities remain unknown. Configured routing is not a fixed serving model.
The later v3 judge evaluator is separately
`e3545f82672e7197c6e567f2`, bound by the [sanitized pilot freeze/status](fixed_judge_pilot_status_20261031.json);
it is not the evaluator identity of these runtime groups.

## Engineering example: Researcher context handoff

WP4.4 investigated a contextual Medusa case whose saved baseline took a local
answer path; that record alone could not establish whether its evidence was
substantively sufficient or diagnose a Researcher failure.
A separate offline actual-node reproduction showed that, after a valid local
gap cleared context, Researcher received the follow-up without prior topic
history or the older session summary.
The focused correction passes existing bounded history and summary through
[Researcher message assembly](../../backend/workflows/learner.py) and selects the
[shared v6 prompt](../../backend/llm/prompts/learner_workflow/researcher/v6.yaml),
without lowering adequacy thresholds.
[Scripted regression cases](../../tests/test_researcher_context.py) cover both
retrieval modes, context-preserving web fallback, adequate evidence staying
local and source exhaustion; the saved focused gate was 27 passed / 1 expected
failure, which demonstrates wiring rather than live model adherence.
Sanitized evidence identities are the accepted WP4.4 source tree
`3dbc6f3c7e8f9d553385f9007d64e3165a53dfba` and patch SHA256
`25b5b8b82b953ca64ee8c309dc10a99f2e28fc239824dfc4552d290d2e9742e1`;
the later [accepted source-freeze identity](fixed_judge_pilot_status_20261031.json)
retains the source lineage without publishing private trace excerpts.
The current live contextual Medusa observation passed its deterministic contract,
while RetNet still failed the contextual web-topic requirement, so substantive
answerability, model compliance and unresolved graph focus remain limitations,
with no causal overall quality improvement established.

## Retrieval comparison and limits

The [retrieval comparison](../../docs/retrieval_ablation.md) and its [aggregate
and provenance](retrieval_ablation_20261030.json) retain development and frozen
cohorts, definitions, denominators and matched differences. The 176/176
Precision and 145/176 Recall counts above describe score availability, not
quality values: the [cohort/arm means table](../../docs/retrieval_ablation.md#arms-and-observed-evidence)
reports development A/B/C/D Precision 49.44/48.33/45.00/36.67% and frozen
33.57/34.29/32.14/28.57%; available-only development Recall is 100% in each
arm, and frozen Recall is 85.71/85.71/78.57/100%. Precision uses relevance
weights 1/0.5/0 over the actual first ten unique sources, divided by fixed ten
(absent slots contribute zero); Recall counts required points with a fully
supported source in that list, with unknown mappings making exact Recall N/A. Graph-assisted B
minus source-only A was −1.11 percentage points Precision@10 on development
(27 pairs) and +0.71 on frozen (21 pairs). C minus B was −3.33 points on
development (18 pairs) and −2.14 on frozen (14 pairs). Development Recall is
available-only; the 31 N/A observations prevent complete-cohort claims. Two of
24 measured runs were interrupted. Labels combine prior reviewed references
and AI-assisted decisions endorsed by the user, not independent expert ratings.

Different prompts, contracts, evaluator versions and variable free-model
routing limit baseline/final comparisons. The sample is small; deployment,
vector-index and cloud parity are incomplete. Source support does not prove
answerability, and generated-answer correctness was not measured by the local
retrieval experiment. All 12 completed final runtime records lack complete token
totals; absent metadata is not zero. The prior historical published scorecards
remain separate and unchanged.

The controlled runtime inventory SHA256 is
`7e63ed839833b45304c00d100ffd475517cc37e2f7c7c961d32aff5ef9d5df0f`;
the pilot stop freeze is recorded in the sanitized status. Their underlying
private evidence remains private. Further independent final-report and M2
assessment is required; this report does not claim M2 completion.
