# Evaluation overview

The [latest system-evaluation report](latest.md) presents two separate historical
editions. These are measurements of the configurations and dates recorded in the
report, not of every later revision or a controlled before/after comparison.

| System evaluation | Representative result | Coverage and limit |
| --- | --- | --- |
| August 2026 runtime | 6 / 12 imported-trace observations and 10 / 15 live executions passed deterministic contracts | Trace source configuration needs verification; a pass is not answer correctness. |
| August 2026 retrieval | 33.33% case pass rate over 12 benchmark cases | Relevance labels were incomplete; individual metric denominators are not all recorded. |
| October 2026 runtime | 10 / 12 completed live executions passed; two failed and three of 15 planned were interrupted | Three evaluator identities remain separate in the detailed report. |
| October 2026 answer and graph quality | **Unavailable**: one HTTP 429 attempt, zero successful scores, three unstarted of four eligible | No judge rerun or answer-quality score; token usage is incomplete. |

The [structured system results](latest.json) and [provenance](latest_provenance.json)
match the [detailed report](latest.md); the [publication index](index.md) retains
earlier snapshots. October system retrieval-benchmark quality is unavailable;
its workflow retrieval-status checks do not measure evidence precision.

The separate [optimization experiment catalog](experiments/README.md) includes
the local A/B/C/D retrieval comparison and its limitations. See the
[evaluation methodology](../../docs/runtime_evaluation.md) and
[execution and publication instructions](../../docs/full_evaluation_report_runbook.md).
Raw traces and evidence remain private. Missing values are not zero.
