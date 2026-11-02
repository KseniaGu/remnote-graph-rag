# Evaluation overview

## System Evaluation

System Evaluation describes how the assembled application behaves on defined
cases. Retrieval metrics assess evidence and graph support; runtime metrics cover
routing, agents, tools, arguments, fallback and completion. Reliability and
efficiency record failures, latency, attempts, retries, web use and available
token usage. Optional judges assess answer and graph quality. Coverage and
provenance identify what was evaluated and under which configuration.

| Example | Result | Population and units |
| --- | ---: | --- |
| Runtime deterministic success | 10 / 12 | October 2026 completed live executions; 3 of 15 planned were interrupted |
| Retrieval case pass rate | 33.3% | August 2026, 12 benchmark cases; relevance labels incomplete |
| Judge answer and graph quality | N/A | October pilot: 4 eligible inputs, no successful scores |

See the [complete system report](latest.md), [structured results](latest.json),
[provenance](latest_provenance.json) and [history](index.md). These observations
apply to their recorded versions and dates, not every later revision. Imported
traces and live executions have different origins and must be interpreted separately.

A deterministic pass covers gating checks; worker-step and Tavily-search bounds
can gate, while several other budgets are report-only. Each metric uses its own
eligible denominator. N/A means unavailable or inapplicable, not zero. Workflow
compliance does not prove that retrieved evidence supports an answer, and evidence
support does not establish generated-answer quality.

## Optimization Experiments

Experiments compare targeted design choices under explicitly recorded conditions,
cohorts and limitations. The [experiment catalog](experiments/README.md) includes
the retrieval comparison; its measurements are not system-evaluation scores.

## Automated Software Checks

Synthetic and mocked tests exercise application and evaluator contracts without
turning CI into a live evaluation. See the [offline testing instructions](../../README.md)
and [CI workflow](../../.github/workflows/quality.yml). Methodology and execution
are described in the [evaluation guide](../../docs/runtime_evaluation.md) and
[report runbook](../../docs/full_evaluation_report_runbook.md).
