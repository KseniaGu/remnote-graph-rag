# System evaluation: historical editions

August and October are separate observations with different configurations and evaluator identities. They are not a controlled before/after comparison.

## August evaluation

- August 22, 2026 UTC. Fingerprint: `f8d6f3b6f3e0348c75a194bf`. The original scorecard contains 12 imported-trace observations and 15 live executions.
- Imported traces: 6 passed / 12 scored; live: 10 passed / 15 scored.
- Recorded live prompt selections: Orchestrator routing v5 / graph-index v2, Retriever v6, Researcher v5, Analyst v6, Mentor v4. Analyst retrieval: legacy_vector_context; Visualizer: optimized.
- Recorded live models: Orchestrator nemotron-3-super:cloud; Retriever nemotron-3-super:cloud; Analyst qwen3.5:cloud; Mentor nemotron-3-super:cloud. Provider and remaining role details are in the unchanged provenance.
- The imported trace configuration requires verification. The following historical scorecard rates combine the August sources only; metric-specific eligibility counts were not recorded.
- The trace's source configuration is not established by the recorded live-configuration snapshot.

### Recorded runtime behavior

| Metric | Recorded value | Population |
| --- | ---: | --- |
| task success rate | 59.26% | 27 total; eligible count not recorded |
| routing correctness | 66.67% | 27 total; eligible count not recorded |
| required agent compliance | 65.38% | 27 total; eligible count not recorded |
| forbidden agent compliance | 80.77% | 27 total; eligible count not recorded |
| tool selection correctness | 81.48% | 27 total; eligible count not recorded |
| tool argument validity | 66.67% | 27 total; eligible count not recorded |
| one tool per worker | 100.00% | 27 total; eligible count not recorded |
| retrieval status correctness | 93.33% | 27 total; eligible count not recorded |
| local to web fallback correctness | 75.00% | 27 total; eligible count not recorded |
| source exhaustion correctness | 85.71% | 27 total; eligible count not recorded |
| unnecessary web rate | 20.00% | 27 total; eligible count not recorded |
| final response rate | 85.19% | 27 total; eligible count not recorded |
| modality correctness | 85.19% | 27 total; eligible count not recorded |
| termination rate | 96.30% | 27 total; eligible count not recorded |
| graph contract rate | 83.33% | 27 total; eligible count not recorded |

### Retrieval, reliability, efficiency, and judge

- Retrieval benchmark: 12 cases; recorded case pass rate 33.33%. Metric-specific eligibility counts are unavailable in the saved scorecard.
- evidence chunk recall: 56.39% (eligible count not recorded).
- concept recall: 53.53% (eligible count not recorded).
- relation recall: 36.11% (eligible count not recorded).
- context precision at 10: 36.88% (eligible count not recorded).
- context recall: 63.12% (eligible count not recorded).
- Reliability: 27 recorded observations; failure counts: {'invalid_retriever_output': 2, 'model_output_truncated': 3, 'structured_output_parse_error': 1, 'sources_exhausted': 3}.
- logical llm calls: mean 3.44 over 27 recorded observations.
- provider attempts: mean 3.44 over 27 recorded observations.
- retries: mean 0.00 over 27 recorded observations.
- tavily searches: mean 0.56 over 27 recorded observations.
- latency seconds: mean 39.66 over 27 recorded observations.
- total tokens: mean 6692.85 over 27 recorded observations.
- Historical token counters lack modern field-level availability evidence and are not complete billing totals.
- Judge quality: N/A; saved dimensions contain errors or skipped prerequisites, with zero successful scores.
- Retrieval relevance-label coverage was incomplete. See the [unchanged August scorecard](history/2026-08-22_f8d6f3b6f3e0/scorecard.md) and [provenance](history/2026-08-22_f8d6f3b6f3e0/provenance.json) for recorded definitions and configurations.

## October evaluation

- Live system evaluation, October 30–31, 2026 UTC: 15 planned executions across 13 scenarios; 12 completed, 10 passed, 2 failed and 3 interrupted/unscored.
- Interrupted observations have no scored record. No result here measures later revisions. Budget checks are report-only and do not determine deterministic pass/fail.
- Failures: Mentor continuation routing/worker-step contract and RetNet contextual web-topic contract. Three interrupted observations have no runtime record or score.
- Configuration: Orchestrator routing v6 / graph-index v2, Retriever v7, Researcher v6, Analyst v6 and Mentor v4. Generative roles used the variable openrouter/free route; serving identities were not always observed.
- Analyst used legacy vector context; Visualizer used optimized retrieval. Document, index and graph stores were local, vector storage Pinecone, and checkpoints MongoDB.
- All 12 completed runtime records lack complete token totals. Runtime attempt reservations were 45 of 54; full effort used 79 of 92, including failures and retries.

### Metrics by original evaluator fingerprint

#### `1095d6342e69b04143ef971d`

Scored 7 of 7 planned observations; passed 6 / 7. Other observations are interrupted and unscored.

| Recorded check | Pass / eligible | Not observed |
| --- | ---: | ---: |
| answer.final_response_present | 7 / 7 | 0 |
| budget.latency_seconds | 7 / 7 | 0 |
| budget.logical_llm_calls | 6 / 7 | 0 |
| budget.provider_attempts | 5 / 7 | 0 |
| budget.retries | 7 / 7 | 0 |
| budget.tavily_searches | 7 / 7 | 0 |
| budget.total_tokens | N/A — no observed eligible checks | 7 |
| budget.worker_steps | 6 / 7 | 0 |
| fallback.sources_exhausted | 5 / 5 | 0 |
| graph.artifact_present | 3 / 3 | 0 |
| graph.edge_count_max | 3 / 3 | 0 |
| graph.edge_count_min | 3 / 3 | 0 |
| graph.no_dangling_edges | 3 / 3 | 0 |
| graph.node_count_max | 3 / 3 | 0 |
| graph.node_count_min | 3 / 3 | 0 |
| graph.required_nodes | 1 / 1 | 0 |
| modality.requested_modality | 7 / 7 | 0 |
| reliability.forbidden_failures | 7 / 7 | 0 |
| retrieval.retrieval_outcome | 4 / 4 | 0 |
| retrieval.retrieval_status | 5 / 5 | 0 |
| routing.basic_loop_detector | 7 / 7 | 0 |
| routing.forbidden_agents | 6 / 7 | 0 |
| routing.required_agents | 6 / 7 | 0 |
| routing.worker_sequence | 6 / 7 | 0 |
| scope.request_scope | 7 / 7 | 0 |
| termination.terminated | 7 / 7 | 0 |
| tools.deep_web_research.topic | 1 / 1 | 0 |
| tools.forbidden_tools | 6 / 7 | 0 |
| tools.get_subgraphs_to_visualize.queries | 3 / 3 | 0 |
| tools.one_tool_per_worker | 7 / 7 | 0 |
| tools.required_tools | 5 / 5 | 0 |
| tools.search_knowledge_base.queries | N/A — no check recorded | 0 |
| tools.search_knowledge_base.required_topics | N/A — no check recorded | 0 |

| Efficiency metric | Mean | Available / scored |
| --- | ---: | ---: |
| worker steps | 2.00 | 7 / 7 |
| logical llm calls | 2.71 | 7 / 7 |
| provider attempts | 2.86 | 7 / 7 |
| retries | 0.14 | 7 / 7 |
| tavily searches | 0.14 | 7 / 7 |
| latency seconds | 30.79 | 7 / 7 |
| input tokens | N/A — not captured in scored records | 0 / 7 |
| output tokens | N/A — not captured in scored records | 0 / 7 |
| total tokens | N/A — not captured in scored records | 0 / 7 |

Recorded classified failure counts: none (check failures are shown separately above).
- loop detected: 0 / 7 observed.
- output limit hit: 0 / 7 observed.

#### `2b1301ae85cf5a5a77713187`

Scored 2 of 2 planned observations; passed 2 / 2. Other observations are interrupted and unscored.

| Recorded check | Pass / eligible | Not observed |
| --- | ---: | ---: |
| answer.final_response_present | 2 / 2 | 0 |
| budget.latency_seconds | 2 / 2 | 0 |
| budget.logical_llm_calls | 2 / 2 | 0 |
| budget.provider_attempts | 2 / 2 | 0 |
| budget.retries | 2 / 2 | 0 |
| budget.tavily_searches | 2 / 2 | 0 |
| budget.total_tokens | N/A — no observed eligible checks | 2 |
| budget.worker_steps | 2 / 2 | 0 |
| fallback.sources_exhausted | 1 / 1 | 0 |
| graph.artifact_present | N/A — no check recorded | 0 |
| graph.edge_count_max | N/A — no check recorded | 0 |
| graph.edge_count_min | N/A — no check recorded | 0 |
| graph.no_dangling_edges | N/A — no check recorded | 0 |
| graph.node_count_max | N/A — no check recorded | 0 |
| graph.node_count_min | N/A — no check recorded | 0 |
| graph.required_nodes | N/A — no check recorded | 0 |
| modality.requested_modality | 2 / 2 | 0 |
| reliability.forbidden_failures | 2 / 2 | 0 |
| retrieval.retrieval_outcome | N/A — no check recorded | 0 |
| retrieval.retrieval_status | N/A — no check recorded | 0 |
| routing.basic_loop_detector | 2 / 2 | 0 |
| routing.forbidden_agents | 2 / 2 | 0 |
| routing.required_agents | 1 / 1 | 0 |
| routing.worker_sequence | 2 / 2 | 0 |
| scope.request_scope | 2 / 2 | 0 |
| termination.terminated | 2 / 2 | 0 |
| tools.deep_web_research.topic | N/A — no check recorded | 0 |
| tools.forbidden_tools | 2 / 2 | 0 |
| tools.get_subgraphs_to_visualize.queries | N/A — no check recorded | 0 |
| tools.one_tool_per_worker | 2 / 2 | 0 |
| tools.required_tools | N/A — no check recorded | 0 |
| tools.search_knowledge_base.queries | N/A — no check recorded | 0 |
| tools.search_knowledge_base.required_topics | N/A — no check recorded | 0 |

| Efficiency metric | Mean | Available / scored |
| --- | ---: | ---: |
| worker steps | 0.50 | 2 / 2 |
| logical llm calls | 1.50 | 2 / 2 |
| provider attempts | 1.50 | 2 / 2 |
| retries | 0.00 | 2 / 2 |
| tavily searches | 0.00 | 2 / 2 |
| latency seconds | 6.17 | 2 / 2 |
| input tokens | N/A — not captured in scored records | 0 / 2 |
| output tokens | N/A — not captured in scored records | 0 / 2 |
| total tokens | N/A — not captured in scored records | 0 / 2 |

Recorded classified failure counts: none (check failures are shown separately above).
- loop detected: 0 / 2 observed.
- output limit hit: 0 / 2 observed.

#### `aacdf873baacf605652888e1`

Scored 3 of 6 planned observations; passed 2 / 3. Other observations are interrupted and unscored.

| Recorded check | Pass / eligible | Not observed |
| --- | ---: | ---: |
| answer.final_response_present | 3 / 3 | 0 |
| budget.latency_seconds | 3 / 3 | 0 |
| budget.logical_llm_calls | 3 / 3 | 0 |
| budget.provider_attempts | 3 / 3 | 0 |
| budget.retries | 3 / 3 | 0 |
| budget.tavily_searches | 3 / 3 | 0 |
| budget.total_tokens | N/A — no observed eligible checks | 3 |
| budget.worker_steps | 3 / 3 | 0 |
| fallback.sources_exhausted | 3 / 3 | 0 |
| graph.artifact_present | N/A — no check recorded | 0 |
| graph.edge_count_max | N/A — no check recorded | 0 |
| graph.edge_count_min | N/A — no check recorded | 0 |
| graph.no_dangling_edges | N/A — no check recorded | 0 |
| graph.node_count_max | N/A — no check recorded | 0 |
| graph.node_count_min | N/A — no check recorded | 0 |
| graph.required_nodes | N/A — no check recorded | 0 |
| modality.requested_modality | 3 / 3 | 0 |
| reliability.forbidden_failures | 3 / 3 | 0 |
| retrieval.retrieval_outcome | N/A — no check recorded | 0 |
| retrieval.retrieval_status | 1 / 1 | 0 |
| routing.basic_loop_detector | 3 / 3 | 0 |
| routing.forbidden_agents | 2 / 2 | 0 |
| routing.required_agents | 3 / 3 | 0 |
| routing.worker_sequence | 3 / 3 | 0 |
| scope.request_scope | 3 / 3 | 0 |
| termination.terminated | 3 / 3 | 0 |
| tools.deep_web_research.topic | 2 / 3 | 0 |
| tools.forbidden_tools | 3 / 3 | 0 |
| tools.get_subgraphs_to_visualize.queries | N/A — no check recorded | 0 |
| tools.one_tool_per_worker | 3 / 3 | 0 |
| tools.required_tools | 3 / 3 | 0 |
| tools.search_knowledge_base.queries | 1 / 1 | 0 |
| tools.search_knowledge_base.required_topics | 1 / 1 | 0 |

| Efficiency metric | Mean | Available / scored |
| --- | ---: | ---: |
| worker steps | 2.33 | 3 / 3 |
| logical llm calls | 4.33 | 3 / 3 |
| provider attempts | 4.67 | 3 / 3 |
| retries | 0.33 | 3 / 3 |
| tavily searches | 1.00 | 3 / 3 |
| latency seconds | 69.76 | 3 / 3 |
| input tokens | N/A — not captured in scored records | 0 / 3 |
| output tokens | N/A — not captured in scored records | 0 / 3 |
| total tokens | N/A — not captured in scored records | 0 / 3 |

Recorded classified failure counts: none (check failures are shown separately above).
- loop detected: 0 / 3 observed.
- output limit hit: 0 / 3 observed.

### Answer and graph quality

**Unavailable.** 4 eligible inputs; 1 HTTP 429 attempt; 0 successful scores; 3 unstarted. Judge score and serving identity are unavailable.
Judge evaluator fingerprint: `e3545f82672e7197c6e567f2`; configured model `xingchen-agi/xing4.0-29b-a4b:free` with Phala only, no fallback routing. No judge rerun was performed.

### Retrieval and limitations

October system retrieval-benchmark quality: N/A — no separate system-benchmark scorecard was saved for these live runs. Retrieval status and tool checks above describe workflow contracts, not evidence precision.
October aggregate token usage: N/A — all 12 completed records lack complete token totals. Missing usage is not zero. The small sample, interrupted executions, unresolved graph focus, and unverified deployment/vector parity limit conclusions.
The separate [retrieval optimization experiment](experiments/retrieval-comparison/report.md) reports its own cohorts, denominators and timing; its scores are not system-evaluation results.

## Provenance and derivation

October metrics aggregate original saved case-result checks and recorded metrics, grouped by original evaluator fingerprint; the source hashes and derivation are in [provenance](history/system-evaluation-october/provenance.json). No historical observation was rescored.
August values are copied from its preserved published scorecard, without recomputation. Execution status, retrieval support and generated-answer quality remain separate.
