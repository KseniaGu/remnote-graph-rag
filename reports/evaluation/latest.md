# System evaluation: consolidated report

The editions below have distinct populations, configurations and evaluator identities. They are not a controlled before/after comparison.

## Reference evaluation — 2026-08-22 UTC

Imported traces: 12; live executions: 15. The imported-trace configuration is not established by the recorded live configuration.

## Graph RAG Evaluation Scorecard

- Fingerprint: `f8d6f3b6f3e0348c75a194bf`
- Generated: 2026-08-23T20:14:33.266128+00:00
- Historical runs: 10
- Contributing current runs: 10

This report intentionally has no overall score or overall status.

### Campaign coverage

**Campaign completeness:** Incomplete
**Missing requirements:** Retrieval relevance labels

| Requirement | Status | Observed | Expected |
| --- | --- | --- | --- |
| Retrieval baseline | Complete | Yes | mode=both; Analyst=legacy_vector_context; Visualizer=optimized |
| Retrieval relevance labels | Missing | 9 | reviewed Context Precision@10 and Context Recall for every Analyst case |
| Offline runtime | Complete | 11 entries — see breakdown below | 11 entries — see breakdown below |
| Reviewed live suite | Complete | 13 entries — see breakdown below | 13 entries — see breakdown below |
| Semantic dimensions | Complete | 5 entries — see breakdown below | 5 items |
| Agentevals when requested | Complete | 12 | results required only when --framework agentevals was requested |
| Invocation and configuration provenance | Complete | 10 | every contributing run has invocation and configuration snapshots |

#### Offline runtime

| Item | Observed | Expected |
| --- | --- | --- |
| `local_transformer_answer_success` | 1 | 1 |
| `contextual_mamba_visualization` | 1 | 1 |
| `explicit_mamba_visualization_success` | 1 | 1 |
| `initial_attention_quiz_route` | 1 | 1 |
| `mentor_stuck_continuation` | 1 | 1 |
| `researcher_structured_output_truncation_reliability` | 2 | 2 |
| `ambiguous_eagle_without_history` | 1 | 1 |
| `contextual_medusa_web_query` | 1 | 1 |
| `explicit_medusa_web_fallback_success` | 1 | 1 |
| `transformer_encoder_graph_coverage` | 1 | 1 |
| `colbert_guidance_mode_continuation` | 1 | 1 |

#### Reviewed live suite

| Item | Observed | Expected |
| --- | --- | --- |
| `local_transformer_answer_success` | 1 | 1 |
| `contextual_mamba_visualization` | 1 | 1 |
| `explicit_mamba_visualization_success` | 1 | 1 |
| `initial_attention_quiz_route` | 1 | 1 |
| `mentor_stuck_continuation` | 1 | 1 |
| `transformer_encoder_graph_coverage` | 1 | 1 |
| `contextual_medusa_web_query` | 1 | 1 |
| `colbert_guidance_mode_continuation` | 1 | 1 |
| `out_of_scope_biological_cats` | 1 | 1 |
| `ambiguous_eagle_without_history` | 1 | 1 |
| `explicit_medusa_web_fallback_success` | 1 | 1 |
| `in_scope_retnet_local_miss_fallback` | 1 | 1 |
| `researcher_structured_output_truncation_reliability` | 3 | 3 |

#### Semantic dimensions

| Dimension | Results | Success | Skipped | Errors | Provider attempts | Accounted | Expected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Claim faithfulness | 1 | 0 | 0 | 1 | 1 | Yes | Yes |
| Analyst usefulness | 1 | 0 | 0 | 1 | 1 | Yes | Yes |
| Mentor pedagogy | 1 | 0 | 1 | 0 | 0 | Yes | Yes |
| Conversational continuity | 1 | 0 | 1 | 0 | 0 | Yes | Yes |
| Graph usefulness | 1 | 0 | 0 | 1 | 1 | Yes | Yes |

### Retrieval quality

<details><summary>Source runs (1)</summary>

- `20260822T011922Z_retrieval_887cc3ed`

</details>

| Metric | Result |
| --- | --- |
| Cases | 12 |
| Case pass rate | 33.3% |
| Evidence chunk recall | 56.4% |
| Concept recall | 53.5% |
| Relation recall | 36.1% |
| Forbidden evidence count | 0 |
| Forbidden evidence rate | 0.0% |
| Retrieval adequacy rate | 100.0% |
| Retrieval error rate | 0.0% |
| Dangling edge count | 0 |
| Chunk node count | 0 |
| Context Precision@10 | 36.9% |
| Context Recall | 63.1% |

#### Retrieval diagnostics

| Metric | Result |
| --- | --- |
| Concept ID recall | 53.5% |
| Concept label recall | 57.7% |
| Relation ID recall | 36.1% |
| Relation spec recall | 36.1% |

### Runtime task behavior

<details><summary>Source runs (6)</summary>

- `20260822T012029Z_offline_479d922a`
- `20260822T012106Z_live_cd556520`
- `20260822T012229Z_live_81ccde39`
- `20260822T012408Z_live_de469949`
- `20260822T012733Z_live_b83709dd`
- `20260822T013029Z_live_674c635e`

</details>

| Metric | Result |
| --- | --- |
| Runs | 27 |
| Task success rate | 59.3% |
| Routing correctness | 66.7% |
| Required agent compliance | 65.4% |
| Forbidden agent compliance | 80.8% |
| Tool selection correctness | 81.5% |
| Tool argument validity | 66.7% |
| One tool per worker | 100.0% |
| Retrieval status correctness | 93.3% |
| Local to web fallback correctness | 75.0% |
| Source exhaustion correctness | 85.7% |
| Unnecessary web rate | 20.0% |
| Final response rate | 85.2% |
| Modality correctness | 85.2% |
| Termination rate | 96.3% |
| Graph contract rate | 83.3% |

#### Samples by mode

| Mode | Samples |
| --- | --- |
| Offline | 12 |
| Live | 15 |

#### Distributions

| Measurement | Samples | Mean | Median | P95 |
| --- | --- | --- | --- | --- |
| Path efficiency ratio | 23 | 88.4% | 100.0% | 100.0% |

### Reliability

<details><summary>Source runs (6)</summary>

- `20260822T012029Z_offline_479d922a`
- `20260822T012106Z_live_cd556520`
- `20260822T012229Z_live_81ccde39`
- `20260822T012408Z_live_de469949`
- `20260822T012733Z_live_b83709dd`
- `20260822T013029Z_live_674c635e`

</details>

| Metric | Result |
| --- | --- |
| Repetition count | 27 |
| Pass rate | 59.3% |
| 95% confidence interval | 40.7%–75.5% |
| Looping rate | 0.0% |
| Output limit hit rate | 11.1% |

#### Route consistency

**Mean:** 89.1%

| Case | Result |
| --- | --- |
| `local_transformer_answer_success` | 100.0% |
| `contextual_mamba_visualization` | 50.0% |
| `explicit_mamba_visualization_success` | 100.0% |
| `initial_attention_quiz_route` | 100.0% |
| `mentor_stuck_continuation` | 100.0% |
| `researcher_structured_output_truncation_reliability` | 80.0% |
| `ambiguous_eagle_without_history` | 100.0% |
| `contextual_medusa_web_query` | 50.0% |
| `explicit_medusa_web_fallback_success` | 100.0% |
| `transformer_encoder_graph_coverage` | 100.0% |
| `colbert_guidance_mode_continuation` | 100.0% |

#### Evidence-set Jaccard stability

**Mean:** 78.7%

| Case | Result |
| --- | --- |
| `local_transformer_answer_success` | 33.3% |
| `contextual_mamba_visualization` | 0.0% |
| `explicit_mamba_visualization_success` | 100.0% |
| `initial_attention_quiz_route` | 100.0% |
| `mentor_stuck_continuation` | 52.9% |
| `researcher_structured_output_truncation_reliability` | 100.0% |
| `ambiguous_eagle_without_history` | 100.0% |
| `contextual_medusa_web_query` | 100.0% |
| `explicit_medusa_web_fallback_success` | 100.0% |
| `transformer_encoder_graph_coverage` | 79.2% |
| `colbert_guidance_mode_continuation` | 100.0% |

#### Failure types

| Failure | Count |
| --- | --- |
| Invalid retriever output | 2 |
| Model output truncated | 3 |
| Structured output parse error | 1 |
| Sources exhausted | 3 |

### Efficiency

<details><summary>Source runs (6)</summary>

- `20260822T012029Z_offline_479d922a`
- `20260822T012106Z_live_cd556520`
- `20260822T012229Z_live_81ccde39`
- `20260822T012408Z_live_de469949`
- `20260822T012733Z_live_b83709dd`
- `20260822T013029Z_live_674c635e`

</details>

| Metric | Result |
| --- | --- |
| Retry rate | 0.0% |
| Tokens per successful run | 11,294.19 |
| Gating | No |

#### Distributions

| Measurement | Samples | Mean | Median | P95 |
| --- | --- | --- | --- | --- |
| Worker steps | 27 | 2.07 | 2 | 3 |
| Logical LLM calls | 27 | 3.44 | 4 | 5 |
| Provider attempts | 27 | 3.44 | 4 | 5 |
| Retries | 27 | 0 | 0 | 0 |
| Tavily searches | 27 | 0.56 | 1 | 1 |
| Input tokens | 27 | 4,791.22 | 4,311 | 9,464 |
| Output tokens | 27 | 1,901.63 | 2,095 | 4,160 |
| Total tokens | 27 | 6,692.85 | 6,502 | 13,474 |
| Latency seconds | 27 | 39.66 | 40.37 | 79.29 |

### Optional semantic quality

<details><summary>Source runs (3)</summary>

- `20260822T013434Z_judge_0e58731e`
- `20260822T013523Z_judge_7017fb24`
- `20260822T013547Z_judge_1a421fad`

</details>

| Dimension | Status | Results | Success | Skipped | Errors | Provider attempts | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Claim faithfulness | N/A | 1 | 0 | 0 | 1 | 1 | all attempted scores unavailable because judge execution failed |
| Analyst usefulness | N/A | 1 | 0 | 0 | 1 | 1 | all attempted scores unavailable because judge execution failed |
| Mentor pedagogy | N/A | 1 | 0 | 1 | 0 | 0 | semantic prerequisites were unavailable |
| Conversational continuity | N/A | 1 | 0 | 1 | 0 | 0 | semantic prerequisites were unavailable |
| Graph usefulness | N/A | 1 | 0 | 0 | 1 | 1 | all attempted scores unavailable because judge execution failed |

#### Judge execution identities

| Dimension | Provider/model | Rubric | Success | Skipped | Errors | Truncations |
| --- | --- | --- | --- | --- | --- | --- |
| Analyst usefulness | ollama/qwen3.5:cloud | runtime-semantic-v1 | 0 | 0 | 1 | 0 |
| Claim faithfulness | ollama/qwen3.5:cloud | runtime-semantic-v1 | 0 | 0 | 1 | 0 |
| Conversational continuity | ollama/qwen3.5:cloud | runtime-semantic-v1 | 0 | 1 | 0 | 0 |
| Graph usefulness | ollama/qwen3.5:cloud | runtime-semantic-v1 | 0 | 0 | 1 | 0 |
| Mentor pedagogy | ollama/qwen3.5:cloud | runtime-semantic-v1 | 0 | 1 | 0 | 0 |

### Diagnostics

| Diagnostic | Count |
| --- | --- |
| Failed runtime checks | 77 |
| AgentEvals rows | 12 |

#### Failure types

| Failure | Count |
| --- | --- |
| Invalid retriever output | 2 |
| Model output truncated | 3 |
| Structured output parse error | 1 |
| Sources exhausted | 3 |

#### Runtime modes

| Mode | Samples | Passed | Failed |
| --- | --- | --- | --- |
| Offline | 12 | 6 | 6 |
| Live | 15 | 10 | 5 |

#### Runtime failure types by mode

| Mode | Failure | Count |
| --- | --- | --- |
| Offline | Invalid retriever output | 2 |
| Offline | Model output truncated | 3 |
| Offline | Structured output parse error | 1 |
| Offline | Sources exhausted | 2 |
| Live | Sources exhausted | 1 |

#### Provenance diagnostics

| Metric | Result |
| --- | --- |
| Manifest count | 10 |
| Invocation present count | 10 |
| Configuration snapshot present count | 10 |

##### Runtime source mode counts

| Metric | Result |
| --- | --- |
| Offline | 12 |
| Live | 15 |

##### Offline trace configuration status counts

| Metric | Result |
| --- | --- |
| Needs verification | 12 |

##### Offline runs needing configuration verification

- `20260822T012029Z_offline_479d922a`

<details><summary>Run artifacts (10)</summary>

| Run | Kind | Directory |
| --- | --- | --- |
| `20260822T011922Z_retrieval_887cc3ed` | Retrieval | `data/evaluation/runs/20260822T011922Z_retrieval_887cc3ed` |
| `20260822T012029Z_offline_479d922a` | Offline | `data/evaluation/runs/20260822T012029Z_offline_479d922a` |
| `20260822T012106Z_live_cd556520` | Live | `data/evaluation/runs/20260822T012106Z_live_cd556520` |
| `20260822T012229Z_live_81ccde39` | Live | `data/evaluation/runs/20260822T012229Z_live_81ccde39` |
| `20260822T012408Z_live_de469949` | Live | `data/evaluation/runs/20260822T012408Z_live_de469949` |
| `20260822T012733Z_live_b83709dd` | Live | `data/evaluation/runs/20260822T012733Z_live_b83709dd` |
| `20260822T013029Z_live_674c635e` | Live | `data/evaluation/runs/20260822T013029Z_live_674c635e` |
| `20260822T013434Z_judge_0e58731e` | Judge | `data/evaluation/runs/20260822T013434Z_judge_0e58731e` |
| `20260822T013523Z_judge_7017fb24` | Judge | `data/evaluation/runs/20260822T013523Z_judge_7017fb24` |
| `20260822T013547Z_judge_1a421fad` | Judge | `data/evaluation/runs/20260822T013547Z_judge_1a421fad` |

</details>

<details><summary>Failed runtime checks (77)</summary>

| Scenario | Dimension | Check | Gating | Reason |
| --- | --- | --- | --- | --- |
| `contextual_mamba_visualization` | Routing | Worker sequence | Yes | actual=['retriever', 'researcher', 'analyst']; allowed=[['retriever', 'visualizer']] |
| `contextual_mamba_visualization` | Routing | Required agents | Yes | missing=['visualizer'] |
| `contextual_mamba_visualization` | Routing | Forbidden agents | Yes | present=['analyst', 'researcher'] |
| `contextual_mamba_visualization` | Tools | Required tools | Yes | missing=['get_subgraphs_to_visualize'] |
| `contextual_mamba_visualization` | Tools | Forbidden tools | Yes | present=['deep_web_research'] |
| `contextual_mamba_visualization` | Retrieval | Retrieval status | Yes | actual='no_results'; allowed=['adequate'] |
| `contextual_mamba_visualization` | Retrieval | Retrieval outcome | Yes | actual='invalid_model_output'; expected='adequate' |
| `contextual_mamba_visualization` | Modality | Requested modality | Yes | actual='text'; expected='graph' |
| `contextual_mamba_visualization` | Answer | Final response present | Yes | present=True; expected=False |
| `contextual_mamba_visualization` | Graph | Artifact present | Yes | artifact_count=0 |
| `contextual_mamba_visualization` | Graph | Node count min | Yes | actual=0; expected >= 3 |
| `contextual_mamba_visualization` | Graph | Edge count min | Yes | actual=0; expected >= 1 |
| `contextual_mamba_visualization` | Reliability | Forbidden failures | Yes | present=['invalid_retriever_output', 'model_output_truncated'] |
| `contextual_mamba_visualization` | Budget | Worker steps | Yes | actual=3; expected <= 2 |
| `contextual_mamba_visualization` | Budget | Logical LLM calls | No | actual=5; expected <= 2 |
| `contextual_mamba_visualization` | Budget | Provider attempts | No | actual=5; expected <= 3 |
| `contextual_mamba_visualization` | Budget | Tavily searches | Yes | actual=1; expected <= 0 |
| `contextual_mamba_visualization` | Budget | Total tokens | No | actual=13253; expected <= 12000 |
| `mentor_stuck_continuation` | Routing | Worker sequence | Yes | actual=['retriever', 'analyst']; allowed=[['mentor']] |
| `mentor_stuck_continuation` | Routing | Required agents | Yes | missing=['mentor'] |
| `mentor_stuck_continuation` | Routing | Forbidden agents | Yes | present=['analyst', 'retriever'] |
| `mentor_stuck_continuation` | Tools | Forbidden tools | Yes | present=['search_knowledge_base'] |
| `mentor_stuck_continuation` | Budget | Worker steps | Yes | actual=2; expected <= 1 |
| `mentor_stuck_continuation` | Budget | Logical LLM calls | No | actual=3; expected <= 2 |
| `mentor_stuck_continuation` | Budget | Provider attempts | No | actual=3; expected <= 2 |
| `mentor_stuck_continuation` | Budget | Total tokens | No | actual=13864; expected <= 7000 |
| `researcher_structured_output_truncation_reliability` | Routing | Worker sequence | Yes | actual=['researcher']; allowed=[['researcher', 'analyst']] |
| `researcher_structured_output_truncation_reliability` | Routing | Required agents | Yes | missing=['analyst'] |
| `researcher_structured_output_truncation_reliability` | Modality | Requested modality | Yes | actual='error'; expected='text' |
| `researcher_structured_output_truncation_reliability` | Answer | Final response present | Yes | present=False; expected=True |
| `researcher_structured_output_truncation_reliability` | Reliability | Forbidden failures | Yes | present=['model_output_truncated', 'structured_output_parse_error'] |
| `researcher_structured_output_truncation_reliability` | Termination | Terminated | Yes | terminated=False |
| `ambiguous_eagle_without_history` | Routing | Worker sequence | Yes | actual=['retriever', 'researcher']; allowed=[['retriever', 'researcher', 'analyst']] |
| `ambiguous_eagle_without_history` | Routing | Required agents | Yes | missing=['analyst'] |
| `ambiguous_eagle_without_history` | Tools | Deep web research.topic | Yes | values=['EAGLE']; missing_all=False; missing_any=True; forbidden_terms=[]; forbidden_exact=['EAGLE'] |
| `ambiguous_eagle_without_history` | Fallback | Sources exhausted | Yes | actual=True; expected=False |
| `ambiguous_eagle_without_history` | Modality | Requested modality | Yes | actual='none'; expected='text' |
| `ambiguous_eagle_without_history` | Answer | Final response present | Yes | present=False; expected=True |
| `contextual_medusa_web_query` | Routing | Worker sequence | Yes | actual=['retriever', 'researcher']; allowed=[['retriever', 'researcher', 'analyst']] |
| `contextual_medusa_web_query` | Routing | Required agents | Yes | missing=['analyst'] |
| `contextual_medusa_web_query` | Tools | Deep web research.topic | Yes | values=['Medusa']; missing_all=True; missing_any=False; forbidden_terms=[]; forbidden_exact=['Medusa'] |
| `contextual_medusa_web_query` | Fallback | Sources exhausted | Yes | actual=True; expected=False |
| `contextual_medusa_web_query` | Modality | Requested modality | Yes | actual='none'; expected='text' |
| `contextual_medusa_web_query` | Answer | Final response present | Yes | present=False; expected=True |
| `colbert_guidance_mode_continuation` | Routing | Worker sequence | Yes | actual=['retriever', 'researcher', 'analyst']; allowed=[['mentor']] |
| `colbert_guidance_mode_continuation` | Routing | Required agents | Yes | missing=['mentor'] |
| `colbert_guidance_mode_continuation` | Routing | Forbidden agents | Yes | present=['analyst', 'researcher', 'retriever'] |
| `colbert_guidance_mode_continuation` | Tools | Forbidden tools | Yes | present=['deep_web_research'] |
| `colbert_guidance_mode_continuation` | Reliability | Forbidden failures | Yes | present=['invalid_retriever_output', 'model_output_truncated'] |
| `colbert_guidance_mode_continuation` | Budget | Worker steps | Yes | actual=3; expected <= 1 |
| `colbert_guidance_mode_continuation` | Budget | Logical LLM calls | No | actual=5; expected <= 2 |
| `colbert_guidance_mode_continuation` | Budget | Provider attempts | No | actual=5; expected <= 2 |
| `colbert_guidance_mode_continuation` | Budget | Tavily searches | Yes | actual=1; expected <= 0 |
| `colbert_guidance_mode_continuation` | Budget | Total tokens | No | actual=13474; expected <= 7000 |
| `mentor_stuck_continuation` | Routing | Worker sequence | Yes | actual=['retriever', 'analyst']; allowed=[['mentor']] |
| `mentor_stuck_continuation` | Routing | Required agents | Yes | missing=['mentor'] |
| `mentor_stuck_continuation` | Routing | Forbidden agents | Yes | present=['analyst', 'retriever'] |
| `mentor_stuck_continuation` | Tools | Forbidden tools | Yes | present=['search_knowledge_base'] |
| `mentor_stuck_continuation` | Budget | Worker steps | Yes | actual=2; expected <= 1 |
| `mentor_stuck_continuation` | Budget | Logical LLM calls | No | actual=3; expected <= 2 |
| `mentor_stuck_continuation` | Budget | Provider attempts | No | actual=3; expected <= 2 |
| `mentor_stuck_continuation` | Budget | Total tokens | No | actual=10078; expected <= 7000 |
| `contextual_medusa_web_query` | Tools | Deep web research.topic | Yes | values=['Medusa']; missing_all=True; missing_any=False; forbidden_terms=[]; forbidden_exact=['Medusa'] |
| `colbert_guidance_mode_continuation` | Routing | Worker sequence | Yes | actual=['retriever', 'researcher', 'analyst']; allowed=[['mentor']] |
| `colbert_guidance_mode_continuation` | Routing | Required agents | Yes | missing=['mentor'] |
| `colbert_guidance_mode_continuation` | Routing | Forbidden agents | Yes | present=['analyst', 'researcher', 'retriever'] |
| `colbert_guidance_mode_continuation` | Tools | Forbidden tools | Yes | present=['deep_web_research', 'search_knowledge_base'] |
| `colbert_guidance_mode_continuation` | Budget | Worker steps | Yes | actual=3; expected <= 1 |
| `colbert_guidance_mode_continuation` | Budget | Logical LLM calls | No | actual=5; expected <= 2 |
| `colbert_guidance_mode_continuation` | Budget | Provider attempts | No | actual=5; expected <= 2 |
| `colbert_guidance_mode_continuation` | Budget | Tavily searches | Yes | actual=1; expected <= 0 |
| `ambiguous_eagle_without_history` | Routing | Worker sequence | Yes | actual=['retriever', 'researcher']; allowed=[['retriever', 'researcher', 'analyst']] |
| `ambiguous_eagle_without_history` | Routing | Required agents | Yes | missing=['analyst'] |
| `ambiguous_eagle_without_history` | Tools | Deep web research.topic | Yes | values=['EAGLE']; missing_all=False; missing_any=True; forbidden_terms=[]; forbidden_exact=['EAGLE'] |
| `ambiguous_eagle_without_history` | Fallback | Sources exhausted | Yes | actual=True; expected=False |
| `in_scope_retnet_local_miss_fallback` | Tools | Deep web research.topic | Yes | values=['RetNet']; missing_all=False; missing_any=True; forbidden_terms=[]; forbidden_exact=[] |
| `in_scope_retnet_local_miss_fallback` | Scope | Request scope | Yes | actual='ambiguous'; expected='in_scope' |

</details>

### Metric definitions

<details><summary>Show metric definitions</summary>

#### Retrieval quality

| Measurement | Definition |
| --- | --- |
| Case pass rate | Share of retrieval cases satisfying every applicable benchmark contract. |
| Evidence chunk recall | Required source chunk IDs returned divided by required source chunk IDs. |
| Concept recall | Required graph concepts returned divided by required graph concepts. |
| Relation recall | Required graph relations returned divided by required graph relations. |
| Forbidden evidence count/rate | Observed regression-backed forbidden evidence items and the share of cases containing any. |
| Retrieval adequacy rate/error rate | Share reported adequate and mean retrieval error count across benchmark cases. |
| Dangling edge count/chunk node count | Structural graph diagnostics for dangling edges and leaked chunk nodes. |
| Context Precision@10 | Weighted reviewed relevance in the first ten ranked Analyst source slots divided by ten. |
| Context Recall | Reviewed required answer points supported by returned Analyst evidence divided by all required answer points. |
| Concept/relation diagnostics | Separate ID, label, and relation-spec recall used to explain aggregate misses. |

#### Runtime task behavior

| Measurement | Definition |
| --- | --- |
| Task success rate | Runs passing every applicable gating contract divided by evaluated runs. |
| Routing correctness | Share of applicable runs whose worker sequence matches an allowed route. |
| Required/forbidden agent compliance | Share of applicable runs containing every required agent or avoiding every forbidden agent. |
| Path efficiency ratio | Shortest allowed completed worker path divided by actual worker steps, capped at one; incomplete runs are N/A. |
| Tool selection correctness | Share of applicable runs satisfying required and forbidden tool contracts. |
| Tool argument validity | Share of observed required tool calls satisfying every argument constraint. |
| One tool per worker | Share of applicable worker steps that comply with the one-tool boundary. |
| Retrieval status correctness | Share of applicable runs with an allowed local retrieval status and outcome. |
| Local to web fallback correctness | Share of applicable runs that escalate, or avoid escalation, according to the scenario contract. |
| Unnecessary web rate | Tavily-using runs divided by scenarios that explicitly forbid web use. |
| Source exhaustion correctness | Share of applicable runs whose source-exhaustion state matches the contract. |
| Final response/modality/termination rates | Shares satisfying final-response presence, requested output modality, and workflow termination contracts. |
| Graph contract rate | Share of graph-applicable runs passing all structural and required-anchor checks. |

#### Reliability

| Measurement | Definition |
| --- | --- |
| Repetition count | Number of runtime observations contributing to this fingerprint. |
| Pass rate and Wilson interval | Functional pass proportion with sample count represented by a 95% Wilson interval. |
| Route consistency | Per repeated scenario, fraction following its most common worker route. |
| Evidence-set Jaccard stability | Mean pairwise Jaccard overlap of evidence IDs for repetitions of the same scenario. |
| Looping rate | Runs with an identical worker/action signature repeated after Orchestrator return divided by applicable runs. |
| Output limit hit rate | Runs observing a provider output-limit stop or classified truncation divided by runtime runs. |
| Failure type frequency | Counts of classified provider, parser, tool, storage, timeout, truncation, recursion, and workflow failures. |

#### Efficiency

| Measurement | Definition |
| --- | --- |
| Worker steps | Distribution of non-Orchestrator worker executions per run. |
| Logical LLM calls | Distribution of logical model operations before provider-level retries. |
| Provider attempts | Distribution of actual provider attempts, including retries. |
| Retries/retry rate | Retry counts and total retries divided by total provider attempts. |
| Tavily searches | Distribution of Tavily tool calls per run. |
| Input/output/total tokens | Provider-reported token distributions; unavailable observations are omitted, not zero-filled. |
| Tokens per successful run | Total tokens across evaluated runs divided by functionally successful runs; N/A when none succeed. |
| Latency seconds | End-to-end time-to-resolution distribution with mean, median, and p95. |
| Gating | False means efficiency limits are reported but do not initially fail functional task success. |

#### Optional semantic quality

| Measurement | Definition |
| --- | --- |
| Claim faithfulness | Counts supported, partially supported, and unsupported factual claims against captured bounded evidence. |
| Grounded claim rate | Supported claims plus half-weighted partial claims divided by all classified factual claims. |
| Analyst usefulness | Boolean judge rate for direct, clear, non-substitutive Analyst answers. |
| Mentor pedagogy | Boolean judge rate for learner-aware hints and appropriate instructional next steps. |
| Conversational continuity | Boolean judge rate for preserving topic and interaction mode from recent state. |
| Graph usefulness | Boolean judge rate for relevant, useful labeled graph relationships. |
| Judge execution diagnostics | Per-dimension result, success, error, and actual provider-attempt counts with privacy-safe failure reasons. |

</details>

### Metric interpretation

- Functional task success excludes non-gating token and latency limits.
- Context Precision@10 and Context Recall remain N/A until human labels are reviewed.
- Live repetitions accumulate; deterministic invocations use their latest successful run.
- Optional semantic judges never change deterministic task success.

## Current evaluation — 2026-10-31 UTC

15 planned executions across 13 scenarios; 12 completed, 10 passed, 2 failed and 3 interrupted/unscored.
Worker-step and Tavily-search bounds can gate deterministic success; logical calls, provider attempts, retries, token usage and latency bounds are report-only. Saved outcomes are unchanged.
- Failures: Mentor continuation routing/worker-step contract and RetNet contextual web-topic contract. Three interrupted observations have no runtime record or score.
- Configuration: Orchestrator routing v6 / graph-index v2, Retriever v7, Researcher v6, Analyst v6 and Mentor v4. Generative roles used the variable openrouter/free route; serving identities were not always observed.
- Analyst used legacy vector context; Visualizer used optimized retrieval. Document, index and graph stores were local, vector storage Pinecone, and checkpoints MongoDB.
- All 12 completed runtime records lack complete token totals. Runtime attempt reservations were 45 of 54; full effort used 79 of 92, including failures and retries.

### Coverage and runtime behavior

#### Evaluator `1095d6342e69b04143ef971d`

7 / 7 planned observations scored; 6 / 7 passed.

| Runtime summary | Result | Eligible observations/checks |
| --- | ---: | ---: |
| Routing Correctness | 85.7% | 7 |
| Required Agent Compliance | 85.7% | 7 |
| Forbidden Agent Compliance | 85.7% | 7 |
| One Tool Per Worker | 100.0% | 7 |
| Retrieval Status Correctness | 100.0% | 5 |
| Source Exhaustion Correctness | 100.0% | 5 |
| Final Response Rate | 100.0% | 7 |
| Modality Correctness | 100.0% | 7 |
| Termination Rate | 100.0% | 7 |
| Tool Selection Correctness | 85.7% | 7 |
| Tool Argument Validity | 100.0% | 4 |
| Local To Web Fallback Correctness | 100.0% | 1 |
| Unnecessary Web Rate | 0.0% | 6 |
| Task Success Rate | 85.7% | 7 |
| Graph Contract Rate | 100.0% | 3 |

<details><summary>Recorded contract checks and gating</summary>

| Workflow metric | Passed / eligible | Not observed | Gating metadata |
| --- | ---: | ---: | --- |
| answer: final response present | 7 / 7 | 0 | true: 7 |
| budget: latency seconds | 7 / 7 | 0 | false: 7 |
| budget: logical llm calls | 6 / 7 | 0 | false: 7 |
| budget: provider attempts | 5 / 7 | 0 | false: 7 |
| budget: retries | 7 / 7 | 0 | false: 7 |
| budget: tavily searches | 7 / 7 | 0 | true: 7 |
| budget: total tokens | N/A — no observed eligible checks | 7 | false: 7 |
| budget: worker steps | 6 / 7 | 0 | true: 7 |
| fallback: sources exhausted | 5 / 5 | 0 | true: 5 |
| graph: artifact present | 3 / 3 | 0 | true: 3 |
| graph: edge count max | 3 / 3 | 0 | true: 3 |
| graph: edge count min | 3 / 3 | 0 | true: 3 |
| graph: no dangling edges | 3 / 3 | 0 | true: 3 |
| graph: node count max | 3 / 3 | 0 | true: 3 |
| graph: node count min | 3 / 3 | 0 | true: 3 |
| graph: required nodes | 1 / 1 | 0 | true: 1 |
| modality: requested modality | 7 / 7 | 0 | true: 7 |
| reliability: forbidden failures | 7 / 7 | 0 | true: 7 |
| retrieval: retrieval outcome | 4 / 4 | 0 | true: 4 |
| retrieval: retrieval status | 5 / 5 | 0 | true: 5 |
| routing: basic loop detector | 7 / 7 | 0 | true: 7 |
| routing: forbidden agents | 6 / 7 | 0 | true: 7 |
| routing: required agents | 6 / 7 | 0 | true: 7 |
| routing: worker sequence | 6 / 7 | 0 | true: 7 |
| scope: request scope | 7 / 7 | 0 | true: 7 |
| termination: terminated | 7 / 7 | 0 | true: 7 |
| tools: deep web research: topic | 1 / 1 | 0 | true: 1 |
| tools: forbidden tools | 6 / 7 | 0 | true: 7 |
| tools: get subgraphs to visualize: queries | 3 / 3 | 0 | true: 3 |
| tools: one tool per worker | 7 / 7 | 0 | true: 7 |
| tools: required tools | 5 / 5 | 0 | true: 5 |

</details>

##### Efficiency distributions

| Metric | Available / scored | Mean | Median | P95 |
| --- | ---: | ---: | ---: | ---: |
| worker steps | 7 / 7 | 2.00 | 2.00 | 3.00 |
| logical llm calls | 7 / 7 | 2.71 | 2.00 | 5.00 |
| provider attempts | 7 / 7 | 2.86 | 2.00 | 5.00 |
| retries | 7 / 7 | 0.14 | 0.00 | 1.00 |
| tavily searches | 7 / 7 | 0.14 | 0.00 | 1.00 |
| latency seconds | 7 / 7 | 30.79 | 19.09 | 72.50 |
| input tokens | 0 / 7 | N/A — not captured | N/A — not captured | N/A — not captured |
| output tokens | 0 / 7 | N/A — not captured | N/A — not captured | N/A — not captured |
| total tokens | 0 / 7 | N/A — not captured | N/A — not captured | N/A — not captured |

##### Reliability

| Metric | Result | Population |
| --- | --- | --- |
| loop detected | 0 | 7 observed |
| output limit hit | 0 | 7 observed |
| Classified failures | None recorded | 7 scored |
| route consistency | N/A — no repeated scenario runs | No eligible repeated scenarios |
| evidence set jaccard stability | N/A — no repeated scenario pairs | No eligible repeated scenarios |

#### Evaluator `2b1301ae85cf5a5a77713187`

2 / 2 planned observations scored; 2 / 2 passed.

| Runtime summary | Result | Eligible observations/checks |
| --- | ---: | ---: |
| Routing Correctness | 100.0% | 2 |
| Required Agent Compliance | 100.0% | 1 |
| Forbidden Agent Compliance | 100.0% | 2 |
| One Tool Per Worker | 100.0% | 2 |
| Retrieval Status Correctness | N/A — no eligible observations | 0 |
| Source Exhaustion Correctness | 100.0% | 1 |
| Final Response Rate | 100.0% | 2 |
| Modality Correctness | 100.0% | 2 |
| Termination Rate | 100.0% | 2 |
| Tool Selection Correctness | 100.0% | 2 |
| Tool Argument Validity | N/A — no eligible observations | 0 |
| Local To Web Fallback Correctness | N/A — no eligible observations | 0 |
| Unnecessary Web Rate | 0.0% | 2 |
| Task Success Rate | 100.0% | 2 |
| Graph Contract Rate | N/A — no eligible observations | 0 |

<details><summary>Recorded contract checks and gating</summary>

| Workflow metric | Passed / eligible | Not observed | Gating metadata |
| --- | ---: | ---: | --- |
| answer: final response present | 2 / 2 | 0 | true: 2 |
| budget: latency seconds | 2 / 2 | 0 | false: 2 |
| budget: logical llm calls | 2 / 2 | 0 | false: 2 |
| budget: provider attempts | 2 / 2 | 0 | false: 2 |
| budget: retries | 2 / 2 | 0 | false: 2 |
| budget: tavily searches | 2 / 2 | 0 | true: 2 |
| budget: total tokens | N/A — no observed eligible checks | 2 | false: 2 |
| budget: worker steps | 2 / 2 | 0 | true: 2 |
| fallback: sources exhausted | 1 / 1 | 0 | true: 1 |
| modality: requested modality | 2 / 2 | 0 | true: 2 |
| reliability: forbidden failures | 2 / 2 | 0 | true: 2 |
| routing: basic loop detector | 2 / 2 | 0 | true: 2 |
| routing: forbidden agents | 2 / 2 | 0 | true: 2 |
| routing: required agents | 1 / 1 | 0 | true: 1 |
| routing: worker sequence | 2 / 2 | 0 | true: 2 |
| scope: request scope | 2 / 2 | 0 | true: 2 |
| termination: terminated | 2 / 2 | 0 | true: 2 |
| tools: forbidden tools | 2 / 2 | 0 | true: 2 |
| tools: one tool per worker | 2 / 2 | 0 | true: 2 |

</details>

##### Efficiency distributions

| Metric | Available / scored | Mean | Median | P95 |
| --- | ---: | ---: | ---: | ---: |
| worker steps | 2 / 2 | 0.50 | 0.50 | 1.00 |
| logical llm calls | 2 / 2 | 1.50 | 1.50 | 2.00 |
| provider attempts | 2 / 2 | 1.50 | 1.50 | 2.00 |
| retries | 2 / 2 | 0.00 | 0.00 | 0.00 |
| tavily searches | 2 / 2 | 0.00 | 0.00 | 0.00 |
| latency seconds | 2 / 2 | 6.17 | 6.17 | 8.56 |
| input tokens | 0 / 2 | N/A — not captured | N/A — not captured | N/A — not captured |
| output tokens | 0 / 2 | N/A — not captured | N/A — not captured | N/A — not captured |
| total tokens | 0 / 2 | N/A — not captured | N/A — not captured | N/A — not captured |

##### Reliability

| Metric | Result | Population |
| --- | --- | --- |
| loop detected | 0 | 2 observed |
| output limit hit | 0 | 2 observed |
| Classified failures | None recorded | 2 scored |
| route consistency | N/A — no repeated scenario runs | No eligible repeated scenarios |
| evidence set jaccard stability | N/A — no repeated scenario pairs | No eligible repeated scenarios |

#### Evaluator `aacdf873baacf605652888e1`

3 / 6 planned observations scored; 2 / 3 passed.

| Runtime summary | Result | Eligible observations/checks |
| --- | ---: | ---: |
| Routing Correctness | 100.0% | 3 |
| Required Agent Compliance | 100.0% | 3 |
| Forbidden Agent Compliance | 100.0% | 2 |
| One Tool Per Worker | 100.0% | 3 |
| Retrieval Status Correctness | 100.0% | 1 |
| Source Exhaustion Correctness | 100.0% | 3 |
| Final Response Rate | 100.0% | 3 |
| Modality Correctness | 100.0% | 3 |
| Termination Rate | 100.0% | 3 |
| Tool Selection Correctness | 100.0% | 3 |
| Tool Argument Validity | 66.7% | 3 |
| Local To Web Fallback Correctness | 100.0% | 3 |
| Unnecessary Web Rate | N/A — no eligible observations | 0 |
| Task Success Rate | 66.7% | 3 |
| Graph Contract Rate | N/A — no eligible observations | 0 |

<details><summary>Recorded contract checks and gating</summary>

| Workflow metric | Passed / eligible | Not observed | Gating metadata |
| --- | ---: | ---: | --- |
| answer: final response present | 3 / 3 | 0 | true: 3 |
| budget: latency seconds | 3 / 3 | 0 | false: 3 |
| budget: logical llm calls | 3 / 3 | 0 | false: 3 |
| budget: provider attempts | 3 / 3 | 0 | false: 3 |
| budget: retries | 3 / 3 | 0 | false: 3 |
| budget: tavily searches | 3 / 3 | 0 | true: 3 |
| budget: total tokens | N/A — no observed eligible checks | 3 | false: 3 |
| budget: worker steps | 3 / 3 | 0 | true: 3 |
| fallback: sources exhausted | 3 / 3 | 0 | true: 3 |
| modality: requested modality | 3 / 3 | 0 | true: 3 |
| reliability: forbidden failures | 3 / 3 | 0 | true: 3 |
| retrieval: retrieval status | 1 / 1 | 0 | true: 1 |
| routing: basic loop detector | 3 / 3 | 0 | true: 3 |
| routing: forbidden agents | 2 / 2 | 0 | true: 2 |
| routing: required agents | 3 / 3 | 0 | true: 3 |
| routing: worker sequence | 3 / 3 | 0 | true: 3 |
| scope: request scope | 3 / 3 | 0 | true: 3 |
| termination: terminated | 3 / 3 | 0 | true: 3 |
| tools: deep web research: topic | 2 / 3 | 0 | true: 3 |
| tools: forbidden tools | 3 / 3 | 0 | true: 3 |
| tools: one tool per worker | 3 / 3 | 0 | true: 3 |
| tools: required tools | 3 / 3 | 0 | true: 3 |
| tools: search knowledge base: queries | 1 / 1 | 0 | true: 1 |
| tools: search knowledge base: required topics | 1 / 1 | 0 | true: 1 |

</details>

##### Efficiency distributions

| Metric | Available / scored | Mean | Median | P95 |
| --- | ---: | ---: | ---: | ---: |
| worker steps | 3 / 3 | 2.33 | 2.00 | 3.00 |
| logical llm calls | 3 / 3 | 4.33 | 4.00 | 5.00 |
| provider attempts | 3 / 3 | 4.67 | 4.00 | 6.00 |
| retries | 3 / 3 | 0.33 | 0.00 | 1.00 |
| tavily searches | 3 / 3 | 1.00 | 1.00 | 1.00 |
| latency seconds | 3 / 3 | 69.76 | 73.36 | 97.66 |
| input tokens | 0 / 3 | N/A — not captured | N/A — not captured | N/A — not captured |
| output tokens | 0 / 3 | N/A — not captured | N/A — not captured | N/A — not captured |
| total tokens | 0 / 3 | N/A — not captured | N/A — not captured | N/A — not captured |

##### Reliability

| Metric | Result | Population |
| --- | --- | --- |
| loop detected | 0 | 3 observed |
| output limit hit | 0 | 3 observed |
| Classified failures | None recorded | 3 scored |
| route consistency | 1.000 | 1 repeated scenarios |
| evidence set jaccard stability | 1.000 | 1 repeated scenarios |

Evidence-set Jaccard uses the established convention that two empty evidence sets have similarity 1; this does not establish evidence quality.

### Retrieval quality and diagnostics

| Metric | Result | Population |
| --- | --- | --- |
| Evidence recall, precision and graph retrieval quality | N/A — no system retrieval benchmark was saved for these executions | Not measured |

Workflow retrieval checks above do not measure evidence precision. The [retrieval optimization experiment](experiments/retrieval-comparison/report.md) has separate cohorts and results.

### Answer and graph quality

| Dimension | Score | Status | Coverage |
| --- | ---: | --- | --- |
| Graph Usefulness | N/A | Unavailable | 2 eligible; 1 HTTP 429 error; 1 unstarted |
| Claim Faithfulness | N/A | Unavailable | 2 eligible; 0 attempted; 2 unstarted |
| Analyst Usefulness | N/A | Unavailable | Not assessed in this pilot |
| Mentor Pedagogy | N/A | Unavailable | Not assessed in this pilot |
| Conversational Continuity | N/A | Unavailable | Not assessed in this pilot |

Eligible inputs: 4; attempts: 1; successful scores: 0; unstarted: 3; HTTP 429 errors: 1. Missing scores are not zero.
Judge evaluator: `e3545f82672e7197c6e567f2`; configured model: `xingchen-agi/xing4.0-29b-a4b:free`; observed serving identity: unknown.

### Diagnostics and interpretation

Unavailable usage remains N/A; deterministic compliance does not establish evidence support or answer quality. See [provenance](history/system-evaluation-v2-controlled-observations-1640fed21174/provenance.json) for original identities and source hashes. No historical observation was rescored.
