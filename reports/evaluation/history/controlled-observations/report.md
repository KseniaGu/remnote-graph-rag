# Controlled evaluation findings

Observations through **2026-10-31 UTC**. These historical results do not measure subsequent revisions or production traffic.

## Deterministic runtime compliance

- Completed: 12 / 15 planned observations across 13 scenarios.
- Passed: 10 / 12 completed; failed: 2; interrupted and unscored: 3.
- A contract pass measures the recorded route and tool behavior; it is not an answer-correctness score.
- Original runtime evaluator fingerprints: `1095d6342e69b04143ef971d`, `2b1301ae85cf5a5a77713187`, `aacdf873baacf605652888e1`.
- Failures: Mentor continuation routing/worker-step contract and RetNet contextual web-topic contract. Three interrupted observations have no runtime record or score.
- Configuration: Orchestrator routing v6 / graph-index v2, Retriever v7, Researcher v6, Analyst v6 and Mentor v4. Generative roles used the variable openrouter/free route; serving identities were not always observed.
- Analyst used legacy vector context; Visualizer used optimized retrieval. Document, index and graph stores were local, vector storage Pinecone, and checkpoints MongoDB.
- All 12 completed runtime records lack complete token totals. Runtime attempt reservations were 45 of 54; full effort used 79 of 92, including failures and retries.

| Runtime evaluator fingerprint | Observations | Completed | Interrupted |
| --- | ---: | ---: | ---: |
| `1095d6342e69b04143ef971d` | 7 | 7 | 0 |
| `2b1301ae85cf5a5a77713187` | 2 | 2 | 0 |
| `aacdf873baacf605652888e1` | 6 | 3 | 3 |

## Retrieved-evidence quality

- Precision@10 available for 176 observations; required-point Recall available for 145. These are availability counts, not quality scores.
- Retrieval source tree: `339884ef4f716dfc497033c5e3428acec3d3e50f`; storage manifest SHA256: `48c0d76bad4b1f8563199c8e398c0b2ce6d80543be30071138ad67d9ea5d6db7`.
- Embedding model: all-MiniLM-L6-v2; local reranker: Qwen3-Reranker-0.6B.
- Cases: 9 development and 7 frozen. Repetitions are not independent samples.

| Cohort / arm | Precision@10 | Precision scored / observed | Recall (available only) | Recall scored / observed |
| --- | ---: | ---: | ---: | ---: |
| development:A | 49.44% | 27 / 27 | 100.00% | 21 / 27 |
| development:B | 48.33% | 27 / 27 | 100.00% | 18 / 27 |
| development:C | 45.00% | 18 / 18 | 100.00% | 14 / 18 |
| development:D | 36.67% | 27 / 27 | 100.00% | 15 / 27 |
| frozen:A | 33.57% | 21 / 21 | 85.71% | 21 / 21 |
| frozen:B | 34.29% | 21 / 21 | 85.71% | 21 / 21 |
| frozen:C | 32.14% | 14 / 14 | 78.57% | 14 / 14 |
| frozen:D | 28.57% | 21 / 21 | 100.00% | 21 / 21 |

Development Recall is unavailable for 31 observations; unavailable values are not zero.

- Measured October 28–29, 2026 UTC on local prepared storage; scoring exported October 30. A is source-only, B graph-assisted, C adds local reranking, and D is the legacy pipeline with different budgets.
- Graph-assisted B minus A Precision@10: −1.11 percentage points on development (27 pairs) and +0.71 on frozen (21 pairs). C minus B: −3.33 (18 pairs) and −2.14 (14 pairs).
- Fresh-process CPU retrieval-call timing C minus B: +274.266 seconds development (18 pairs), +204.258 seconds frozen (14 pairs); this is not production latency.
- Twenty-two of 24 planned measurements completed. Precision uses weighted relevance over ten slots; Recall requires a fully supported source. Labels are AI-assisted/user-endorsed and not independent expert ratings.

## Generated-answer quality

- Eligible judge inputs: 4; attempted: 1; successfully scored: 0; unstarted: 3.
- HTTP 429 errors: 1. Judge quality score: N/A.
- Judge evaluator fingerprint: `e3545f82672e7197c6e567f2`. Configured and observed serving identities are distinct; unavailable usage is not zero.
- Configured model: `xingchen-agi/xing4.0-29b-a4b:free`; configured route: Phala only, no fallback. Observed serving identity: unknown.
- Generated-answer correctness and completeness were not measured. The one attempted judge request returned HTTP 429; its origin and cooldown are unknown. Three further eligible assessments were unstarted.
- The configured judge model and endpoint do not establish observed serving identity. Token availability was 0 of 1 attempts for input, output and total; missing is not zero.
- The sample is small; source support does not establish answerability. Graph focus remains unresolved, and deployment, cloud-vector and production parity were not established.

## Evidence and provenance

- [Runtime summary](./runtime.json)
- [Retrieval aggregate](./retrieval.json)
- [Judge status](./judge.json)
- [Detailed saved findings](./source-report.md)

The original runtime, retrieval, and judge populations and configurations are separate. No combined fingerprint or pooled quality score is implied.
