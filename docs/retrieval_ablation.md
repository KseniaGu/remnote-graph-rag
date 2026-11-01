# Local retrieval comparison

The controlled local retrieval comparison produced mixed results. Graph-assisted
retrieval improved frozen Precision@10 slightly over source-only retrieval, but
that difference reversed on development. The local reranker added substantial
CPU retrieval time and did not improve the measured evidence metrics. The current
legacy reference had lower frozen precision and higher required-point coverage.
These results do not select a new default or evaluate generated answers. The
[sanitized aggregate and provenance](../reports/evaluation/history/controlled-observations/retrieval.json)
contains the underlying aggregate values; raw case text and evidence are private.

## Arms and observed evidence

A uses graph-free source retrieval with the model reranker disabled. It retains
the shared deterministic lexical and source-metadata scoring and diversification
used by B/C, so its ordering is not based solely on vector similarity. B adds
the optimized graph-assisted path; C adds the local reranker to B; D is the
current legacy vector-context pipeline with its existing settings. A/B/C share the source-first
retrieval and applicable budgets. D uses different normalization, traversal and
budgets, so D comparisons are whole-pipeline comparisons. D is not a matched
pre-fix baseline. Production defaults remain legacy Analyst and optimized
Visualizer.

| Cohort | Arm | Precision@10 mean | Scored / observed | Required-point Recall mean | Scored / observed |
| --- | --- | ---: | ---: | ---: | ---: |
| Development | A | 49.44% | 27 / 27 | 100%* | 21 / 27 |
| Development | B | 48.33% | 27 / 27 | 100%* | 18 / 27 |
| Development | C | 45.00% | 18 / 18 | 100%* | 14 / 18 |
| Development | D | 36.67% | 27 / 27 | 100%* | 15 / 27 |
| Frozen | A | 33.57% | 21 / 21 | 85.71% | 21 / 21 |
| Frozen | B | 34.29% | 21 / 21 | 85.71% | 21 / 21 |
| Frozen | C | 32.14% | 14 / 14 | 78.57% | 14 / 14 |
| Frozen | D | 28.57% | 21 / 21 | 100% | 21 / 21 |

\* Development Recall is an available-only mean. **31 of 99 development
observations are N/A**, so these values do not establish full-cohort coverage or
equal recall across arms. Frozen Recall is available for 77 of 77 observations.
Observations are case bundles and repetitions, not independent queries or samples.
The development set has nine cases, some with ordered multi-query bundles; the
frozen set has seven single-query cases. No per-query attribution is inferred
from a bundled result.

Precision@10 sums relevance weights of 1, 0.5 and 0 over the actual first ten
unique sources and divides by **10**; absent slots contribute zero. Required-point
Recall is the fraction of unchanged required points with a fully supported source
in the actual top ten. Partial support earns no full-point credit and partials
are not combined. If no full witness exists and an in-list support mapping is
unknown, exact Recall is N/A; unknown is not treated as unsupported. The 98 new
required pair labels and a later point decision were AI-assisted and endorsed by
the user; 77 prior development labels and eight frozen reference approvals were
reused. These are not independent expert or judge ratings.

## Matched differences and resources

The table reports left minus right, with precision and recall differences in
percentage points. Recall comparisons use only pairs with both values available;
their smaller development subsets can change the conclusions.

| Cohort | Contrast | Precision difference (pairs) | Recall difference (available pairs) | Retrieval time difference, seconds (pairs) |
| --- | --- | ---: | ---: | ---: |
| Development | B − A | −1.11 (27) | 0.00 (15) | +1.721 (27) |
| Development | C − B | −3.33 (18) | 0.00 (10) | +274.266 (18) |
| Development | D − B | −11.67 (27) | 0.00 (12) | −1.116 (27) |
| Frozen | B − A | +0.71 (21) | 0.00 (21) | +0.847 (21) |
| Frozen | C − B | −2.14 (14) | −7.14 (14) | +204.258 (14) |
| Frozen | D − B | −5.71 (21) | +14.29 (21) | −0.469 (21) |

The time differences are means of matched case-bundle observations in fresh CPU
processes. They cover retrieval calls and exclude initialization, evidence
normalization and scoring; they are not warm-cache or production latency. C's
completed measurements recorded effective reranking. D's validity and resource
counters were not instrumented. Per-case peak memory and provider/judge token
metadata are unavailable; any recorded cgroup memory values are point samples,
not peaks.

A saved frozen container-development case illustrates the tradeoff: the number
of completed measured observations with a full in-list witness for its required
point was A 0/3, B 0/3, C 0/2 and D 3/3. This evidence-coverage result
helps explain why these local results do not justify switching the default; it
does not establish what an answer generator would say or fix application behavior.

## Protocol and limits

The measured manifests date from October 28–29, 2026 UTC; annotations date from
October 29–30, while scoring and the aggregate export are dated October 30. The accepted measured source tree is
`339884ef4f716dfc497033c5e3428acec3d3e50f`, with snapshot manifest SHA-256
`48c0d76bad4b1f8563199c8e398c0b2ce6d80543be30071138ad67d9ea5d6db7`.
Local model assets were `all-MiniLM-L6-v2` and `Qwen3-Reranker-0.6B`. The
source-first configuration used 30 source candidates, six final sources, a
7,000-character context limit, and a ten-candidate rerank limit with 1,200
characters per candidate. The attachment records source hashes and other aggregate
provenance. Hash agreement verifies retained artifacts, not an independently
established pre-execution snapshot.

Twenty-two of 24 planned measured runs completed. The development C third and
frozen C first measurements were interrupted and remain missing; warmups are
excluded. Some early exit and environment details cannot be reconstructed. The
small frozen set includes an SVM topic with prior exposure and unproven
disambiguation, and lacks a true-miss case. Repetitions are not independent, and
no statistical significance or general graph/reranker superiority is established.

These controlled real-model retrieval observations are distinct from historical
retrospective traces and mocked tests. Current canonical task success,
generated-answer quality and production traffic remain unmeasured in this
retrieval slice. Retrieval evidence coverage is not generated-answer correctness
or completeness, and local run completion is not workflow reliability. Full
runtime evaluation and generated-answer quality are separate questions.
