# RemNote Graph RAG Project Overview

This document is a concise overview of the current project architecture. It
uses the older review documents as background, but reflects the current codebase
as of the optimized data-processing and retrieval pipeline.

For operational commands to run the full parser, LLM post-processing, and final
storage build, see `docs/optimized_pipeline_runbook.md`.

## Purpose

RemNote Graph RAG is a personal learning and technical-interview preparation
system built around a RemNote knowledge base. It turns exported notes and
external artifacts into searchable retrieval chunks, a concept graph, and
interactive graph visualizations, then exposes them through a multi-agent chat
interface.

The application supports five main user experiences:

- Answering questions from the personal knowledge base.
- Falling back to web research when local notes are insufficient.
- Producing structured technical explanations through an Analyst agent.
- Generating focused knowledge-graph visualizations through a Visualizer agent.
- Running interview-style practice through a Mentor agent.

## Architecture At A Glance

```text
RemNote exports + external artifacts
        |
        v
Optimized parser IR + retrieval chunks
        |
        v
LLM post-processing sidecars
  - quality decisions
  - graph extraction
  - concept registry
  - graph projection
        |
        v
Final LlamaIndex storage
  - docstore
  - index store
  - vector store with embedding passages
  - property graph store
        |
        v
LangGraph learner workflow
  - Retriever
  - Researcher
  - Analyst
  - Mentor
  - Visualizer
        |
        v
Reflex web application
```

The project deliberately separates offline data preparation from online user
serving. Expensive parsing, OCR, LLM post-processing, and graph materialization
are run ahead of time. The application runtime loads prepared storage and uses
the configured Analyst and Visualizer retrieval modes to ground agent responses.

## Main Components

| Area | Current modules |
| --- | --- |
| Web UI | `app/app.py`, `app/state.py`, `app/components.py` |
| Workflow orchestration | `backend/workflows/learner.py`, `backend/workflows/learner_reflex.py` |
| Agent models/tools | `backend/workflows/agents/factory.py`, `tools.py`, `schemas.py` |
| Optimized retrieval | `analyst_retrieval.py`, `visualizer_retrieval.py`, `retrieval_access.py` |
| Storage/index access | `backend/knowledge_graph/storage.py`, `indexer.py` |
| Optimized data pipeline | `parser_optimized.py`, `llm_postprocess.py`, `llm_postprocess_runner.py`, `concept_registry.py` |
| Final storage build | `scripts/build_postprocessed_graph_storage.py` |
| Deployment | `deploy/Dockerfile`, `deploy/Dockerfile.frontend`, `.github/workflows/deploy.yml` |

The older parser and graph-index scripts still exist, but the production
optimized path is the parser IR plus LLM sidecar flow documented in the runbook.

## Data And Knowledge Pipeline

The optimized pipeline starts from RemNote Markdown exports under
`data/raw/AI Research`. `RemNoteParserOptimized` converts the notes and parsed
external artifacts into a typed intermediate representation:

- `SourceDocument`
- `RemNoteBlock`
- `ExternalResource`
- `ParsedArtifact`
- `ArtifactGateDecision`
- `RetrievalChunk`

The parser preserves provenance while emitting coherent retrieval chunks. It can
prepare external artifacts through network/OCR work, or reuse reviewed cached
Markdown artifacts when requested.

LLM post-processing then writes sidecar files instead of mutating storage
directly. The current production prompt version is `v1`, using separate quality
and graph passes plus concept resolution. Concept resolution uses a controlled
type set:

```text
CONCEPT, METHOD, MODEL, COMPONENT, FORMULA, PARAMETER,
METRIC, DATASET, TASK, PROBLEM, TOOL, PAPER
```

The final storage builder consumes the optimized IR and sidecars, materializes
postprocessed retrieval chunks, imports concept nodes and semantic relations into
the property graph, creates embedding-sized passage children for retrieval
chunks, embeds those passages and concept nodes, and persists the storage used
by the app. Parent retrieval chunks remain the evidence units shown to agents;
passage nodes are internal vector-search records mapped back to stable parent
chunk IDs.

## Runtime Retrieval

Runtime retrieval is split by user-facing task.

The Analyst tool defaults to `legacy_vector_context`. It uses LlamaIndex's
`VectorContextRetriever` with source text, vector similarity, and bounded graph
traversal, then normalizes and formats the returned evidence for the Analyst
agent. The source-first `AnalystRetrievalPipeline` remains available by setting
`KG_SEARCH_ANALYST_RETRIEVAL_MODE=optimized`; that mode adds passage-to-parent
mapping, quarantine filtering, global source dedupe, reranking, and
source-grounded relation selection.

The Visualizer tool defaults to `VisualizerRetrievalPipeline`. It is
concept-first: it resolves query anchors, retrieves supporting source chunks and
concept candidates, uses passage-aware source support, expands semantic graph
edges, filters noisy or ungrounded relations, and returns the graph tuple
consumed by the Visualizer node. Its legacy mode remains available for explicit
comparison.

The optimized pipelines share low-level store access through
`RetrievalStoreAccess`, which centralizes vector-store fallback, graph relation
fallback, docstore lookup, and retrieval health events.

The deterministic retrieval benchmark uses `legacy_vector_context` as the
configured Analyst baseline because that is the application default. Optimized
Analyst retrieval remains an explicit comparison variant. The runner can score
either retrieval implementation against the same reviewed evidence contract;
it records the resolved variant in every case result and run manifest.

## Agent Workflow

The online workflow is a LangGraph `StateGraph` with a central Orchestrator and
five worker agents:

- `Retriever`: decides whether to call knowledge-base search or graph
  visualization retrieval.
- `Researcher`: performs Tavily web research and synthesizes structured findings
  when local retrieval is empty or insufficient.
- `Analyst`: writes grounded technical explanations from retrieved context.
- `Mentor`: conducts interview-style practice using retrieved context as ground
  truth.
- `Visualizer`: converts retrieved graph tuples into Plotly figures.

All worker nodes return to the Orchestrator. The Orchestrator uses deterministic
fast paths when context already implies the next step, and falls back to one
structured-output LLM call for initial routing and broad request-scope
classification. Clearly unrelated requests terminate at the existing end path;
AI/ML-adjacent mathematics, statistics, data, programming, and systems requests
remain in scope. Unknown technical-looking names and acronyms are classified as
ambiguous and continue to retrieval rather than being rejected.

For knowledge searches, Retriever `v6` can provide up to three named-topic alias
groups. After either Analyst retrieval mode formats evidence, a deterministic
exact-token check requires at least one alias from every group. Query and result
headers are excluded, while source text, source paths, and relations count as
evidence. Topic mismatch supplements the existing numeric thresholds and routes
to Researcher through the same `retriever_empty` transition. Broad searches with
no topic requirements retain their previous behavior.

The workflow state includes messages, retrieved context, a compact older-session
summary, visual artifacts, routing state, `request_scope`, `retrieval_status`,
and retrieval failure flags. Retrieval status distinguishes `not_run`,
`adequate`, `no_results`, `below_threshold`, and `topic_mismatch`. The wrapper
initializes these turn-scoped fields for every submitted message. Researcher
`no_relevant_info` or blank findings set `sources_exhausted` and terminate before
Analyst. Final fallback selection reads state directly, prioritizing out-of-scope,
all-sources-exhausted, visualization, and no-results outcomes.

## Frontend And Session Experience

The UI is a Reflex application with a chat-first layout, streaming responses,
agent status, and recent graph navigation. The frontend can render Markdown,
LaTeX, Mermaid blocks, and Plotly graph artifacts.

`ReflexLearnerWorkflow` lazily initializes the heavy workflow singleton and
streams workflow events back to `AppState`. Session memory is MongoDB-backed:
recent messages, older extractive summaries, session history, recent maps, and
visual artifacts can be restored across browser sessions. A MongoDB-backed
quick-action cache can replay configured quick-action responses without
initializing the full workflow.

Anonymous browser identity and chat admission are deliberately separate from a
conversation. Reflex stores an opaque UUID `visitor_id` in browser LocalStorage
and a replaceable `session_id` in session storage. Clearing a conversation
rotates only the session ID, so it does not reset usage counters. This is a cost
control rather than an identity guarantee: clearing site storage creates a new
anonymous visitor and bypasses prior visitor quotas.

When `CHAT_LIMITS_SHARED_QUOTAS_ENABLED=true`, MongoDB coordinates admission
across Cloud Run instances. A visitor lease enforces one active turn and a
ten-second cooldown, while two fixed global lease slots bound expensive graph
workflows without reducing Cloud Run's WebSocket concurrency. Provider-backed
turns and provider attempts use atomic, UTC-period-qualified counters. Quick
action cache hits still observe the visitor lease and cooldown but do not charge
visitor daily, global-workflow, Ollama, or Tavily quotas.

Each accepted turn has a request-scoped context carried through LangGraph with
`contextvars`. It bounds logical LLM calls, provider attempts, retries, Tavily
searches, and content sizes without putting mutable counters on the shared
workflow singleton. Content-free usage events retain IDs, timestamps, status,
routes, attempt counts, and token totals for 35 days; prompts, retrieved text,
answers, secrets, and raw provider errors are excluded.

Shared enforcement fails closed when MongoDB is unavailable. The application
exposes `/healthz`, which reports healthy only after the workflow singleton has
initialized and, when shared quotas are enabled, the quota collection is
reachable. Local development leaves shared MongoDB enforcement disabled by
default while retaining per-turn input, workflow, tool, retry, and context
bounds.

## Storage Model

The project uses LlamaIndex storage abstractions with local and remote
implementations:

| Store | Purpose | Code default | Other supported options |
| --- | --- | --- | --- |
| Document store | Retrieval chunks and metadata | Local `SimpleDocumentStore` under `storage` | Redis |
| Index store | LlamaIndex index metadata | Redis | Local `SimpleIndexStore` |
| Vector store | Chunk and concept embeddings | Pinecone | Local `SimpleVectorStore` or Redis |
| Property graph | Concepts, relations, chunk links | Local `SimplePropertyGraphStore` under `storage` | Memgraph or Neo4j |
| Checkpoints/session memory/chat limits | Workflow state, UI memory, atomic counters, leases, and usage telemetry | MongoDB | Process-memory fallback for checkpoints only |

`CustomMemgraphPropertyGraphStore` and `CustomNeo4jPropertyGraphStore` extend
LlamaIndex behavior so graph traversal can include chunk nodes as well as entity
nodes.

The default `StorageSettings()` are mixed: local document storage, Redis index
storage, Pinecone vector storage, local property-graph storage, and MongoDB
checkpoint storage. Environment-backed settings can change connection details,
while entry points and scripts can construct different backend selections
explicitly. Runtime serving is load-only: storage migration, index construction,
and embedding backfill belong to offline scripts.

## Models, Prompts, And Configuration

Model configuration is centralized in `backend/configs/models.py`.

- Default LLM pipeline: `LLM_PIPELINE=ollama`.
- Default routing, retrieval, research, and Mentor model:
  `nemotron-3-super:cloud` via Ollama-compatible APIs.
- Default Analyst model: `qwen3.5:cloud` with prompt `v6` and an 8,192-token
  generation limit.
- Optional pipeline: `LLM_PIPELINE=vllm`, using OpenAI-compatible endpoints with
  Google Cloud Run ID-token authentication.
- Embedder: local HuggingFace model configured by `EMBEDDER_MODEL_PATH`.
- Optimized Analyst reranker: local Qwen reranker configured by
  `RERANKER_MODEL_PATH` and used through `sentence-transformers` CrossEncoder
  when optimized Analyst retrieval is selected.

Current prompt selections:

| Role / purpose | Default `ollama` pipeline | Optional `vllm` pipeline |
| --- | --- | --- |
| Orchestrator routing | `v5` | `v5` |
| Orchestrator offline/legacy graph indexing | `v2` | `v2` |
| Retriever | `v6` | `v6` |
| Researcher | `v5` | `v5` |
| Analyst | `v6` | `v4` |
| Mentor | `v4` | `v4` |
| LLM postprocess quality/graph | `v1` | `v1` |
| Concept resolution | `v1` | `v1` |

Search behavior is configured in `backend/configs/search.py`. The Analyst
defaults to `legacy_vector_context`, while the Visualizer defaults to
`optimized`. The optimized Analyst mode is opt-in. When it is enabled, its
reranker defaults to `sentence_transformers` and can be disabled or switched to
an Ollama LLM rerank mode through configuration.

## Deployment

The main deployment path is Google Cloud Run:

- Backend image: Reflex backend-only service on port `8000`.
- Frontend image: static Reflex export served by Caddy on port `8080`.
- GitHub Actions builds both images and deploys them to Cloud Run.
- Retrieval models are downloaded from a Cloud Storage bucket into the Docker
  build context and copied into `/app/models`.

The backend container sets offline HuggingFace flags and default model paths:

```text
EMBEDDER_MODEL_PATH=/app/models/all-MiniLM-L6-v2
RERANKER_MODEL_PATH=/app/models/Qwen3-Reranker-0.6B
```

Cloud runtime secrets and environment variables configure Redis, Pinecone,
Memgraph, MongoDB, Ollama, Tavily, and LangSmith.

## Testing And Evaluation

The project now has a focused regression suite covering the optimized parser,
LLM post-processing, concept resolution, final storage materialization, Analyst
retrieval, Visualizer retrieval, retrieval evidence formatting, model settings,
workflow routing, and storage smoke behavior.

Notable test areas:

- Parser chunking, external artifact handling, and three-source parser
  regression.
- LLM postprocess defaults, sidecar writing, prompt contracts, concept
  canonicalization, alias hygiene, cluster bounds, and concept-resolution-only
  reruns.
- Final storage metadata, graph import, and vector metadata safety.
- Embedding passage splitting and parent-chunk mapping for retrieval.
- Analyst and Visualizer retrieval scoring, grounding, dedupe, budgets, and
  fallback behavior.
- Workflow scope routing, named-topic adequacy, web exhaustion, fallback
  priority, and optimized-vs-legacy retrieval selection.

The suite is not a full end-to-end production evaluation harness, but it now
covers the core contracts that were previously only documented in review notes.
Retrieval quality is also checked with the deterministic benchmark runner in
`scripts/evaluate_retrieval_pipeline.py`. Reviewed cases live in
`evals/retrieval/benchmark_cases.jsonl` and use mode-neutral, set-based recall
for supporting chunks, concepts, and relations, plus forbidden-evidence and
Visualizer graph-shape checks. Exact ID and semantic-label/spec matches remain
separate diagnostics. The runner supports configured, optimized, and
`legacy_vector_context` variants without requiring Ragas, LangSmith, or an LLM
evaluator.

## Current Design Boundaries

The project is strongest when expensive data processing is treated as offline
work and the online application only loads prepared storage. The runtime agent
workflow expects that final storage already contains postprocessed retrieval
chunks, concept nodes, semantic relations, and embeddings.

Important boundaries to preserve:

- Parser/data-processing code should produce auditable IR and sidecars, not
  directly decide online retrieval behavior.
- Concept resolution should be conservative; false merges are more damaging than
  duplicate concepts.
- Analyst retrieval should stay source-grounded and relation expansion should
  require evidence from retrieved chunks.
- Parent retrieval chunks should remain the displayed evidence units. Passage
  nodes are an embedding/search implementation detail.
- Visualizer retrieval should produce readable focused graphs, not hide bad
  registry data through aggressive label dedupe.
- Workflow agents should consume formatted context, while retrieval modules own
  store access, reranking, and evidence selection.

## Known Limitations

- The workflow still passes retrieval evidence to generation primarily as
  formatted text, not as a fully typed evidence object.
- Default storage is mixed across local, Redis, Pinecone, and MongoDB backends,
  so every application environment must provide the prepared dependencies selected
  by its effective settings.
- Production-mode secret validation is still limited; some settings have
  development defaults.
- The legacy graph-indexing path remains available to explicit offline scripts,
  but runtime startup fails closed when prepared index storage is missing or
  invalid.
- Scope classification and creation of named-topic requirements rely on the
  existing Orchestrator and Retriever model calls. Their prompt contracts are
  tested, but model compliance remains stochastic and should be measured in the
  future runtime evaluation pipeline.
- Exact-token adequacy checks intentionally do not infer synonyms. They can only
  enforce aliases supplied by the Retriever and do not filter broad queries with
  an empty `required_topics` list.
- Long-form Analyst and Mentor calls remain the most latency-sensitive runtime
  components.
- Web research improves the current turn but is not persisted back into the
  knowledge graph.

## Where To Look Next

- `docs/optimized_pipeline_runbook.md`: how to run processing and build final
  storage.
- `docs/remnote_parser_optimization_review.md`: parser optimization rationale.
- `docs/remnote_llm_postprocessing_review.md`: post-processing rationale.
- `docs/remnote_retrieval_optimization_review.md`: retrieval optimization
  rationale.
- `tests/`: executable examples of the main current contracts.
