# AI Practice UI redesign handoff

AI Practice organizes the same browser-tab conversation into Welcome, Study,
and Graph. Welcome offers sample prompts, Study gives streamed answers a
dedicated reading pane, and Graph makes retained concept maps explorable.
Current-session history links back to turns and maps. Sources and execution
details are disclosed separately from the answer. Mentor remains a
conversational agent. The interface supports light and dark themes, keyboard
navigation and reduced motion. On smaller screens, auxiliary panels move into
drawers so the reading pane and composer stay usable.

## Reflex and React boundaries

- [Reflex components](../app/components.py) and [AppState](../app/state.py)
  own the page, session state, streaming, admission, and cross-view navigation.
  The local [PracticeShell](../app/practice.jsx) handles browser-side scrolling,
  focus, drawer behavior, and Mermaid rendering. The app uses local styles
  and fonts in [assets](../assets/).
- The [Reflex graph wrapper](../app/ui/knowledge_graph.py) exposes versioned
  graph data, known sources, visibility, and selection commands to the
  [React workspace](../app/ui/knowledge_graph.jsx). Reflex packages the JSX,
  stylesheet, [layout](../app/ui/graph_layout.mjs), and
  [camera](../app/ui/graph_camera.mjs) as shared local assets. Layout and
  measurement depend on visibility and available fonts and run in the browser
  after rendering. Pointer movement, camera changes, filters, and selection
  stay in React. AppState validates session-owned artifacts and relevant
  concept and source IDs when handling actions across that boundary.
- A map keeps its camera and selection when switching between Study and Graph.
  Hidden graph work is suspended. A late result stays available through its
  Study row and the map selector. Switching artifacts resets the workspace.
  Explain this prepares an editable draft in the normal composer. Study links
  an evidence concept to a retained map by stored concept ID and prefers
  same-turn matches.

## Presentation and session contracts

- [Presentation models](../backend/workflows/presentation.py) define turn and
  response identities, response evidence, execution, and graph artifacts.
  Graph schema 1 preserves stable node IDs, directed edges, synthetic edge
  flags, and available source associations. Its contract admits up to 512
  nodes and 2,048 edges. Interactive Map is capped at 48 nodes and 96 edges.
  Larger valid artifacts open in Relations with their relationship rows
  available. Plotly remains available for offline exports and historical
  trace decoding.
- Analyst and Mentor capture admitted evidence for each response before
  workflow state can clear it. Note rows can show stored titles, summaries,
  and bounded previews for notes referenced by admitted relationships. Web
  rows reflect URLs and titles supplied with Researcher findings. Graph sources
  belong to the artifact and its concepts. These records describe context used
  by a response or map. Claim-level citations are outside this contract.
- The [streaming bridge](../backend/workflows/learner_reflex.py) uses run IDs
  and event ancestry to associate agent invocations with provisional and final
  messages. Execution reports retain invocation order, monotonic timing
  offsets, and observed outcomes, including partial and graph-only turns.
  Displayed durations are elapsed stage observations. Cached turns show their
  replay origin and have no measured invocation stages.
- [Quick-action cache](../backend/utils/cache.py) and
  [session snapshots](../backend/utils/session_memory.py) use schema 5.
  Incompatible records are treated as misses. Eligible first-turn cache
  replays pass admission and receive fresh session-specific turn, response,
  artifact, and DOM IDs. Snapshots restore the browser tab's conversation,
  execution reports, and up to ten maps. Reads and writes are best-effort.
- A stable anonymous visitor ID is stored separately from the tab session ID.
  Clear rotates the session ID, invalidates active work, drains in-flight
  writes, and retires the old snapshot with a content-free marker. This marker
  fences stale snapshot writes until its TTL expires. Visitor admission
  history survives Clear. A disconnected turn may finish within its timeout
  and be restored on reload. Task cancellation coordination is process-local.

## Verification and maintenance

- Use the [offline suite](project_overview.md#setup-and-offline-checks) for
  presentation, source provenance, execution and cancellation, BSON
  persistence, cache replay, and graph routing. The
  [browser guide](../tests/browser/README.md) covers Node geometry tests and
  Chromium against the compiled Reflex app with synthetic workflow events,
  including Study streaming, responsive layouts, graph interactions, and
  snapshot restore. Keep synthetic fixtures independent of local design files.
- The implementation stages exercised both development and prerendered builds
  in Chromium and rehearsed schema cutover with synthetic records. A real-store
  cutover, multi-worker reconnects, Firefox, Safari, assistive technology, and
  physical touch or low-power devices remain open. Graph-source enrichment is
  bounded to 128 stored node mentions. Cold remote docstore read latency remains
  unmeasured.
- When changing presentation records, check live streaming, snapshot restore,
  and cache replay together. Recheck export and hydration after changes to
  shared JSX or assets. Verify both sides of the Map ceiling and keep browser
  interaction state local.
- For a future approved state reset, use the
  [disposable rehearsal](../tests/test_presentation_persistence.py) as a
  reference. Inventory the target environment, application-owned records, and
  state-manager namespace. Quiesce old writers before resetting application
  state. Preserve visitor quotas, checkpoints, prepared stores, and evaluation
  evidence. Published evaluation results are historical. See the
  [runtime evaluation guide](runtime_evaluation.md) for those contracts.
