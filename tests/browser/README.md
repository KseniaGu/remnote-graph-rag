# AI Practice browser checks

Use the [locked Python environment](../../docs/project_overview.md#setup-and-offline-checks)
and Node.js. These checks exercise the compiled Reflex page and its WebSocket
state. The browser fixture serves in-memory session snapshots and deterministic
workflow events without provider credentials or an existing MongoDB database.
Interactions cover reloading and clearing while a response is streaming.

Install the tested Playwright version and Chromium into temporary storage, or
point `PLAYWRIGHT_MODULE` at an existing compatible Playwright installation.
Export the same settings in each terminal that runs browser checks.

```bash
browser_tools=$(mktemp -d)
npm install --prefix "$browser_tools" --no-save playwright@1.65.1
export PLAYWRIGHT_MODULE="$browser_tools/node_modules/playwright"
export PLAYWRIGHT_BROWSERS_PATH="$browser_tools/browsers"
node "$PLAYWRIGHT_MODULE/cli.js" install chromium
```

From the repository root, start the development frontend:

```bash
PYTHON_DOTENV_DISABLED=1 REFLEX_TELEMETRY_ENABLED=false \
  REFLEX_USE_NPM=true REFLEX_API_URL=http://127.0.0.1:8300 \
  uv run --no-sync reflex run --frontend-only --frontend-port 8301
```

In another terminal, serve the test backend:

```bash
PYTHON_DOTENV_DISABLED=1 REFLEX_TELEMETRY_ENABLED=false \
  REFLEX_API_URL=http://127.0.0.1:8300 REFLEX_STATE_MANAGER_MODE=memory \
  CHAT_LIMITS_SHARED_QUOTAS_ENABLED=false \
  PYTHONPATH="$PWD/tests/browser:$PWD" uv run --no-sync granian --interface asgi \
  --factory --host 127.0.0.1 --port 8300 serve_stage1:create_app
```

Run the interactions and responsive matrix:

```bash
node tests/browser/stage1.cjs http://127.0.0.1:8301/
```

Set `PRACTICE_TEST_MODE` to `interactions` or `layout` to run just that group,
or use `stage3` for sources and execution. The Stage 3 check holds a source
selection WebSocket event to verify the pending state, and reloads with a fresh
Reflex token while keeping the app's tab session ID to exercise snapshot restore.
Set `PRACTICE_EVIDENCE` to a writable directory to save screenshots.
To check static output, stop the development frontend and run:

```bash
PYTHON_DOTENV_DISABLED=1 REFLEX_TELEMETRY_ENABLED=false \
  REFLEX_USE_NPM=true REFLEX_API_URL=http://127.0.0.1:8300 \
  uv run --no-sync reflex export --frontend-only --no-zip --ssr
uv run --no-sync python -m http.server 8301 --bind 127.0.0.1 --directory .web/build/client
```

Keep the fixture backend running and rerun the browser checks at port 8301.

## Stage 4 Balanced graph fixture

The graph fixture compiles the actual wrapper and local React files in a small
Reflex app. Its state contains synthetic, typed Stage 2b records and receives
explanation actions through the Reflex event boundary. Set up the fixture separately from the
running application:

```bash
repo="$PWD"
python_bin="$repo/.venv/bin/python"
reflex_bin="$repo/.venv/bin/reflex"
stage4_dir=$(mktemp -d)
mkdir -p "$stage4_dir/stage4_probe" "$stage4_dir/assets"
touch "$stage4_dir/stage4_probe/__init__.py"
cp tests/browser/stage4_app.py "$stage4_dir/stage4_probe/stage4_probe.py"
cp -a assets/fonts assets/practice.css "$stage4_dir/assets/"
node tests/browser/stage4-fixture.cjs > "$stage4_dir/stage4-fixture.json"
cat > "$stage4_dir/rxconfig.py" <<'PY'
import reflex as rx

config = rx.Config(
    app_name="stage4_probe",
    api_url="http://127.0.0.1:8380",
    deploy_url="http://127.0.0.1:8381",
    telemetry_enabled=False,
)
PY
cd "$stage4_dir"
export PYTHONPATH="$repo" PYTHON_DOTENV_DISABLED=1 REFLEX_USE_NPM=true
export REFLEX_DIR="$stage4_dir/reflex-home" REFLEX_CHECK_LATEST_VERSION=false
"$reflex_bin" run --frontend-port 8381 --backend-port 8380 --backend-host 127.0.0.1
```

From the repository root in another terminal:

```bash
node --test tests/browser/graph_layout.test.mjs
node tests/browser/stage4.cjs http://127.0.0.1:8381/
```

To check prerendering, stop development and, from the fixture directory with
the same environment, run:

```bash
REFLEX_DEPLOY_URL=http://127.0.0.1:8382 \
  "$reflex_bin" export --frontend-only --no-zip --ssr
"$reflex_bin" run --env prod --backend-only --backend-port 8380 --backend-host 127.0.0.1
```

Serve `.web/build/client` via `"$python_bin" -m http.server 8382 --bind
127.0.0.1 --directory .web/build/client` in a separate terminal and point the
same browser command to port 8382. The test reports fixture-to-paint times and
frame gaps sampled during real pointer drags. Set `STAGE4_EVIDENCE` to save
timings as JSON.

## Integrated graph and state checks

Run the regular development frontend from this guide. Generate the approved
fixture's structured records and start the test backend with process-memory
Reflex state and the graph fixture enabled.

```bash
stage5_fixture=$(mktemp)
node tests/browser/stage4-fixture.cjs > "$stage5_fixture"
PYTHON_DOTENV_DISABLED=1 REFLEX_TELEMETRY_ENABLED=false \
  REFLEX_API_URL=http://127.0.0.1:8300 REFLEX_STATE_MANAGER_MODE=memory \
  CHAT_LIMITS_SHARED_QUOTAS_ENABLED=false PRACTICE_GRAPH_FIXTURE="$stage5_fixture" \
  PYTHONPATH="$PWD/tests/browser:$PWD" uv run --no-sync granian --interface asgi \
  --factory --host 127.0.0.1 --port 8300 serve_stage1:create_app
```

Run Chromium against the development URL and again against the prerendered
static export with the same API URL. `PRACTICE_EVIDENCE` optionally saves
screenshots and timings. Run the browser processes sequentially on hosts with
small process limits. The Stage 1 and Stage 3 suites still exercise existing
Study interactions against the same test backend.

```bash
node tests/browser/stage5.cjs http://127.0.0.1:8301/
```
