"""
Parses a LangSmith traces JSON export (made with export_langsmith_logs.py) into a human-readable analysis text file.

Extracts key information for analysing agent behaviour and prompt formatting:
  - LLM calls per graph node (system prompt, context, conversation history, response)
  - Orchestrator routing decisions with reasoning
  - Token / latency stats per step
  - Deduplication: long repeated blocks (system prompts, retriever context) are
    shown in full on first occurrence and replaced with "[→ LABEL]" afterwards.

Usage:
    python scripts/parse_langsmith_traces.py <input.json> [output.txt]
"""

import json
import re
import sys
import hashlib
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Tunable limits (chars)
# ---------------------------------------------------------------------------
MAX_SYSTEM_PROMPT = 5_500   # system prompt shown in full on first occurrence
MAX_CTX_VALUE = 4_000       # per-value cap inside the context dict
MAX_SOURCE_TEXT = 400       # chars of each [SOURCE] block kept
MAX_HISTORY_MSG = 600       # chars per conversation-history message
MAX_AGENT_RESPONSE = 4_000  # final agent response
MAX_ROUTING_RESPONSE = 800  # orchestrator raw JSON response
MAX_TOOL_OUTPUT = 4_000     # tool run output cap

SEPARATOR_THICK = "=" * 80
SEPARATOR_THIN = "-" * 60


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class DedupTracker:
    """Replace repeated text blocks with a short reference label."""

    def __init__(self) -> None:
        self._seen: dict[str, str] = {}   # md5 -> label
        self._counters: dict[str, int] = defaultdict(int)

    def _fp(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:10]

    def label_of(self, text: str) -> Optional[str]:
        return self._seen.get(self._fp(text))

    def register(self, text: str, prefix: str) -> str:
        """Register text, return a label string (e.g. '[SYS_1]')."""
        fp = self._fp(text)
        if fp in self._seen:
            return self._seen[fp]
        self._counters[prefix] += 1
        label = f"{prefix}_{self._counters[prefix]}"
        self._seen[fp] = label
        return label

    def process(self, text: str, prefix: str, max_chars: int) -> str:
        """Return formatted text – full on first occurrence, ref label on repeats."""
        existing = self.label_of(text)
        if existing:
            preview = text[:90].replace("\n", " ").strip()
            return f"[→ {existing}]  # {preview}..."

        label = self.register(text, prefix)
        if len(text) > max_chars:
            return f"[{label}]\n{text[:max_chars]}\n... [+{len(text) - max_chars} chars truncated]"
        return f"[{label}]\n{text}"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _indent(text: str, n: int = 2) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in text.split("\n"))


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [+{len(text) - max_chars} chars truncated]"


_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Remove surrounding ```json ... ``` or ``` ... ``` wrappers if present."""
    m = _FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text


# ---------------------------------------------------------------------------
# Content-aware formatters
# ---------------------------------------------------------------------------

def _format_retriever_results(raw: str) -> str:
    """Keep QUERY + [RELATION] lines; truncate each [SOURCE] block."""
    lines = raw.split("\n")
    out: list[str] = []
    in_source = False
    source_buf: list[str] = []

    def _flush_source() -> None:
        if source_buf:
            combined = "\n".join(source_buf)
            if len(combined) > MAX_SOURCE_TEXT:
                combined = (
                    combined[:MAX_SOURCE_TEXT]
                    + f" ... [+{len(combined) - MAX_SOURCE_TEXT} chars SOURCE TRUNCATED]"
                )
            out.append(combined)
        source_buf.clear()

    for line in lines:
        stripped = line.strip()
        # [SOURCE], [SOURCE PATH], [SOURCE_xxx] all start a new source block.
        if stripped.startswith("[SOURCE"):
            _flush_source()
            in_source = True
            source_buf.append(line)
        elif stripped.startswith("[RELATION]") or stripped.startswith("QUERY:") or stripped.startswith("RETRIEVER RESULTS"):
            _flush_source()
            in_source = False
            out.append(line)
        elif in_source:
            source_buf.append(line)
        else:
            out.append(line)

    _flush_source()
    return "\n".join(out)


_RETRIEVER_KEYS = ("search_knowledge_base", "get_subgraphs_to_visualize")


def _format_context_json(context_str: str) -> str:
    """Render the context as a multiline dict, truncating only values.

    Produces output like:
        {
          "search_knowledge_base": <<<
            RETRIEVER RESULTS:
            ...
          >>>,
          "no_results": false,
        }
    The surrounding braces make it scannable as a dict; per-value truncation
    preserves dict structure even when individual values are huge.
    """
    context_str = context_str.strip()
    try:
        ctx = json.loads(context_str)
    except (json.JSONDecodeError, TypeError):
        return _truncate(context_str, MAX_CTX_VALUE)

    if not isinstance(ctx, dict) or not ctx:
        return "{}  (empty context)"

    lines: list[str] = ["{"]
    for key, value in ctx.items():
        key_repr = json.dumps(key)
        if isinstance(value, str) and key in _RETRIEVER_KEYS:
            formatted = _format_retriever_results(value)
            formatted = _truncate(formatted, MAX_CTX_VALUE)
            lines.append(f"  {key_repr}: <<<")
            lines.append(_indent(formatted, 4))
            lines.append("  >>>,")
        elif isinstance(value, str):
            truncated = _truncate(value, MAX_CTX_VALUE)
            if "\n" in truncated:
                lines.append(f"  {key_repr}: <<<")
                lines.append(_indent(truncated, 4))
                lines.append("  >>>,")
            else:
                lines.append(f"  {key_repr}: {json.dumps(truncated)},")
        else:
            # bool / None / number / list / dict -> compact JSON
            compact = json.dumps(value, ensure_ascii=False)
            lines.append(f"  {key_repr}: {_truncate(compact, MAX_CTX_VALUE)},")
    lines.append("}")
    return "\n".join(lines)


def _format_xml_history(history_xml: str) -> str:
    """Parse <message role=X><content>...</content></message> history.

    Single-line messages are rendered inline: '  [ROLE]: content'.
    Multi-line messages get a hanging indent so continuation lines align
    under the first line of content, not under the role label.
    """
    if not history_xml.strip():
        return "(empty)"
    messages = re.findall(
        r"<message\s+role=(\w+)>\s*<content>(.*?)</content>\s*</message>",
        history_xml,
        re.DOTALL,
    )
    if not messages:
        return _truncate(history_xml, MAX_HISTORY_MSG * 4)

    blocks: list[str] = []
    for role, content in messages:
        content = _truncate(content.strip(), MAX_HISTORY_MSG)
        label = f"  [{role.upper()}]:"
        if "\n" not in content:
            blocks.append(f"{label} {content}")
        else:
            # Place content on its own lines indented 4 spaces so it sits
            # cleanly under the role label regardless of label width.
            blocks.append(f"{label}\n{_indent(content, 4)}")
    return "\n".join(blocks)


_SECTION_HEADERS = (
    "# Current context",
    "# Ground truth context",
    "# Conversation history",
    "# User last message",
    "# User previous message",
)


def _split_human_message(content: str) -> dict[str, str]:
    """Split the human prompt into its named sections.

    Returns a dict with any of the keys: 'context', 'history', 'user_last'.
    Falls back gracefully when headers are missing (e.g. agent variants
    that omit context, or retriever prompts that add '# User last message').
    """
    # Locate every known header in the order they appear in the content.
    hits: list[tuple[int, str]] = []
    for header in _SECTION_HEADERS:
        idx = content.find(header)
        if idx != -1:
            hits.append((idx, header))
    hits.sort()

    sections: dict[str, str] = {}
    if not hits:
        if "<message role=" in content:
            sections["history"] = content.strip()
        else:
            sections["context"] = content.strip()
        return sections

    header_to_key = {
        "# Current context": "context",
        "# Ground truth context": "context",
        "# Conversation history": "history",
        "# User last message": "user_last",
        "# User previous message": "user_prev",
    }
    for i, (start, header) in enumerate(hits):
        body_start = start + len(header)
        body_end = hits[i + 1][0] if i + 1 < len(hits) else len(content)
        body = content[body_start:body_end]
        # Drop leading 'dashed' separators used by some prompts.
        body = re.sub(r"^[\s\-]*\n", "", body, count=1)
        body = body.rstrip().rstrip("-").rstrip()
        sections[header_to_key[header]] = body.strip()
    return sections


def _extract_llm_messages(raw_messages) -> tuple[str, str]:
    """
    Extract (system_content, human_content) from the various shapes that
    LangSmith stores LLM input messages in.
    """
    system, human = "", ""
    if not raw_messages:
        return system, human

    # Unwrap outer list if present: messages[[...]]
    msg_list = raw_messages
    if isinstance(msg_list, list) and msg_list and isinstance(msg_list[0], list):
        msg_list = msg_list[0]

    for msg in msg_list:
        if not isinstance(msg, dict):
            continue

        # ---- Format A: LangChain constructor dicts ----
        if msg.get("type") == "constructor":
            kwargs = msg.get("kwargs", {})
            content = kwargs.get("content", "")
            type_path = msg.get("id", [])
            type_str = str(type_path)
            if "SystemMessage" in type_str:
                system = content
            elif "HumanMessage" in type_str:
                human = content
            continue

        # ---- Format B: simple {"type": "system"|"human", "content": ...} ----
        msg_type = msg.get("type", "")
        content = msg.get("content", "")
        if msg_type == "system":
            system = content
        elif msg_type in ("human", "user"):
            human = content

    return system, human


def _extract_response_text(outputs: dict) -> str:
    """Pull the response text from an LLM run's outputs."""
    gens = outputs.get("generations", [[]])
    if gens and gens[0]:
        g = gens[0][0]
        if isinstance(g, dict):
            msg = g.get("message", {})
            if isinstance(msg, dict):
                kwargs = msg.get("kwargs", {})
                return kwargs.get("content", "") or msg.get("content", "")
            return g.get("text", "")
    return ""


def _extract_tool_calls(outputs: dict) -> list[dict]:
    """Pull structured tool_calls from an LLM run's outputs (if any)."""
    gens = outputs.get("generations", [[]])
    if not (gens and gens[0]):
        return []
    g = gens[0][0]
    if not isinstance(g, dict):
        return []
    msg = g.get("message", {})
    if not isinstance(msg, dict):
        return []
    calls = msg.get("kwargs", {}).get("tool_calls") or []
    return [c for c in calls if isinstance(c, dict)]


def _format_tool_calls(calls: list[dict]) -> str:
    parts = []
    for c in calls:
        name = c.get("name", "?")
        args = c.get("args", {})
        try:
            args_repr = json.dumps(args, ensure_ascii=False)
        except (TypeError, ValueError):
            args_repr = str(args)
        parts.append(f"TOOL CALL: {name}({_truncate(args_repr, 600)})")
    return "\n".join(parts)


def _get_langgraph_meta(run: dict) -> dict:
    meta = run.get("extra", {}).get("metadata", {})
    return {
        "node": meta.get("langgraph_node", ""),
        "step": meta.get("langgraph_step", ""),
        "thread_id": meta.get("thread_id", ""),
        "depth": meta.get("ls_run_depth", 0),
    }


def _get_model(run: dict) -> str:
    gens = run.get("outputs", {}).get("generations", [[]])
    if gens and gens[0] and isinstance(gens[0][0], dict):
        model = gens[0][0].get("generation_info", {}).get("model", "")
        if model:
            return model
    return run.get("extra", {}).get("invocation_params", {}).get("model", "?")


# ---------------------------------------------------------------------------
# Run-tree helpers
# ---------------------------------------------------------------------------

def _build_children(runs: list[dict]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for r in runs:
        pid = r.get("parent_run_id")
        if pid:
            children[pid].append(r["_id"])
    return children


def _find_node_for_run(run_id: str, runs_by_id: dict, max_depth: int = 8) -> dict:
    """Walk up parent chain to find the langgraph node context."""
    current = run_id
    for _ in range(max_depth):
        r = runs_by_id.get(current)
        if not r:
            break
        meta = _get_langgraph_meta(r)
        if meta["node"]:
            return meta
        current = r.get("parent_run_id", "")
    return {"node": "?", "step": "?", "thread_id": "", "depth": 0}


# ---------------------------------------------------------------------------
# Main trace formatter
# ---------------------------------------------------------------------------

def format_trace(trace_id: str, runs_by_id: dict, dedup: DedupTracker) -> str:
    trace_runs = [r for r in runs_by_id.values() if r.get("trace_id") == trace_id]
    if not trace_runs:
        return ""
    trace_runs.sort(key=lambda r: r.get("start_time", ""))

    # Gather metadata
    session_id = trace_runs[0].get("session_id", "")
    thread_id = next(
        (_get_langgraph_meta(r)["thread_id"] for r in trace_runs if _get_langgraph_meta(r)["thread_id"]),
        "",
    )
    start_ts = trace_runs[0].get("start_time", "")[:19]
    end_ts = trace_runs[-1].get("end_time", "")[:19]

    # Aggregate token totals across LLM runs in this trace
    llm_runs = [r for r in trace_runs if r.get("run_type") == "llm"]
    total_in = sum(r.get("prompt_tokens") or 0 for r in llm_runs)
    total_out = sum(r.get("completion_tokens") or 0 for r in llm_runs)

    out: list[str] = []
    out.append(f"\n{SEPARATOR_THICK}")
    out.append(f"TRACE : {trace_id}")
    out.append(f"Thread: {thread_id or 'n/a'}  |  Session: {session_id}")
    out.append(f"Time  : {start_ts} → {end_ts}")
    out.append(f"Tokens: {total_in} in + {total_out} out = {total_in + total_out} total  ({len(llm_runs)} LLM call(s))")
    out.append(SEPARATOR_THICK)

    # One block per LLM call, in chronological order
    for llm_run in sorted(llm_runs, key=lambda r: r.get("start_time", "")):
        node_meta = _find_node_for_run(llm_run["_id"], runs_by_id)
        node = node_meta["node"] or "unknown"
        step = node_meta["step"]

        tokens_in = llm_run.get("prompt_tokens") or 0
        tokens_out = llm_run.get("completion_tokens") or 0
        latency = llm_run.get("latency") or 0
        model = _get_model(llm_run)
        ts = llm_run.get("start_time", "")[:19]
        status = llm_run.get("status", "?")

        out.append(f"\n{'─' * 60}")
        out.append(
            f"## STEP {step} | {node.upper()}"
            f"  [{ts}  {latency:.1f}s  {tokens_in}↑{tokens_out}↓  {model}]"
        )
        if status != "success":
            out.append(f"   STATUS: {status}")
        if llm_run.get("error"):
            out.append(f"   ERROR: {llm_run['error']}")
        out.append("")

        # --- Extract messages ---
        raw_msgs = llm_run.get("inputs", {}).get("messages", [])
        system_content, human_content = _extract_llm_messages(raw_msgs)

        # SYSTEM PROMPT
        if system_content:
            out.append("### SYSTEM PROMPT:")
            deduped = dedup.process(system_content, "SYS", MAX_SYSTEM_PROMPT)
            out.append(_indent(deduped, 2))
            out.append("")

        # HUMAN MESSAGE → split into named sections
        if human_content:
            sections = _split_human_message(human_content)

            ctx_str = sections.get("context", "")
            if ctx_str:
                out.append("### CONTEXT (retriever / tool results):")
                formatted_ctx = _format_context_json(ctx_str)
                # Dedup label without length cap: the dict itself isn't
                # truncated (only its values were, inside _format_context_json).
                deduped_ctx = dedup.process(formatted_ctx, "CTX", 10**9)
                out.append(_indent(deduped_ctx, 2))
                out.append("")

            hist_str = sections.get("history", "")
            if hist_str:
                out.append("### CONVERSATION HISTORY (passed to LLM):")
                formatted_hist = _format_xml_history(hist_str)
                deduped_hist = dedup.process(formatted_hist, "HIST", 10**9)
                out.append(_indent(deduped_hist, 2))
                out.append("")

            user_last = sections.get("user_last", "")
            if user_last:
                # Never deduped / suppressed - this is the actionable ask.
                out.append("### USER LAST MESSAGE:")
                out.append(_indent(_truncate(user_last, MAX_HISTORY_MSG * 2), 2))
                out.append("")

            user_prev = sections.get("user_prev", "")
            if user_prev:
                out.append("### USER PREVIOUS MESSAGE:")
                out.append(_indent(_truncate(user_prev, MAX_HISTORY_MSG * 2), 2))
                out.append("")

        # LLM RESPONSE (text and/or tool_calls)
        response_text = _extract_response_text(llm_run.get("outputs", {}))
        tool_calls = _extract_tool_calls(llm_run.get("outputs", {}))
        if response_text or tool_calls:
            out.append("### LLM RESPONSE:")
            if response_text:
                # Try to parse as routing JSON (orchestrator pattern),
                # tolerating ```json ... ``` fenced wrappers.
                candidate = _strip_code_fences(response_text)
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and "next_step" in parsed:
                        out.append(f"  ROUTING  → {parsed['next_step']}")
                        reasoning = parsed.get("reasoning", "")
                        out.append(f"  REASONING: {reasoning}")
                    else:
                        out.append(_indent(_truncate(response_text, MAX_ROUTING_RESPONSE), 2))
                except (json.JSONDecodeError, ValueError):
                    out.append(_indent(_truncate(response_text, MAX_AGENT_RESPONSE), 2))
            if tool_calls:
                out.append(_indent(_format_tool_calls(tool_calls), 2))
            out.append("")

        # TOOL RUNS that belong to this LLM step (immediate retriever tools
        # executed because of this LLM's tool_calls). Match by parent chain.
        for tool_run in _tool_runs_for_step(llm_run, runs_by_id):
            out.append(_format_tool_run(tool_run))

    return "\n".join(out)


def _tool_runs_for_step(llm_run: dict, runs_by_id: dict) -> list[dict]:
    """Find tool runs whose langgraph_node+step match this LLM run and
    whose start_time is at or after the LLM run's start."""
    meta = _find_node_for_run(llm_run["_id"], runs_by_id)
    if not meta["node"]:
        return []
    llm_start = llm_run.get("start_time", "")
    matches = []
    for r in runs_by_id.values():
        if r.get("run_type") != "tool":
            continue
        if r.get("trace_id") != llm_run.get("trace_id"):
            continue
        tm = _find_node_for_run(r["_id"], runs_by_id)
        if tm["node"] == meta["node"] and tm["step"] == meta["step"] \
                and r.get("start_time", "") >= llm_start:
            matches.append(r)
    matches.sort(key=lambda r: r.get("start_time", ""))
    return matches


def _format_tool_run(tool_run: dict) -> str:
    name = tool_run.get("name", "?")
    ts = tool_run.get("start_time", "")[:19]
    latency = tool_run.get("latency") or 0
    status = tool_run.get("status", "?")
    inputs = tool_run.get("inputs", {})
    input_repr = inputs.get("input") if isinstance(inputs, dict) else None
    if input_repr is None:
        try:
            input_repr = json.dumps(inputs, ensure_ascii=False)
        except (TypeError, ValueError):
            input_repr = str(inputs)
    outputs = tool_run.get("outputs", {}) or {}
    output_text = outputs.get("output") if isinstance(outputs, dict) else ""
    if not isinstance(output_text, str):
        try:
            output_text = json.dumps(output_text, ensure_ascii=False)
        except (TypeError, ValueError):
            output_text = str(output_text)

    formatted_output = _format_retriever_results(output_text) if output_text else ""
    formatted_output = _truncate(formatted_output, MAX_TOOL_OUTPUT)

    lines = ["### TOOL RUN:"]
    lines.append(f"  {name}  [{ts}  {latency:.1f}s  status={status}]")
    lines.append(f"  INPUT:  {_truncate(str(input_repr), 600)}")
    if tool_run.get("error"):
        lines.append(f"  ERROR:  {tool_run['error']}")
    if formatted_output:
        lines.append("  OUTPUT:")
        lines.append(_indent(formatted_output, 4))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Latency analysis
# ---------------------------------------------------------------------------

def _analyze_latency(runs: list[dict]) -> str:
    """Compute and format latency statistics by agent/node and overall."""
    llm_runs = [r for r in runs if r.get("run_type") == "llm"]
    tool_runs = [r for r in runs if r.get("run_type") == "tool"]

    if not llm_runs and not tool_runs:
        return ""

    lines = [f"\n{SEPARATOR_THICK}", "LATENCY ANALYSIS", SEPARATOR_THICK, ""]

    # Per-node (agent) latency for LLM runs
    node_latencies: dict[str, list[float]] = defaultdict(list)
    for r in llm_runs:
        meta = _get_langgraph_meta(r)
        node = meta.get("node") or "unknown"
        latency = r.get("latency") or 0
        node_latencies[node].append(latency)

    if node_latencies:
        lines.append("## LLM Calls by Agent")
        lines.append("")
        # Sort by total time (descending)
        sorted_nodes = sorted(
            node_latencies.items(),
            key=lambda kv: sum(kv[1]),
            reverse=True,
        )
        for node, latencies in sorted_nodes:
            count = len(latencies)
            total = sum(latencies)
            mean = total / count if count else 0
            median = statistics.median(latencies) if latencies else 0
            min_lat = min(latencies) if latencies else 0
            max_lat = max(latencies) if latencies else 0
            lines.append(
                f"  {node.upper():15s}  "
                f"count={count:2d}  "
                f"total={total:7.1f}s  "
                f"mean={mean:6.2f}s  "
                f"median={median:6.2f}s  "
                f"min={min_lat:6.2f}s  "
                f"max={max_lat:6.2f}s"
            )
        lines.append("")

    # Per-tool latency
    tool_latencies: dict[str, list[float]] = defaultdict(list)
    for r in tool_runs:
        name = r.get("name") or "unknown"
        latency = r.get("latency") or 0
        tool_latencies[name].append(latency)

    if tool_latencies:
        lines.append("## Tool Calls by Type")
        lines.append("")
        sorted_tools = sorted(
            tool_latencies.items(),
            key=lambda kv: sum(kv[1]),
            reverse=True,
        )
        for tool_name, latencies in sorted_tools:
            count = len(latencies)
            total = sum(latencies)
            mean = total / count if count else 0
            median = statistics.median(latencies) if latencies else 0
            min_lat = min(latencies) if latencies else 0
            max_lat = max(latencies) if latencies else 0
            lines.append(
                f"  {tool_name:25s}  "
                f"count={count:2d}  "
                f"total={total:7.1f}s  "
                f"mean={mean:6.2f}s  "
                f"median={median:6.2f}s  "
                f"min={min_lat:6.2f}s  "
                f"max={max_lat:6.2f}s"
            )
        lines.append("")

    # Overall summary
    all_latencies = [r.get("latency") or 0 for r in llm_runs + tool_runs]
    if all_latencies:
        lines.append("## Overall Summary")
        lines.append("")
        total_latency = sum(all_latencies)
        mean_latency = total_latency / len(all_latencies)
        median_latency = statistics.median(all_latencies)
        lines.append(f"  Total runs        : {len(all_latencies)}")
        lines.append(f"  Total latency     : {total_latency:.1f}s")
        lines.append(f"  Mean latency      : {mean_latency:.2f}s")
        lines.append(f"  Median latency    : {median_latency:.2f}s")
        lines.append(f"  Min latency       : {min(all_latencies):.2f}s")
        lines.append(f"  Max latency       : {max(all_latencies):.2f}s")
        lines.append("")

    lines.append(SEPARATOR_THICK)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python parse_langsmith_traces.py <input.json> [output.txt]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else input_path.with_suffix(".analysis.txt")

    print(f"Loading {input_path} …")
    with open(input_path, encoding="utf-8") as f:
        runs: list[dict] = json.load(f)
    print(f"  {len(runs)} run records loaded")

    runs_by_id: dict[str, dict] = {r["_id"]: r for r in runs}

    # Collect unique trace IDs ordered by first-seen start_time
    trace_first_seen: dict[str, str] = {}
    for r in runs:
        tid = r.get("trace_id")
        if tid:
            st = r.get("start_time", "")
            if tid not in trace_first_seen or st < trace_first_seen[tid]:
                trace_first_seen[tid] = st
    trace_ids = sorted(trace_first_seen, key=lambda t: trace_first_seen[t])
    print(f"  {len(trace_ids)} distinct trace(s)")

    dedup = DedupTracker()

    header_lines = [
        "LangSmith Traces — Human-Readable Analysis",
        f"Source : {input_path}",
        f"Runs   : {len(runs)}",
        f"Traces : {len(trace_ids)}",
        SEPARATOR_THICK,
        "NOTE: Long repeated blocks (system prompts, retriever context) are shown",
        "in full on first occurrence and referenced as [→ LABEL] afterwards.",
        f"Truncation limits: system={MAX_SYSTEM_PROMPT}c  ctx_value={MAX_CTX_VALUE}c",
        f"  history_msg={MAX_HISTORY_MSG}c  response={MAX_AGENT_RESPONSE}c  tool_output={MAX_TOOL_OUTPUT}c",
        SEPARATOR_THICK,
    ]

    parts = ["\n".join(header_lines)]

    for i, trace_id in enumerate(trace_ids, 1):
        print(f"  Processing trace {i}/{len(trace_ids)}: {trace_id}")
        parts.append(format_trace(trace_id, runs_by_id, dedup))

    # Add latency analysis at the end
    latency_analysis = _analyze_latency(runs)
    if latency_analysis:
        parts.append(latency_analysis)

    full_output = "\n".join(parts)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_output)

    print(f"\nOutput written to: {output_path}")
    print(f"Output size      : {len(full_output):,} chars  ({len(full_output) // 1024} KB)")


if __name__ == "__main__":
    main()
