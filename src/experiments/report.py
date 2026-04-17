"""Generate a self-contained HTML report from a run's JSONL artifacts.

Usage:
    python -m src.experiments.report outputs/<scenario>_<timestamp>
"""

import html
import json
import re
import sys
from pathlib import Path
from typing import Any

_DIR_NAME_RE = re.compile(r"^(?P<name>.+)_(?P<ts>\d{8}_\d{6})$")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_agents(run_dir: Path) -> list[dict[str, Any]]:
    agents: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*.jsonl")):
        if path.name == "actions.jsonl":
            continue
        entries = _read_jsonl(path)
        if not entries:
            continue
        meta = entries[0]
        if meta.get("_type") != "agent_metadata":
            continue
        agents.append({
            "id": meta.get("agent_id", path.stem),
            "profile": meta.get("profile", {}),
            "total_messages": meta.get("total_messages", len(entries) - 1),
            "turns": entries[1:],
        })
    return agents


def _parse_scenario_meta(run_dir: Path) -> tuple[str, str]:
    match = _DIR_NAME_RE.match(run_dir.name)
    if match:
        return match.group("name"), match.group("ts")
    return run_dir.name, ""


def _principal(profile: dict[str, Any]) -> str:
    principal = profile.get("principal")
    if isinstance(principal, dict):
        return principal.get("organization") or principal.get("name") or ""
    if isinstance(principal, str):
        return principal
    return ""


# --- HTML helpers ------------------------------------------------------------

def _esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)


def _pretty_maybe_json(s: Any) -> str:
    """If `s` is a JSON-encoded string, pretty-print it; otherwise return as-is."""
    if isinstance(s, str):
        try:
            return json.dumps(json.loads(s), indent=2, ensure_ascii=False)
        except (TypeError, ValueError, json.JSONDecodeError):
            return s
    return _pretty(s)


def _truncate(s: str, n: int = 160) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


# --- Timeline rendering ------------------------------------------------------

def _action_summary(action: dict[str, Any]) -> tuple[str, str, str]:
    """Return (kind, badge, one_line_summary)."""
    name = action.get("request_name", "")
    params = action.get("request_parameters", {}) or {}
    if name == "SendMessage":
        to = params.get("to_agent_id", "?")
        content = (params.get("message") or {}).get("content", "")
        return "dm", "DM", f"→ {to}: {_truncate(content)}"
    if name == "ChannelMessage":
        channel = params.get("channel", "?")
        content = (params.get("message") or {}).get("content", "")
        return "channel", f"#{channel}", _truncate(content)
    if name == "CreateChannel":
        channel = params.get("channel", "?")
        members = params.get("member_ids", [])
        return "channel", "create", f"#{channel} with {members}"
    if name == "ExecuteCommand":
        cmd = params.get("command", [])
        return "command", "CMD", _truncate(" ".join(cmd) if isinstance(cmd, list) else str(cmd))
    if name == "FetchMessages":
        return "fetch", "FETCH", ""
    return "other", name or "?", ""


def _render_timeline(actions: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    parts.append('<section class="section">')
    parts.append('<div class="section-header">')
    parts.append('<h2>Timeline</h2>')
    parts.append('<label class="toggle"><input type="checkbox" id="show-fetch"> show FetchMessages</label>')
    parts.append('</div>')
    parts.append('<div class="timeline">')
    for action in actions:
        kind, badge, summary = _action_summary(action)
        is_error = bool(action.get("result_is_error"))
        agent_id = action.get("agent_id", "?")
        idx = action.get("index", "?")
        ts = str(action.get("created_at", ""))
        classes = f"card kind-{kind}"
        if kind == "fetch":
            classes += " fetch-row"
        if is_error:
            classes += " is-error"

        parts.append(f'<div class="{classes}" onclick="toggleCard(this)">')
        parts.append('<div class="card-head">')
        parts.append(f'<span class="idx">#{_esc(idx)}</span>')
        parts.append(f'<span class="badge badge-{kind}">{_esc(badge)}</span>')
        parts.append(f'<span class="agent">{_esc(agent_id)}</span>')
        parts.append(f'<span class="summary">{_esc(summary)}</span>')
        parts.append(f'<span class="ts">{_esc(ts)}</span>')
        parts.append('</div>')
        parts.append('<div class="card-body">')
        parts.append('<div class="kv"><span class="kv-k">parameters</span>')
        parts.append(f'<pre>{_esc(_pretty(action.get("request_parameters")))}</pre></div>')
        parts.append('<div class="kv"><span class="kv-k">result</span>')
        parts.append(f'<pre>{_esc(_pretty(action.get("result_content")))}</pre></div>')
        parts.append('</div>')
        parts.append('</div>')
    parts.append('</div>')
    parts.append('</section>')
    return "".join(parts)


# --- Per-agent conversation rendering ----------------------------------------

def _render_turn(turn: dict[str, Any]) -> str:
    role = turn.get("role", "")
    parts: list[str] = []

    if role == "user":
        parts.append('<div class="turn turn-user">')
        parts.append('<div class="turn-head">user</div>')
        parts.append(f'<div class="turn-body"><pre class="prose">{_esc(turn.get("content", ""))}</pre></div>')
        parts.append('</div>')

    elif role == "assistant":
        parts.append('<div class="turn turn-assistant">')
        parts.append('<div class="turn-head">assistant</div>')
        text = turn.get("text") or turn.get("content") or ""
        if text:
            parts.append(f'<div class="turn-body"><pre class="prose">{_esc(text)}</pre></div>')
        for tc in turn.get("tool_calls") or []:
            tc_id = tc.get("id", "")
            parts.append('<div class="tool-call">')
            parts.append('<div class="tool-call-head">')
            parts.append(f'<span class="tool-name">{_esc(tc.get("name", ""))}</span>')
            if tc_id:
                parts.append(f'<span class="tool-id" title="tool_use_id">{_esc(tc_id)}</span>')
            parts.append('</div>')
            parts.append(f'<pre>{_esc(_pretty(tc.get("arguments", {})))}</pre>')
            parts.append('</div>')
        parts.append('</div>')

    elif role == "tool":
        parts.append('<div class="turn turn-tool">')
        parts.append('<div class="turn-head">')
        parts.append('tool result')
        name = turn.get("name")
        if name:
            parts.append(f' <span class="tool-name">{_esc(name)}</span>')
        tc_id = turn.get("tool_use_id") or turn.get("tool_call_id")
        if tc_id:
            parts.append(f' <span class="tool-id" title="tool_use_id">{_esc(tc_id)}</span>')
        parts.append('</div>')
        parts.append(f'<pre>{_esc(_pretty_maybe_json(turn.get("content", "")))}</pre>')
        parts.append('</div>')

    else:
        parts.append(f'<div class="turn turn-other"><div class="turn-head">{_esc(role or "?")}</div>')
        parts.append(f'<pre>{_esc(_pretty(turn))}</pre></div>')

    return "".join(parts)


def _render_agent_panel(agent: dict[str, Any]) -> str:
    principal = _principal(agent["profile"])
    parts: list[str] = []
    parts.append('<details class="agent-panel" open>')
    parts.append('<summary>')
    parts.append(f'<span class="agent-id">{_esc(agent["id"])}</span>')
    if principal:
        parts.append(f'<span class="agent-principal">{_esc(principal)}</span>')
    parts.append(f'<span class="turn-count">{_esc(len(agent["turns"]))} turns</span>')
    parts.append('</summary>')
    parts.append('<div class="agent-body">')
    for turn in agent["turns"]:
        parts.append(_render_turn(turn))
    parts.append('</div>')
    parts.append('</details>')
    return "".join(parts)


# --- Top-level ---------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 0; padding: 24px; background: #fafafa; color: #111827; }
h1 { margin: 0 0 4px; font-size: 22px; }
h2 { margin: 0; font-size: 16px; font-weight: 600; }
pre { margin: 0; padding: 8px 10px; background: #f3f4f6; border-radius: 4px;
      font-family: ui-monospace, 'SF Mono', Menlo, monospace; font-size: 12px;
      white-space: pre-wrap; word-break: break-word; overflow-x: auto; }
pre.prose { background: transparent; padding: 0; font-family: inherit; font-size: 13px; }
.wrap { max-width: 1100px; margin: 0 auto; }
.header { background: white; border: 1px solid #e5e7eb; border-radius: 8px;
          padding: 16px 20px; margin-bottom: 16px; }
.header .sub { color: #6b7280; font-size: 13px; margin-top: 2px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.chip { background: #eef2ff; color: #3730a3; padding: 3px 8px; border-radius: 10px;
        font-size: 12px; }
.chip .principal { color: #6b7280; margin-left: 6px; }
.totals { display: flex; gap: 12px; margin-top: 10px; font-size: 12px; color: #374151; }
.total { background: #f3f4f6; padding: 3px 8px; border-radius: 4px; }
.total.err { background: #fee2e2; color: #991b1b; }
.section { background: white; border: 1px solid #e5e7eb; border-radius: 8px;
           padding: 14px 16px; margin-bottom: 16px; }
.section-header { display: flex; justify-content: space-between; align-items: center;
                  margin-bottom: 10px; }
.toggle { font-size: 12px; color: #6b7280; cursor: pointer; user-select: none; }
.timeline { display: flex; flex-direction: column; gap: 4px; }
.card { border: 1px solid #e5e7eb; border-left: 3px solid #9ca3af;
        border-radius: 4px; padding: 6px 10px; cursor: pointer; font-size: 13px; }
.card:hover { background: #f9fafb; }
.card.kind-dm { border-left-color: #3b82f6; }
.card.kind-channel { border-left-color: #10b981; }
.card.kind-command { border-left-color: #f59e0b; }
.card.kind-fetch { border-left-color: #d1d5db; color: #9ca3af; }
.card.is-error { border-left-color: #ef4444 !important; background: #fef2f2; }
.card.fetch-row { display: none; }
body.show-fetch .card.fetch-row { display: block; }
.card-head { display: flex; align-items: center; gap: 8px; flex-wrap: nowrap; }
.idx { color: #9ca3af; font-family: ui-monospace, monospace; font-size: 11px;
       min-width: 32px; }
.badge { padding: 1px 7px; border-radius: 3px; font-size: 11px; font-weight: 600;
         color: white; background: #9ca3af; white-space: nowrap; }
.badge-dm { background: #3b82f6; }
.badge-channel { background: #10b981; }
.badge-command { background: #f59e0b; }
.badge-fetch { background: #d1d5db; color: #6b7280; }
.agent { color: #374151; font-weight: 500; font-size: 12px; }
.summary { flex: 1; color: #4b5563; overflow: hidden; text-overflow: ellipsis;
           white-space: nowrap; }
.ts { color: #9ca3af; font-size: 11px; font-family: ui-monospace, monospace; }
.card-body { display: none; margin-top: 8px; padding-top: 8px;
             border-top: 1px dashed #e5e7eb; }
.card.open .card-body { display: block; }
.kv { margin-bottom: 6px; }
.kv-k { font-size: 11px; color: #6b7280; text-transform: uppercase;
        letter-spacing: 0.05em; display: block; margin-bottom: 2px; }
.agent-panel { background: white; border: 1px solid #e5e7eb; border-radius: 8px;
               margin-bottom: 10px; }
.agent-panel summary { padding: 10px 16px; cursor: pointer; display: flex;
                       align-items: center; gap: 10px; font-size: 14px; }
.agent-panel summary::-webkit-details-marker { color: #9ca3af; }
.agent-id { font-weight: 600; color: #111827; }
.agent-principal { color: #6b7280; font-size: 12px; }
.turn-count { margin-left: auto; color: #9ca3af; font-size: 12px; }
.agent-body { padding: 8px 16px 16px; display: flex; flex-direction: column; gap: 8px; }
.turn { border-radius: 4px; padding: 8px 10px; font-size: 13px; }
.turn-user { background: #f3f4f6; }
.turn-assistant { background: #eff6ff; border-left: 3px solid #3b82f6; }
.turn-tool { background: #fef3c7; border-left: 3px solid #f59e0b;
             margin-left: 20px; }
.turn-other { background: #fafafa; }
.turn-head { font-size: 11px; color: #6b7280; text-transform: uppercase;
             letter-spacing: 0.05em; margin-bottom: 4px; }
.turn-body { margin-bottom: 4px; }
.tool-call { margin-top: 6px; padding: 6px 8px; background: white;
             border: 1px solid #dbeafe; border-radius: 3px; }
.tool-call-head { display: flex; gap: 8px; align-items: center; margin-bottom: 4px; }
.tool-name { font-family: ui-monospace, monospace; font-size: 12px; font-weight: 600;
             color: #1e40af; }
.tool-id { font-family: ui-monospace, monospace; font-size: 10px; color: #9ca3af; }
"""

_JS = """
function toggleCard(el) { el.classList.toggle('open'); }
document.getElementById('show-fetch').addEventListener('change', function(e) {
  document.body.classList.toggle('show-fetch', e.target.checked);
});
"""


def _render_header(
    scenario_name: str,
    timestamp: str,
    agents: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> str:
    dms = channels = commands = errors = 0
    for a in actions:
        name = a.get("request_name", "")
        if a.get("result_is_error"):
            errors += 1
        if name == "SendMessage":
            dms += 1
        elif name == "ChannelMessage":
            channels += 1
        elif name == "ExecuteCommand":
            commands += 1

    parts: list[str] = []
    parts.append('<header class="header">')
    parts.append(f'<h1>{_esc(scenario_name)}</h1>')
    parts.append(f'<div class="sub">run {_esc(timestamp)} · {_esc(len(agents))} agents · {_esc(len(actions))} actions</div>')
    parts.append('<div class="chips">')
    for a in agents:
        principal = _principal(a["profile"])
        parts.append('<span class="chip">')
        parts.append(_esc(a["id"]))
        if principal:
            parts.append(f'<span class="principal">{_esc(principal)}</span>')
        parts.append('</span>')
    parts.append('</div>')
    parts.append('<div class="totals">')
    parts.append(f'<span class="total">{dms} DMs</span>')
    parts.append(f'<span class="total">{channels} channel msgs</span>')
    parts.append(f'<span class="total">{commands} commands</span>')
    if errors:
        parts.append(f'<span class="total err">{errors} errors</span>')
    parts.append('</div>')
    parts.append('</header>')
    return "".join(parts)


def generate_report(run_dir: Path | str) -> Path:
    """Read a run directory's JSONL artifacts and write `report.html` into it."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    scenario_name, timestamp = _parse_scenario_meta(run_dir)
    agents = _load_agents(run_dir)

    actions_path = run_dir / "actions.jsonl"
    actions = _read_jsonl(actions_path) if actions_path.exists() else []

    body_parts: list[str] = []
    body_parts.append('<div class="wrap">')
    body_parts.append(_render_header(scenario_name, timestamp, agents, actions))
    body_parts.append(_render_timeline(actions))
    body_parts.append('<section class="section">')
    body_parts.append('<div class="section-header"><h2>Agent conversations</h2></div>')
    for agent in agents:
        body_parts.append(_render_agent_panel(agent))
    body_parts.append('</section>')
    body_parts.append('</div>')

    title = f"{scenario_name} · {timestamp}" if timestamp else scenario_name
    doc = (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        f"<title>{_esc(title)}</title>"
        f"<style>{_CSS}</style>"
        "</head><body>"
        + "".join(body_parts)
        + f"<script>{_JS}</script>"
        + "</body></html>"
    )

    out_path = run_dir / "report.html"
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.experiments.report <run_dir>")
        sys.exit(1)
    out = generate_report(sys.argv[1])
    print(f"Report written to: {out}")


if __name__ == "__main__":
    main()
