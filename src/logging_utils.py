# src/logging_utils.py
from datetime import datetime
import uuid
from typing import Any, Dict, Optional

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def attach_log(state: Dict[str, Any], node: str, message: str, level: str = "info", meta: Optional[Dict[str, Any]] = None) -> None:
    """Append a structured log entry into state['logs'] (creates list if missing)."""
    entry = {
        "id": str(uuid.uuid4()),
        "ts": now_iso(),
        "node": node,
        "level": level,
        "message": message,
        "meta": meta or {}
    }
    logs = state.get("logs")
    if logs is None:
        logs = []
    logs.append(entry)
    state["logs"] = logs

def format_trace(state: Dict[str, Any]) -> str:
    """Return a human-readable trace suitable for saving to file or embedding in report."""
    logs = state.get("logs", [])
    lines = []
    run_id = state.get("run_id", "run-unknown")
    lines.append(f"Run ID: {run_id}")
    lines.append(f"Started at: {state.get('started_at', 'unknown')}")
    lines.append("-" * 80)
    for e in logs:
        lines.append(f"[{e['ts']}] [{e['node']}] [{e['level'].upper()}] {e['message']} {e.get('meta') or ''}")
    return "\n".join(lines)
