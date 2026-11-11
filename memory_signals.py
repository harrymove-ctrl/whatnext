"""Safe reader for Hermes persistent user memory (USER.md, MEMORY.md, SOUL.md),
session transcripts (via public ``hermes_state.SessionDB``), and workspace markers.

Never reads ``.env``, ``auth.json``, or credential stores (Catalog Rule 11).
Scrubs secret-shaped substrings before returning any memory or preference block.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_SECRET_SCRUB_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}"
    r"|AKIA[0-9A-Z]{16}|glpat-[A-Za-z0-9_-]{16,}|xox[baprs]-[A-Za-z0-9-]{16,}"
    r"|AIza[0-9A-Za-z_-]{25,}|Bearer\s+[A-Za-z0-9._-]{16,})"
)

_WORKSPACE_MANIFESTS = (
    ("pyproject.toml", "Python"),
    ("Cargo.toml", "Rust"),
    ("package.json", "TypeScript/Node"),
    ("go.mod", "Go"),
    ("plugin.yaml", "Hermes Plugin"),
    ("Move.toml", "Sui Move"),
    ("Dockerfile", "Docker"),
)


def scrub_text(text: str) -> str:
    if not text:
        return ""
    return _SECRET_SCRUB_RE.sub("[REDACTED]", text)


def resolve_hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        raw = os.environ.get("HERMES_HOME", "").strip()
        return Path(raw) if raw else (Path.home() / ".hermes")


def _read_bounded_markdown(path: Path, max_chars: int = 1400) -> str:
    try:
        if not path.is_file() or path.is_symlink():
            return ""
        if path.stat().st_size > 256 * 1024:
            return ""
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    cleaned = scrub_text(raw)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + " ..."


def extract_bullets(markdown_text: str, limit: int = 8) -> List[str]:
    bullets: List[str] = []
    for raw_line in (markdown_text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("<!--"):
            continue
        line = re.sub(r"^[-*•\d.)]+\s*", "", line).strip()
        if len(line) >= 6:
            bullets.append(line[:180])
        if len(bullets) >= limit:
            break
    return bullets


def read_user_memory_bundle(*, enabled: bool = True) -> Dict[str, Any]:
    """Read Hermes USER.md, MEMORY.md, and SOUL.md from the active profile home."""
    if not enabled:
        return {
            "enabled": False,
            "has_memory": False,
            "user_md": "",
            "memory_md": "",
            "soul_md": "",
            "highlights": [],
            "preferences_text": "",
        }

    home = resolve_hermes_home()
    memories_dir = home / "memories"
    user_md = _read_bounded_markdown(memories_dir / "USER.md", max_chars=1300)
    memory_md = _read_bounded_markdown(memories_dir / "MEMORY.md", max_chars=1500)
    soul_md = _read_bounded_markdown(home / "SOUL.md", max_chars=600)

    highlights = (
        extract_bullets(user_md, limit=5)
        + extract_bullets(memory_md, limit=5)
    )[:8]

    sections: List[str] = []
    if user_md:
        sections.append(f"User profile (USER.md):\n{user_md}")
    if memory_md:
        sections.append(f"Operating memory (MEMORY.md):\n{memory_md}")
    if soul_md:
        sections.append(f"Agent persona (SOUL.md):\n{soul_md}")

    preferences_text = "\n\n".join(sections)[:3600]
    return {
        "enabled": True,
        "has_memory": bool(user_md or memory_md or soul_md),
        "user_md": user_md,
        "memory_md": memory_md,
        "soul_md": soul_md,
        "highlights": highlights,
        "preferences_text": preferences_text,
    }


def inspect_workspace_context(cwd: Optional[str]) -> Dict[str, Any]:
    """Read safe, zero-subprocess metadata about the user's current working directory."""
    raw = (cwd or "").strip()
    if not raw:
        return {"project_name": "", "stacks": [], "has_tests": False, "has_git": False, "summary": ""}
    try:
        root = Path(raw).expanduser()
        if not root.is_dir():
            return {"project_name": root.name, "stacks": [], "has_tests": False, "has_git": False, "summary": root.name}
        stacks: List[str] = []
        for fname, label in _WORKSPACE_MANIFESTS:
            if (root / fname).exists():
                stacks.append(label)
        has_tests = any((root / d).is_dir() for d in ("tests", "test", "__tests__", "spec"))
        has_git = (root / ".git").exists()
        parts = [f"workspace={root.name}"]
        if stacks:
            parts.append("stack=" + "/".join(stacks[:3]))
        if has_tests:
            parts.append("has_tests=yes")
        if has_git:
            parts.append("git=yes")
        return {
            "project_name": root.name,
            "stacks": stacks,
            "has_tests": has_tests,
            "has_git": has_git,
            "summary": ", ".join(parts),
        }
    except OSError:
        return {"project_name": "", "stacks": [], "has_tests": False, "has_git": False, "summary": ""}


def load_session_turns(session_id: str, max_messages: int = 12) -> Dict[str, Any]:
    """Read recent user/assistant exchanges for ``session_id`` through ``hermes_state.SessionDB``."""
    if not session_id or session_id == "new":
        return {"messages": [], "cwd": ""}
    try:
        from hermes_state import SessionDB
        with SessionDB() as db:
            meta = db.get_session(session_id) or {}
            rows = db.get_messages(session_id) or []
    except Exception:
        return {"messages": [], "cwd": ""}

    normalized: List[Dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "")
        if role not in ("user", "assistant"):
            continue
        content = row.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        text = scrub_text(str(content).strip())
        if not text:
            continue
        if normalized and normalized[-1]["role"] == role == "assistant":
            normalized[-1]["content"] = text[:2400]
        else:
            normalized.append({"role": role, "content": text[:2400]})

    keep = max(2, min(30, int(max_messages)))
    return {
        "messages": normalized[-keep:],
        "cwd": str(meta.get("cwd") or ""),
    }
