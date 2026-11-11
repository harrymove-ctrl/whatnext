"""Profile-scoped SQLite persistence, revision fencing, evaluation ledger, and movement graph.

Uses ``plugins.plugin_storage.plugin_db("whatnext")`` when running inside Hermes so state
lives under ``$HERMES_HOME/plugin-data/whatnext/whatnext.db``, survives plugin updates,
and follows profile switches automatically. Falls back to ``$HERMES_HOME/plugin-data/whatnext``
when exercised from standalone tests or the local demo playground.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PLUGIN_NAME = "whatnext"
DB_FILENAME = "whatnext.db"
STALE_TTL_SECONDS = 7 * 24 * 3600
MAX_EVENTS_STORED = 2000
MAX_MOVEMENTS_STORED = 1000

_INIT_LOCK = threading.Lock()
_INITIALIZED_PATHS: set[str] = set()

INTENT_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("verify", re.compile(r"\b(test|pytest|vitest|cargo test|verify|check|lint|validate|benchmark|qa|smoke)\b", re.I)),
    ("ship", re.compile(r"\b(commit|push|pr|pull request|deploy|release|merge|ship|publish|changelog)\b", re.I)),
    ("debug", re.compile(r"\b(fix|bug|error|trace|traceback|crash|failing|broken|why is|investigate)\b", re.I)),
    ("refactor", re.compile(r"\b(refactor|clean up|simplify|extract|rename|tighten|dedupe|optimize)\b", re.I)),
    ("review", re.compile(r"\b(review|audit|diff|critique|inspect|security|edge case)\b", re.I)),
    ("implement", re.compile(r"\b(build|create|add|implement|write|wire|scaffold|integrate|update)\b", re.I)),
    ("explore", re.compile(r"\b(explain|how does|where is|architecture|overview|summarize|find|search)\b", re.I)),
    ("answer", re.compile(r"^(yes|no|yep|nope|sure|go ahead|do it|option [123abc]|use the|let's go with|sounds good)\b", re.I)),
)


def classify_intent(text: str) -> str:
    """Classify a user prompt into a stable workflow movement category."""
    cleaned = (text or "").strip()
    if not cleaned:
        return "explore"
    for label, pattern in INTENT_PATTERNS:
        if pattern.search(cleaned):
            return label
    if cleaned.endswith("?"):
        return "explore"
    return "implement"


def detect_language(text: str) -> str:
    """Lightweight script and stopword check so predictions match the user's language."""
    sample = (text or "").strip()
    if not sample:
        return "en"
    if re.search(r"[\u3040-\u30ff]", sample):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", sample):
        return "ko"
    if re.search(r"[\u4e00-\u9fff]", sample):
        return "zh"
    lowered = f" {sample.lower()} "
    if any(tok in lowered for tok in (" mình ", " không ", " giúp ", " của ", " được ", " nhé ")):
        return "vi"
    if any(tok in lowered for tok in (" por favor ", " hacer ", " cómo ", " para ", " gracias ")):
        return "es"
    if any(tok in lowered for tok in (" bitte ", " danke ", " nicht ", " mach ", " füge ")):
        return "de"
    if any(tok in lowered for tok in (" merci ", " avec ", " pour ", " faire ", " s'il ")):
        return "fr"
    return "en"


def normalized_edit_distance(a: str, b: str) -> float:
    """Normalized Levenshtein distance in [0.0, 1.0] bounded to 512 chars for speed."""
    s1 = (a or "").strip()[:512]
    s2 = (b or "").strip()[:512]
    if s1 == s2:
        return 0.0
    if not s1 or not s2:
        return 1.0
    rows = len(s1) + 1
    cols = len(s2) + 1
    prev = list(range(cols))
    for i in range(1, rows):
        curr = [i] + [0] * (cols - 1)
        c1 = s1[i - 1]
        for j in range(1, cols):
            cost = 0 if c1 == s2[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return round(prev[-1] / max(len(s1), len(s2)), 4)


def _fallback_data_dir() -> Path:
    override = os.environ.get("WHATNEXT_DATA_DIR", "").strip()
    if override:
        root = Path(override)
    else:
        hermes_home = os.environ.get("HERMES_HOME", "").strip()
        base = Path(hermes_home) if hermes_home else (Path.home() / ".hermes")
        root = base / "plugin-data" / PLUGIN_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_data_dir() -> Path:
    try:
        from plugins.plugin_storage import plugin_data_dir
        return plugin_data_dir(PLUGIN_NAME)
    except Exception:
        return _fallback_data_dir()


def _connect() -> sqlite3.Connection:
    conn: Optional[sqlite3.Connection] = None
    db_key = ""
    try:
        from plugins.plugin_storage import plugin_db
        conn = plugin_db(PLUGIN_NAME, DB_FILENAME)
        db_key = str(get_data_dir() / DB_FILENAME)
    except Exception:
        path = _fallback_data_dir() / DB_FILENAME
        db_key = str(path)
        conn = sqlite3.connect(str(path), timeout=10.0, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            pass
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn, db_key)
    return conn


def _ensure_schema(conn: sqlite3.Connection, db_key: str) -> None:
    if db_key in _INITIALIZED_PATHS:
        return
    with _INIT_LOCK:
        if db_key in _INITIALIZED_PATHS:
            return
        with conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS session_revisions (
                    session_id TEXT PRIMARY KEY,
                    context_revision INTEGER NOT NULL DEFAULT 1,
                    draft_revision INTEGER NOT NULL DEFAULT 0,
                    last_origin TEXT NOT NULL DEFAULT 'human',
                    pending_accepted_text TEXT NOT NULL DEFAULT '',
                    pending_request_id TEXT NOT NULL DEFAULT '',
                    pending_mode TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_predictions (
                    session_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    context_revision INTEGER NOT NULL,
                    draft_revision INTEGER NOT NULL,
                    draft_prefix TEXT NOT NULL DEFAULT '',
                    candidates_json TEXT NOT NULL DEFAULT '[]',
                    backend TEXT NOT NULL DEFAULT 'hermes_llm',
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    error_kind TEXT NOT NULL DEFAULT '',
                    memory_used INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (session_id, mode)
                );

                CREATE TABLE IF NOT EXISTS observation_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL DEFAULT '',
                    mode TEXT NOT NULL DEFAULT 'next_prompt',
                    candidate_index INTEGER NOT NULL DEFAULT 0,
                    edit_distance REAL,
                    latency_ms INTEGER,
                    origin TEXT NOT NULL DEFAULT 'human',
                    detail TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_obs_type_time
                    ON observation_events(event_type, created_at);

                CREATE TABLE IF NOT EXISTS human_movements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    prev_intent TEXT NOT NULL DEFAULT '',
                    char_len INTEGER NOT NULL,
                    word_count INTEGER NOT NULL,
                    language TEXT NOT NULL DEFAULT 'en',
                    opener TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_movements_time
                    ON human_movements(created_at DESC);

                CREATE TABLE IF NOT EXISTS movement_transitions (
                    prev_intent TEXT NOT NULL,
                    next_intent TEXT NOT NULL,
                    weight INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (prev_intent, next_intent)
                );

                CREATE TABLE IF NOT EXISTS dismissed_phrases (
                    phrase_hash TEXT PRIMARY KEY,
                    phrase_preview TEXT NOT NULL,
                    session_id TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );
                """
            )
        _INITIALIZED_PATHS.add(db_key)


def _phrase_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:24]


def get_revisions(session_id: str) -> Dict[str, Any]:
    if not session_id:
        return {
            "context_revision": 1,
            "draft_revision": 0,
            "last_origin": "human",
            "pending_accepted_text": "",
            "pending_request_id": "",
            "pending_mode": "",
        }
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM session_revisions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return {
                "context_revision": 1,
                "draft_revision": 0,
                "last_origin": "human",
                "pending_accepted_text": "",
                "pending_request_id": "",
                "pending_mode": "",
            }
        return dict(row)
    finally:
        conn.close()


def bump_context_revision(session_id: str, *, origin: str = "human") -> int:
    """Advance context_revision on a turn boundary or preference change and clear stale predictions."""
    if not session_id:
        return 1
    now = time.time()
    conn = _connect()
    try:
        with conn:
            row = conn.execute(
                "SELECT context_revision FROM session_revisions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            next_rev = (int(row["context_revision"]) + 1) if row else 1
            conn.execute(
                """
                INSERT INTO session_revisions (
                    session_id, context_revision, draft_revision, last_origin,
                    pending_accepted_text, pending_request_id, pending_mode, updated_at
                ) VALUES (?, ?, 0, ?, '', '', '', ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    context_revision = excluded.context_revision,
                    last_origin = excluded.last_origin,
                    updated_at = excluded.updated_at
                """,
                (session_id, next_rev, origin, now),
            )
            conn.execute("DELETE FROM session_predictions WHERE session_id = ?", (session_id,))
        return next_rev
    finally:
        conn.close()


def mark_accepted_candidate(
    session_id: str,
    *,
    request_id: str,
    mode: str,
    accepted_text: str,
    candidate_index: int = 0,
    origin: str = "suggestion",
) -> None:
    """Record that the user accepted a candidate into the composer so send-time middleware/hooks
    can measure exact edit_distance and avoid treating machine-generated sends as human preference."""
    now = time.time()
    conn = _connect()
    try:
        with conn:
            if session_id:
                row = conn.execute(
                    "SELECT context_revision, draft_revision FROM session_revisions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                ctx_rev = int(row["context_revision"]) if row else 1
                dr_rev = int(row["draft_revision"]) if row else 0
                conn.execute(
                    """
                    INSERT INTO session_revisions (
                        session_id, context_revision, draft_revision, last_origin,
                        pending_accepted_text, pending_request_id, pending_mode, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        last_origin = excluded.last_origin,
                        pending_accepted_text = excluded.pending_accepted_text,
                        pending_request_id = excluded.pending_request_id,
                        pending_mode = excluded.pending_mode,
                        updated_at = excluded.updated_at
                    """,
                    (session_id, ctx_rev, dr_rev, origin, accepted_text[:4000], request_id, mode, now),
                )
            _insert_event(
                conn,
                event_type="accepted",
                session_id=session_id or "new",
                request_id=request_id,
                mode=mode,
                candidate_index=candidate_index,
                origin=origin,
                created_at=now,
            )
    finally:
        conn.close()


def save_prediction(
    session_id: str,
    *,
    mode: str,
    request_id: str,
    context_revision: int,
    draft_revision: int = 0,
    draft_prefix: str = "",
    candidates: List[Dict[str, Any]],
    backend: str = "hermes_llm",
    latency_ms: int = 0,
    memory_used: bool = False,
    error_kind: str = "",
) -> Optional[Dict[str, Any]]:
    """Store a prediction bundle only if its (context_revision, draft_revision) is not stale."""
    if not session_id:
        return None
    now = time.time()
    conn = _connect()
    try:
        with conn:
            rev_row = conn.execute(
                "SELECT context_revision, draft_revision FROM session_revisions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            current_ctx_rev = int(rev_row["context_revision"]) if rev_row else 1
            current_draft_rev = int(rev_row["draft_revision"]) if rev_row else 0
            if context_revision < current_ctx_rev:
                return None
            if mode == "complete_draft" and draft_revision < current_draft_rev:
                return None

            existing = conn.execute(
                "SELECT context_revision, draft_revision FROM session_predictions WHERE session_id = ? AND mode = ?",
                (session_id, mode),
            ).fetchone()
            if existing is not None:
                if context_revision < int(existing["context_revision"]):
                    return None
                if (
                    context_revision == int(existing["context_revision"])
                    and draft_revision < int(existing["draft_revision"])
                ):
                    return None

            conn.execute(
                """
                INSERT INTO session_revisions (
                    session_id, context_revision, draft_revision, last_origin,
                    pending_accepted_text, pending_request_id, pending_mode, updated_at
                ) VALUES (?, ?, ?, 'human', '', '', '', ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    context_revision = MAX(session_revisions.context_revision, excluded.context_revision),
                    draft_revision = CASE
                        WHEN ? = 'complete_draft' THEN MAX(session_revisions.draft_revision, excluded.draft_revision)
                        ELSE session_revisions.draft_revision
                    END,
                    updated_at = excluded.updated_at
                """,
                (session_id, context_revision, draft_revision, now, mode),
            )

            conn.execute(
                """
                INSERT INTO session_predictions (
                    session_id, mode, request_id, context_revision, draft_revision,
                    draft_prefix, candidates_json, backend, latency_ms, error_kind,
                    memory_used, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, mode) DO UPDATE SET
                    request_id = excluded.request_id,
                    context_revision = excluded.context_revision,
                    draft_revision = excluded.draft_revision,
                    draft_prefix = excluded.draft_prefix,
                    candidates_json = excluded.candidates_json,
                    backend = excluded.backend,
                    latency_ms = excluded.latency_ms,
                    error_kind = excluded.error_kind,
                    memory_used = excluded.memory_used,
                    created_at = excluded.created_at
                """,
                (
                    session_id,
                    mode,
                    request_id,
                    context_revision,
                    draft_revision,
                    draft_prefix[:4000],
                    json.dumps(candidates, ensure_ascii=False),
                    backend,
                    int(latency_ms),
                    error_kind,
                    1 if memory_used else 0,
                    now,
                ),
            )
            conn.execute(
                "DELETE FROM session_predictions WHERE created_at < ?",
                (now - STALE_TTL_SECONDS,),
            )
        return get_prediction(session_id, mode=mode)
    finally:
        conn.close()


def get_prediction(session_id: str, *, mode: str = "next_prompt") -> Dict[str, Any]:
    if not session_id:
        return {"session_id": "", "mode": mode, "candidates": [], "error_kind": ""}
    now = time.time()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM session_predictions WHERE session_id = ? AND mode = ?",
            (session_id, mode),
        ).fetchone()
        rev = conn.execute(
            "SELECT context_revision, draft_revision FROM session_revisions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        ctx_rev = int(rev["context_revision"]) if rev else 1
        dr_rev = int(rev["draft_revision"]) if rev else 0
        if row is None or (now - float(row["created_at"])) > STALE_TTL_SECONDS:
            return {
                "session_id": session_id,
                "mode": mode,
                "context_revision": ctx_rev,
                "draft_revision": dr_rev,
                "candidates": [],
                "error_kind": "",
            }
        try:
            candidates = json.loads(row["candidates_json"] or "[]")
        except ValueError:
            candidates = []
        return {
            "session_id": session_id,
            "mode": row["mode"],
            "request_id": row["request_id"],
            "context_revision": int(row["context_revision"]),
            "draft_revision": int(row["draft_revision"]),
            "draft_prefix": row["draft_prefix"],
            "candidates": candidates if isinstance(candidates, list) else [],
            "backend": row["backend"],
            "latency_ms": int(row["latency_ms"]),
            "memory_used": bool(row["memory_used"]),
            "error_kind": row["error_kind"] or "",
            "created_at": float(row["created_at"]),
        }
    finally:
        conn.close()


def clear_session_predictions(session_id: str, *, mode: Optional[str] = None) -> None:
    if not session_id:
        return
    conn = _connect()
    try:
        with conn:
            if mode:
                conn.execute(
                    "DELETE FROM session_predictions WHERE session_id = ? AND mode = ?",
                    (session_id, mode),
                )
            else:
                conn.execute("DELETE FROM session_predictions WHERE session_id = ?", (session_id,))
    finally:
        conn.close()


def record_dismissal(
    session_id: str,
    *,
    request_id: str = "",
    mode: str = "next_prompt",
    candidate_text: str = "",
    candidate_index: int = 0,
    feedback_enabled: bool = True,
) -> None:
    now = time.time()
    conn = _connect()
    try:
        with conn:
            if session_id:
                conn.execute(
                    "DELETE FROM session_predictions WHERE session_id = ? AND mode = ?",
                    (session_id, mode),
                )
            _insert_event(
                conn,
                event_type="dismissed",
                session_id=session_id or "new",
                request_id=request_id,
                mode=mode,
                candidate_index=candidate_index,
                origin="human",
                created_at=now,
            )
            if feedback_enabled and candidate_text.strip():
                conn.execute(
                    """
                    INSERT INTO dismissed_phrases (phrase_hash, phrase_preview, session_id, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(phrase_hash) DO UPDATE SET
                        created_at = excluded.created_at,
                        session_id = excluded.session_id
                    """,
                    (_phrase_hash(candidate_text), candidate_text.strip()[:140], session_id or "", now),
                )
    finally:
        conn.close()


def is_dismissed_phrase(text: str) -> bool:
    if not (text or "").strip():
        return False
    conn = _connect()
    try:
        cutoff = time.time() - STALE_TTL_SECONDS
        row = conn.execute(
            "SELECT 1 FROM dismissed_phrases WHERE phrase_hash = ? AND created_at >= ?",
            (_phrase_hash(text), cutoff),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def record_observation_event(
    event_type: str,
    *,
    session_id: str = "",
    request_id: str = "",
    mode: str = "next_prompt",
    candidate_index: int = 0,
    edit_distance: Optional[float] = None,
    latency_ms: Optional[int] = None,
    origin: str = "human",
    detail: str = "",
) -> None:
    if event_type not in {"presented", "accepted", "edited", "dismissed", "submitted", "clone_mode_start", "clone_mode_stop"}:
        return
    conn = _connect()
    try:
        with conn:
            _insert_event(
                conn,
                event_type=event_type,
                session_id=session_id or "new",
                request_id=request_id,
                mode=mode,
                candidate_index=candidate_index,
                edit_distance=edit_distance,
                latency_ms=latency_ms,
                origin=origin,
                detail=detail[:240],
                created_at=time.time(),
            )
    finally:
        conn.close()


def _insert_event(
    conn: sqlite3.Connection,
    *,
    event_type: str,
    session_id: str,
    request_id: str = "",
    mode: str = "next_prompt",
    candidate_index: int = 0,
    edit_distance: Optional[float] = None,
    latency_ms: Optional[int] = None,
    origin: str = "human",
    detail: str = "",
    created_at: Optional[float] = None,
) -> None:
    ts = created_at if created_at is not None else time.time()
    conn.execute(
        """
        INSERT INTO observation_events (
            event_type, session_id, request_id, mode, candidate_index,
            edit_distance, latency_ms, origin, detail, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_type,
            session_id,
            request_id,
            mode,
            int(candidate_index),
            edit_distance,
            latency_ms,
            origin,
            detail,
            ts,
        ),
    )
    conn.execute(
        """
        DELETE FROM observation_events
        WHERE id NOT IN (
            SELECT id FROM observation_events ORDER BY id DESC LIMIT ?
        )
        """,
        (MAX_EVENTS_STORED,),
    )


def observe_submitted_turn(
    session_id: str,
    user_text: str,
    *,
    explicit_origin: str = "",
    feedback_enabled: bool = True,
) -> Dict[str, Any]:
    """Process an outgoing user turn:
    1. Reconciles against any accepted candidate on this session to log ``submitted`` + ``edited``.
    2. Only learns movement transitions & style preferences when the turn is independently
       human-authored (``origin == 'human'``), never when it came from an accepted suggestion
       or Clone Mode delegation (Clone Personalization invariant).
    """
    cleaned = (user_text or "").strip()
    now = time.time()
    conn = _connect()
    try:
        with conn:
            rev = conn.execute(
                "SELECT * FROM session_revisions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            pending_text = (rev["pending_accepted_text"] if rev else "") or ""
            pending_req = (rev["pending_request_id"] if rev else "") or ""
            pending_mode = (rev["pending_mode"] if rev else "") or "next_prompt"
            stored_origin = (rev["last_origin"] if rev else "") or "human"
            next_ctx_rev = (int(rev["context_revision"]) + 1) if rev else 2

            origin = explicit_origin or ("suggestion" if pending_text else stored_origin or "human")
            edit_dist: Optional[float] = None

            if pending_text and cleaned:
                edit_dist = normalized_edit_distance(pending_text, cleaned)
                if edit_dist < 0.75:
                    if edit_dist > 0.02:
                        _insert_event(
                            conn,
                            event_type="edited",
                            session_id=session_id or "new",
                            request_id=pending_req,
                            mode=pending_mode,
                            edit_distance=edit_dist,
                            origin=origin,
                            created_at=now,
                        )
                    _insert_event(
                        conn,
                        event_type="submitted",
                        session_id=session_id or "new",
                        request_id=pending_req,
                        mode=pending_mode,
                        edit_distance=edit_dist,
                        origin=origin,
                        created_at=now,
                    )
                else:
                    # User rewrote almost everything from scratch: treat as independent human input.
                    origin = "human"
            elif explicit_origin == "clone_mode":
                origin = "clone_mode"
                _insert_event(
                    conn,
                    event_type="submitted",
                    session_id=session_id or "new",
                    request_id=pending_req,
                    mode="clone_mode",
                    edit_distance=0.0,
                    origin="clone_mode",
                    created_at=now,
                )

            if session_id:
                conn.execute(
                    """
                    INSERT INTO session_revisions (
                        session_id, context_revision, draft_revision, last_origin,
                        pending_accepted_text, pending_request_id, pending_mode, updated_at
                    ) VALUES (?, ?, 0, ?, '', '', '', ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        context_revision = excluded.context_revision,
                        draft_revision = 0,
                        last_origin = excluded.last_origin,
                        pending_accepted_text = '',
                        pending_request_id = '',
                        pending_mode = '',
                        updated_at = excluded.updated_at
                    """,
                    (session_id, next_ctx_rev, origin, now),
                )
                conn.execute("DELETE FROM session_predictions WHERE session_id = ?", (session_id,))

            learned_intent = classify_intent(cleaned)
            if feedback_enabled and cleaned and origin == "human":
                prev_row = conn.execute(
                    "SELECT intent FROM human_movements WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                    (session_id or "",),
                ).fetchone()
                prev_intent = prev_row["intent"] if prev_row else "session_start"
                words = cleaned.split()
                opener = " ".join(words[:3]).lower()[:48]
                conn.execute(
                    """
                    INSERT INTO human_movements (
                        session_id, intent, prev_intent, char_len, word_count,
                        language, opener, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id or "",
                        learned_intent,
                        prev_intent,
                        len(cleaned),
                        len(words),
                        detect_language(cleaned),
                        opener,
                        now,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO movement_transitions (prev_intent, next_intent, weight, updated_at)
                    VALUES (?, ?, 1, ?)
                    ON CONFLICT(prev_intent, next_intent) DO UPDATE SET
                        weight = movement_transitions.weight + 1,
                        updated_at = excluded.updated_at
                    """,
                    (prev_intent, learned_intent, now),
                )
                conn.execute(
                    """
                    DELETE FROM human_movements
                    WHERE id NOT IN (
                        SELECT id FROM human_movements ORDER BY id DESC LIMIT ?
                    )
                    """,
                    (MAX_MOVEMENTS_STORED,),
                )
            return {
                "origin": origin,
                "intent": learned_intent,
                "edit_distance": edit_dist,
                "context_revision": next_ctx_rev,
            }
    finally:
        conn.close()


def get_movement_profile(current_intent: str = "") -> Dict[str, Any]:
    """Summarize learned human movement habits, style cadence, and likely next workflow moves."""
    conn = _connect()
    try:
        stats = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                AVG(char_len) AS avg_chars,
                AVG(word_count) AS avg_words
            FROM human_movements
            """
        ).fetchone()
        total = int(stats["total"] or 0) if stats else 0
        avg_chars = int(round(float(stats["avg_chars"] or 72))) if stats else 72
        avg_words = int(round(float(stats["avg_words"] or 12))) if stats else 12

        lang_row = conn.execute(
            """
            SELECT language, COUNT(*) AS c
            FROM human_movements
            GROUP BY language
            ORDER BY c DESC
            LIMIT 1
            """
        ).fetchone()
        preferred_language = lang_row["language"] if lang_row else "en"

        top_intents_rows = conn.execute(
            """
            SELECT intent, COUNT(*) AS c
            FROM human_movements
            GROUP BY intent
            ORDER BY c DESC
            LIMIT 5
            """
        ).fetchall()
        top_intents = [{"intent": r["intent"], "count": int(r["c"])} for r in top_intents_rows]

        transitions_rows = conn.execute(
            """
            SELECT prev_intent, next_intent, weight
            FROM movement_transitions
            ORDER BY weight DESC, updated_at DESC
            LIMIT 12
            """
        ).fetchall()
        transitions = [
            {
                "from": r["prev_intent"],
                "to": r["next_intent"],
                "weight": int(r["weight"]),
            }
            for r in transitions_rows
        ]

        predicted_next_intents: List[Dict[str, Any]] = []
        if current_intent:
            next_rows = conn.execute(
                """
                SELECT next_intent, weight
                FROM movement_transitions
                WHERE prev_intent = ?
                ORDER BY weight DESC
                LIMIT 3
                """,
                (current_intent,),
            ).fetchall()
            total_w = sum(int(r["weight"]) for r in next_rows) or 1
            predicted_next_intents = [
                {
                    "intent": r["next_intent"],
                    "probability": round(int(r["weight"]) / total_w, 3),
                    "weight": int(r["weight"]),
                }
                for r in next_rows
            ]

        dismissed_rows = conn.execute(
            "SELECT phrase_preview FROM dismissed_phrases ORDER BY created_at DESC LIMIT 6"
        ).fetchall()

        return {
            "human_turns_learned": total,
            "avg_chars": avg_chars,
            "avg_words": avg_words,
            "preferred_language": preferred_language,
            "top_intents": top_intents,
            "transitions": transitions,
            "predicted_next_intents": predicted_next_intents,
            "recent_dismissed": [r["phrase_preview"] for r in dismissed_rows],
        }
    finally:
        conn.close()


def get_evaluation_metrics() -> Dict[str, Any]:
    """Return Clone-style composer evaluation funnel metrics:
    presentation, acceptance, editing, dismissal, and submission tracked separately."""
    conn = _connect()
    try:
        counts_rows = conn.execute(
            """
            SELECT event_type, COUNT(*) AS c
            FROM observation_events
            GROUP BY event_type
            """
        ).fetchall()
        counts = {r["event_type"]: int(r["c"]) for r in counts_rows}
        presented = counts.get("presented", 0)
        accepted = counts.get("accepted", 0)
        edited = counts.get("edited", 0)
        dismissed = counts.get("dismissed", 0)
        submitted = counts.get("submitted", 0)

        lat_rows = conn.execute(
            """
            SELECT latency_ms FROM session_predictions
            WHERE latency_ms > 0
            ORDER BY created_at DESC
            LIMIT 100
            """
        ).fetchall()
        latencies = sorted(int(r["latency_ms"]) for r in lat_rows)
        if not latencies:
            ev_lat_rows = conn.execute(
                """
                SELECT latency_ms FROM observation_events
                WHERE latency_ms IS NOT NULL AND latency_ms > 0
                ORDER BY id DESC
                LIMIT 100
                """
            ).fetchall()
            latencies = sorted(int(r["latency_ms"]) for r in ev_lat_rows)

        median_latency_ms = latencies[len(latencies) // 2] if latencies else 0

        edit_rows = conn.execute(
            """
            SELECT AVG(edit_distance) AS avg_ed
            FROM observation_events
            WHERE event_type = 'submitted' AND edit_distance IS NOT NULL
            """
        ).fetchone()
        avg_edit_distance = round(float(edit_rows["avg_ed"]), 3) if (edit_rows and edit_rows["avg_ed"] is not None) else 0.0

        by_mode_rows = conn.execute(
            """
            SELECT mode, event_type, COUNT(*) AS c
            FROM observation_events
            WHERE mode IN ('next_prompt', 'complete_draft', 'clone_mode')
            GROUP BY mode, event_type
            """
        ).fetchall()
        by_mode: Dict[str, Dict[str, int]] = {}
        for r in by_mode_rows:
            by_mode.setdefault(r["mode"], {})[r["event_type"]] = int(r["c"])

        recent_events = conn.execute(
            """
            SELECT event_type, session_id, mode, candidate_index, edit_distance, origin, created_at
            FROM observation_events
            ORDER BY id DESC
            LIMIT 15
            """
        ).fetchall()

        return {
            "presented": presented,
            "accepted": accepted,
            "edited": edited,
            "dismissed": dismissed,
            "submitted": submitted,
            "acceptance_rate": round(accepted / presented, 3) if presented else 0.0,
            "submission_rate": round(submitted / presented, 3) if presented else 0.0,
            "verbatim_keep_rate": round(max(0, submitted - edited) / submitted, 3) if submitted else 0.0,
            "avg_edit_distance": avg_edit_distance,
            "median_latency_ms": median_latency_ms,
            "by_mode": by_mode,
            "recent_events": [dict(r) for r in recent_events],
        }
    finally:
        conn.close()


def clear_feedback_memory() -> Dict[str, Any]:
    """Reset scoped feedback, movement history, and dismissed phrases (Clone /v1/prediction-feedback/clear equivalent)."""
    conn = _connect()
    try:
        with conn:
            ev_deleted = conn.execute("DELETE FROM observation_events").rowcount
            mv_deleted = conn.execute("DELETE FROM human_movements").rowcount
            tr_deleted = conn.execute("DELETE FROM movement_transitions").rowcount
            ds_deleted = conn.execute("DELETE FROM dismissed_phrases").rowcount
            conn.execute("DELETE FROM session_predictions")
        return {
            "ok": True,
            "cleared": {
                "events": ev_deleted,
                "movements": mv_deleted,
                "transitions": tr_deleted,
                "dismissed": ds_deleted,
            },
        }
    finally:
        conn.close()
