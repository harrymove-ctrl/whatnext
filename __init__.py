"""WhatNext: Memory-grounded prompt prediction, live draft autocomplete,
human movement learning, and bounded Clone Mode for Hermes Agent.
"""

from __future__ import annotations

import contextvars
import hashlib
import importlib.util
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PLUGIN_ID = "whatnext"
AUX_TASK_KEY = "whatnext"
UI_SURFACES = frozenset({"", "desktop", "tui", "cli", "dashboard"})


def _load_local(stem: str):
    path = Path(__file__).resolve().with_name(f"{stem}.py")
    mod_name = "whatnext_mod_" + stem + "_" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:10]
    mod = sys.modules.get(mod_name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(mod)
        sys.modules[mod_name] = mod
    return mod


store = _load_local("store")
memory_signals = _load_local("memory_signals")
clone_client = _load_local("clone_client")
predictor = _load_local("predictor")

_ctx: Any = None
_llm_task: Optional[str] = None
_lock = threading.Lock()
_session_seq: Dict[str, int] = {}
_last_bg_time: Dict[str, float] = {}

DEFAULT_SETTINGS: Dict[str, Any] = {
    "enabled": True,
    "backend": "auto",
    "enable_draft_completion": True,
    "enable_empty_chat_starters": True,
    "enable_memory_personalization": True,
    "feedback_enabled": True,
    "max_context_messages": 12,
    "candidate_count": 3,
    "cooldown_seconds": 2,
    "clone_mode_max_turns": 3,
    "clone_mode_max_seconds": 300,
    "clone_api_base_url": "https://api.clone.is",
}


def get_effective_settings(ctx_obj: Any = None) -> Dict[str, Any]:
    active = ctx_obj or _ctx
    resolved = dict(DEFAULT_SETTINGS)
    if active is None:
        return resolved
    for key, default_val in DEFAULT_SETTINGS.items():
        try:
            val = active.get_config(key, default_val)
            if val is not None:
                resolved[key] = val
        except Exception:
            continue
    return resolved


def _extract_last_user_text(kwargs: Dict[str, Any]) -> str:
    direct = kwargs.get("user_message") or kwargs.get("prompt") or kwargs.get("message")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    history = kwargs.get("conversation_history") or kwargs.get("messages") or []
    if isinstance(history, list):
        for msg in reversed(history):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(
                        str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                    )
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return ""


def _spawn_in_profile_context(target, *args) -> None:
    try:
        from agent.memory_provider import spawn_context_thread
        spawn_context_thread(target, name="whatnext-worker", args=args).start()
    except ImportError:
        copied = contextvars.copy_context()
        threading.Thread(target=copied.run, args=(target, *args), daemon=True).start()


def _push_session_event(session_id: str, payload: Dict[str, Any]) -> None:
    if not session_id or _ctx is None:
        return
    emit_fn = getattr(_ctx, "emit_to_session", None)
    if callable(emit_fn):
        try:
            emit_fn(session_id, "prediction", payload)
        except Exception:
            logger.debug("whatnext: emit_to_session unavailable for %s", session_id)


def _background_predict(
    session_id: str,
    expected_seq: int,
    messages: List[Dict[str, Any]],
    cwd: str,
    settings: Dict[str, Any],
) -> None:
    with _lock:
        if _session_seq.get(session_id, 0) != expected_seq:
            return
    rev_info = store.get_revisions(session_id)
    llm_handle = getattr(_ctx, "llm", None) if _ctx is not None else None
    profile_name = getattr(_ctx, "profile_name", "default") if _ctx is not None else "default"
    result = predictor.generate_prediction_bundle(
        session_id=session_id,
        mode="next_prompt",
        draft="",
        draft_revision=0,
        context_revision=int(rev_info["context_revision"]),
        messages=messages,
        cwd=cwd,
        settings=settings,
        ctx_llm=llm_handle,
        llm_task=_llm_task,
        profile_name=profile_name,
    )
    with _lock:
        if _session_seq.get(session_id, 0) != expected_seq:
            store.clear_session_predictions(session_id)
            return
        _last_bg_time[session_id] = time.time()
    if result and not result.get("stale"):
        _push_session_event(session_id, result)


# ── Lifecycle Hooks ──────────────────────────────────────────────────────


def _on_session_start(**kwargs) -> None:
    settings = get_effective_settings()
    if not settings.get("enabled", True) or not settings.get("enable_empty_chat_starters", True):
        return
    session_id = str(kwargs.get("session_id") or "").strip()
    if not session_id:
        return
    platform = str(kwargs.get("platform") or "")
    if platform not in UI_SURFACES:
        return
    cwd = ""
    if _ctx is not None and callable(getattr(_ctx, "session_cwd", None)):
        try:
            cwd = str(_ctx.session_cwd() or "")
        except Exception:
            cwd = ""
    with _lock:
        seq = _session_seq.get(session_id, 0) + 1
        _session_seq[session_id] = seq
    _spawn_in_profile_context(_background_predict, session_id, seq, [], cwd, settings)


def _on_pre_llm_call(**kwargs) -> None:
    session_id = str(kwargs.get("session_id") or "").strip()
    settings = get_effective_settings()
    user_text = _extract_last_user_text(kwargs)
    with _lock:
        if session_id:
            _session_seq[session_id] = _session_seq.get(session_id, 0) + 1
    if session_id or user_text:
        store.observe_submitted_turn(
            session_id or "cli",
            user_text,
            feedback_enabled=bool(settings.get("feedback_enabled", True)),
        )


def _on_post_llm_call(**kwargs) -> None:
    settings = get_effective_settings()
    if not settings.get("enabled", True):
        return
    session_id = str(kwargs.get("session_id") or "").strip()
    messages = list(kwargs.get("conversation_history") or [])
    if not session_id or len(messages) < 2:
        return
    platform = str(kwargs.get("platform") or "")
    if platform not in UI_SURFACES:
        return
    cooldown = max(0, int(settings.get("cooldown_seconds", 2)))
    if time.time() - _last_bg_time.get(session_id, 0.0) < cooldown:
        return
    cwd = str(kwargs.get("cwd") or "")
    if not cwd and _ctx is not None and callable(getattr(_ctx, "session_cwd", None)):
        try:
            cwd = str(_ctx.session_cwd() or "")
        except Exception:
            cwd = ""
    with _lock:
        seq = _session_seq.get(session_id, 0)
    _spawn_in_profile_context(_background_predict, session_id, seq, messages, cwd, settings)


# ── Gateway JSON-RPC Handlers (plugin.whatnext.*) ────────────────────────


def _rpc_predict(params: Dict[str, Any], *, session_id: Optional[str] = None, **_kw) -> Dict[str, Any]:
    settings = get_effective_settings()
    sid = str(session_id or params.get("session_id") or "new").strip() or "new"
    draft = str(params.get("draft") or "")
    mode = str(params.get("mode") or ("complete_draft" if draft.strip() else "next_prompt"))
    draft_rev = int(params.get("draft_revision") or 0)
    ctx_rev_raw = params.get("context_revision")
    ctx_rev = int(ctx_rev_raw) if ctx_rev_raw is not None else None
    cwd = str(params.get("cwd") or "")
    messages = params.get("messages") if isinstance(params.get("messages"), list) else None

    llm_handle = getattr(_ctx, "llm", None) if _ctx is not None else None
    profile_name = getattr(_ctx, "profile_name", "default") if _ctx is not None else "default"
    return predictor.generate_prediction_bundle(
        session_id=sid,
        mode=mode,
        draft=draft,
        draft_revision=draft_rev,
        context_revision=ctx_rev,
        messages=messages,
        cwd=cwd,
        settings=settings,
        ctx_llm=llm_handle,
        llm_task=_llm_task,
        profile_name=profile_name,
    )


def _rpc_state(params: Dict[str, Any], *, session_id: Optional[str] = None, **_kw) -> Dict[str, Any]:
    sid = str(session_id or params.get("session_id") or "new").strip() or "new"
    settings = get_effective_settings()
    next_p = store.get_prediction(sid, mode="next_prompt")
    comp_p = store.get_prediction(sid, mode="complete_draft")
    revs = store.get_revisions(sid)
    return {
        "session_id": sid,
        "context_revision": int(revs["context_revision"]),
        "draft_revision": int(revs["draft_revision"]),
        "next_prompt": next_p,
        "complete_draft": comp_p,
        "settings": {
            "enabled": bool(settings.get("enabled", True)),
            "backend": str(settings.get("backend", "auto")),
            "enable_draft_completion": bool(settings.get("enable_draft_completion", True)),
            "enable_empty_chat_starters": bool(settings.get("enable_empty_chat_starters", True)),
            "clone_mode_max_turns": int(settings.get("clone_mode_max_turns", 3)),
            "clone_mode_max_seconds": int(settings.get("clone_mode_max_seconds", 300)),
            "clone_configured": clone_client.is_clone_configured(),
        },
    }


def _rpc_event(params: Dict[str, Any], *, session_id: Optional[str] = None, **_kw) -> Dict[str, Any]:
    settings = get_effective_settings()
    sid = str(session_id or params.get("session_id") or "new").strip() or "new"
    ev_type = str(params.get("event_type") or "")
    req_id = str(params.get("request_id") or "")
    mode = str(params.get("mode") or "next_prompt")
    cand_idx = int(params.get("candidate_index") or 0)
    cand_text = str(params.get("text") or "")
    origin = str(params.get("origin") or "suggestion")

    if ev_type == "accepted":
        store.mark_accepted_candidate(
            sid,
            request_id=req_id,
            mode=mode,
            accepted_text=cand_text,
            candidate_index=cand_idx,
            origin=origin,
        )
    elif ev_type == "dismissed":
        store.record_dismissal(
            sid,
            request_id=req_id,
            mode=mode,
            candidate_text=cand_text,
            candidate_index=cand_idx,
            feedback_enabled=bool(settings.get("feedback_enabled", True)),
        )
    elif ev_type == "submitted_direct":
        store.observe_submitted_turn(
            sid,
            cand_text,
            explicit_origin=origin,
            feedback_enabled=bool(settings.get("feedback_enabled", True)),
        )
    else:
        store.record_observation_event(
            ev_type,
            session_id=sid,
            request_id=req_id,
            mode=mode,
            candidate_index=cand_idx,
            origin=origin,
            detail=str(params.get("detail") or ""),
        )

    if clone_client.is_clone_configured() and ev_type in {"presented", "accepted", "dismissed", "submitted"}:
        clone_client.send_clone_event(
            {
                "event_type": ev_type,
                "request_id": req_id,
                "session_id": sid,
                "mode": mode,
                "candidate_index": cand_idx,
            },
            base_url=str(settings.get("clone_api_base_url") or "https://api.clone.is"),
        )
    return {"ok": True}


def _rpc_dashboard(params: Dict[str, Any], **_kw) -> Dict[str, Any]:
    settings = get_effective_settings()
    mem = memory_signals.read_user_memory_bundle(enabled=bool(settings.get("enable_memory_personalization", True)))
    return {
        "metrics": store.get_evaluation_metrics(),
        "movements": store.get_movement_profile(current_intent=str(params.get("current_intent") or "")),
        "memory": {
            "enabled": mem["enabled"],
            "has_memory": mem["has_memory"],
            "highlights": mem["highlights"],
        },
        "engine": {
            "configured_backend": settings.get("backend", "auto"),
            "clone_api_configured": clone_client.is_clone_configured(),
            "auxiliary_task": _llm_task or "main",
        },
    }


def _rpc_clear_feedback(_params: Dict[str, Any], **_kw) -> Dict[str, Any]:
    settings = get_effective_settings()
    local_res = store.clear_feedback_memory()
    remote_cleared = False
    if clone_client.is_clone_configured():
        remote_cleared = clone_client.clear_clone_feedback(
            base_url=str(settings.get("clone_api_base_url") or "https://api.clone.is")
        )
    return {**local_res, "clone_api_cleared": remote_cleared}


# ── Agent Tools ──────────────────────────────────────────────────────────

PREDICT_TOOL_SCHEMA: Dict[str, Any] = {
    "name": "whatnext_predict",
    "description": (
        "Predict the user's next likely prompts (next_prompt mode) or complete a half-written "
        "user draft (complete_draft mode) using USER.md, MEMORY.md, workspace stack, and "
        "learned human workflow movements."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "draft": {
                "type": "string",
                "description": "Optional partial user draft to complete. Leave empty for next_prompt mode.",
            },
            "session_id": {
                "type": "string",
                "description": "Optional session identifier to read recent conversation turns from.",
            },
            "cwd": {
                "type": "string",
                "description": "Optional workspace path for project stack context.",
            },
        },
        "required": [],
    },
}

ANALYTICS_TOOL_SCHEMA: Dict[str, Any] = {
    "name": "whatnext_analytics",
    "description": (
        "Inspect WhatNext prompt suggestion evaluation metrics (presentation, acceptance, edit "
        "distance, submission rate, latency) and learned user workflow movement transitions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["summary", "clear_feedback"],
                "description": "Read the evaluation summary or clear stored feedback memory.",
            }
        },
        "required": [],
    },
}


def _handle_predict_tool(args: Dict[str, Any], **_kw) -> str:
    draft = str((args or {}).get("draft") or "")
    sid = str((args or {}).get("session_id") or "tool").strip()
    cwd = str((args or {}).get("cwd") or "")
    res = _rpc_predict({"draft": draft, "session_id": sid, "cwd": cwd}, session_id=sid)
    return json.dumps(res, ensure_ascii=False, indent=2)


def _handle_analytics_tool(args: Dict[str, Any], **_kw) -> str:
    action = str((args or {}).get("action") or "summary")
    if action == "clear_feedback":
        return json.dumps(_rpc_clear_feedback({}), ensure_ascii=False, indent=2)
    return json.dumps(_rpc_dashboard({}), ensure_ascii=False, indent=2)


# ── Slash Command (/whatnext) & CLI Subcommand (hermes whatnext) ─────────


def _format_slash_output(raw_args: str, *, session_id: Optional[str] = None) -> str:
    tokens = (raw_args or "").strip()
    lower = tokens.lower()
    sid = (session_id or "cli").strip() or "cli"

    if lower in ("stats", "eval", "metrics"):
        m = store.get_evaluation_metrics()
        mv = store.get_movement_profile()
        lines = [
            "WhatNext Evaluation Scorecard",
            f"1. Presented: {m['presented']} | Accepted: {m['accepted']} ({int(m['acceptance_rate'] * 100)}%) | Submitted: {m['submitted']} ({int(m['submission_rate'] * 100)}%)",
            f"2. Edited before send: {m['edited']} | Avg edit distance: {m['avg_edit_distance']} | Median latency: {m['median_latency_ms']} ms",
            f"3. Human movements learned: {mv['human_turns_learned']} turns (avg {mv['avg_words']} words, lang={mv['preferred_language']})",
        ]
        if mv["transitions"]:
            top_edges = ", ".join(f"{t['from']} -> {t['to']} ({t['weight']}x)" for t in mv["transitions"][:4])
            lines.append(f"4. Top workflow transitions: {top_edges}")
        return "\n".join(lines)

    if lower in ("clear", "reset"):
        res = _rpc_clear_feedback({})
        return f"Cleared WhatNext feedback memory: {res['cleared']}"

    if lower.startswith("complete "):
        draft = tokens[len("complete "):]
        bundle = _rpc_predict({"draft": draft, "mode": "complete_draft"}, session_id=sid)
        cands = bundle.get("candidates") or []
        if not cands:
            return "No draft completion candidate generated for that prefix."
        top = cands[0]
        return (
            f"WhatNext Draft Completion ({bundle.get('backend', 'auto')}, {bundle.get('latency_ms', 0)} ms)\n"
            f"1. Full prompt: {top['text']}\n"
            f"2. Ghost suffix: {top.get('completion_suffix', '')}\n"
            f"3. Reason: {top.get('reason', '')}"
        )

    bundle = _rpc_predict({"mode": "next_prompt"}, session_id=sid)
    cands = bundle.get("candidates") or []
    if not cands:
        return "No next-move suggestions right now. Try `/whatnext complete <partial draft>` or `/whatnext stats`."
    lines = [
        f"WhatNext Predicted Movements ({bundle.get('backend', 'auto')}, {bundle.get('latency_ms', 0)} ms):"
    ]
    for idx, item in enumerate(cands, start=1):
        lines.append(f"{idx}. [{item.get('intent', 'move')}] {item['text']} ({item.get('reason', '')})")
    return "\n".join(lines)


def _slash_whatnext(raw_args: str = "", **kwargs) -> str:
    sid = str(kwargs.get("session_id") or kwargs.get("session_key") or "cli")
    return _format_slash_output(raw_args, session_id=sid)


def _setup_cli(subparser) -> None:
    subparser.add_argument(
        "subaction",
        nargs="?",
        default="status",
        choices=["status", "predict", "eval", "clear"],
        help="Inspect WhatNext status, run a prediction, view evaluation metrics, or clear feedback memory.",
    )
    subparser.add_argument("--draft", default="", help="Partial draft text to complete in complete_draft mode.")
    subparser.add_argument("--cwd", default="", help="Workspace path to condition predictions on.")


def _handle_cli(args) -> int:
    action = getattr(args, "subaction", "status")
    if action == "clear":
        print(json.dumps(_rpc_clear_feedback({}), indent=2))
        return 0
    if action == "eval":
        print(json.dumps(_rpc_dashboard({}), indent=2))
        return 0
    if action == "predict":
        draft = getattr(args, "draft", "") or ""
        cwd = getattr(args, "cwd", "") or ""
        res = _rpc_predict({"draft": draft, "cwd": cwd}, session_id="cli-preview")
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    info = _rpc_dashboard({})
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


# ── Entry Point ──────────────────────────────────────────────────────────


def register(ctx) -> None:
    global _ctx, _llm_task
    _ctx = ctx

    try:
        ctx.register_auxiliary_task(
            AUX_TASK_KEY,
            display_name="WhatNext Prompt Predictor",
            description="Memory-grounded next-prompt and live draft completion predictions.",
        )
        _llm_task = AUX_TASK_KEY
    except (AttributeError, ValueError):
        _llm_task = None

    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)

    ctx.register_tool(
        name="whatnext_predict",
        toolset="whatnext",
        schema=PREDICT_TOOL_SCHEMA,
        handler=_handle_predict_tool,
        description="Predict next user prompts or complete a partial draft using memory and workflow habits.",
    )
    ctx.register_tool(
        name="whatnext_analytics",
        toolset="whatnext",
        schema=ANALYTICS_TOOL_SCHEMA,
        handler=_handle_analytics_tool,
        description="Inspect WhatNext prompt prediction evaluation funnel and learned workflow movements.",
    )

    if callable(getattr(ctx, "register_rpc", None)):
        ctx.register_rpc("predict", _rpc_predict)
        ctx.register_rpc("state", _rpc_state)
        ctx.register_rpc("event", _rpc_event)
        ctx.register_rpc("dashboard", _rpc_dashboard)
        ctx.register_rpc("clear_feedback", _rpc_clear_feedback)

    if callable(getattr(ctx, "register_command", None)):
        ctx.register_command(
            name="whatnext",
            handler=_slash_whatnext,
            description="Predict your next prompts, complete a draft, or view suggestion evaluation metrics.",
            args_hint="[stats | clear | complete <draft>]",
        )

    if callable(getattr(ctx, "register_cli_command", None)):
        ctx.register_cli_command(
            name="whatnext",
            help="Memory-grounded prompt prediction and composer evaluation console",
            setup_fn=_setup_cli,
            handler_fn=_handle_cli,
            description="Inspect WhatNext prompt predictions, movement graph, and evaluation metrics.",
        )

    skill_file = Path(__file__).resolve().parent / "skills" / "movements" / "SKILL.md"
    if skill_file.is_file() and callable(getattr(ctx, "register_skill", None)):
        ctx.register_skill(
            "movements",
            skill_file,
            description="Prompt prediction, draft completion, and workflow movement analytics.",
        )

    logger.info("whatnext: registered plugin across agent, RPC, CLI, and composer surfaces")
