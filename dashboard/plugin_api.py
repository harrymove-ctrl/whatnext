"""FastAPI backend for WhatNext, mounted at ``/api/plugins/whatnext/``.

Shared by Hermes Desktop (``ctx.rest(...)``) and the Hermes Web Dashboard
(``hermes dashboard``). Reads and writes the profile-scoped SQLite database in
``store.py`` so state stays coherent even when the web host and gateway turns
run in separate processes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Query

router = APIRouter()


def _load_root_module(stem: str):
    path = Path(__file__).resolve().parents[1] / f"{stem}.py"
    key = "whatnext_mod_" + stem + "_" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:10]
    mod = sys.modules.get(key)
    if mod is None:
        spec = importlib.util.spec_from_file_location(key, path)
        mod = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(mod)
        sys.modules[key] = mod
    return mod


@router.get("/state")
def get_state(session_id: str = Query("new")) -> Dict[str, Any]:
    store = _load_root_module("store")
    clone_client = _load_root_module("clone_client")
    sid = (session_id or "new").strip() or "new"
    revs = store.get_revisions(sid)
    return {
        "session_id": sid,
        "context_revision": int(revs["context_revision"]),
        "draft_revision": int(revs["draft_revision"]),
        "next_prompt": store.get_prediction(sid, mode="next_prompt"),
        "complete_draft": store.get_prediction(sid, mode="complete_draft"),
        "clone_configured": clone_client.is_clone_configured(),
    }


@router.post("/predict")
def post_predict(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    predictor = _load_root_module("predictor")
    sid = str(payload.get("session_id") or "new").strip() or "new"
    draft = str(payload.get("draft") or "")
    mode = str(payload.get("mode") or ("complete_draft" if draft.strip() else "next_prompt"))
    draft_rev = int(payload.get("draft_revision") or 0)
    ctx_rev_raw = payload.get("context_revision")
    ctx_rev = int(ctx_rev_raw) if ctx_rev_raw is not None else None
    cwd = str(payload.get("cwd") or "")
    messages: Optional[List[Dict[str, Any]]] = (
        payload.get("messages") if isinstance(payload.get("messages"), list) else None
    )
    return predictor.generate_prediction_bundle(
        session_id=sid,
        mode=mode,
        draft=draft,
        draft_revision=draft_rev,
        context_revision=ctx_rev,
        messages=messages,
        cwd=cwd,
    )


@router.post("/events")
def post_event(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    store = _load_root_module("store")
    clone_client = _load_root_module("clone_client")
    sid = str(payload.get("session_id") or "new").strip() or "new"
    ev_type = str(payload.get("event_type") or "")
    req_id = str(payload.get("request_id") or "")
    mode = str(payload.get("mode") or "next_prompt")
    cand_idx = int(payload.get("candidate_index") or 0)
    cand_text = str(payload.get("text") or "")
    origin = str(payload.get("origin") or "suggestion")

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
        )
    elif ev_type == "submitted_direct":
        store.observe_submitted_turn(sid, cand_text, explicit_origin=origin)
    else:
        store.record_observation_event(
            ev_type,
            session_id=sid,
            request_id=req_id,
            mode=mode,
            candidate_index=cand_idx,
            origin=origin,
            detail=str(payload.get("detail") or ""),
        )

    if clone_client.is_clone_configured() and ev_type in {"presented", "accepted", "dismissed", "submitted"}:
        clone_client.send_clone_event(
            {
                "event_type": ev_type,
                "request_id": req_id,
                "session_id": sid,
                "mode": mode,
                "candidate_index": cand_idx,
            }
        )
    return {"ok": True}


@router.get("/dashboard")
def get_dashboard(current_intent: str = Query("")) -> Dict[str, Any]:
    store = _load_root_module("store")
    memory_signals = _load_root_module("memory_signals")
    clone_client = _load_root_module("clone_client")
    mem = memory_signals.read_user_memory_bundle(enabled=True)
    return {
        "metrics": store.get_evaluation_metrics(),
        "movements": store.get_movement_profile(current_intent=current_intent),
        "memory": {
            "enabled": mem["enabled"],
            "has_memory": mem["has_memory"],
            "highlights": mem["highlights"],
        },
        "engine": {
            "clone_api_configured": clone_client.is_clone_configured(),
        },
    }


@router.post("/feedback/clear")
def post_clear_feedback() -> Dict[str, Any]:
    store = _load_root_module("store")
    clone_client = _load_root_module("clone_client")
    res = store.clear_feedback_memory()
    remote_cleared = clone_client.clear_clone_feedback() if clone_client.is_clone_configured() else False
    return {**res, "clone_api_cleared": remote_cleared}
