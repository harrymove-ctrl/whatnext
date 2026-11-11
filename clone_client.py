"""Client for the Clone Labs Prompt Prediction HTTP API (https://api.clone.is).

Reads ``CLONE_API_KEY`` strictly via ``agent.secret_scope.get_secret("CLONE_API_KEY")``
so profile credential isolation is preserved and no browser/renderer bundle ever sees
the key. Validates target URLs with ``tools.url_safety.is_safe_url`` when running inside
Hermes and enforces ``https://`` in production.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional


DEFAULT_CLONE_BASE_URL = "https://api.clone.is"


def get_clone_api_key() -> str:
    """Resolve CLONE_API_KEY through Hermes profile-scoped secret_scope first."""
    try:
        from agent.secret_scope import get_secret
        val = get_secret("CLONE_API_KEY")
        return (val or "").strip()
    except Exception:
        # Standalone unit test / demo fallback when Hermes agent package is absent.
        if os.environ.get("WHATNEXT_STANDALONE_DEMO") == "1":
            return os.environ.get("CLONE_API_KEY", "").strip()
        return ""


def is_clone_configured() -> bool:
    return bool(get_clone_api_key())


def _validated_endpoint(base_url: str, path: str) -> str:
    clean_base = (base_url or DEFAULT_CLONE_BASE_URL).strip().rstrip("/")
    parsed = urllib.parse.urlparse(clean_base)
    if parsed.scheme != "https":
        # Allow http://127.0.0.1 only inside deterministic local unit tests.
        if not (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}):
            raise ValueError("Clone API base_url must use https://")
    full_url = f"{clean_base}{path}"
    if parsed.scheme == "https":
        try:
            from tools.url_safety import is_safe_url
            if not is_safe_url(full_url):
                raise ValueError("Clone API URL rejected by Hermes SSRF guard")
        except ImportError:
            pass
    return full_url


def _post_json(
    base_url: str,
    path: str,
    payload: Dict[str, Any],
    *,
    api_key: str,
    timeout_seconds: float = 12.0,
) -> Dict[str, Any]:
    url = _validated_endpoint(base_url, path)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "hermes-whatnext/0.1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw.strip() else {}


def request_clone_prediction(
    *,
    request_id: str,
    user_id: str,
    session_id: str,
    mode: str,
    draft: str,
    draft_revision: int,
    context_revision: int,
    messages: List[Dict[str, Any]],
    user_preferences: str = "",
    language: str = "auto",
    feedback_enabled: bool = True,
    base_url: str = DEFAULT_CLONE_BASE_URL,
    timeout_seconds: float = 12.0,
) -> Dict[str, Any]:
    """Call POST /v1/predictions on Clone API and normalize the response into WhatNext candidates."""
    api_key = get_clone_api_key()
    if not api_key:
        raise RuntimeError("CLONE_API_KEY is not configured")

    wire_messages = []
    for msg in messages[-30:]:
        role = str(msg.get("role") or "user")
        content = str(msg.get("content") or "")[:8000]
        if not content:
            continue
        wire_messages.append(
            {
                "role": role,
                "content": content,
                "origin": msg.get("origin") or ("human" if role == "user" else "agent"),
            }
        )

    payload: Dict[str, Any] = {
        "request_id": request_id,
        "user_id": user_id or "hermes-default",
        "session_id": session_id or "new",
        "mode": "complete_draft" if mode == "complete_draft" else "next_prompt",
        "draft": {
            "text": (draft or "")[:8000],
            "revision": int(draft_revision),
        },
        "context_revision": int(context_revision),
        "messages": wire_messages,
        "language": language or "auto",
        "feedback_enabled": bool(feedback_enabled),
    }
    if user_preferences:
        payload["user_preferences"] = user_preferences[:4000]

    data = _post_json(
        base_url,
        "/v1/predictions",
        payload,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )

    candidates: List[Dict[str, Any]] = []
    raw_candidates = data.get("candidates")
    if isinstance(raw_candidates, list):
        for item in raw_candidates[:3]:
            if isinstance(item, dict):
                txt = str(item.get("completion") or item.get("text") or "").strip()
                if txt:
                    candidates.append(
                        {
                            "text": txt,
                            "intent": str(item.get("intent") or "implement"),
                            "confidence": float(item.get("confidence") or 0.85),
                            "reason": str(item.get("reason") or "Predicted via Clone API"),
                        }
                    )
    else:
        single = data.get("candidate") if isinstance(data.get("candidate"), dict) else data
        txt = str(
            (single or {}).get("completion")
            or (single or {}).get("prediction")
            or (single or {}).get("text")
            or ""
        ).strip()
        if txt:
            candidates.append(
                {
                    "text": txt,
                    "intent": "implement",
                    "confidence": float((single or {}).get("confidence") or 0.88),
                    "reason": "Predicted via Clone API",
                }
            )

    return {"candidates": candidates, "raw": data}


def cancel_clone_prediction(
    request_id: str,
    *,
    base_url: str = DEFAULT_CLONE_BASE_URL,
) -> bool:
    api_key = get_clone_api_key()
    if not api_key or not request_id:
        return False
    try:
        _post_json(
            base_url,
            f"/v1/predictions/{urllib.parse.quote(request_id, safe='')}/cancel",
            {},
            api_key=api_key,
            timeout_seconds=4.0,
        )
        return True
    except Exception:
        return False


def send_clone_event(
    event_payload: Dict[str, Any],
    *,
    base_url: str = DEFAULT_CLONE_BASE_URL,
) -> bool:
    api_key = get_clone_api_key()
    if not api_key:
        return False
    try:
        _post_json(
            base_url,
            "/v1/prediction-events",
            event_payload,
            api_key=api_key,
            timeout_seconds=4.0,
        )
        return True
    except Exception:
        return False


def clear_clone_feedback(*, base_url: str = DEFAULT_CLONE_BASE_URL) -> bool:
    api_key = get_clone_api_key()
    if not api_key:
        return False
    try:
        _post_json(
            base_url,
            "/v1/prediction-feedback/clear",
            {},
            api_key=api_key,
            timeout_seconds=5.0,
        )
        return True
    except Exception:
        return False
