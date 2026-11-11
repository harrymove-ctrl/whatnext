"""Memory-grounded prompt prediction and live draft completion engine for WhatNext.

Implements both Clone Prompt Prediction modes:
1. ``next_prompt`` (empty draft):
   a. Brand-new empty chat starter prompts personalized from USER.md, MEMORY.md,
      workspace stack, and learned human workflow movements.
   b. Post-turn follow-up predictions committing to likely answers or the user's
      habitual next workflow step (verify, implement, debug, review, ship).
2. ``complete_draft`` (non-empty draft):
   Live autocomplete continuation of a partially typed user draft, returning both
   ``completion_suffix`` and merged ``text`` while respecting monotonic revisions.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_PLACEHOLDER_RE = re.compile(r"\[[^\[\]]{1,40}\]|<[^<>]{1,40}>|_{3,}")
_LEADING_LABEL_RE = re.compile(
    r"^(?:suggestion|next prompt|next move|user|draft|completion)\s*:\s*",
    re.I,
)
_ABSTAIN_TOKENS = frozenset({"NULL", "NONE", "ABSTAIN", "NO_SUGGESTION", "N/A"})

NEXT_PROMPT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "abstain": {"type": "boolean"},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "intent": {
                        "type": "string",
                        "enum": [
                            "answer",
                            "verify",
                            "implement",
                            "debug",
                            "review",
                            "refactor",
                            "ship",
                            "explore",
                        ],
                    },
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["text", "intent"],
            },
        },
    },
    "required": ["abstain", "candidates"],
}

COMPLETE_DRAFT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "abstain": {"type": "boolean"},
        "full_text": {"type": "string"},
        "intent": {"type": "string"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["abstain", "full_text"],
}


def _load_sibling(module_stem: str):
    import importlib.util
    import sys

    path = Path(__file__).resolve().with_name(f"{module_stem}.py")
    key = "whatnext_mod_" + module_stem + "_" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:10]
    mod = sys.modules.get(key)
    if mod is None:
        spec = importlib.util.spec_from_file_location(key, path)
        mod = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(mod)
        sys.modules[key] = mod
    return mod


def classify_error_kind(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in (401, 403):
            return "auth"
        if exc.code == 429:
            return "rate_limit"
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    name = type(exc).__name__.lower()
    if status in (401, 403) or "auth" in name or "permission" in name:
        return "auth"
    if status == 429 or "ratelimit" in name:
        return "rate_limit"
    if "timeout" in name or isinstance(exc, TimeoutError):
        return "timeout"
    return "other"


def clean_candidate_text(raw: str, *, max_chars: int = 220) -> str:
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    text = _LEADING_LABEL_RE.sub("", lines[0]).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'", "`", "“"}:
        text = text[1:-1].strip()
    if text.upper().rstrip(".!") in _ABSTAIN_TOKENS:
        return ""
    if _PLACEHOLDER_RE.search(text):
        return ""
    if len(text) < 4 or len(text) > max_chars:
        return ""
    return text


def compute_completion_suffix(draft: str, candidate_text: str) -> Tuple[str, str]:
    """Align ``candidate_text`` with the user's live ``draft`` so Tab acceptance never
    duplicates prefixes or mangles spacing. Returns ``(merged_full_text, ghost_suffix)``."""
    raw_draft = draft or ""
    stripped_draft = raw_draft.strip()
    cand = (candidate_text or "").strip()
    if not stripped_draft or not cand:
        return "", ""

    if cand.lower().startswith(raw_draft.lower()):
        suffix = cand[len(raw_draft):]
        return (raw_draft + suffix, suffix) if suffix.strip() else ("", "")

    if cand.lower().startswith(stripped_draft.lower()):
        tail = cand[len(stripped_draft):]
        if not tail.strip():
            return "", ""
        if raw_draft.endswith(" ") and tail.startswith(" "):
            tail = tail[1:]
        return raw_draft + tail, tail

    # Overlap on trailing words of draft -> leading words of candidate
    draft_lower = raw_draft.lower()
    cand_lower = cand.lower()
    max_overlap = min(len(raw_draft), len(cand))
    for k in range(max_overlap, 2, -1):
        if draft_lower.endswith(cand_lower[:k]):
            suffix = cand[k:]
            if suffix.strip():
                return raw_draft + suffix, suffix

    # Treat candidate as a direct continuation suffix if it doesn't repeat the start
    sep = "" if raw_draft.endswith((" ", "\n", "\t")) or cand.startswith((",", ".", "?", ":", ";")) else " "
    merged = raw_draft + sep + cand
    if len(merged) > 320:
        return "", ""
    return merged, sep + cand


def _assistant_asked_question(last_assistant: str) -> Optional[List[str]]:
    """Detect if the assistant ended with a concrete question or explicit option choices."""
    if not last_assistant:
        return None
    tail = last_assistant[-700:]
    if "?" not in tail:
        return None
    options: List[str] = []
    for m in re.finditer(r"(?:^|\n)\s*(?:[1-3a-cA-C][.)]|[-*•])\s+\*{0,2}([^:\n*]{3,64})\*{0,2}", tail):
        opt = m.group(1).strip()
        if opt and not _PLACEHOLDER_RE.search(opt):
            options.append(opt)
    return options[:3]


def build_heuristic_predictions(
    *,
    mode: str,
    draft: str,
    messages: List[Dict[str, Any]],
    memory_bundle: Dict[str, Any],
    movement_profile: Dict[str, Any],
    workspace: Dict[str, Any],
    candidate_count: int = 3,
    is_dismissed_fn=None,
) -> List[Dict[str, Any]]:
    """Deterministic, zero-network movement predictor grounded in user memory,
    learned workflow transitions, workspace stack, and conversation state."""
    store = _load_sibling("store")
    dismissed = is_dismissed_fn or (lambda _t: False)
    proj = workspace.get("project_name") or "this project"
    stacks = workspace.get("stacks") or []
    stack_label = stacks[0] if stacks else "the codebase"
    has_tests = bool(workspace.get("has_tests"))
    highlights = memory_bundle.get("highlights") or []
    mem_hint = highlights[0][:72] if highlights else ""

    if mode == "complete_draft":
        raw_draft = draft or ""
        low = raw_draft.strip().lower()
        if len(low) < 2:
            return []

        continuations: List[Tuple[str, str, str]] = []
        if low.startswith(("can you ", "could you ", "please ")):
            continuations.append(("walk through the relevant files and propose the cleanest fix?", "explore", "Matches polite request opener"))
        if any(low.startswith(p) for p in ("run ", "run the", "test", "check ", "verify")):
            test_cmd = "pytest and report any failing assertions" if "Python" in stacks else (
                "cargo test and summarize any warnings" if "Rust" in stacks else "the test suite and fix any failures"
            )
            if low.startswith("run "):
                continuations.append((f"run {test_cmd}", "verify", "Grounded in workspace test stack"))
            else:
                continuations.append((f"{raw_draft.strip()} across {proj} and show the output", "verify", "Completes verification move"))
        if any(low.startswith(p) for p in ("fix ", "debug ", "why is", "investigate")):
            continuations.append((f"{raw_draft.strip()} by tracing the root cause before changing code", "debug", "Grounded in root-cause debugging flow"))
        if any(low.startswith(p) for p in ("commit", "open a pr", "ship", "push")):
            continuations.append((f"{raw_draft.strip()} after verifying git status and passing tests", "ship", "Matches ship-after-verify habit"))
        if any(low.startswith(p) for p in ("refactor", "clean up", "simplify")):
            continuations.append((f"{raw_draft.strip()} while keeping existing public interfaces intact", "refactor", "Safe refactor completion"))
        if any(low.startswith(p) for p in ("make it", "rewrite", "shorten", "tighten")):
            continuations.append((f"{raw_draft.strip()} more concise and concrete with clear next steps", "review", "Editorial refinement completion"))

        if not continuations:
            predicted_next = movement_profile.get("predicted_next_intents") or []
            top_next = predicted_next[0]["intent"] if predicted_next else "implement"
            if top_next == "verify":
                continuations.append((f"{raw_draft.strip()} and verify it with automated tests", "verify", "Follows your learned verify movement"))
            elif top_next == "ship":
                continuations.append((f"{raw_draft.strip()} and prepare a clean summary of changes", "ship", "Follows your learned ship movement"))
            else:
                continuations.append((f"{raw_draft.strip()} in {proj} and show the exact file changes", "implement", f"Scoped to {proj}"))

        results: List[Dict[str, Any]] = []
        for cand_full, intent, reason in continuations:
            merged, suffix = compute_completion_suffix(raw_draft, cand_full)
            if suffix and not dismissed(merged):
                results.append(
                    {
                        "text": merged,
                        "completion_suffix": suffix,
                        "intent": intent,
                        "confidence": 0.82,
                        "reason": reason,
                    }
                )
                break
        return results

    # next_prompt mode (empty draft)
    candidates: List[Dict[str, Any]] = []

    if not messages:
        # Brand-new empty chat: starter movements grounded in USER.md / MEMORY.md + workspace + habits
        if stacks:
            candidates.append(
                {
                    "text": f"Inspect {proj} ({stack_label}) and summarize what is currently in progress.",
                    "completion_suffix": "",
                    "intent": "explore",
                    "confidence": 0.86,
                    "reason": f"Workspace starter ({stack_label} detected)",
                }
            )
        if mem_hint:
            candidates.append(
                {
                    "text": f"Pick up my current priority in {proj} keeping in mind: {mem_hint}",
                    "completion_suffix": "",
                    "intent": "implement",
                    "confidence": 0.84,
                    "reason": "Grounded in Hermes USER.md / MEMORY.md",
                }
            )
        if has_tests:
            candidates.append(
                {
                    "text": f"Run the test suite in {proj} and flag anything broken or flaky.",
                    "completion_suffix": "",
                    "intent": "verify",
                    "confidence": 0.80,
                    "reason": "Test directory detected in workspace",
                }
            )
        candidates.append(
            {
                "text": f"Check recent git changes in {proj} and propose the highest-impact next step.",
                "completion_suffix": "",
                "intent": "review",
                "confidence": 0.77,
                "reason": "Starter movement from session history",
            }
        )
    else:
        last_user = ""
        last_assistant = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and not last_assistant:
                last_assistant = str(msg.get("content") or "")
            elif msg.get("role") == "user" and not last_user:
                last_user = str(msg.get("content") or "")
            if last_user and last_assistant:
                break

        options = _assistant_asked_question(last_assistant)
        if options is not None:
            if options:
                candidates.append(
                    {
                        "text": f"Go with {options[0]}, and keep the implementation minimal and well-tested.",
                        "completion_suffix": "",
                        "intent": "answer",
                        "confidence": 0.90,
                        "reason": "Commits to option 1 from the assistant's question",
                    }
                )
                if len(options) > 1:
                    candidates.append(
                        {
                            "text": f"Use {options[1]} instead, and walk me through the trade-offs briefly.",
                            "completion_suffix": "",
                            "intent": "answer",
                            "confidence": 0.81,
                            "reason": "Alternative choice from the assistant's question",
                        }
                    )
            else:
                candidates.append(
                    {
                        "text": "Yes, go ahead with that approach and verify it end to end.",
                        "completion_suffix": "",
                        "intent": "answer",
                        "confidence": 0.87,
                        "reason": "Direct confirmation to the assistant's question",
                    }
                )

        last_intent = store.classify_intent(last_user) if last_user else "explore"
        predicted_intents = [
            item["intent"] for item in (movement_profile.get("predicted_next_intents") or [])
        ]
        # Sensible natural engineering workflow progression when cold-starting
        default_chain = {
            "explore": ["implement", "review", "verify"],
            "implement": ["verify", "review", "ship"],
            "debug": ["verify", "review", "ship"],
            "verify": ["ship", "refactor", "implement"],
            "review": ["implement", "verify", "ship"],
            "refactor": ["verify", "ship", "review"],
            "answer": ["verify", "implement", "ship"],
            "ship": ["explore", "verify", "implement"],
        }
        ordered_intents = []
        for it in predicted_intents + default_chain.get(last_intent, ["verify", "implement", "ship"]):
            if it not in ordered_intents:
                ordered_intents.append(it)

        templates = {
            "verify": (
                f"Run the tests and validation checks for {proj} to prove the changes pass.",
                "Learned workflow move: verify after changes" if predicted_intents else "Natural next step: verify behavior",
            ),
            "review": (
                "Review the diff for edge cases, regressions, or missing error handling.",
                "Learned workflow move: adversarial diff review" if predicted_intents else "Check edge cases before shipping",
            ),
            "ship": (
                "Summarize what changed, verify git status is clean, and prepare the commit.",
                "Learned workflow move: package and ship" if predicted_intents else "Ready-to-ship checkpoint",
            ),
            "implement": (
                f"Implement the recommended approach in {proj} and keep changes focused.",
                "Learned workflow move: move from plan to implementation",
            ),
            "debug": (
                "Trace the failure path step by step and isolate the exact root cause.",
                "Root-cause debugging follow-up",
            ),
            "refactor": (
                "Tighten the implementation and remove any duplicated logic without changing behavior.",
                "Cleanup and simplification follow-up",
            ),
            "explore": (
                f"Show me how this connects to the rest of {proj} and what else is affected.",
                "Impact and architecture check",
            ),
        }

        for idx, intent_key in enumerate(ordered_intents):
            if intent_key in templates:
                txt, why = templates[intent_key]
                candidates.append(
                    {
                        "text": txt,
                        "completion_suffix": "",
                        "intent": intent_key,
                        "confidence": round(max(0.70, 0.88 - idx * 0.06), 2),
                        "reason": why + (f" · memory active" if memory_bundle.get("has_memory") else ""),
                    }
                )

    filtered: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for c in candidates:
        key = c["text"].lower()
        if key in seen or dismissed(c["text"]):
            continue
        seen.add(key)
        filtered.append(c)
        if len(filtered) >= max(1, min(3, int(candidate_count))):
            break
    return filtered


def _format_llm_context(
    *,
    messages: List[Dict[str, Any]],
    memory_bundle: Dict[str, Any],
    movement_profile: Dict[str, Any],
    workspace: Dict[str, Any],
) -> str:
    blocks: List[str] = []
    if workspace.get("summary"):
        blocks.append(f"Workspace: {workspace['summary']}")

    if memory_bundle.get("preferences_text"):
        blocks.append(f"Persistent User Memory:\n{memory_bundle['preferences_text'][:1800]}")

    learned = int(movement_profile.get("human_turns_learned") or 0)
    if learned > 0:
        next_moves = ", ".join(
            f"{m['intent']} ({int(m['probability'] * 100)}%)"
            for m in (movement_profile.get("predicted_next_intents") or [])
        )
        habit_line = (
            f"Learned User Movement Profile ({learned} human turns): "
            f"avg_length={movement_profile.get('avg_words', 12)} words, "
            f"preferred_language={movement_profile.get('preferred_language', 'en')}"
        )
        if next_moves:
            habit_line += f", likely_next_intents=[{next_moves}]"
        blocks.append(habit_line)

    dismissed = movement_profile.get("recent_dismissed") or []
    if dismissed:
        blocks.append("Recently dismissed by user (do NOT suggest similar phrasing): " + " | ".join(dismissed[:4]))

    if messages:
        convo_lines: List[str] = []
        for msg in messages[-10:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            body = str(msg.get("content") or "").strip()
            if len(body) > 900:
                body = body[:260].rstrip() + " ... " + body[-580:].lstrip()
            convo_lines.append(f"[{role}]: {body}")
        blocks.append("Recent Conversation:\n" + "\n\n".join(convo_lines))
    else:
        blocks.append("Recent Conversation: (Brand-new empty chat session)")

    return "\n\n".join(blocks)


def _predict_with_hermes_llm(
    ctx_llm: Any,
    *,
    llm_task: Optional[str],
    mode: str,
    draft: str,
    messages: List[Dict[str, Any]],
    memory_bundle: Dict[str, Any],
    movement_profile: Dict[str, Any],
    workspace: Dict[str, Any],
    candidate_count: int,
    is_dismissed_fn,
) -> List[Dict[str, Any]]:
    context_blob = _format_llm_context(
        messages=messages,
        memory_bundle=memory_bundle,
        movement_profile=movement_profile,
        workspace=workspace,
    )
    task_kw = {"task": llm_task} if llm_task else {}

    if mode == "complete_draft":
        instructions = (
            "You complete the message a human user is actively typing into their AI coding and research assistant. "
            "Write strictly in the user's voice, matching their language, conciseness, and persistent memory preferences. "
            "Return full_text containing the user's draft completed naturally into one crisp, ready-to-send sentence or clause "
            "(under 200 characters total). Never include bracketed placeholders like [file] or <name>. "
            "Set abstain=true if the draft is already complete or ambiguous."
        )
        prompt_text = f"{context_blob}\n\nCurrent partial user draft:\n{draft}"
        res = ctx_llm.complete_structured(
            instructions=instructions,
            input=[{"type": "text", "text": prompt_text}],
            json_schema=COMPLETE_DRAFT_SCHEMA,
            temperature=0.2,
            max_tokens=160,
            purpose="whatnext.complete_draft",
            **task_kw,
        )
        parsed = getattr(res, "parsed", None)
        if isinstance(parsed, dict):
            if parsed.get("abstain"):
                return []
            cleaned = clean_candidate_text(str(parsed.get("full_text") or ""), max_chars=280)
            merged, suffix = compute_completion_suffix(draft, cleaned)
            if suffix and not is_dismissed_fn(merged):
                return [
                    {
                        "text": merged,
                        "completion_suffix": suffix,
                        "intent": str(parsed.get("intent") or "implement"),
                        "confidence": float(parsed.get("confidence") or 0.86),
                        "reason": str(parsed.get("reason") or "Live draft completion"),
                    }
                ]
        return []

    want = max(1, min(3, int(candidate_count)))
    instructions = (
        f"You predict the next {want} ranked prompt(s) the human user will most likely want to send to Hermes Agent. "
        "Suggestions appear under the chat composer and never send automatically. "
        "Rules:\n"
        "1. Write from the USER's first-person perspective in the user's language and preferred directness.\n"
        "2. If the assistant just asked a question or offered choices, candidate 1 MUST commit to the concrete answer the user would pick.\n"
        "3. Condition on Persistent User Memory (USER.md / MEMORY.md) and the user's Learned Movement Profile.\n"
        "4. Each candidate must be 15 to 150 characters, specific, zero bracketed placeholders ([x], <y>), and distinct in intent.\n"
        "5. Set abstain=true only when silence is clearly appropriate (such as a final goodbye)."
    )
    res = ctx_llm.complete_structured(
        instructions=instructions,
        input=[{"type": "text", "text": context_blob}],
        json_schema=NEXT_PROMPT_SCHEMA,
        temperature=0.3,
        max_tokens=280,
        purpose="whatnext.next_prompt",
        **task_kw,
    )
    parsed = getattr(res, "parsed", None)
    if not isinstance(parsed, dict) or parsed.get("abstain"):
        return []
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw_item in (parsed.get("candidates") or [])[:want]:
        if not isinstance(raw_item, dict):
            continue
        txt = clean_candidate_text(str(raw_item.get("text") or ""), max_chars=180)
        if not txt or txt.lower() in seen or is_dismissed_fn(txt):
            continue
        seen.add(txt.lower())
        out.append(
            {
                "text": txt,
                "completion_suffix": "",
                "intent": str(raw_item.get("intent") or "implement"),
                "confidence": round(float(raw_item.get("confidence") or 0.85), 2),
                "reason": str(raw_item.get("reason") or "Grounded in conversation & user memory")[:120],
            }
        )
    return out


def generate_prediction_bundle(
    *,
    session_id: str,
    mode: str = "next_prompt",
    draft: str = "",
    draft_revision: int = 0,
    context_revision: Optional[int] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    cwd: str = "",
    settings: Optional[Dict[str, Any]] = None,
    ctx_llm: Any = None,
    llm_task: Optional[str] = None,
    profile_name: str = "default",
) -> Dict[str, Any]:
    """End-to-end prediction coordinator used by hooks, JSON-RPC, REST API, slash commands, and tools."""
    store = _load_sibling("store")
    mem_mod = _load_sibling("memory_signals")
    clone_mod = _load_sibling("clone_client")

    cfg = {
        "enabled": True,
        "backend": "auto",
        "enable_draft_completion": True,
        "enable_empty_chat_starters": True,
        "enable_memory_personalization": True,
        "feedback_enabled": True,
        "max_context_messages": 12,
        "candidate_count": 3,
        "clone_api_base_url": "https://api.clone.is",
        **(settings or {}),
    }

    sid = (session_id or "new").strip() or "new"
    clean_mode = "complete_draft" if (mode == "complete_draft" or bool((draft or "").strip())) else "next_prompt"

    if not cfg.get("enabled", True):
        return {"session_id": sid, "mode": clean_mode, "candidates": [], "error_kind": "disabled"}

    if clean_mode == "complete_draft" and not cfg.get("enable_draft_completion", True):
        return {"session_id": sid, "mode": clean_mode, "candidates": [], "error_kind": ""}

    started = time.monotonic()
    request_id = f"wn_{uuid.uuid4().hex[:16]}"
    rev_state = store.get_revisions(sid)
    ctx_rev = int(context_revision) if context_revision is not None else int(rev_state["context_revision"])
    dr_rev = int(draft_revision)

    if messages is None:
        loaded = mem_mod.load_session_turns(sid, max_messages=int(cfg.get("max_context_messages", 12)))
        turn_messages = loaded["messages"]
        if not cwd:
            cwd = loaded["cwd"]
    else:
        turn_messages = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "")
            if role not in ("user", "assistant"):
                continue
            content = m.get("content") or ""
            if isinstance(content, list):
                content = " ".join(
                    str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            txt = mem_mod.scrub_text(str(content).strip())
            if txt:
                if turn_messages and turn_messages[-1]["role"] == role == "assistant":
                    turn_messages[-1]["content"] = txt[:2400]
                else:
                    turn_messages.append({"role": role, "content": txt[:2400]})
        turn_messages = turn_messages[-max(2, min(30, int(cfg.get("max_context_messages", 12)))):]

    if clean_mode == "next_prompt" and not turn_messages and not cfg.get("enable_empty_chat_starters", True):
        return {"session_id": sid, "mode": clean_mode, "candidates": [], "error_kind": ""}

    memory_bundle = mem_mod.read_user_memory_bundle(enabled=bool(cfg.get("enable_memory_personalization", True)))
    last_user_text = next((m["content"] for m in reversed(turn_messages) if m["role"] == "user"), "")
    last_intent = store.classify_intent(last_user_text) if last_user_text else "session_start"
    movement_profile = (
        store.get_movement_profile(current_intent=last_intent)
        if cfg.get("feedback_enabled", True)
        else {"human_turns_learned": 0, "predicted_next_intents": [], "recent_dismissed": []}
    )
    workspace = mem_mod.inspect_workspace_context(cwd)

    backend_choice = str(cfg.get("backend") or "auto").lower()
    candidates: List[Dict[str, Any]] = []
    used_backend = "local_heuristics"
    error_kind = ""

    # 1. Optional Clone Labs Official HTTP API
    if backend_choice in ("auto", "clone_api") and clone_mod.is_clone_configured():
        try:
            clone_res = clone_mod.request_clone_prediction(
                request_id=request_id,
                user_id=f"hermes:{profile_name}",
                session_id=sid,
                mode=clean_mode,
                draft=draft,
                draft_revision=dr_rev,
                context_revision=ctx_rev,
                messages=turn_messages,
                user_preferences=memory_bundle.get("preferences_text", ""),
                language=movement_profile.get("preferred_language", "auto"),
                feedback_enabled=bool(cfg.get("feedback_enabled", True)),
                base_url=str(cfg.get("clone_api_base_url") or "https://api.clone.is"),
            )
            raw_cands = clone_res.get("candidates") or []
            for item in raw_cands:
                if clean_mode == "complete_draft":
                    merged, suffix = compute_completion_suffix(draft, item["text"])
                    if suffix and not store.is_dismissed_phrase(merged):
                        candidates.append({**item, "text": merged, "completion_suffix": suffix})
                else:
                    cleaned = clean_candidate_text(item["text"])
                    if cleaned and not store.is_dismissed_phrase(cleaned):
                        candidates.append({**item, "text": cleaned, "completion_suffix": ""})
            if candidates:
                used_backend = "clone_api"
        except Exception as exc:
            error_kind = classify_error_kind(exc)
            if backend_choice == "clone_api":
                latency_ms = int((time.monotonic() - started) * 1000)
                saved = store.save_prediction(
                    sid,
                    mode=clean_mode,
                    request_id=request_id,
                    context_revision=ctx_rev,
                    draft_revision=dr_rev,
                    draft_prefix=draft if clean_mode == "complete_draft" else "",
                    candidates=[],
                    backend="clone_api",
                    latency_ms=latency_ms,
                    memory_used=bool(memory_bundle.get("has_memory")),
                    error_kind=error_kind,
                )
                return saved or {
                    "session_id": sid,
                    "mode": clean_mode,
                    "candidates": [],
                    "error_kind": error_kind,
                }

    # 2. Host-owned Hermes LLM (zero extra API keys required)
    if not candidates and backend_choice in ("auto", "hermes_llm") and ctx_llm is not None:
        try:
            candidates = _predict_with_hermes_llm(
                ctx_llm,
                llm_task=llm_task,
                mode=clean_mode,
                draft=draft,
                messages=turn_messages,
                memory_bundle=memory_bundle,
                movement_profile=movement_profile,
                workspace=workspace,
                candidate_count=int(cfg.get("candidate_count", 3)),
                is_dismissed_fn=store.is_dismissed_phrase,
            )
            if candidates:
                used_backend = "hermes_llm"
                error_kind = ""
        except Exception as exc:
            error_kind = classify_error_kind(exc)

    # 3. Deterministic Memory & Movement Heuristic Fallback (guarantees zero dead-air offline or in sandbox)
    if not candidates and (backend_choice == "local_heuristics" or (backend_choice == "auto" and ctx_llm is None)):
        candidates = build_heuristic_predictions(
            mode=clean_mode,
            draft=draft,
            messages=turn_messages,
            memory_bundle=memory_bundle,
            movement_profile=movement_profile,
            workspace=workspace,
            candidate_count=int(cfg.get("candidate_count", 3)),
            is_dismissed_fn=store.is_dismissed_phrase,
        )
        if candidates:
            used_backend = "movement_engine"
            error_kind = ""

    latency_ms = max(1, int((time.monotonic() - started) * 1000))
    saved = store.save_prediction(
        sid,
        mode=clean_mode,
        request_id=request_id,
        context_revision=ctx_rev,
        draft_revision=dr_rev,
        draft_prefix=draft if clean_mode == "complete_draft" else "",
        candidates=candidates,
        backend=used_backend,
        latency_ms=latency_ms,
        memory_used=bool(memory_bundle.get("has_memory") or movement_profile.get("human_turns_learned", 0) > 0),
        error_kind=error_kind if not candidates else "",
    )
    if saved is None:
        # Stale revision discarded cleanly
        return {
            "session_id": sid,
            "mode": clean_mode,
            "stale": True,
            "candidates": [],
            "error_kind": "",
        }
    return saved
