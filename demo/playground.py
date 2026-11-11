#!/usr/bin/env python3
"""Interactive local playground for WhatNext (Hermes Agent Plugin).

Runs the real WhatNext Python prediction engine, SQLite store, memory reader,
movement transition learner, and evaluation ledger behind a local loopback
HTTP server so you can test and record the full composer experience:
1. Empty-draft starter suggestions grounded in USER.md and MEMORY.md
2. Live Tab autocomplete while typing a draft (complete_draft mode)
3. Post-turn ranked movement predictions (next_prompt mode)
4. Bounded Clone Mode delegation with visible countdown and instant Stop
5. Real-time Evaluation Scorecard and Human Movement Transition Graph

Usage:
    python3 demo/playground.py --port 8765
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import memory_signals  # noqa: E402
import predictor  # noqa: E402
import store  # noqa: E402

DEFAULT_USER_MD = """# User Profile
1. Prefers concise, direct engineering prompts with concrete file paths.
2. Always runs automated tests (`pytest` or `vitest`) immediately after code edits.
3. Prefers clean atomic git commits after diff review.
"""

DEFAULT_MEMORY_MD = """# Operating Memory
1. Active repository: whatnext (Hermes Agent Plugin for prompt prediction and movement learning).
2. Validation command: `hermes plugins validate .` must pass all 16 admission checks with zero warnings.
3. Keep composer suggestions non-blocking; Tab inserts into draft and never auto-sends.
"""

PLAYGROUND_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>WhatNext · Prompt Suggestions & Movement Predictor for Hermes Agent</title>
<style>
  :root {
    --bg: #090b10;
    --panel: #11151f;
    --panel-elev: #171c29;
    --border: rgba(148, 163, 184, 0.16);
    --border-strong: rgba(148, 163, 184, 0.32);
    --text: #f1f5f9;
    --muted: #94a3b8;
    --dim: #64748b;
    --accent: #818cf8;
    --emerald: #34d399;
    --amber: #fbbf24;
    --rose: #fb7185;
    --sky: #38bdf8;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: radial-gradient(circle at 15% 0%, rgba(99, 102, 241, 0.12), transparent 42%), var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
    min-height: 100vh;
  }
  .topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 24px;
    border-bottom: 1px solid var(--border);
    background: rgba(9, 11, 16, 0.82);
    backdrop-filter: blur(12px);
    position: sticky;
    top: 0;
    z-index: 10;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .brand-dot {
    width: 10px; height: 10px; border-radius: 999px;
    background: var(--emerald);
    box-shadow: 0 0 12px rgba(52, 211, 153, 0.65);
  }
  .brand h1 { font-size: 15px; font-weight: 650; margin: 0; letter-spacing: -0.01em; }
  .brand span { font-size: 12px; color: var(--muted); }
  .badge {
    font-size: 11px; font-weight: 600; padding: 3px 9px; border-radius: 999px;
    border: 1px solid rgba(129, 140, 248, 0.35);
    background: rgba(129, 140, 248, 0.12);
    color: #c7d2fe;
  }
  .layout {
    max-width: 1360px;
    margin: 0 auto;
    padding: 20px 24px 36px;
    display: grid;
    grid-template-columns: 1.45fr 1fr;
    gap: 20px;
  }
  @media (max-width: 980px) { .layout { grid-template-columns: 1fr; } }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
  }
  .card-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }
  .card-title { font-size: 12px; font-weight: 650; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
  .transcript {
    height: 360px;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .empty-splash {
    margin: auto;
    text-align: center;
    max-width: 420px;
    color: var(--muted);
  }
  .empty-splash h2 { color: var(--text); font-size: 18px; margin: 0 0 6px; }
  .empty-splash p { font-size: 13px; line-height: 1.5; margin: 0; }
  .msg {
    max-width: 88%;
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 13.5px;
    line-height: 1.5;
    white-space: pre-wrap;
  }
  .msg.user {
    align-self: flex-end;
    background: rgba(99, 102, 241, 0.2);
    border: 1px solid rgba(129, 140, 248, 0.32);
  }
  .msg.assistant {
    align-self: flex-start;
    background: var(--panel-elev);
    border: 1px solid var(--border);
  }
  .msg-meta { font-size: 10.5px; color: var(--dim); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.04em; }
  .composer-zone {
    padding: 14px 16px 16px;
    border-top: 1px solid var(--border);
    background: rgba(15, 19, 29, 0.9);
  }
  .composer-box {
    position: relative;
    border: 1px solid var(--border-strong);
    border-radius: 12px;
    background: #0b0e15;
    padding: 10px 12px;
    transition: border-color 0.15s;
  }
  .composer-box:focus-within { border-color: var(--accent); }
  .ghost-layer {
    position: absolute;
    top: 10px; left: 12px; right: 90px;
    pointer-events: none;
    font-family: inherit;
    font-size: 14px;
    line-height: 1.45;
    white-space: pre-wrap;
    word-break: break-word;
    color: transparent;
  }
  .ghost-layer .ghost-tail { color: rgba(148, 163, 184, 0.48); }
  textarea#composer {
    position: relative;
    width: 100%;
    min-height: 52px;
    resize: none;
    border: none;
    outline: none;
    background: transparent;
    color: var(--text);
    font-family: inherit;
    font-size: 14px;
    line-height: 1.45;
    padding-right: 78px;
  }
  .composer-actions {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 6px;
    font-size: 11.5px;
    color: var(--dim);
  }
  .btn {
    border: 1px solid var(--border-strong);
    background: var(--panel-elev);
    color: var(--text);
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 550;
    cursor: pointer;
  }
  .btn:hover { border-color: var(--accent); }
  .btn.primary { background: #4f46e5; border-color: #6366f1; color: #fff; }
  .btn.danger { background: rgba(244, 63, 94, 0.15); border-color: rgba(244, 63, 94, 0.4); color: #fda4af; }
  /* Hermes Desktop COMPOSER_AREAS.underside strip */
  .underside {
    margin-top: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    min-height: 32px;
  }
  .pill {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 10px;
    border-radius: 999px;
    border: 1px solid var(--border-strong);
    background: var(--panel-elev);
    font-size: 12.5px;
  }
  .pill.clone-active {
    border-color: rgba(251, 191, 36, 0.5);
    background: rgba(251, 191, 36, 0.1);
  }
  .intent-tag {
    flex-shrink: 0;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 2px 7px;
    border-radius: 999px;
    background: rgba(56, 189, 248, 0.14);
    border: 1px solid rgba(56, 189, 248, 0.35);
    color: var(--sky);
  }
  .pill-text {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    cursor: pointer;
    color: #e2e8f0;
    background: none;
    border: none;
    text-align: left;
    font-size: 12.5px;
    padding: 0;
  }
  .pill-text:hover { color: #fff; }
  .kbd {
    font-size: 10.5px;
    padding: 1px 6px;
    border-radius: 5px;
    border: 1px solid var(--border-strong);
    color: var(--muted);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .pill-reason {
    margin-top: 5px;
    font-size: 11px;
    color: var(--dim);
    padding-left: 6px;
  }
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    padding: 12px 16px;
  }
  .kpi {
    padding: 10px 12px;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: rgba(11, 14, 21, 0.65);
  }
  .kpi .k-label { font-size: 10.5px; text-transform: uppercase; color: var(--dim); letter-spacing: 0.05em; }
  .kpi .k-val { font-size: 19px; font-weight: 700; margin-top: 3px; font-variant-numeric: tabular-nums; }
  .kpi .k-sub { font-size: 11px; color: var(--muted); margin-top: 2px; }
  .section-body { padding: 12px 16px; display: flex; flex-direction: column; gap: 8px; }
  .edge-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 10px; border-radius: 8px; background: rgba(11, 14, 21, 0.55);
    font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .mem-editor {
    width: 100%; min-height: 74px; border-radius: 8px;
    border: 1px solid var(--border); background: #0b0e15; color: #cbd5e1;
    padding: 8px 10px; font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
</style>
</head>
<body>
  <div class="topbar">
    <div class="brand">
      <div class="brand-dot"></div>
      <div>
        <h1>WhatNext for Hermes Agent</h1>
        <span>Prompt Suggestions &amp; Predict Your Movements · Clone-Compatible Composer Engine</span>
      </div>
    </div>
    <div style="display:flex; gap:8px; align-items:center;">
      <span class="badge" id="engineBadge">Engine: Movement + Memory</span>
      <button class="btn" onclick="newChat()">New Empty Chat</button>
      <button class="btn danger" onclick="clearFeedback()">Clear Feedback Memory</button>
    </div>
  </div>

  <div class="layout">
    <div class="card" style="display:flex; flex-direction:column;">
      <div class="card-header">
        <span class="card-title" id="sessionHeader">Hermes Desktop Session · Empty Draft Starter Mode</span>
        <div style="display:flex; gap:6px;">
          <button class="btn" onclick="seedQuestionScenario()" title="Simulate assistant asking an architecture choice">Simulate Agent Question</button>
        </div>
      </div>

      <div class="transcript" id="transcript"></div>

      <div class="composer-zone">
        <div class="composer-box">
          <div class="ghost-layer" id="ghostLayer"></div>
          <textarea id="composer" placeholder="Type to see live Tab draft completion (try: 'run ', 'fix ', 'commit ', 'make it ')..."></textarea>
          <div class="composer-actions">
            <span><strong>Tab</strong> accepts into draft (never sends) · <strong>⌘+Click</strong> sends · <strong>⌥]</strong> cycles moves</span>
            <button class="btn primary" onclick="sendDraft('human')">Send Turn</button>
          </div>
        </div>

        <div class="underside" id="underside"></div>
        <div class="pill-reason" id="reasonLine"></div>
      </div>
    </div>

    <div style="display:flex; flex-direction:column; gap:16px;">
      <div class="card">
        <div class="card-header">
          <span class="card-title">Composer Evaluation Scorecard</span>
          <span style="font-size:11px; color:var(--dim);">Live SQLite Ledger</span>
        </div>
        <div class="kpi-grid">
          <div class="kpi"><div class="k-label">Presented</div><div class="k-val" id="mPresented">0</div><div class="k-sub">Suggestions shown</div></div>
          <div class="kpi"><div class="k-label">Accepted</div><div class="k-val" id="mAccepted">0%</div><div class="k-sub" id="mAcceptedSub">0 into draft</div></div>
          <div class="kpi"><div class="k-label">Submitted</div><div class="k-val" id="mSubmitted">0%</div><div class="k-sub" id="mSubmittedSub">0 sent</div></div>
          <div class="kpi"><div class="k-label">Edit Distance</div><div class="k-val" id="mEditDist">0.0</div><div class="k-sub" id="mEditedSub">0 edited</div></div>
          <div class="kpi"><div class="k-label">Median Latency</div><div class="k-val" id="mLatency">0 ms</div><div class="k-sub">Non-blocking</div></div>
          <div class="kpi"><div class="k-label">Human Turns</div><div class="k-val" id="mHumanTurns">0</div><div class="k-sub">Origin isolated</div></div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <span class="card-title">Learned Workflow Movements (Human-Only)</span>
          <span style="font-size:11px; color:var(--emerald);">Generated sends excluded</span>
        </div>
        <div class="section-body" id="movementEdges"></div>
      </div>

      <div class="card">
        <div class="card-header">
          <span class="card-title">Hermes Persistent Memory (USER.md &amp; MEMORY.md)</span>
          <button class="btn" onclick="saveMemory()">Save &amp; Re-Predict</button>
        </div>
        <div class="section-body">
          <label style="font-size:11px; color:var(--muted);">memories/USER.md</label>
          <textarea class="mem-editor" id="userMdInput"></textarea>
          <label style="font-size:11px; color:var(--muted);">memories/MEMORY.md</label>
          <textarea class="mem-editor" id="memoryMdInput"></textarea>
        </div>
      </div>
    </div>
  </div>

<script>
let sessionId = 'demo-new-1';
let messages = [];
let currentBundle = null;
let candidateIdx = 0;
let draftRevision = 0;
let debounceTimer = null;
let presentedIds = new Set();
let cloneMode = { active: false, remaining: 0, countdownMs: 0, timer: null };

const composerEl = document.getElementById('composer');
const ghostEl = document.getElementById('ghostLayer');

function escapeHtml(s) {
  return (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderTranscript() {
  const el = document.getElementById('transcript');
  const hdr = document.getElementById('sessionHeader');
  hdr.textContent = messages.length === 0
    ? `Hermes Desktop · ${sessionId} (Brand-New Empty Chat)`
    : `Hermes Desktop · ${sessionId} (${messages.length} messages)`;
  if (messages.length === 0) {
    el.innerHTML = `<div class="empty-splash">
      <h2>Predict Your Next Move</h2>
      <p>Even on an empty chat, WhatNext reads <code>USER.md</code>, <code>MEMORY.md</code>, and your workspace stack to suggest personalized starter prompts below, or start typing for live Tab completion.</p>
    </div>`;
    return;
  }
  el.innerHTML = messages.map(m => `
    <div class="msg ${m.role}">
      <div class="msg-meta">${m.role === 'user' ? ('User · origin=' + (m.origin || 'human')) : 'Hermes Agent'}</div>
      <div>${escapeHtml(m.content)}</div>
    </div>
  `).join('');
  el.scrollTop = el.scrollHeight;
}

function updateGhostOverlay() {
  const draft = composerEl.value;
  const cand = currentBundle && currentBundle.candidates && currentBundle.candidates[candidateIdx];
  if (draft.trim().length >= 2 && currentBundle && currentBundle.mode === 'complete_draft' && cand && cand.completion_suffix) {
    ghostEl.innerHTML = escapeHtml(draft) + `<span class="ghost-tail">${escapeHtml(cand.completion_suffix)}</span>`;
  } else {
    ghostEl.innerHTML = '';
  }
}

function renderUnderside() {
  const wrap = document.getElementById('underside');
  const reasonEl = document.getElementById('reasonLine');
  updateGhostOverlay();
  const cands = (currentBundle && currentBundle.candidates) || [];
  if (cands.length === 0) {
    wrap.innerHTML = `<span style="font-size:12px; color:var(--dim);">Waiting for next movement signal...</span>`;
    reasonEl.textContent = '';
    return;
  }
  const idx = candidateIdx % cands.length;
  const c = cands[idx];
  const reqId = currentBundle.request_id || '';
  if (reqId && !presentedIds.has(reqId)) {
    presentedIds.add(reqId);
    fetch('/api/events', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ session_id: sessionId, request_id: reqId, mode: currentBundle.mode, candidate_index: idx, event_type: 'presented' })
    }).then(() => refreshDashboard());
  }

  const badgePrefix = cloneMode.active
    ? `CLONE ${(cloneMode.countdownMs / 1000).toFixed(1)}s`
    : (currentBundle.mode === 'complete_draft' ? 'COMPLETE DRAFT' : (messages.length === 0 ? 'STARTER' : 'NEXT MOVE'));

  wrap.innerHTML = `
    <div class="pill ${cloneMode.active ? 'clone-active' : ''}">
      <span class="intent-tag">${escapeHtml(badgePrefix)} · ${escapeHtml(c.intent || 'move')}</span>
      <button class="pill-text" id="acceptPillBtn" title="Click or press Tab to accept into draft; Cmd/Ctrl+Click to send immediately">
        ${escapeHtml(c.text)}
      </button>
      ${cands.length > 1 ? `<button class="btn" style="padding:2px 7px; font-size:11px;" onclick="cycleCandidate()">${idx + 1}/${cands.length}</button>` : ''}
      <span class="kbd">Tab</span>
      <button class="btn" style="padding:2px 7px; font-size:11px;" onclick="dismissCandidate()">×</button>
    </div>
    ${messages.length > 0 && currentBundle.mode === 'next_prompt'
      ? (cloneMode.active
          ? `<button class="btn danger" onclick="stopClone('user_stop')">Stop (${cloneMode.remaining})</button>`
          : `<button class="btn" onclick="startClone()">Clone Mode (3x)</button>`)
      : ''}
  `;
  document.getElementById('acceptPillBtn').onclick = (ev) => {
    acceptCandidate(Boolean(ev.metaKey || ev.ctrlKey));
  };
  reasonEl.textContent = `Why this move: ${c.reason || 'Context + Memory'} · Engine: ${currentBundle.backend || 'movement_engine'} (${currentBundle.latency_ms || 1} ms)`;
}

async function requestPrediction(mode) {
  const draft = composerEl.value;
  const rev = ++draftRevision;
  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      session_id: sessionId,
      mode: mode,
      draft: mode === 'complete_draft' ? draft : '',
      draft_revision: rev,
      messages: messages
    })
  });
  const data = await res.json();
  if (mode === 'complete_draft' && rev < draftRevision) return;
  currentBundle = data;
  candidateIdx = 0;
  renderUnderside();
  refreshDashboard();
}

async function acceptCandidate(sendNow = false) {
  const cands = (currentBundle && currentBundle.candidates) || [];
  if (!cands.length) return;
  const idx = candidateIdx % cands.length;
  const c = cands[idx];
  const reqId = currentBundle.request_id || '';
  const mode = currentBundle.mode || 'next_prompt';

  await fetch('/api/events', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      session_id: sessionId,
      request_id: reqId,
      mode: mode,
      candidate_index: idx,
      text: c.text,
      origin: 'suggestion',
      event_type: 'accepted'
    })
  });

  composerEl.value = c.text;
  currentBundle = null;
  renderUnderside();
  composerEl.focus();
  if (sendNow) {
    await sendDraft('suggestion');
  } else {
    refreshDashboard();
  }
}

function cycleCandidate() {
  const cands = (currentBundle && currentBundle.candidates) || [];
  if (cands.length <= 1) return;
  candidateIdx = (candidateIdx + 1) % cands.length;
  renderUnderside();
}

async function dismissCandidate() {
  const cands = (currentBundle && currentBundle.candidates) || [];
  if (!cands.length) return;
  const idx = candidateIdx % cands.length;
  const c = cands[idx];
  await fetch('/api/events', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      session_id: sessionId,
      request_id: currentBundle.request_id || '',
      mode: currentBundle.mode || 'next_prompt',
      candidate_index: idx,
      text: c.text,
      event_type: 'dismissed'
    })
  });
  currentBundle = null;
  renderUnderside();
  refreshDashboard();
}

async function sendDraft(originHint = '') {
  const text = composerEl.value.trim();
  if (!text) return;
  composerEl.value = '';
  currentBundle = null;
  renderUnderside();

  const res = await fetch('/api/send', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ session_id: sessionId, text: text, origin: originHint, messages: messages })
  });
  const payload = await res.json();
  messages.push({ role: 'user', content: text, origin: payload.submitted.origin });
  messages.push({ role: 'assistant', content: payload.assistant_reply });
  renderTranscript();
  currentBundle = payload.prediction;
  candidateIdx = 0;
  renderUnderside();
  refreshDashboard();

  if (cloneMode.active) {
    scheduleCloneTick();
  }
}

function startClone() {
  if (cloneMode.active) return;
  cloneMode.active = true;
  cloneMode.remaining = 3;
  cloneMode.countdownMs = 2400;
  scheduleCloneTick();
}

function stopClone(reason) {
  cloneMode.active = false;
  cloneMode.remaining = 0;
  if (cloneMode.timer) clearInterval(cloneMode.timer);
  cloneMode.timer = null;
  renderUnderside();
}

function scheduleCloneTick() {
  if (cloneMode.timer) clearInterval(cloneMode.timer);
  if (!cloneMode.active || cloneMode.remaining <= 0) {
    stopClone('done');
    return;
  }
  const cands = (currentBundle && currentBundle.candidates) || [];
  if (!cands.length) {
    stopClone('no_candidate');
    return;
  }
  const started = Date.now();
  cloneMode.timer = setInterval(() => {
    if (!cloneMode.active) { clearInterval(cloneMode.timer); return; }
    const left = Math.max(0, 2400 - (Date.now() - started));
    cloneMode.countdownMs = left;
    renderUnderside();
    if (left <= 0) {
      clearInterval(cloneMode.timer);
      cloneMode.remaining -= 1;
      composerEl.value = cands[0].text;
      if (cloneMode.remaining <= 0) cloneMode.active = false;
      sendDraft('clone_mode');
    }
  }, 120);
}

async function seedQuestionScenario() {
  stopClone('scenario');
  messages.push({
    role: 'user',
    content: 'Add caching to the session lookup path in store.py.',
    origin: 'human'
  });
  messages.push({
    role: 'assistant',
    content: 'I inspected store.py. Which caching strategy should we use?\n1. Bounded in-memory LRU cache invalidated on context_revision bump\n2. SQLite prepared statement read pool\n\nLet me know which option you prefer.'
  });
  renderTranscript();
  await requestPrediction('next_prompt');
}

function newChat() {
  stopClone('new_chat');
  sessionId = 'demo-new-' + Math.floor(Math.random() * 9000 + 1000);
  messages = [];
  composerEl.value = '';
  renderTranscript();
  requestPrediction('next_prompt');
}

async function clearFeedback() {
  stopClone('clear');
  await fetch('/api/feedback/clear', { method: 'POST' });
  await refreshDashboard();
  await requestPrediction(composerEl.value.trim().length >= 2 ? 'complete_draft' : 'next_prompt');
}

async function loadMemory() {
  const res = await fetch('/api/memory');
  const data = await res.json();
  document.getElementById('userMdInput').value = data.user_md || '';
  document.getElementById('memoryMdInput').value = data.memory_md || '';
}

async function saveMemory() {
  await fetch('/api/memory', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      user_md: document.getElementById('userMdInput').value,
      memory_md: document.getElementById('memoryMdInput').value
    })
  });
  await requestPrediction(composerEl.value.trim().length >= 2 ? 'complete_draft' : 'next_prompt');
}

async function refreshDashboard() {
  const res = await fetch('/api/dashboard');
  const d = await res.json();
  const m = d.metrics || {};
  const mv = d.movements || {};
  document.getElementById('mPresented').textContent = m.presented || 0;
  document.getElementById('mAccepted').textContent = Math.round((m.acceptance_rate || 0) * 100) + '%';
  document.getElementById('mAcceptedSub').textContent = (m.accepted || 0) + ' into draft';
  document.getElementById('mSubmitted').textContent = Math.round((m.submission_rate || 0) * 100) + '%';
  document.getElementById('mSubmittedSub').textContent = (m.submitted || 0) + ' sent';
  document.getElementById('mEditDist').textContent = (m.avg_edit_distance || 0).toFixed(2);
  document.getElementById('mEditedSub').textContent = (m.edited || 0) + ' edited before send';
  document.getElementById('mLatency').textContent = (m.median_latency_ms || 1) + ' ms';
  document.getElementById('mHumanTurns').textContent = mv.human_turns_learned || 0;

  const edgesEl = document.getElementById('movementEdges');
  const edges = mv.transitions || [];
  if (!edges.length) {
    edgesEl.innerHTML = `<div style="font-size:12px; color:var(--dim);">Type and send 2+ human prompts (for example: implement -> run tests -> commit) to watch WhatNext learn your workflow movements.</div>`;
  } else {
    edgesEl.innerHTML = edges.slice(0, 6).map(e => `
      <div class="edge-row">
        <span>${escapeHtml(e.from)} -&gt; ${escapeHtml(e.to)}</span>
        <span style="color:var(--emerald);">${e.weight}x</span>
      </div>
    `).join('');
  }
}

composerEl.addEventListener('input', () => {
  if (cloneMode.active) stopClone('manual_typing');
  clearTimeout(debounceTimer);
  const val = composerEl.value;
  if (val.trim().length === 0) {
    requestPrediction('next_prompt');
    return;
  }
  if (val.trim().length < 2) {
    currentBundle = null;
    renderUnderside();
    return;
  }
  debounceTimer = setTimeout(() => requestPrediction('complete_draft'), 180);
});

composerEl.addEventListener('keydown', (e) => {
  if (e.key === 'Tab' && currentBundle && currentBundle.candidates && currentBundle.candidates.length > 0) {
    e.preventDefault();
    acceptCandidate(false);
  } else if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendDraft('');
  } else if (e.key === ']' && e.altKey) {
    e.preventDefault();
    cycleCandidate();
  }
});

(async function boot() {
  renderTranscript();
  await loadMemory();
  await requestPrediction('next_prompt');
  await refreshDashboard();
})();
</script>
</body>
</html>
"""


def _synthesize_assistant_reply(user_text: str) -> str:
    intent = store.classify_intent(user_text)
    if intent == "explore":
        return (
            "I mapped the workspace modules (`__init__.py`, `predictor.py`, `store.py`, and `desktop/plugin.js`). "
            "Everything stays inside Hermes's public SDK surfaces. Ready to implement changes or run validation."
        )
    if intent == "implement":
        return (
            "Updated the target module cleanly with monotonic revision checks and profile-scoped SQLite storage. "
            "No core overrides or DOM selectors were introduced."
        )
    if intent == "verify":
        return (
            "Validation suite finished: 16/16 `hermes plugins validate` checks passed (`security scan: safe`, 0 warnings) "
            "and all unit + E2E assertions are green."
        )
    if intent == "review":
        return (
            "Reviewed the diff across Python and Desktop ESM surfaces: revision fencing discards stale drafts, "
            "and machine-accepted prompts are isolated from human movement learning."
        )
    if intent == "ship":
        return (
            "Git working tree verified and catalog entry `whatnext.yaml` is ready to pin to the exact 40-character commit SHA."
        )
    return (
        "Applied your instructions and updated session state. Would you like to run the verification checks next "
        "or refine another surface?"
    )


class PlaygroundHandler(BaseHTTPRequestHandler):
    workspace_cwd: str = str(REPO_ROOT)

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        return json.loads(raw) if raw.strip() else {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = PLAYGROUND_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/dashboard":
            mem = memory_signals.read_user_memory_bundle(enabled=True)
            self._send_json(
                {
                    "metrics": store.get_evaluation_metrics(),
                    "movements": store.get_movement_profile(),
                    "memory": {"has_memory": mem["has_memory"], "highlights": mem["highlights"]},
                }
            )
            return

        if parsed.path == "/api/memory":
            home = memory_signals.resolve_hermes_home()
            mem_dir = home / "memories"
            mem_dir.mkdir(parents=True, exist_ok=True)
            user_md = (mem_dir / "USER.md").read_text(encoding="utf-8") if (mem_dir / "USER.md").is_file() else ""
            memory_md = (mem_dir / "MEMORY.md").read_text(encoding="utf-8") if (mem_dir / "MEMORY.md").is_file() else ""
            self._send_json({"user_md": user_md, "memory_md": memory_md})
            return

        if parsed.path == "/api/state":
            qs = parse_qs(parsed.query)
            sid = (qs.get("session_id") or ["new"])[0]
            self._send_json(
                {
                    "session_id": sid,
                    "next_prompt": store.get_prediction(sid, mode="next_prompt"),
                    "complete_draft": store.get_prediction(sid, mode="complete_draft"),
                }
            )
            return

        self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        data = self._read_json()

        if parsed.path == "/api/predict":
            sid = str(data.get("session_id") or "new")
            draft = str(data.get("draft") or "")
            mode = str(data.get("mode") or ("complete_draft" if draft.strip() else "next_prompt"))
            draft_rev = int(data.get("draft_revision") or 0)
            msgs: List[Dict[str, Any]] = data.get("messages") if isinstance(data.get("messages"), list) else []
            bundle = predictor.generate_prediction_bundle(
                session_id=sid,
                mode=mode,
                draft=draft,
                draft_revision=draft_rev,
                messages=msgs,
                cwd=self.workspace_cwd,
            )
            self._send_json(bundle)
            return

        if parsed.path == "/api/events":
            sid = str(data.get("session_id") or "new")
            ev_type = str(data.get("event_type") or "")
            req_id = str(data.get("request_id") or "")
            mode = str(data.get("mode") or "next_prompt")
            idx = int(data.get("candidate_index") or 0)
            txt = str(data.get("text") or "")
            origin = str(data.get("origin") or "suggestion")
            if ev_type == "accepted":
                store.mark_accepted_candidate(
                    sid, request_id=req_id, mode=mode, accepted_text=txt, candidate_index=idx, origin=origin
                )
            elif ev_type == "dismissed":
                store.record_dismissal(sid, request_id=req_id, mode=mode, candidate_text=txt, candidate_index=idx)
            else:
                store.record_observation_event(
                    ev_type, session_id=sid, request_id=req_id, mode=mode, candidate_index=idx, origin=origin
                )
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/send":
            sid = str(data.get("session_id") or "new")
            txt = str(data.get("text") or "")
            explicit_origin = str(data.get("origin") or "")
            msgs = list(data.get("messages") or [])
            submitted_info = store.observe_submitted_turn(sid, txt, explicit_origin=explicit_origin)
            reply = _synthesize_assistant_reply(txt)
            updated_msgs = msgs + [
                {"role": "user", "content": txt},
                {"role": "assistant", "content": reply},
            ]
            pred = predictor.generate_prediction_bundle(
                session_id=sid,
                mode="next_prompt",
                draft="",
                context_revision=submitted_info["context_revision"],
                messages=updated_msgs,
                cwd=self.workspace_cwd,
            )
            self._send_json(
                {
                    "submitted": submitted_info,
                    "assistant_reply": reply,
                    "prediction": pred,
                }
            )
            return

        if parsed.path == "/api/memory":
            home = memory_signals.resolve_hermes_home()
            mem_dir = home / "memories"
            mem_dir.mkdir(parents=True, exist_ok=True)
            (mem_dir / "USER.md").write_text(str(data.get("user_md") or ""), encoding="utf-8")
            (mem_dir / "MEMORY.md").write_text(str(data.get("memory_md") or ""), encoding="utf-8")
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/feedback/clear":
            self._send_json(store.clear_feedback_memory())
            return

        self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)


def ensure_sandbox_home(custom_home: str = "") -> Path:
    if custom_home:
        home = Path(custom_home).expanduser().resolve()
    elif os.environ.get("HERMES_HOME"):
        home = Path(os.environ["HERMES_HOME"]).expanduser().resolve()
    else:
        home = Path(tempfile.gettempdir()) / "whatnext-playground-hermes-home"
    os.environ["HERMES_HOME"] = str(home)
    os.environ["WHATNEXT_STANDALONE_DEMO"] = "1"
    mem_dir = home / "memories"
    mem_dir.mkdir(parents=True, exist_ok=True)
    if not (mem_dir / "USER.md").is_file():
        (mem_dir / "USER.md").write_text(DEFAULT_USER_MD, encoding="utf-8")
    if not (mem_dir / "MEMORY.md").is_file():
        (mem_dir / "MEMORY.md").write_text(DEFAULT_MEMORY_MD, encoding="utf-8")
    return home


def create_server(host: str = "127.0.0.1", port: int = 8765, hermes_home: str = "") -> ThreadingHTTPServer:
    ensure_sandbox_home(hermes_home)
    return ThreadingHTTPServer((host, port), PlaygroundHandler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the WhatNext interactive composer playground.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port number (default: 8765)")
    parser.add_argument("--hermes-home", default="", help="Optional HERMES_HOME path")
    parsed_args = parser.parse_args()

    srv = create_server(host=parsed_args.host, port=parsed_args.port, hermes_home=parsed_args.hermes_home)
    actual_port = srv.server_address[1]
    print(f"WhatNext Playground running at http://{parsed_args.host}:{actual_port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping WhatNext Playground.")
    finally:
        srv.server_close()
