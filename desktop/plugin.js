/**
 * WhatNext Desktop Plugin for Hermes Agent.
 *
 * Implements Clone-style contextual prompt prediction and movement intelligence
 * inside Hermes Desktop using strictly public `@hermes/plugin-sdk` surfaces:
 * 1. `COMPOSER_AREAS.underside`:
 *    - `next_prompt` mode when composer draft is empty (brand-new empty chat
 *      starters + post-turn ranked follow-up movements grounded in USER.md,
 *      MEMORY.md, and human workflow transitions).
 *    - `complete_draft` mode when typing in the composer (live ghost-suffix
 *      autocomplete with monotonic revision fencing).
 *    - Bounded `Clone Mode` autopilot (explicit Start only, visible countdown,
 *      hard turn/time caps, instant Stop on any manual edit or session switch).
 * 2. `COMPOSER_AREAS.middleware`: send-time funnel tracking (never blocks sends).
 * 3. `COMPOSER_AREAS.contextMenu`: right-click draft completion & refresh.
 * 4. `ROUTES_AREA` (`/whatnext`), `SIDEBAR_NAV_AREA`, `registerSettingsPage`:
 *    full Evaluation Scorecard, Sandbox Tester, and Learned Movement Graph.
 * 5. `STATUSBAR_AREAS.right`, `PALETTE_AREA`, `KEYBINDS_AREA`.
 */

import {
  atom,
  Badge,
  Button,
  cn,
  Codicon,
  COMPOSER_AREAS,
  haptic,
  host,
  Input,
  Kbd,
  KEYBINDS_AREA,
  PALETTE_AREA,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
  STATUSBAR_AREAS,
  Tip,
  useQuery,
  useQueryClient,
  useValue,
} from '@hermes/plugin-sdk'
import { useCallback, useEffect, useRef, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const PLUGIN_ID = 'whatnext'
const ACCEPT_COMBO = 'mod+shift+space'
const CYCLE_COMBO = 'alt+]'
const DISMISS_COMBO = 'mod+shift+backspace'
const STOP_CLONE_COMBO = 'mod+shift+.'

const DRAFT_POLL_MS = 380
const DRAFT_DEBOUNCE_MS = 340
const CLONE_REVIEW_MS = 2500

let pluginCtx = null

const $selectedCandidateIdx = atom(0)
const $dismissedKeys = atom({})
const $liveDraftBundle = atom(null)
const $pushedNextBundle = atom(null)
const $cloneMode = atom({
  active: false,
  sessionId: null,
  remainingTurns: 0,
  expiresAtMs: 0,
  countdownMs: 0,
  candidateText: '',
  requestId: '',
})

let draftRevisionCounter = 0
let presentedTracker = new Set()
let activeVisibleCandidate = null
let pendingCloneOriginSend = false

// ── Transport helpers (Gateway RPC first, scoped REST fallback) ──────────

async function callBackendPredict(payload) {
  if (host && typeof host.request === 'function') {
    try {
      const rpcRes = await host.request('plugin.whatnext.predict', payload)
      if (rpcRes && typeof rpcRes === 'object') return rpcRes
    } catch {
      // Fall back to plugin_api.py REST route
    }
  }
  if (pluginCtx && typeof pluginCtx.rest === 'function') {
    return await pluginCtx.rest('/predict', {
      method: 'POST',
      body: JSON.stringify(payload),
      headers: { 'Content-Type': 'application/json' },
    })
  }
  return { candidates: [] }
}

async function fetchSessionState(sessionKey) {
  if (host && typeof host.request === 'function') {
    try {
      const rpcRes = await host.request('plugin.whatnext.state', { session_id: sessionKey })
      if (rpcRes && typeof rpcRes === 'object') return rpcRes
    } catch {
      // Fall back to REST
    }
  }
  if (pluginCtx && typeof pluginCtx.rest === 'function') {
    return await pluginCtx.rest(`/state?session_id=${encodeURIComponent(sessionKey)}`)
  }
  return null
}

async function fetchDashboardData() {
  if (host && typeof host.request === 'function') {
    try {
      const rpcRes = await host.request('plugin.whatnext.dashboard', {})
      if (rpcRes && typeof rpcRes === 'object') return rpcRes
    } catch {
      // Fall back to REST
    }
  }
  if (pluginCtx && typeof pluginCtx.rest === 'function') {
    return await pluginCtx.rest('/dashboard')
  }
  return null
}

function postObservationEvent(payload) {
  if (host && typeof host.request === 'function') {
    host.request('plugin.whatnext.event', payload).catch(() => {
      if (pluginCtx) {
        pluginCtx
          .rest('/events', {
            method: 'POST',
            body: JSON.stringify(payload),
            headers: { 'Content-Type': 'application/json' },
          })
          .catch(() => {})
      }
    })
    return
  }
  if (pluginCtx) {
    pluginCtx
      .rest('/events', {
        method: 'POST',
        body: JSON.stringify(payload),
        headers: { 'Content-Type': 'application/json' },
      })
      .catch(() => {})
  }
}

async function clearFeedbackOnBackend() {
  if (host && typeof host.request === 'function') {
    try {
      return await host.request('plugin.whatnext.clear_feedback', {})
    } catch {
      // Fall back
    }
  }
  if (pluginCtx) {
    return await pluginCtx.rest('/feedback/clear', { method: 'POST' })
  }
  return { ok: false }
}

// ── Composer helpers (100% SDK `host.composer`, zero DOM access) ─────────

function resolveComposerTarget(storedId) {
  if (!storedId || storedId === 'new') return 'new'
  return storedId
}

async function readLiveComposerDraft(storedId) {
  const composer = host && host.composer
  if (!composer || typeof composer.getDraft !== 'function') return ''
  try {
    const target = storedId ? storedId : null
    const val = await composer.getDraft(target)
    return typeof val === 'string' ? val : ''
  } catch {
    return ''
  }
}

async function applyCandidateToComposer(storedId, text, mode) {
  const composer = host && host.composer
  if (!composer) return false
  try {
    if (mode === 'complete_draft' && typeof composer.setDraft === 'function') {
      const addr = storedId ? resolveComposerTarget(storedId) : null
      if (await composer.setDraft(addr, text)) {
        if (typeof composer.focus === 'function') composer.focus(addr)
        return true
      }
      if (await composer.setDraft(null, text)) {
        if (typeof composer.focus === 'function') composer.focus(null)
        return true
      }
    }
    if (typeof composer.insertText === 'function') {
      if (await composer.insertText(null, text, { mode: 'block' })) {
        if (typeof composer.focus === 'function') composer.focus(null)
        return true
      }
    }
    if (typeof composer.setDraft === 'function') {
      if (await composer.setDraft(null, text)) return true
    }
  } catch {
    return false
  }
  return false
}

function submitCandidateNow(storedId, text) {
  const composer = host && host.composer
  if (!composer || typeof composer.submit !== 'function') return false
  try {
    const addr = storedId ? resolveComposerTarget(storedId) : null
    return composer.submit(addr, text) === true || composer.submit(null, text) === true
  } catch {
    return false
  }
}

// ── Clone Mode Lifecycle (Bounded Delegation) ────────────────────────────

function stopCloneMode(reason = 'manual') {
  const cur = $cloneMode.get()
  if (!cur.active) return
  $cloneMode.set({
    active: false,
    sessionId: null,
    remainingTurns: 0,
    expiresAtMs: 0,
    countdownMs: 0,
    candidateText: '',
    requestId: '',
  })
  postObservationEvent({
    session_id: cur.sessionId || 'new',
    event_type: 'clone_mode_stop',
    mode: 'clone_mode',
    detail: reason,
  })
}

function startCloneMode(sessionId, maxTurns = 3, maxSeconds = 300) {
  if (!sessionId || sessionId === 'new') {
    host.notify({ kind: 'info', message: 'Send one starter prompt first before starting Clone Mode.' })
    return
  }
  const safeTurns = Math.max(1, Math.min(10, Number(maxTurns) || 3))
  const safeSec = Math.max(30, Math.min(3600, Number(maxSeconds) || 300))
  $cloneMode.set({
    active: true,
    sessionId,
    remainingTurns: safeTurns,
    expiresAtMs: Date.now() + safeSec * 1000,
    countdownMs: CLONE_REVIEW_MS,
    candidateText: '',
    requestId: '',
  })
  haptic('tap')
  postObservationEvent({
    session_id: sessionId,
    event_type: 'clone_mode_start',
    mode: 'clone_mode',
    detail: `turns=${safeTurns},sec=${safeSec}`,
  })
}

// ── Global Action Invokers (for Keybinds, Palette, Pill Clicks) ──────────

async function acceptActiveCandidate({ sendImmediately = false } = {}) {
  const cur = activeVisibleCandidate
  if (!cur || host.state.busy.get()) return false
  const { sessionKey, requestId, mode, candidate, index } = cur
  if (!candidate || !candidate.text) return false

  haptic('tap')
  const dismissKey = `${sessionKey}:${mode}:${requestId}`
  $dismissedKeys.set({ ...$dismissedKeys.get(), [dismissKey]: true })
  $liveDraftBundle.set(null)

  postObservationEvent({
    session_id: sessionKey,
    request_id: requestId,
    mode,
    candidate_index: index,
    text: candidate.text,
    origin: 'suggestion',
    event_type: 'accepted',
  })

  if (sendImmediately && submitCandidateNow(sessionKey, candidate.text)) {
    return true
  }

  const ok = await applyCandidateToComposer(sessionKey, candidate.text, mode)
  if (ok) return true

  if (pluginCtx && pluginCtx.os && typeof pluginCtx.os.writeClipboard === 'function') {
    try {
      await pluginCtx.os.writeClipboard(candidate.text)
      host.notify({ kind: 'info', message: 'Copied WhatNext suggestion to clipboard' })
      return true
    } catch {
      // ignore
    }
  }
  return false
}

function cycleActiveCandidate() {
  const cur = activeVisibleCandidate
  if (!cur || !cur.total || cur.total <= 1) return
  haptic('tap')
  $selectedCandidateIdx.set((cur.index + 1) % cur.total)
}

function dismissActiveCandidate() {
  const cur = activeVisibleCandidate
  if (!cur) return
  haptic('tap')
  const dismissKey = `${cur.sessionKey}:${cur.mode}:${cur.requestId}`
  $dismissedKeys.set({ ...$dismissedKeys.get(), [dismissKey]: true })
  if (cur.mode === 'complete_draft') {
    $liveDraftBundle.set(null)
  }
  postObservationEvent({
    session_id: cur.sessionKey,
    request_id: cur.requestId,
    mode: cur.mode,
    candidate_index: cur.index,
    text: cur.candidate ? cur.candidate.text : '',
    event_type: 'dismissed',
  })
}

// ── Underside Composer Strip Component ───────────────────────────────────

const INTENT_TONE = {
  answer: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10',
  verify: 'text-sky-400 border-sky-500/30 bg-sky-500/10',
  ship: 'text-violet-400 border-violet-500/30 bg-violet-500/10',
  debug: 'text-amber-400 border-amber-500/30 bg-amber-500/10',
  review: 'text-cyan-400 border-cyan-500/30 bg-cyan-500/10',
  implement: 'text-indigo-400 border-indigo-500/30 bg-indigo-500/10',
  refactor: 'text-teal-400 border-teal-500/30 bg-teal-500/10',
  explore: 'text-slate-300 border-slate-400/30 bg-slate-500/10',
}

function WhatNextUndersideStrip() {
  const focusedStoredId = useValue(host.state.focusedStoredSessionId)
  const busy = useValue(host.state.busy)
  const cwd = useValue(host.state.cwd)
  const selectedIdx = useValue($selectedCandidateIdx)
  const dismissedMap = useValue($dismissedKeys)
  const liveDraftBundle = useValue($liveDraftBundle)
  const pushedNext = useValue($pushedNextBundle)
  const cloneMode = useValue($cloneMode)

  const sessionKey = focusedStoredId || 'new'
  const [liveDraft, setLiveDraft] = useState('')
  const lastDraftRef = useRef('')
  const debounceTimerRef = useRef(null)
  const starterFetchedRef = useRef({})

  // Stop Clone Mode immediately if user switches sessions
  useEffect(() => {
    if (cloneMode.active && cloneMode.sessionId !== focusedStoredId) {
      stopCloneMode('session_switch')
    }
    $selectedCandidateIdx.set(0)
    $liveDraftBundle.set(null)
  }, [focusedStoredId])

  // Fetch session state via React Query
  const stateQuery = useQuery({
    queryKey: [PLUGIN_ID, 'state', sessionKey],
    queryFn: () => fetchSessionState(sessionKey),
    enabled: !busy,
    refetchInterval: busy ? false : 3500,
    staleTime: 1500,
  })

  // Generate empty-chat starters once per empty session
  useEffect(() => {
    if (busy) return
    const key = `${sessionKey}:${cwd || ''}`
    const existingCands =
      stateQuery.data &&
      stateQuery.data.next_prompt &&
      Array.isArray(stateQuery.data.next_prompt.candidates)
        ? stateQuery.data.next_prompt.candidates
        : []
    if (sessionKey === 'new' && existingCands.length === 0 && !starterFetchedRef.current[key]) {
      starterFetchedRef.current[key] = true
      callBackendPredict({
        session_id: 'new',
        mode: 'next_prompt',
        draft: '',
        cwd: cwd || '',
      })
        .then(res => {
          if (res && Array.isArray(res.candidates) && res.candidates.length > 0) {
            $pushedNextBundle.set(res)
          }
        })
        .catch(() => {})
    }
  }, [sessionKey, cwd, busy, stateQuery.data])

  // Poll live composer draft via host.composer.getDraft & trigger complete_draft
  useEffect(() => {
    if (busy) {
      setLiveDraft('')
      lastDraftRef.current = ''
      return
    }
    let cancelled = false
    const tick = async () => {
      const currentText = await readLiveComposerDraft(focusedStoredId)
      if (cancelled) return
      if (currentText !== lastDraftRef.current) {
        const previous = lastDraftRef.current
        lastDraftRef.current = currentText
        setLiveDraft(currentText)

        // Any manual typing immediately halts Clone Mode (Clone Mode spec)
        if (cloneMode.active && currentText.trim() !== '' && previous !== currentText) {
          stopCloneMode('manual_draft_edit')
        }

        if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current)
        const trimmed = currentText.trim()
        if (trimmed.length < 3) {
          $liveDraftBundle.set(null)
          return
        }

        const nextRev = ++draftRevisionCounter
        debounceTimerRef.current = setTimeout(() => {
          callBackendPredict({
            session_id: sessionKey,
            mode: 'complete_draft',
            draft: currentText,
            draft_revision: nextRev,
            cwd: cwd || '',
          })
            .then(bundle => {
              if (cancelled) return
              if (nextRev < draftRevisionCounter) return
              if (lastDraftRef.current !== currentText) return
              if (bundle && Array.isArray(bundle.candidates) && bundle.candidates.length > 0) {
                $liveDraftBundle.set(bundle)
              } else {
                $liveDraftBundle.set(null)
              }
            })
            .catch(() => {})
        }, DRAFT_DEBOUNCE_MS)
      }
    }

    const timer = setInterval(tick, DRAFT_POLL_MS)
    tick()
    return () => {
      cancelled = true
      clearInterval(timer)
      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current)
    }
  }, [focusedStoredId, sessionKey, busy, cwd, cloneMode.active])

  // Resolve active bundle: complete_draft when draft has text, else next_prompt
  const hasDraft = liveDraft.trim().length >= 3
  let activeBundle = null
  if (hasDraft) {
    if (
      liveDraftBundle &&
      liveDraftBundle.session_id === sessionKey &&
      Array.isArray(liveDraftBundle.candidates) &&
      liveDraftBundle.candidates.length > 0
    ) {
      activeBundle = liveDraftBundle
    }
  } else {
    const pushedMatches =
      pushedNext &&
      pushedNext.session_id === sessionKey &&
      Array.isArray(pushedNext.candidates) &&
      pushedNext.candidates.length > 0
        ? pushedNext
        : null
    const queried =
      stateQuery.data &&
      stateQuery.data.next_prompt &&
      Array.isArray(stateQuery.data.next_prompt.candidates) &&
      stateQuery.data.next_prompt.candidates.length > 0
        ? stateQuery.data.next_prompt
        : null
    if (pushedMatches && queried) {
      activeBundle =
        (pushedMatches.created_at || 0) >= (queried.created_at || 0) ? pushedMatches : queried
    } else {
      activeBundle = pushedMatches || queried
    }
  }

  const mode = hasDraft ? 'complete_draft' : 'next_prompt'
  const requestId = (activeBundle && activeBundle.request_id) || ''
  const dismissKey = `${sessionKey}:${mode}:${requestId}`
  const candidates =
    activeBundle && !dismissedMap[dismissKey] && Array.isArray(activeBundle.candidates)
      ? activeBundle.candidates
      : []

  const safeIdx = candidates.length > 0 ? selectedIdx % candidates.length : 0
  const candidate = candidates[safeIdx] || null

  // Record `presented` once per unique request_id
  useEffect(() => {
    if (!candidate || !requestId || busy) return
    if (presentedTracker.has(requestId)) return
    presentedTracker.add(requestId)
    if (presentedTracker.size > 400) {
      const first = presentedTracker.values().next().value
      presentedTracker.delete(first)
    }
    postObservationEvent({
      session_id: sessionKey,
      request_id: requestId,
      mode,
      candidate_index: safeIdx,
      event_type: 'presented',
    })
  }, [requestId, candidate, busy, sessionKey, mode, safeIdx])

  //Bounded Clone Mode countdown ticker
  useEffect(() => {
    if (!cloneMode.active || busy) return
    if (Date.now() >= cloneMode.expiresAtMs || cloneMode.remainingTurns <= 0) {
      stopCloneMode('bounds_reached')
      return
    }
    if (!candidate || hasDraft) return

    const startedAt = Date.now()
    const candText = candidate.text
    const interval = setInterval(() => {
      const cur = $cloneMode.get()
      if (!cur.active) {
        clearInterval(interval)
        return
      }
      if (Date.now() >= cur.expiresAtMs || cur.remainingTurns <= 0) {
        clearInterval(interval)
        stopCloneMode('bounds_reached')
        return
      }
      const elapsed = Date.now() - startedAt
      const left = Math.max(0, CLONE_REVIEW_MS - elapsed)
      $cloneMode.set({
        ...cur,
        countdownMs: left,
        candidateText: candText,
        requestId,
      })
      if (left <= 0) {
        clearInterval(interval)
        pendingCloneOriginSend = true
        const sent = submitCandidateNow(cur.sessionId, candText)
        if (!sent) {
          pendingCloneOriginSend = false
          stopCloneMode('submit_rejected')
          return
        }
        const nextRemaining = cur.remainingTurns - 1
        const nextKey = `${sessionKey}:next_prompt:${requestId}`
        $dismissedKeys.set({ ...$dismissedKeys.get(), [nextKey]: true })
        if (nextRemaining <= 0) {
          stopCloneMode('turns_completed')
        } else {
          $cloneMode.set({
            ...cur,
            remainingTurns: nextRemaining,
            countdownMs: CLONE_REVIEW_MS,
            candidateText: '',
          })
        }
      }
    }, 120)

    return () => clearInterval(interval)
  }, [cloneMode.active, busy, candidate, hasDraft, requestId, sessionKey])

  if (busy) {
    activeVisibleCandidate = null
    if (!cloneMode.active) return null
    return jsx('div', {
      className: 'flex items-center justify-between gap-2 pt-1.5 px-1 text-xs',
      children: jsxs('div', {
        className:
          'inline-flex items-center gap-2 rounded-full border border-amber-500/30 bg-amber-500/10 px-2.5 py-0.5 text-[0.72rem] text-amber-300',
        children: [
          jsx(Codicon, { name: 'pulse', className: 'text-[0.75rem]' }),
          jsx('span', {
            children: `Clone Mode active · ${cloneMode.remainingTurns} turn(s) remaining`,
          }),
          jsx('button', {
            type: 'button',
            onClick: () => stopCloneMode('user_stop'),
            className:
              'ml-1 rounded px-1.5 py-0.5 font-medium text-rose-300 hover:bg-rose-500/20',
            children: 'Stop',
          }),
        ],
      }),
    })
  }

  if (!candidate) {
    activeVisibleCandidate = null
    const errKind =
      (activeBundle && activeBundle.error_kind) ||
      (stateQuery.data && stateQuery.data.next_prompt && stateQuery.data.next_prompt.error_kind) ||
      ''
    if (!errKind || errKind === 'disabled') return null
    const errLabel =
      errKind === 'auth'
        ? 'WhatNext: provider authentication needed'
        : errKind === 'rate_limit'
          ? 'WhatNext: rate limited, cooling down'
          : errKind === 'timeout'
            ? 'WhatNext: prediction timed out'
            : 'WhatNext: prediction unavailable'
    return jsx('div', {
      className: 'pt-1.5 px-1 text-[0.72rem] text-(--ui-text-tertiary)',
      children: errLabel,
    })
  }

  activeVisibleCandidate = {
    sessionKey,
    requestId,
    mode,
    candidate,
    index: safeIdx,
    total: candidates.length,
  }

  const intent = candidate.intent || 'move'
  const toneClass = INTENT_TONE[intent] || INTENT_TONE.explore
  const modeLabel =
    cloneMode.active
      ? `CLONE ${(cloneMode.countdownMs / 1000).toFixed(1)}s`
      : mode === 'complete_draft'
        ? 'COMPLETE'
        : sessionKey === 'new'
          ? 'STARTER'
          : 'NEXT MOVE'

  const maxTurnsCfg =
    (stateQuery.data && stateQuery.data.settings && stateQuery.data.settings.clone_mode_max_turns) || 3
  const maxSecCfg =
    (stateQuery.data && stateQuery.data.settings && stateQuery.data.settings.clone_mode_max_seconds) || 300

  return jsxs('div', {
    className: 'flex w-full items-center justify-between gap-2 pt-1.5 px-0.5 select-none',
    children: [
      jsxs('div', {
        className: cn(
          'group flex min-w-0 flex-1 items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors',
          cloneMode.active
            ? 'border-amber-500/40 bg-amber-500/10'
            : 'border-(--ui-border-subtle) bg-(--ui-bg-elevated)/80 hover:border-(--ui-border-strong)'
        ),
        children: [
          jsx('span', {
            className: cn(
              'shrink-0 rounded-full border px-1.5 py-0.2 text-[0.625rem] font-semibold tracking-wide uppercase',
              toneClass
            ),
            children: `${modeLabel} · ${intent}`,
          }),
          jsx(Tip, {
            label: `${candidate.reason || 'Predicted from context & memory'} · Click or ⌘⇧Space to insert · ⌘Click to send`,
            children: jsx('button', {
              type: 'button',
              onClick: ev => {
                const immediate = Boolean(ev && (ev.metaKey || ev.ctrlKey))
                void acceptActiveCandidate({ sendImmediately: immediate })
              },
              className:
                'min-w-0 flex-1 truncate text-left text-[0.76rem] text-(--ui-text-secondary) hover:text-(--ui-text-primary)',
              children:
                mode === 'complete_draft' && candidate.completion_suffix
                  ? jsxs('span', {
                      children: [
                        jsx('span', {
                          className: 'text-(--ui-text-tertiary)',
                          children: liveDraft,
                        }),
                        jsx('span', {
                          className: 'font-medium text-(--ui-text-primary) underline decoration-dotted underline-offset-2',
                          children: candidate.completion_suffix,
                        }),
                      ],
                    })
                  : candidate.text,
            }),
          }),
          candidates.length > 1
            ? jsx(Tip, {
                label: 'Cycle ranked movements (⌥])',
                children: jsx('button', {
                  type: 'button',
                  onClick: cycleActiveCandidate,
                  className:
                    'shrink-0 rounded px-1 py-0.5 text-[0.65rem] tabular-nums text-(--ui-text-tertiary) hover:bg-(--ui-bg-hover) hover:text-(--ui-text-primary)',
                  children: `${safeIdx + 1}/${candidates.length}`,
                }),
              })
            : null,
          jsx(Kbd, {
            className: 'hidden sm:inline-flex shrink-0 text-[0.62rem] opacity-65',
            children: '⌘⇧Space',
          }),
          jsx(Tip, {
            label: 'Dismiss suggestion',
            children: jsx('button', {
              type: 'button',
              onClick: dismissActiveCandidate,
              className:
                'shrink-0 rounded p-0.5 text-(--ui-text-tertiary) hover:bg-(--ui-bg-hover) hover:text-(--ui-text-primary)',
              children: jsx(Codicon, { name: 'close', className: 'text-[0.7rem]' }),
            }),
          }),
        ],
      }),
      sessionKey !== 'new' && !hasDraft
        ? cloneMode.active
          ? jsx('button', {
              type: 'button',
              onClick: () => stopCloneMode('user_stop'),
              className:
                'shrink-0 rounded-full border border-rose-500/40 bg-rose-500/15 px-2.5 py-1 text-[0.68rem] font-medium text-rose-300 hover:bg-rose-500/25',
              children: `Stop (${cloneMode.remainingTurns})`,
            })
          : jsx(Tip, {
              label: `Start bounded Clone Mode (${maxTurnsCfg} turns max · visible 2.5s review before each send)`,
              children: jsx('button', {
                type: 'button',
                onClick: () => startCloneMode(sessionKey, maxTurnsCfg, maxSecCfg),
                className:
                  'shrink-0 rounded-full border border-(--ui-border-subtle) bg-(--ui-bg-elevated)/70 px-2 py-1 text-[0.68rem] text-(--ui-text-tertiary) hover:border-(--ui-border-strong) hover:text-(--ui-text-primary)',
                children: 'Clone Mode',
              }),
            })
        : null,
    ],
  })
}

// ── Full Evaluation & Movement Studio Page (/whatnext & Settings) ────────

function MetricTile({ label, value, detail }) {
  return jsxs('div', {
    className:
      'flex flex-col gap-1 rounded-xl border border-(--ui-border-subtle) bg-(--ui-bg-elevated)/60 p-3.5',
    children: [
      jsx('div', {
        className: 'text-[0.68rem] font-medium uppercase tracking-wider text-(--ui-text-tertiary)',
        children: label,
      }),
      jsx('div', {
        className: 'text-xl font-semibold tabular-nums text-(--ui-text-primary)',
        children: value,
      }),
      detail
        ? jsx('div', {
            className: 'text-[0.72rem] text-(--ui-text-tertiary)',
            children: detail,
          })
        : null,
    ],
  })
}

function WhatNextStudioPage() {
  const qc = useQueryClient()
  const cwd = useValue(host.state.cwd)
  const [sandboxDraft, setSandboxDraft] = useState('run ')
  const [sandboxOut, setSandboxOut] = useState(null)
  const [testing, setTesting] = useState(false)

  const dashQuery = useQuery({
    queryKey: [PLUGIN_ID, 'dashboard'],
    queryFn: fetchDashboardData,
    refetchInterval: 6000,
  })

  const runSandbox = useCallback(
    async modeOverride => {
      setTesting(true)
      try {
        const res = await callBackendPredict({
          session_id: 'studio-sandbox',
          mode: modeOverride,
          draft: modeOverride === 'complete_draft' ? sandboxDraft : '',
          draft_revision: ++draftRevisionCounter,
          cwd: cwd || '',
        })
        setSandboxOut(res)
        void qc.invalidateQueries({ queryKey: [PLUGIN_ID, 'dashboard'] })
      } finally {
        setTesting(false)
      }
    },
    [sandboxDraft, cwd, qc]
  )

  const handleClear = useCallback(async () => {
    await clearFeedbackOnBackend()
    haptic('tap')
    host.notify({ kind: 'info', message: 'Cleared WhatNext feedback and movement memory.' })
    void qc.invalidateQueries({ queryKey: [PLUGIN_ID, 'dashboard'] })
  }, [qc])

  const data = dashQuery.data || {}
  const metrics = data.metrics || {}
  const movements = data.movements || {}
  const memory = data.memory || {}
  const engine = data.engine || {}

  return jsxs('div', {
    className: 'mx-auto flex h-full max-w-5xl flex-col gap-5 overflow-y-auto p-6 text-sm',
    children: [
      jsxs('div', {
        className: 'flex flex-wrap items-start justify-between gap-3',
        children: [
          jsxs('div', {
            className: 'flex flex-col gap-1',
            children: [
              jsxs('div', {
                className: 'flex items-center gap-2',
                children: [
                  jsx('h1', {
                    className: 'text-lg font-semibold text-(--ui-text-primary)',
                    children: 'WhatNext Studio',
                  }),
                  jsx(Badge, {
                    children: engine.clone_api_configured
                      ? 'Clone API + Hermes Memory'
                      : `Hermes LLM (${engine.auxiliary_task || 'active'})`,
                  }),
                ],
              }),
              jsx('p', {
                className: 'text-xs text-(--ui-text-tertiary)',
                children:
                  'Memory-grounded prompt suggestions, live draft completion, human movement learning, and Clone evaluation funnel.',
              }),
            ],
          }),
          jsx(Button, {
            variant: 'outline',
            size: 'sm',
            onClick: handleClear,
            children: 'Clear Feedback Memory',
          }),
        ],
      }),

      jsxs('div', {
        className: 'grid grid-cols-2 gap-3 sm:grid-cols-5',
        children: [
          jsx(MetricTile, {
            label: 'Presented',
            value: String(metrics.presented || 0),
            detail: 'Suggestions shown',
          }),
          jsx(MetricTile, {
            label: 'Acceptance',
            value: `${Math.round((metrics.acceptance_rate || 0) * 100)}%`,
            detail: `${metrics.accepted || 0} placed in draft`,
          }),
          jsx(MetricTile, {
            label: 'Submitted',
            value: `${Math.round((metrics.submission_rate || 0) * 100)}%`,
            detail: `${metrics.submitted || 0} sent after review`,
          }),
          jsx(MetricTile, {
            label: 'Edit Distance',
            value: String(metrics.avg_edit_distance || 0),
            detail: `${metrics.edited || 0} refined before send`,
          }),
          jsx(MetricTile, {
            label: 'Median Latency',
            value: `${metrics.median_latency_ms || 0} ms`,
            detail: 'Prediction round-trip',
          }),
        ],
      }),

      jsxs('div', {
        className:
          'flex flex-col gap-3 rounded-xl border border-(--ui-border-subtle) bg-(--ui-bg-elevated)/50 p-4',
        children: [
          jsx('div', {
            className: 'text-xs font-semibold uppercase tracking-wider text-(--ui-text-secondary)',
            children: 'Interactive Composer Prediction Sandbox',
          }),
          jsxs('div', {
            className: 'flex flex-wrap items-center gap-2',
            children: [
              jsx('div', {
                className: 'min-w-[240px] flex-1',
                children: jsx(Input, {
                  value: sandboxDraft,
                  onChange: e => setSandboxDraft(e.target.value),
                  placeholder: 'Type a partial prompt to test complete_draft...',
                }),
              }),
              jsx(Button, {
                size: 'sm',
                disabled: testing,
                onClick: () => void runSandbox('complete_draft'),
                children: 'Test Draft Completion',
              }),
              jsx(Button, {
                size: 'sm',
                variant: 'outline',
                disabled: testing,
                onClick: () => void runSandbox('next_prompt'),
                children: 'Predict Empty-Draft Moves',
              }),
            ],
          }),
          sandboxOut && Array.isArray(sandboxOut.candidates) && sandboxOut.candidates.length > 0
            ? jsx('div', {
                className: 'flex flex-col gap-1.5 pt-1',
                children: sandboxOut.candidates.map((cand, idx) =>
                  jsxs(
                    'div',
                    {
                      className:
                        'flex items-center justify-between gap-2 rounded-lg border border-(--ui-border-subtle) bg-(--ui-bg-base)/60 px-3 py-2 text-xs',
                      children: [
                        jsxs('div', {
                          className: 'min-w-0 flex-1',
                          children: [
                            jsx('span', {
                              className: 'mr-2 font-semibold uppercase text-(--ui-text-tertiary)',
                              children: `[${cand.intent || 'move'}]`,
                            }),
                            jsx('span', {
                              className: 'text-(--ui-text-primary)',
                              children: cand.text,
                            }),
                            cand.reason
                              ? jsx('span', {
                                  className: 'ml-2 text-[0.7rem] text-(--ui-text-tertiary)',
                                  children: `(${cand.reason})`,
                                })
                              : null,
                          ],
                        }),
                        jsx('span', {
                          className: 'shrink-0 tabular-nums text-[0.68rem] text-(--ui-text-tertiary)',
                          children: `${sandboxOut.backend || 'engine'} · ${sandboxOut.latency_ms || 0}ms`,
                        }),
                      ],
                    },
                    idx
                  )
                ),
              })
            : null,
        ],
      }),

      jsxs('div', {
        className: 'grid grid-cols-1 gap-4 md:grid-cols-2',
        children: [
          jsxs('div', {
            className:
              'flex flex-col gap-2 rounded-xl border border-(--ui-border-subtle) bg-(--ui-bg-elevated)/50 p-4',
            children: [
              jsx('div', {
                className: 'text-xs font-semibold uppercase tracking-wider text-(--ui-text-secondary)',
                children: `Learned Human Movement Graph (${movements.human_turns_learned || 0} human turns)`,
              }),
              jsx('p', {
                className: 'text-[0.72rem] text-(--ui-text-tertiary)',
                children:
                  'Only independently human-authored prompts update transitions; accepted suggestions never pollute preference learning.',
              }),
              Array.isArray(movements.transitions) && movements.transitions.length > 0
                ? jsx('div', {
                    className: 'flex flex-col gap-1.5 pt-1',
                    children: movements.transitions.map((edge, idx) =>
                      jsxs(
                        'div',
                        {
                          className:
                            'flex items-center justify-between rounded-md bg-(--ui-bg-base)/50 px-2.5 py-1.5 text-xs',
                          children: [
                            jsx('span', {
                              className: 'font-mono text-(--ui-text-primary)',
                              children: `${edge.from} -> ${edge.to}`,
                            }),
                            jsx('span', {
                              className: 'tabular-nums text-(--ui-text-tertiary)',
                              children: `${edge.weight}x`,
                            }),
                          ],
                        },
                        idx
                      )
                    ),
                  })
                : jsx('div', {
                    className: 'py-3 text-xs text-(--ui-text-tertiary)',
                    children: 'Send a few prompts to build your personalized workflow transition matrix.',
                  }),
            ],
          }),

          jsxs('div', {
            className:
              'flex flex-col gap-2 rounded-xl border border-(--ui-border-subtle) bg-(--ui-bg-elevated)/50 p-4',
            children: [
              jsx('div', {
                className: 'text-xs font-semibold uppercase tracking-wider text-(--ui-text-secondary)',
                children: 'Active Hermes Memory Signals (USER.md & MEMORY.md)',
              }),
              Array.isArray(memory.highlights) && memory.highlights.length > 0
                ? jsx('div', {
                    className: 'flex flex-col gap-1.5 pt-1',
                    children: memory.highlights.map((item, idx) =>
                      jsx(
                        'div',
                        {
                          className:
                            'rounded-md bg-(--ui-bg-base)/50 px-2.5 py-1.5 text-xs text-(--ui-text-secondary)',
                          children: `${idx + 1}. ${item}`,
                        },
                        idx
                      )
                    ),
                  })
                : jsx('div', {
                    className: 'py-3 text-xs text-(--ui-text-tertiary)',
                    children:
                      'No USER.md or MEMORY.md entries detected yet. As Hermes remembers your preferences, WhatNext automatically conditions predictions on them.',
                  }),
            ],
          }),
        ],
      }),
    ],
  })
}

// ── Statusbar Chip ───────────────────────────────────────────────────────

function WhatNextStatusChip() {
  const cloneMode = useValue($cloneMode)
  return jsx(Tip, {
    label: cloneMode.active
      ? 'Clone Mode active: click to open WhatNext Studio'
      : 'Open WhatNext Prompt Prediction & Movement Studio',
    children: jsxs('button', {
      type: 'button',
      onClick: () => host.navigate('/whatnext'),
      className: cn(
        'inline-flex items-center gap-1 px-1.5 text-[0.6875rem] transition-colors',
        cloneMode.active
          ? 'text-amber-300'
          : 'text-(--ui-text-tertiary) hover:text-(--ui-text-primary)'
      ),
      children: [
        jsx(Codicon, { name: 'sparkle', className: 'text-[0.72rem]' }),
        jsx('span', {
          children: cloneMode.active ? `whatnext · clone (${cloneMode.remainingTurns})` : 'whatnext',
        }),
      ],
    }),
  })
}

// ── Plugin Export ────────────────────────────────────────────────────────

export default {
  id: PLUGIN_ID,
  name: 'WhatNext',

  register(ctx) {
    pluginCtx = ctx

    // Subscribe to instant gateway push after assistant turns
    ctx.onEvent('plugin.whatnext.prediction', event => {
      const payload = event && (event.payload || event.data || event)
      if (payload && Array.isArray(payload.candidates)) {
        $pushedNextBundle.set(payload)
      }
    })

    ctx.onDispose(() => {
      stopCloneMode('plugin_unload')
      pluginCtx = null
      activeVisibleCandidate = null
    })

    ctx.registerSettingsPage({
      id: 'studio',
      title: 'WhatNext',
      icon: 'sparkle',
      render: () => jsx(WhatNextStudioPage, {}),
    })

    ctx.registerMany([
      {
        id: 'underside',
        area: COMPOSER_AREAS.underside,
        order: 20,
        render: () => jsx(WhatNextUndersideStrip, {}),
      },
      {
        id: 'send-observer',
        area: COMPOSER_AREAS.middleware,
        order: 90,
        data: {
          handler: draft => {
            if (draft && typeof draft.text === 'string' && draft.text.trim()) {
              const sid = host.state.focusedStoredSessionId.get() || 'new'
              if (pendingCloneOriginSend) {
                pendingCloneOriginSend = false
                postObservationEvent({
                  session_id: sid,
                  event_type: 'submitted_direct',
                  mode: 'clone_mode',
                  origin: 'clone_mode',
                  text: draft.text,
                })
              }
            }
            return draft
          },
        },
      },
      {
        id: 'context-menu',
        area: COMPOSER_AREAS.contextMenu,
        data: {
          items: ({ selection }) => [
            {
              label: selection
                ? 'WhatNext: Complete from selection'
                : 'WhatNext: Predict next movements',
              onSelect: () => {
                const sid = host.state.focusedStoredSessionId.get() || 'new'
                const cwd = host.state.cwd.get() || ''
                void readLiveComposerDraft(sid).then(draft => {
                  const hasText = draft.trim().length >= 3
                  return callBackendPredict({
                    session_id: sid,
                    mode: hasText ? 'complete_draft' : 'next_prompt',
                    draft: hasText ? draft : '',
                    draft_revision: ++draftRevisionCounter,
                    cwd,
                  }).then(bundle => {
                    if (hasText) $liveDraftBundle.set(bundle)
                    else $pushedNextBundle.set(bundle)
                  })
                })
              },
            },
          ],
        },
      },
      {
        id: 'route',
        area: ROUTES_AREA,
        data: { path: '/whatnext' },
        render: () => jsx(WhatNextStudioPage, {}),
      },
      {
        id: 'sidebar-nav',
        area: SIDEBAR_NAV_AREA,
        order: 85,
        data: {
          path: '/whatnext',
          label: 'WhatNext',
          codicon: 'sparkle',
        },
      },
      {
        id: 'statusbar',
        area: STATUSBAR_AREAS.right,
        order: 125,
        render: () => jsx(WhatNextStatusChip, {}),
      },
      {
        id: 'kb-accept',
        area: KEYBINDS_AREA,
        data: {
          id: 'whatnext.accept',
          label: 'WhatNext: Accept prompt suggestion',
          combo: ACCEPT_COMBO,
          allowInEditable: true,
          handler: () => {
            if (!activeVisibleCandidate) return false
            void acceptActiveCandidate({ sendImmediately: false })
            return true
          },
        },
      },
      {
        id: 'kb-cycle',
        area: KEYBINDS_AREA,
        data: {
          id: 'whatnext.cycle',
          label: 'WhatNext: Cycle ranked candidate',
          combo: CYCLE_COMBO,
          allowInEditable: true,
          handler: () => {
            if (!activeVisibleCandidate || activeVisibleCandidate.total <= 1) return false
            cycleActiveCandidate()
            return true
          },
        },
      },
      {
        id: 'kb-dismiss',
        area: KEYBINDS_AREA,
        data: {
          id: 'whatnext.dismiss',
          label: 'WhatNext: Dismiss prompt suggestion',
          combo: DISMISS_COMBO,
          allowInEditable: true,
          handler: () => {
            if (!activeVisibleCandidate) return false
            dismissActiveCandidate()
            return true
          },
        },
      },
      {
        id: 'kb-stop-clone',
        area: KEYBINDS_AREA,
        data: {
          id: 'whatnext.stopClone',
          label: 'WhatNext: Stop Clone Mode',
          combo: STOP_CLONE_COMBO,
          allowInEditable: true,
          handler: () => {
            if (!$cloneMode.get().active) return false
            stopCloneMode('shortcut_stop')
            return true
          },
        },
      },
      {
        id: 'palette-accept',
        area: PALETTE_AREA,
        data: {
          id: 'whatnext.palette.accept',
          label: 'WhatNext: Accept current suggestion into composer',
          shortcut: ACCEPT_COMBO,
          onSelect: () => void acceptActiveCandidate({ sendImmediately: false }),
        },
      },
      {
        id: 'palette-studio',
        area: PALETTE_AREA,
        data: {
          id: 'whatnext.palette.studio',
          label: 'WhatNext: Open Evaluation & Movement Studio',
          onSelect: () => host.navigate('/whatnext'),
        },
      },
    ])
  },
}
