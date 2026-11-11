/* Hermes Web Dashboard bundle for WhatNext (/whatnext tab).
 * Uses window.__HERMES_PLUGIN_SDK__ when hosted inside `hermes dashboard`.
 */
(function () {
  const sdk = window.__HERMES_PLUGIN_SDK__;
  if (!sdk || typeof sdk.register !== 'function') return;

  const React = sdk.React;
  const h = React.createElement;

  function StatCard(props) {
    return h(
      'div',
      {
        style: {
          padding: '14px 16px',
          borderRadius: '10px',
          border: '1px solid rgba(148, 163, 184, 0.2)',
          background: 'rgba(15, 23, 42, 0.45)',
        },
      },
      h('div', { style: { fontSize: '11px', textTransform: 'uppercase', opacity: 0.65, letterSpacing: '0.05em' } }, props.label),
      h('div', { style: { fontSize: '24px', fontWeight: 700, marginTop: '4px', fontVariantNumeric: 'tabular-nums' } }, props.value),
      props.sub ? h('div', { style: { fontSize: '12px', opacity: 0.7, marginTop: '2px' } }, props.sub) : null
    );
  }

  function WhatNextDashboardPage() {
    const [data, setData] = React.useState(null);
    const [draft, setDraft] = React.useState('run ');
    const [sandboxResult, setSandboxResult] = React.useState(null);
    const [busy, setBusy] = React.useState(false);

    const refresh = React.useCallback(function () {
      fetch('/api/plugins/whatnext/dashboard')
        .then(function (r) { return r.json(); })
        .then(setData)
        .catch(function () {});
    }, []);

    React.useEffect(function () {
      refresh();
      const id = setInterval(refresh, 8000);
      return function () { clearInterval(id); };
    }, [refresh]);

    function runSandboxPredict(modeOverride) {
      setBusy(true);
      const body = modeOverride === 'next_prompt'
        ? { session_id: 'dashboard-sandbox', mode: 'next_prompt', draft: '' }
        : { session_id: 'dashboard-sandbox', mode: 'complete_draft', draft: draft, draft_revision: Date.now() % 100000 };
      fetch('/api/plugins/whatnext/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
        .then(function (r) { return r.json(); })
        .then(function (res) { setSandboxResult(res); setBusy(false); refresh(); })
        .catch(function () { setBusy(false); });
    }

    function clearFeedback() {
      fetch('/api/plugins/whatnext/feedback/clear', { method: 'POST' })
        .then(function () { refresh(); });
    }

    const metrics = (data && data.metrics) || {};
    const movements = (data && data.movements) || {};
    const memory = (data && data.memory) || {};

    return h(
      'div',
      { style: { padding: '24px', maxWidth: '1120px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '20px' } },
      h(
        'div',
        { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' } },
        h(
          'div',
          null,
          h('h1', { style: { fontSize: '22px', fontWeight: 700, margin: 0 } }, 'WhatNext: Prompt Prediction & Movement Studio'),
          h('p', { style: { fontSize: '13px', opacity: 0.75, margin: '4px 0 0 0' } },
            'Memory-grounded next_prompt suggestions, live complete_draft autocomplete, and composer evaluation funnel.')
        ),
        h(
          'button',
          {
            type: 'button',
            onClick: clearFeedback,
            style: {
              padding: '7px 12px',
              borderRadius: '8px',
              border: '1px solid rgba(248, 113, 113, 0.35)',
              background: 'rgba(248, 113, 113, 0.12)',
              color: 'inherit',
              cursor: 'pointer',
              fontSize: '12px',
            },
          },
          'Clear Feedback Memory'
        )
      ),
      h(
        'div',
        { style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px' } },
        h(StatCard, { label: 'Presented', value: String(metrics.presented || 0), sub: 'Candidates shown' }),
        h(StatCard, { label: 'Accepted', value: Math.round((metrics.acceptance_rate || 0) * 100) + '%', sub: (metrics.accepted || 0) + ' accepted into draft' }),
        h(StatCard, { label: 'Submitted', value: Math.round((metrics.submission_rate || 0) * 100) + '%', sub: (metrics.submitted || 0) + ' sent after review' }),
        h(StatCard, { label: 'Avg Edit Distance', value: String(metrics.avg_edit_distance || 0), sub: (metrics.edited || 0) + ' edited before send' }),
        h(StatCard, { label: 'Median Latency', value: (metrics.median_latency_ms || 0) + ' ms', sub: 'Prediction round-trip' })
      ),
      h(
        'div',
        { style: { padding: '16px', borderRadius: '10px', border: '1px solid rgba(148, 163, 184, 0.2)' } },
        h('h3', { style: { margin: '0 0 10px 0', fontSize: '15px' } }, 'Live Prediction Sandbox'),
        h(
          'div',
          { style: { display: 'flex', gap: '8px', flexWrap: 'wrap' } },
          h('input', {
            value: draft,
            onChange: function (e) { setDraft(e.target.value); },
            placeholder: 'Type a partial prompt to test complete_draft...',
            style: {
              flex: '1 1 280px',
              padding: '8px 12px',
              borderRadius: '8px',
              border: '1px solid rgba(148, 163, 184, 0.3)',
              background: 'transparent',
              color: 'inherit',
            },
          }),
          h('button', {
            type: 'button',
            disabled: busy,
            onClick: function () { runSandboxPredict('complete_draft'); },
            style: { padding: '8px 14px', borderRadius: '8px', cursor: 'pointer', border: '1px solid rgba(99, 102, 241, 0.5)', background: 'rgba(99, 102, 241, 0.2)', color: 'inherit' },
          }, 'Complete Draft'),
          h('button', {
            type: 'button',
            disabled: busy,
            onClick: function () { runSandboxPredict('next_prompt'); },
            style: { padding: '8px 14px', borderRadius: '8px', cursor: 'pointer', border: '1px solid rgba(16, 185, 129, 0.5)', background: 'rgba(16, 185, 129, 0.2)', color: 'inherit' },
          }, 'Predict Empty-Draft Moves')
        ),
        sandboxResult && sandboxResult.candidates && sandboxResult.candidates.length > 0
          ? h(
              'div',
              { style: { marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '6px' } },
              sandboxResult.candidates.map(function (c, i) {
                return h(
                  'div',
                  { key: i, style: { padding: '8px 10px', borderRadius: '8px', background: 'rgba(148, 163, 184, 0.1)', fontSize: '13px' } },
                  h('strong', null, '[' + (c.intent || 'move') + '] '),
                  c.text,
                  c.reason ? h('span', { style: { opacity: 0.65, marginLeft: '8px', fontSize: '11px' } }, '(' + c.reason + ')') : null
                );
              })
            )
          : null
      ),
      h(
        'div',
        { style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '14px' } },
        h(
          'div',
          { style: { padding: '16px', borderRadius: '10px', border: '1px solid rgba(148, 163, 184, 0.2)' } },
          h('h3', { style: { margin: '0 0 8px 0', fontSize: '15px' } }, 'Learned Human Workflow Transitions (' + (movements.human_turns_learned || 0) + ' turns)'),
          (movements.transitions && movements.transitions.length > 0)
            ? movements.transitions.map(function (t, i) {
                return h('div', { key: i, style: { fontSize: '13px', padding: '4px 0' } },
                  t.from + ' -> ' + t.to + ' (' + t.weight + 'x)');
              })
            : h('div', { style: { fontSize: '13px', opacity: 0.65 } }, 'Send human-authored messages to build your workflow movement graph.')
        ),
        h(
          'div',
          { style: { padding: '16px', borderRadius: '10px', border: '1px solid rgba(148, 163, 184, 0.2)' } },
          h('h3', { style: { margin: '0 0 8px 0', fontSize: '15px' } }, 'Active Memory Signals (USER.md & MEMORY.md)'),
          (memory.highlights && memory.highlights.length > 0)
            ? memory.highlights.map(function (line, i) {
                return h('div', { key: i, style: { fontSize: '13px', padding: '4px 0' } }, (i + 1) + '. ' + line);
              })
            : h('div', { style: { fontSize: '13px', opacity: 0.65 } }, 'Add facts to ~/.hermes/memories/USER.md or MEMORY.md to personalize predictions.')
        )
      )
    );
  }

  sdk.register('whatnext', {
    Component: WhatNextDashboardPage,
  });
})();
