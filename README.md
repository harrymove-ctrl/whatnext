<img src="assets/icon.svg" width="64" height="64" alt="WhatNext Icon">

# WhatNext: Prompt Suggestions & Predict Your Movements for Hermes Agent

WhatNext brings Clone-style contextual prompt prediction and workflow movement learning to Hermes Agent across Hermes Desktop, the Hermes Web Dashboard, CLI, and chat gateways. When your composer is empty, WhatNext predicts up to three ranked next moves conditioned on `USER.md`, `MEMORY.md`, `SOUL.md`, workspace stack, and your personal workflow history. When you start typing, WhatNext switches into live `complete_draft` autocomplete and offers a ghost continuation you can accept into the draft with one shortcut without ever blocking normal typing or sending.

## Core Capabilities

| Capability | Mode | How It Works in Hermes |
| :--- | :--- | :--- |
| **Empty-Chat Starter Movements** | `next_prompt` | Reads `USER.md`, `MEMORY.md`, and active workspace manifests (`pyproject.toml`, `Cargo.toml`, `package.json`) to suggest personalized starters on a brand-new chat. |
| **Post-Turn Follow-Ups & Answers** | `next_prompt` | Fires after `post_llm_call` and pushes ranked next moves to Desktop over `ctx.emit_to_session`. Commits to a concrete option when Hermes asks a question. |
| **Live Draft Autocomplete** | `complete_draft` | Reads live composer text through `host.composer.getDraft`, debounces keystrokes, fences stale responses with monotonic `(context_revision, draft_revision)`, and merges completions via `host.composer.setDraft`. |
| **Human-Only Movement Graph** | Feedback Loop | Learns workflow transitions (such as `implement -> verify -> ship`) strictly from `origin: human` turns so accepted or machine-generated prompts never pollute personal preference learning. |
| **Bounded Clone Mode** | Delegation | Optional explicit user Start button with a 3-turn and 5-minute cap, a visible 2.5-second review countdown, and instant Stop on any manual edit, session switch, or approval boundary. |
| **Composer Evaluation Studio** | `/whatnext` | Tracks `presented`, `accepted`, `edited`, `dismissed`, and `submitted` counts separately alongside normalized character edit distance and median latency. |

## Dual Prediction Engines

1. **Zero-Key Hermes LLM (`auto` / `hermes_llm`)**: Uses `ctx.llm.complete_structured` on its own `auxiliary.whatnext` task slot, inheriting whatever provider and model you already use in Hermes with zero extra credentials. Point `auxiliary.whatnext` at a smaller fast model anytime through `hermes model`.
2. **Official Clone Labs API (`clone_api`)**: When `CLONE_API_KEY` is stored in your Hermes profile credentials, WhatNext calls `https://api.clone.is/v1/predictions`, forwards observation events to `/v1/prediction-events`, and supports `/v1/prediction-feedback/clear`.
3. **Offline Movement Fallback (`local_heuristics`)**: Guarantees sub-5ms deterministic movement predictions and draft completions when offline or inside deterministic evaluation runs.

## Try the Interactive Composer Playground Right Now

You can test empty-chat starters, live `Tab` draft completions, post-turn movement predictions, Clone Mode countdowns, and the live SQLite evaluation scorecard in your browser immediately:

```bash
python3 demo/playground.py --port 8765
```

Open `http://127.0.0.1:8765` to interact with the composer:

1. Click any `STARTER` pill on the empty chat or press `Tab` to place it into the draft without sending.
2. Type `run `, `fix `, `commit `, or `make it ` to watch live ghost-suffix autocomplete appear inside the input box, then press `Tab` to accept and edit before sending.
3. Click **Simulate Agent Question** to see WhatNext commit to concrete option answers (`1/3`, cycle with `Alt+]`).
4. Edit `USER.md` or `MEMORY.md` on the right panel and click **Save & Re-Predict** to watch suggestions adapt immediately.

## Install in Hermes Agent

```bash
hermes plugins install https://github.com/harrymove-ctrl/whatnext
hermes plugins enable whatnext
```

After enabling in Hermes Desktop (`Settings -> Plugins -> WhatNext`):

1. **Click** the underside pill or press `Cmd/Ctrl+Shift+Space` to accept a suggestion into the composer draft for review.
2. **Cmd/Ctrl+Click** the pill to submit immediately.
3. Press `Alt+]` to cycle through ranked movement candidates (`1/3`, `2/3`, `3/3`).
4. Press `Cmd/Ctrl+Shift+Backspace` or click `x` to dismiss a candidate and suppress similar phrasing.
5. Open `/whatnext` from the left sidebar, status bar chip, or Command Palette (`WhatNext: Open Evaluation & Movement Studio`) to inspect funnel metrics or clear feedback memory.
6. In the Hermes CLI, TUI, or messaging gateways, run `/whatnext`, `/whatnext complete <partial draft>`, `/whatnext stats`, or `/whatnext clear`, or run `hermes whatnext eval` from your shell.

## Validation & Tests

```bash
python3 -m unittest discover -s tests -v
hermes plugins validate . --install-deps
```

## Security & Privacy Disclosures (Catalog Rule 13)

1. **Local Storage**: Stores active session candidates, revision counters, observation funnel events, dismissed phrase hashes, and human workflow intent transitions in `$HERMES_HOME/plugin-data/whatnext/whatnext.db` (SQLite WAL). One-click reset is available via `/whatnext clear`, the Studio button, or `whatnext_analytics(action="clear_feedback")`.
2. **Reads Outside Plugin Data**: When `enable_memory_personalization` is enabled (default `true`), reads `$HERMES_HOME/memories/USER.md`, `$HERMES_HOME/memories/MEMORY.md`, `$HERMES_HOME/SOUL.md`, recent session messages via `hermes_state.SessionDB`, and top-level workspace filenames (`pyproject.toml`, `Cargo.toml`, `package.json`, `.git` presence) to ground predictions. Secret-shaped tokens are scrubbed prior to prompt assembly. Never reads `.env` or `auth.json`.
3. **Network Calls**: Uses host-owned `ctx.llm` by default (sending bounded conversation turns and scrubbed memory highlights to your active Hermes LLM provider). Only when `CLONE_API_KEY` is configured via Hermes profile secrets (`agent.secret_scope.get_secret`), sends HTTPS requests to `https://api.clone.is/v1/predictions`, `/v1/prediction-events`, and `/v1/prediction-feedback/clear`.
4. **Approvals & Unattended Safety**: Suggestions only populate the composer draft unless you explicitly `Cmd/Ctrl+Click` or start bounded Clone Mode (`3` turns and `5` minutes default, visible 2.5s review bar, instant Stop on any manual edit or session switch). Clone Mode uses `host.composer.submit` and never bypasses Hermes tool approval gates. No shell commands, no background daemons, no telemetry, and no self-updater.
