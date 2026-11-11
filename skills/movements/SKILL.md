---
name: movements
description: Inspect predicted next user prompts, draft completions, learned workflow movement patterns, and composer evaluation metrics with WhatNext.
---

# WhatNext Prompt Prediction and Movement Intelligence

Use WhatNext tools when the user asks what to do next in a project, wants follow-up prompt ideas tailored to their working style, or wants to audit suggestion acceptance and movement habits.

## Available Tools

1. `whatnext_predict`: Generates ranked `next_prompt` candidates for an empty composer or `complete_draft` continuations for a half-written draft, conditioned on `USER.md`, `MEMORY.md`, workspace stack, and learned human workflow transitions.
2. `whatnext_analytics`: Returns composer evaluation funnel metrics (presented, accepted, edited, dismissed, submitted, edit distance, median latency) together with the user's top workflow movement transitions.

## Operating Rules

1. Keep suggestions written in the user's own voice rather than assistant narration.
2. Never auto-execute destructive commands from a predicted prompt; predictions only propose what the user may review, edit, and send.
3. Respect privacy boundaries: machine-accepted or Clone Mode prompts are never recorded as new human preference evidence.
