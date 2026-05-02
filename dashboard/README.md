# Dashboard

Single-page monitoring UI served by the firewall at `GET /dashboard`.

## What it shows

- A multi-turn **conversation panel** that calls `POST /v1/chat/completions` with a
  per-conversation `conversation_id`, renders each turn inline, and tracks the
  running cumulative `P(injection)` against the configured threshold via a gauge.
  Pressing **+ New conversation** issues `DELETE /v1/conversations/{id}` and
  starts fresh.
- The currently configured upstream URL and the registered input/output
  classifiers (from `GET /api/config`).
- A live decision log driven by Server-Sent Events (`GET /api/stream`) — no
  polling. The server emits a `snapshot` event on connect and a `decision`
  event whenever a new entry is logged, each carrying authoritative aggregate
  stats so the UI never has to reconcile counts locally.
- Aggregate counts (`allowed` / `blocked` / `dropped` / `errors`) and the average
  classifier-only latency.

## Files

- `index.html` — entire UI in one file (HTML + inline CSS + inline JS, dark theme).

If/when this UI grows beyond a single screen, split it into separate `styles.css` and
`app.js`, or replace the static page with a small framework (Vite + React/Svelte).
