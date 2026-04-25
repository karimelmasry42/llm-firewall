# Dashboard

Single-page monitoring UI served by the firewall at `GET /dashboard`.

## What it shows

- A prompt submission form that calls `POST /v1/chat/completions` and renders the
  upstream response (or refusal message).
- The currently configured upstream URL and the registered input/output classifiers
  (sourced from `GET /api/config`).
- A live decision log polled from `GET /api/logs` (capped at the most recent 500
  entries — the limit is enforced server-side in
  [`llm_firewall/api/_processing.py`](../llm_firewall/api/_processing.py)).
- Aggregate counts (`allowed` / `blocked` / `dropped` / `errors`) and average
  end-to-end latency from `GET /api/stats`.

## Files

- `index.html` — entire UI in one file (HTML + inline CSS + inline JS, dark theme).

If/when this UI grows beyond a single screen, split it into separate `styles.css` and
`app.js`, or replace the static page with a small framework (Vite + React/Svelte).
