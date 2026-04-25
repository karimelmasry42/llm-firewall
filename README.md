# LLM Firewall

This project The project implements a simple LLM firewall that screens prompts before they reach an upstream model and screens model responses before they are shown to the user.

## Team Members

- Karim Elmasry
- Ahmed Yasser
- Omar Selim
- Ammar Osama

## Hackathon Context

This challenge focuses on building a real-time semantic filtering layer for a black-box LLM. The firewall must detect malicious prompts such as prompt injection, jailbreak attempts, and system prompt extraction attempts using only the model inputs and outputs.

## Project Overview

The application exposes an OpenAI-compatible `/v1/chat/completions` endpoint and a browser dashboard.

Request flow:

1. The user submits a prompt from the dashboard or through the API.
2. A language router classifies the prompt as English or Spanish: for longer prompts it uses `fasttext` when the `lid.176.bin` model is available, for shorter prompts it uses `lingua`, and if detectors are unavailable or confidence is low it falls back to heuristics or the English route. Unsupported languages fall back to the English route.
3. The firewall runs the prompt through the input classifier(s) registered for that language in `llm_firewall/classifiers/registry.py`.
4. If the input classifier blocks, the API returns: `Sorry, I cannot answer this prompt`.
5. If the input passes, the request is forwarded to the configured upstream LLM URL.
6. The upstream response is run through a regex PII masker that replaces emails, phone numbers, URLs, credit cards, API keys, private keys, and similar entities with `*` characters in place.
7. The masked response is then checked by every output classifier registered in `llm_firewall/classifiers/registry.py` (skipped entirely if `LLM_FIREWALL_ENABLE_OUTPUT_CLASSIFIERS=false`).
8. If any output classifier blocks, the same refusal message is returned. Otherwise the (possibly masked) response is returned to the caller.

A batch testing endpoint at `/v1/chat/completions/batch` accepts up to 1000 prompts in one request. For local end-to-end testing, the repository includes a dummy upstream LLM that always replies with a fixed configurable message.

### Classifiers and filters

| Stage | Component (label in logs/dashboard) | Source |
|---|---|---|
| Input (English route) | `linear_svm_input_classifier` | `data/models/linear_svm_input_classifier.pkl` |
| Input (Spanish route) | `linear_svm_spanish` | `data/models/linear_svm_spanish.pkl` |
| Response masking | regex PII masker (always on) | `llm_firewall/filters/pii.py` |
| Output | `Tiny-Toxic-Detector` | Hugging Face: `AssistantsLab/Tiny-Toxic-Detector` |

Update [llm_firewall/classifiers/registry.py](llm_firewall/classifiers/registry.py) when you add, remove, or replace classifiers. Each entry defines the label shown in logs and the dashboard (`display_name` when set, otherwise `name`), the pickle path or Hugging Face id, and the preprocessing function applied before prediction.

## Setup Instructions

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install the package

```bash
make install      # equivalent to: pip install -e ".[dev]"
```

This installs the firewall in editable mode together with the development extras
(`pytest`, `pytest-asyncio`, `respx`). The base dependencies pull in `torch`,
`transformers`, and `huggingface-hub` for the Hugging Face output classifier, plus
`fasttext-wheel` and `lingua-language-detector` for the input language router.

### 3. Configure runtime settings

The app reads from a local `.env` file and from shell environment variables. Shell variables override `.env` if both are set.

Copy the example file and edit:

```bash
cp .env.example .env
```

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_FIREWALL_UPSTREAM_CHAT_COMPLETIONS_URL` | `https://api.openai.com/v1/chat/completions` | Upstream LLM endpoint |
| `LLM_FIREWALL_UPSTREAM_API_KEY` | `""` | Server-side upstream key. If empty, the firewall forwards the caller's bearer token. |
| `LLM_FIREWALL_DEFAULT_MODEL_ID` | `firewall-demo` | Default model name returned to clients |
| `LLM_FIREWALL_ENABLE_OUTPUT_CLASSIFIERS` | `true` | Set to `false` to skip output validation entirely |
| `LLM_FIREWALL_REFUSAL_MESSAGE` | `Sorry, I cannot answer this prompt` | Returned when input or output is blocked |

Optional dummy-upstream variables: `DUMMY_LLM_API_KEY`, `DUMMY_LLM_RESPONSE_TEXT`. See `.env.example` for the canonical list and shipped defaults (which target Google Gemini's OpenAI-compatible endpoint).

Shell exports also work — use `export VAR=...` (bash/zsh), `$env:VAR=...` (PowerShell), or `set VAR=...` (cmd) for the same variable names.

### 4. Run the server

```bash
make run          # equivalent to: uvicorn llm_firewall.api.app:app --reload --port 8000
```

If `LLM_FIREWALL_ENABLE_OUTPUT_CLASSIFIERS=true`, the first startup downloads the Hugging Face output model into the local cache.

## Usage Guide

### Dashboard

Open `http://localhost:8000/dashboard` to:

- submit prompts through the firewall
- view the last response returned to the user
- inspect the configured upstream URL and registered classifiers
- monitor request logs and aggregate counts

### In-Memory Logs

The firewall keeps a lightweight in-memory decision log inside the running FastAPI process. Per-request fields:

- timestamp and short log entry id
- original prompt and returned response (or refusal message)
- final decision: `ALLOWED`, `BLOCKED`, `DROPPED`, or `ERROR`
- per-model scores and latencies, prefixed with `input:<name>` and `output:<name>`, where `<name>` is the classifier's dashboard/log display name (for example, `output:Tiny-Toxic-Detector`), plus a synthetic `input:Language Router` entry
- detail text and an optional `failed_filters` list when a classifier blocks

Behavior:

- the log is in-memory only — restarting the server clears it
- capped at the most recent `500` entries
- batch requests add one log entry per prompt, not one entry for the whole batch

Endpoints:

```bash
curl "http://localhost:8000/api/logs?limit=20"
curl "http://localhost:8000/api/stats"
curl "http://localhost:8000/api/config"
curl "http://localhost:8000/health"
```

`/api/config` reflects the resolved runtime configuration shown to the dashboard. `/health` returns a simple liveness check.

### API

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "firewall-demo",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

Expected behavior:

- blocked input → refusal message (`decision: BLOCKED`)
- blocked output → refusal message (`decision: DROPPED`)
- safe input and safe output → upstream response, possibly with PII entities masked (`decision: ALLOWED`)

### OpenAI SDK Compatibility

The firewall implements the Chat Completions surface of the OpenAI API:

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /v1/models/{model_id}`

It does not yet implement the Responses API. When the upstream provider does not expose `/v1/models` (e.g. the local dummy), the firewall returns a minimal local fallback list.

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key-or-placeholder",
    base_url="http://localhost:8000/v1",
)

response = client.chat.completions.create(
    model="firewall-demo",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

print(response.choices[0].message.content)
```

If `LLM_FIREWALL_UPSTREAM_API_KEY` is set on the firewall, the SDK key can be a placeholder — the firewall uses its server-side key. If unset, the firewall forwards the caller's bearer token to the upstream.

### Batch Testing API

Use the batch endpoint to test up to 1000 prompts in one call:

```bash
curl -X POST http://localhost:8000/v1/chat/completions/batch \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d @examples/batch_prompts.json
```

Request body fields:

- `model` — model name placed in each generated request body
- `system_message` — optional system prompt added to every request
- `concurrency` — optional integer, how many prompts run at once (default `20`)
- `prompts` — array of prompt strings, max `1000`

Each prompt is wrapped into its own OpenAI-style request body:

```json
{
  "model": "firewall-demo",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ]
}
```

and routed through the same firewall pipeline as `/v1/chat/completions` (language router → input classifier → upstream → PII mask → output classifier). Prompts are processed independently and the `concurrency` setting bounds in-flight work.

Sample input lives at [examples/batch_prompts.json](examples/batch_prompts.json) (and a 100-prompt variant alongside it). The response includes batch metadata, an `allowed`/`blocked`/`dropped`/`errors` summary, and one result per prompt with `index`, `prompt`, `http_status`, `decision`, `content`, `scores`, `detail`, and `failed_filters`.

### Local Dummy LLM

A local OpenAI-compatible upstream lives at [llm_firewall/api/dummy_llm.py](llm_firewall/api/dummy_llm.py). It exposes `POST /v1/chat/completions` and `GET /health`, ignores the prompt content, and always returns the assistant message from `DUMMY_LLM_RESPONSE_TEXT` (default `This is a dummy response.`).

Run it on a separate port:

```bash
make dummy        # equivalent to: uvicorn llm_firewall.api.dummy_llm:app --reload --port 9000
```

Then point the firewall at it:

```dotenv
LLM_FIREWALL_UPSTREAM_CHAT_COMPLETIONS_URL=http://127.0.0.1:9000/v1/chat/completions
LLM_FIREWALL_UPSTREAM_API_KEY=
DUMMY_LLM_API_KEY=
DUMMY_LLM_RESPONSE_TEXT="This is a dummy response."
```

The default `DUMMY_LLM_RESPONSE_TEXT` currently passes with the shipped output classifiers. To require auth instead, set the same shared secret in both `LLM_FIREWALL_UPSTREAM_API_KEY` and `DUMMY_LLM_API_KEY` — the firewall will use its server-side key (see "OpenAI SDK Compatibility" above for the auth fallback rule).

## Architecture

### Repository layout

```
llm-firewall/
├── llm_firewall/
│   ├── api/                FastAPI app, routes, dashboard, dummy upstream
│   ├── core/               Settings + outbound HTTP proxy
│   ├── classifiers/        Registry, ensemble, language router, model backends
│   ├── filters/            FilterResult primitive + PII / toxicity filters
│   └── validators/         InputValidator / OutputValidator wrappers
├── data/models/            Pre-trained classifier artifacts (.pkl)
├── dashboard/              Single-page monitoring UI
├── examples/               Sample batch payloads
├── scripts/                Standalone simulation script
├── tests/
│   ├── unit/               Single-module tests
│   └── integration/        End-to-end tests against the FastAPI app
├── pyproject.toml
├── Makefile
└── README.md
```

### Core components

- [llm_firewall/api/app.py](llm_firewall/api/app.py) — FastAPI application factory and lifecycle wiring.
- [llm_firewall/api/routes.py](llm_firewall/api/routes.py) — chat completions, model listing, and batch endpoints.
- [llm_firewall/api/dashboard.py](llm_firewall/api/dashboard.py) — dashboard, decision-log, stats, config, and health endpoints.
- [llm_firewall/api/_processing.py](llm_firewall/api/_processing.py) — single-prompt firewall flow shared by single + batch routes.
- [llm_firewall/api/dummy_llm.py](llm_firewall/api/dummy_llm.py) — local fixed-response upstream for end-to-end testing.
- [llm_firewall/core/config.py](llm_firewall/core/config.py) — `Settings` loaded from `.env` and the environment.
- [llm_firewall/core/proxy.py](llm_firewall/core/proxy.py) — forwards approved requests to the upstream LLM endpoint and proxies `/v1/models`.
- [llm_firewall/classifiers/language_router.py](llm_firewall/classifiers/language_router.py) — routes English vs Spanish input via fasttext, lingua, and a heuristic fallback.
- [llm_firewall/classifiers/registry.py](llm_firewall/classifiers/registry.py) — hard-coded classifier specs grouped by language for input and as a flat list for output.
- [llm_firewall/classifiers/ensemble.py](llm_firewall/classifiers/ensemble.py) — runs every configured classifier and aggregates scores.
- [llm_firewall/classifiers/pickle_classifier.py](llm_firewall/classifiers/pickle_classifier.py) — sklearn-style pickle/joblib bundle backend.
- [llm_firewall/classifiers/huggingface.py](llm_firewall/classifiers/huggingface.py) — Hugging Face backend used by the `Tiny-Toxic-Detector` output classifier.
- [llm_firewall/validators/input.py](llm_firewall/validators/input.py) — wraps the per-language input classifier ensemble; one instance per routed language.
- [llm_firewall/validators/output.py](llm_firewall/validators/output.py) — wraps the output classifier ensemble.
- [llm_firewall/filters/pii.py](llm_firewall/filters/pii.py) — regex PII masker (and `PiiFilter` detection class) applied to every upstream response before output validation.
- [dashboard/index.html](dashboard/index.html) — UI for prompt submission, response display, and live monitoring.

### Routing logic

The firewall is fail-closed when output classifiers are enabled:

- if the routed input classifier blocks, the request is stopped
- if any output classifier blocks, the response is withheld
- only fully approved requests and responses are returned (with PII masked when detected)

When `LLM_FIREWALL_ENABLE_OUTPUT_CLASSIFIERS=false`, output validation is skipped entirely. Upstream responses are returned (still with PII masked) without that second check, and the decision detail is annotated `Output classifiers disabled`. Input validation always runs.

## Testing Instructions

Install the package first (`make install`), then run:

```bash
make test         # equivalent to: pytest
```

Targeted suites:

```bash
pytest tests/integration/test_integration.py -q   # end-to-end API + batch coverage
pytest tests/unit/test_dummy_llm.py -q            # dummy upstream
pytest tests/unit/test_language_router.py -q
pytest tests/unit/test_pii_filter.py -q
```

Fast syntax check:

```bash
python3 -m compileall llm_firewall tests
```

Notes:

- the tests use lightweight fake pickle models and do not depend on the production classifier files
- integration tests mock the upstream LLM instead of making real network calls
- the dummy LLM has direct tests for fixed-response and API-key behavior
- batch testing returns a full response for up to 1000 prompts, but the in-memory dashboard log remains capped at the most recent 500 entries

## AI Usage Declaration

AI-assisted development tools were used during this project. OpenAI Codex/ChatGPT was used to help draft, refactor, and test parts of the codebase and documentation. All generated content was reviewed and edited by the team before inclusion in the repository.
