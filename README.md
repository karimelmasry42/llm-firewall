# PromptShield Router

This repository is the codebase for our CyberChain Hackathon submission for Challenge 1 at the University of Salamanca and AASTMT. The project implements a simple LLM firewall that screens prompts before they reach an upstream model and screens model responses before they are shown to the user.

## Team Members

- Karim Elmasry
- Mennatallah Wael
- Abdelrhman Mohamed Abdelmoaty
- Shahd Wagdy
- Yosif Qassim

## Hackathon Context

Challenge 1 focuses on building a real-time semantic filtering layer for a black-box LLM. The firewall must detect malicious prompts such as prompt injection, jailbreak attempts, and system prompt extraction attempts using only the model inputs and outputs.

## Project Overview

The application exposes an OpenAI-compatible `/v1/chat/completions` endpoint and a browser dashboard.

Request flow:

1. The user submits a prompt from the dashboard or through the API.
2. A language router classifies the prompt as English or Spanish (fasttext → lingua → heuristic fallback). Unsupported languages fall back to the English route.
3. The firewall runs the prompt through the input classifier(s) registered for that language in `llm_firewall/model_registry.py`.
4. If the input classifier blocks, the API returns: `Sorry, I cannot answer this prompt`.
5. If the input passes, the request is forwarded to the configured upstream LLM URL.
6. The upstream response is run through a regex PII masker that replaces emails, phone numbers, URLs, credit cards, API keys, private keys, and similar entities with `*` characters in place.
7. The masked response is then checked by every output classifier registered in `llm_firewall/model_registry.py` (skipped entirely if `LLM_FIREWALL_ENABLE_OUTPUT_CLASSIFIERS=false`).
8. If any output classifier blocks, the same refusal message is returned. Otherwise the (possibly masked) response is returned to the caller.

A batch testing endpoint at `/v1/chat/completions/batch` accepts up to 1000 prompts in one request. For local end-to-end testing, the repository includes a dummy upstream LLM that always replies with a fixed configurable message.

### Classifiers and filters

| Stage | Component | Source |
|---|---|---|
| Input (English route) | `linear_svm_input_classifier` | `models/input/linear_svm_input_classifier.pkl` |
| Input (Spanish route) | `linear_svm_spanish` | `models/input/linear_svm_spanish.pkl` |
| Response masking | regex PII masker (always on) | `llm_firewall/pii_filter.py` |
| Output | `tiny_toxic_detector` | Hugging Face: `AssistantsLab/Tiny-Toxic-Detector` |

Update [llm_firewall/model_registry.py](llm_firewall/model_registry.py) when you add, remove, or replace classifiers. Each entry defines the model name shown in logs and the dashboard, the pickle path or Hugging Face id, and the preprocessing function applied before prediction.

## Setup Instructions

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs `torch`, `transformers`, and `huggingface-hub` for the Hugging Face output classifier, plus `fasttext-wheel` and `lingua-language-detector` for the input language router.

### 3. Configure runtime settings

The app reads from a local `.env` file and from shell environment variables. Shell variables override `.env` if both are set.

Copy the example file and edit:

```bash
cp .env.example .env
```

Environment variables (prefix `LLM_FIREWALL_`):

| Variable | Default | Purpose |
|---|---|---|
| `UPSTREAM_CHAT_COMPLETIONS_URL` | `https://api.openai.com/v1/chat/completions` | Upstream LLM endpoint |
| `UPSTREAM_API_KEY` | `""` | Server-side upstream key. If empty, the firewall forwards the caller's bearer token. |
| `DEFAULT_MODEL_ID` | `firewall-demo` | Default model name returned to clients |
| `ENABLE_OUTPUT_CLASSIFIERS` | `true` | Set to `false` to skip output validation entirely |
| `REFUSAL_MESSAGE` | `Sorry, I cannot answer this prompt` | Returned when input or output is blocked |

Optional dummy-upstream variables (prefix `DUMMY_LLM_`): `DUMMY_LLM_API_KEY`, `DUMMY_LLM_RESPONSE_TEXT`. See `.env.example` for the canonical list and shipped defaults (which target Google Gemini's OpenAI-compatible endpoint).

Shell exports also work — use `export VAR=...` (bash/zsh), `$env:VAR=...` (PowerShell), or `set VAR=...` (cmd) for the same variable names.

### 4. Run the server

```bash
uvicorn llm_firewall.main:app --reload --port 8000
```

The first startup downloads the Hugging Face output model into the local cache.

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
- per-model scores and latencies, prefixed with `input:<name>`, `output:<name>`, plus a synthetic `input:Language Router` entry
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

A local OpenAI-compatible upstream lives at [llm_firewall/dummy_llm.py](llm_firewall/dummy_llm.py). It exposes `POST /v1/chat/completions` and `GET /health`, ignores the prompt content, and always returns the assistant message from `DUMMY_LLM_RESPONSE_TEXT` (default `This is a dummy response.`).

Run it on a separate port:

```bash
uvicorn llm_firewall.dummy_llm:app --reload --port 9000
```

Then point the firewall at it:

```dotenv
LLM_FIREWALL_UPSTREAM_CHAT_COMPLETIONS_URL=http://127.0.0.1:9000/v1/chat/completions
LLM_FIREWALL_UPSTREAM_API_KEY=
DUMMY_LLM_API_KEY=
DUMMY_LLM_RESPONSE_TEXT="This is a dummy response."
```

The default `DUMMY_LLM_RESPONSE_TEXT` is chosen to pass the current output classifiers. To require auth instead, set the same shared secret in both `LLM_FIREWALL_UPSTREAM_API_KEY` and `DUMMY_LLM_API_KEY` — the firewall will use its server-side key (see "OpenAI SDK Compatibility" above for the auth fallback rule).

## Architecture

### Core components

- [llm_firewall/main.py](llm_firewall/main.py) — FastAPI app, request flow, batch endpoint, dashboard and stats endpoints, and in-memory decision log.
- [llm_firewall/config.py](llm_firewall/config.py) — `Settings` loaded from `.env` and the environment.
- [llm_firewall/language_router.py](llm_firewall/language_router.py) — routes English vs Spanish input via fasttext, lingua, and a heuristic fallback.
- [llm_firewall/model_registry.py](llm_firewall/model_registry.py) — hard-coded classifier specs grouped by language for input and as a flat list for output.
- [llm_firewall/classifiers.py](llm_firewall/classifiers.py) — loads each classifier (pickle or HF backend), runs classification, and aggregates scores.
- [llm_firewall/huggingface_toxicity.py](llm_firewall/huggingface_toxicity.py) — Hugging Face backend used by the `tiny_toxic_detector` output classifier.
- [llm_firewall/validators/input_validator.py](llm_firewall/validators/input_validator.py) — wraps the per-language input classifier ensemble; one instance per routed language.
- [llm_firewall/validators/output_validator.py](llm_firewall/validators/output_validator.py) — wraps the output classifier ensemble.
- [llm_firewall/pii_filter.py](llm_firewall/pii_filter.py) — regex PII masker applied to every upstream response before output validation.
- [llm_firewall/proxy.py](llm_firewall/proxy.py) — forwards approved requests to the upstream LLM endpoint and proxies `/v1/models`.
- [llm_firewall/dummy_llm.py](llm_firewall/dummy_llm.py) — local fixed-response upstream for end-to-end testing.
- [dashboard/index.html](dashboard/index.html) — UI for prompt submission, response display, and live monitoring.

### Routing logic

The firewall is fail-closed:

- if the routed input classifier blocks, the request is stopped
- if any output classifier blocks, the response is withheld
- only fully approved requests and responses are returned (with PII masked when detected)

## Testing Instructions

Install the dependencies first, then run:

```bash
python3 -m pytest -q
```

Targeted suites:

```bash
python3 -m pytest tests/test_integration.py -q   # end-to-end API + batch coverage
python3 -m pytest tests/test_dummy_llm.py -q     # dummy upstream
python3 -m pytest tests/test_language_router.py -q
python3 -m pytest tests/test_pii_filter.py -q
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
