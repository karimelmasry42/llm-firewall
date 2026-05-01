# LLM Firewall

An OpenAI-compatible proxy that screens prompts before they reach an upstream model and screens model responses before they are shown to the user. The goal is real-time semantic filtering for prompt injection, jailbreaks, system-prompt extraction, and toxic outputs — using only the inputs and outputs of a black-box LLM.

The firewall exposes `/v1/chat/completions` (single + batch), `/v1/models`, and a browser dashboard at `/dashboard`.

## How it works

```
client ──▶ language router ──▶ input classifier ──▶ upstream LLM ──▶ PII mask ──▶ output classifier ──▶ client
                                       │                                                  │
                                       └──── refusal message ◀───────────────────────────┘
```

Per request:

1. The **language router** classifies the prompt as English or Spanish using `fasttext` (when `lid.176.bin` is available) or `lingua`, falling back to heuristics or the English route if confidence is low. Unsupported languages take the English route.
2. The routed **input classifier** scores the prompt. If it blocks, the API returns the configured refusal message with `decision: BLOCKED`.
3. Otherwise the request is forwarded to the upstream LLM.
4. A regex **PII masker** redacts emails, phone numbers, URLs, credit cards, API keys, private keys, and similar entities from the response in place. PII masking always runs.
5. Every **output classifier** then scores the masked response. If any blocks, the refusal is returned with `decision: DROPPED`. If `LLM_FIREWALL_ENABLE_OUTPUT_CLASSIFIERS=false`, this stage is skipped (PII masking still applies).
6. Otherwise the masked response is returned with `decision: ALLOWED`. Errors return `decision: ERROR`.

The firewall is fail-closed: only fully-approved requests and responses pass through.

### Shipped classifiers

| Stage | Label (in logs/dashboard) | Source |
|---|---|---|
| Input (all languages, multilingual) | `Llama-Prompt-Guard-2-86M` (threshold=0.001) | Hugging Face `meta-llama/Llama-Prompt-Guard-2-86M` (gated — accept Meta's license) |
| Response masking | regex PII masker (always on) | [llm_firewall/filters/pii.py](llm_firewall/filters/pii.py) |
| Output | `Tiny-Toxic-Detector` | Hugging Face `AssistantsLab/Tiny-Toxic-Detector` |

To add, swap, or remove classifiers, edit [llm_firewall/classifiers/registry.py](llm_firewall/classifiers/registry.py). Each spec defines the dashboard label, a pickle path or Hugging Face id, and the preprocessing function applied before prediction.

For the full story of how the input classifier was built and chosen — datasets, evaluation harness, model bake-off, and reproduction commands — see [docs/input_classifier/](docs/input_classifier/README.md).

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
make install              # pip install -e ".[dev]"
cp .env.example .env      # then edit upstream URL + API key
make run                  # uvicorn on http://localhost:8000
```

`make install` pulls `torch`, `transformers`, and `huggingface-hub` for the output classifier and `fasttext-wheel` + `lingua-language-detector` for the language router. The first `make run` with output classifiers enabled downloads the Hugging Face model into the local cache.

Open `http://localhost:8000/dashboard` to send prompts, watch decisions, and inspect runtime config.

## Configuration

The app reads from `.env` and from shell environment variables (shell wins on conflict).

| Variable | Default | Purpose |
|---|---|---|
| `LLM_FIREWALL_UPSTREAM_CHAT_COMPLETIONS_URL` | `https://api.openai.com/v1/chat/completions` | Upstream LLM endpoint |
| `LLM_FIREWALL_UPSTREAM_API_KEY` | `""` | Server-side upstream key. If empty, the firewall forwards the caller's bearer token. |
| `LLM_FIREWALL_DEFAULT_MODEL_ID` | `firewall-demo` | Default model name returned to clients |
| `LLM_FIREWALL_ENABLE_OUTPUT_CLASSIFIERS` | `true` | Set to `false` to skip output validation (PII masking still runs) |
| `LLM_FIREWALL_REFUSAL_MESSAGE` | `Sorry, I cannot answer this prompt` | Returned when input or output is blocked |
| `LLM_FIREWALL_MODELS_DIR` | unset | Override the directory the input classifiers load `.pkl` artifacts from |
| `DUMMY_LLM_API_KEY` | `""` | Bearer token enforced by `make dummy` (empty = open) |
| `DUMMY_LLM_RESPONSE_TEXT` | `This is a dummy response.` | Canned response from the dummy upstream |

`.env.example` ships with Gemini's OpenAI-compatible endpoint as the default upstream and a commented recipe for pointing at the local dummy.

## API

### `POST /v1/chat/completions`

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "firewall-demo",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

The response shape matches OpenAI's; blocked requests/responses come back as a normal `chat.completion` whose `content` is the configured refusal message. Inspect the dashboard or `/api/logs` to see the actual `decision` (`ALLOWED` / `BLOCKED` / `DROPPED` / `ERROR`).

If `LLM_FIREWALL_UPSTREAM_API_KEY` is set the SDK key can be a placeholder — the firewall uses its server-side key. If unset, the caller's bearer token is forwarded upstream.

### `POST /v1/chat/completions/batch`

Test up to 1000 prompts in one call, each routed through the full pipeline:

```bash
curl -X POST http://localhost:8000/v1/chat/completions/batch \
  -H "Content-Type: application/json" \
  -d @examples/batch_prompts.json
```

Body fields: `prompts` (string array, max 1000), `model`, optional `system_message`, optional `concurrency` (default 20, capped at 100). The response includes batch metadata, an `allowed`/`blocked`/`dropped`/`errors` summary, and one result per prompt with `index`, `prompt`, `http_status`, `decision`, `content`, `scores`, `latencies_ms`, `detail`, and `failed_filters`. Sample payloads live in [examples/](examples/).

### `GET /v1/models`, `GET /v1/models/{id}`

Proxied to the upstream when supported, with a minimal local fallback otherwise (e.g. when running against `make dummy`).

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(api_key="placeholder", base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="firewall-demo",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(response.choices[0].message.content)
```

The Responses API is not implemented.

## Dashboard & observability

`http://localhost:8000/dashboard` provides prompt submission, the last response, the resolved upstream and registered classifiers, and a live decision feed.

Read-only JSON endpoints:

| Endpoint | Returns |
|---|---|
| `GET /api/logs?limit=N` | Most recent `N` decision log entries (1 ≤ N ≤ 500) |
| `GET /api/stats` | Totals + decision counts + average end-to-end latency |
| `GET /api/config` | Resolved runtime config shown to the dashboard |
| `GET /health` | `{"status": "healthy", "service": "promptshield"}` |

Each log entry carries: `timestamp`, short `id`, `prompt`, `response`/refusal, `decision`, per-classifier `scores` and `latencies_ms` keyed `input:<name>` / `output:<name>` (plus a synthetic `input:Language Router` entry), `total_latency_ms`, `detail`, and `failed_filters`.

The log lives in process memory — restarting the server clears it — and is capped at the most recent 500 entries. Batch requests add one entry per prompt.

## Local dummy upstream

[llm_firewall/api/dummy_llm.py](llm_firewall/api/dummy_llm.py) is a fixed-response, OpenAI-compatible upstream useful for end-to-end testing without spending API credits.

```bash
make dummy   # uvicorn on :9000, exposes POST /v1/chat/completions and GET /health
```

Then point the firewall at it (see the commented block in `.env.example`):

```dotenv
LLM_FIREWALL_UPSTREAM_CHAT_COMPLETIONS_URL=http://localhost:9000/v1/chat/completions
LLM_FIREWALL_UPSTREAM_API_KEY=
LLM_FIREWALL_DEFAULT_MODEL_ID=dummy-llm
```

The default `DUMMY_LLM_RESPONSE_TEXT` passes the shipped output classifier. To require auth between firewall and dummy, set the same shared secret in both `LLM_FIREWALL_UPSTREAM_API_KEY` and `DUMMY_LLM_API_KEY`.

## Repository layout

```
llm_firewall/
  api/           FastAPI app, routes, dashboard, dummy upstream, shared processing
  core/          Settings + outbound HTTP proxy
  classifiers/   Registry, ensemble, language router, pickle/HF backends
  filters/       FilterResult primitive + PII / toxicity filters
  validators/    InputValidator / OutputValidator wrappers
data/models/     Pre-trained input classifier artifacts (.pkl)
dashboard/       Single-page monitoring UI
examples/        Sample batch payloads
scripts/         Standalone simulation script
tests/           unit/ + integration/
```

## Development

```bash
make test        # pytest, unit + integration
make simulate    # scripts/simulate.py — standalone validation run
make clean       # remove build, cache, .venv
```

Tests use lightweight fake pickle models, mock the upstream LLM with `respx`, and never make real network calls. Batch processing is exercised end-to-end against the dummy upstream.

## Team

- Karim Elmasry
- Ahmed Yasser
- Omar Selim
- Ammar Osama

## AI Usage Declaration

AI-assisted development tools (OpenAI Codex/ChatGPT, Anthropic Claude) helped draft, refactor, and test parts of the codebase and documentation. All generated content was reviewed and edited by the team before inclusion.
