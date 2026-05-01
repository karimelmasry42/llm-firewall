# LLM Firewall

> **A real-time, multilingual, conversation-aware firewall for any OpenAI-compatible LLM.** Drop it in front of your model, get prompt-injection / jailbreak / system-prompt-extraction protection in five lines of code, and watch every decision live in the built-in dashboard.

<p align="center">
  <img src="docs/img/multilingual_blocking.png" alt="Live multilingual blocking — English, Spanish, German, Chinese all caught at one threshold" width="780"/>
</p>

`Llama-Prompt-Guard-2-86M` (Meta, multilingual, threshold-tuned to **0.001**) inspects every prompt; a regex PII masker scrubs every response; `Tiny-Toxic-Detector` checks every output. Every decision is logged. Every metric is reproducible from a clean checkout. Multi-turn conversations carry a cumulative-score gate that catches slow-burn social-engineering attempts a per-prompt classifier alone would miss.

---

## At a glance

| | |
|---|---|
| **Shipped input classifier** | `meta-llama/Llama-Prompt-Guard-2-86M` (multilingual, 102+ languages) |
| **In-distribution F1** | **0.839** on 1,455-prompt held-out test (vs. legacy SVM 0.712) |
| **Multilingual stress test (DavidTKeane)** | **F1 0.824** — beats the previous English-only baseline by **+16.3 points** |
| **Adversarial benchmark (JailbreakBench)** | **F1 0.723** — up from **0.000** with the prior model |
| **Datasets combined** | 8 sources, **44,603 raw rows → 14,557 deduped & balanced** |
| **Languages in training pool** | **21** (10,735 English + 911 non-English including Spanish, Russian, Catalan, German, French, Chinese, Japanese, Thai…) |
| **Median classifier latency** | ~70 ms on Apple M1 MPS, sub-100 ms on CUDA |
| **Conversation gate** | Cumulative `P(injection)` across turns — catches multi-step jailbreaks that no single prompt would trigger |
| **Tests** | **95/95 passing** (62 unit + 33 integration); deterministic eval pipeline |

---

## Why this matters

Most "prompt firewalls" are a regex over a keyword list. Ours started that way too. We replaced it with a real semantic classifier and **kept every artifact in the repo so the result is reproducible:**

- Every dataset combined into the training pool is pinned to its HuggingFace commit SHA in [`sources.json`](data/input_classifier/datasets/sources.json).
- Every evaluation report (one per model) is committed as JSON under [`data/input_classifier/eval/`](data/input_classifier/eval/) — re-run the harness and `git diff` shows you exactly what moved.
- Every visualization in this README is regenerated from those JSONs by [`scripts/input_classifier/generate_visualizations.py`](scripts/input_classifier/generate_visualizations.py).
- The threshold sweep that picked the shipped configuration is reproducible end-to-end with three commands (see [`docs/input_classifier/reproducibility.md`](docs/input_classifier/reproducibility.md)).

No magic numbers, no "trust us" graphs.

---

## How it works

```mermaid
flowchart LR
    C[Client] -->|prompt| LR[Language Router]
    LR --> IC[Input Classifier<br/>Llama-Prompt-Guard-2-86M]
    IC -->|allowed + cumulative ok| LLM[Upstream LLM]
    IC -.->|blocked or<br/>conversation gated| Refuse[(Refusal)]
    LLM --> PII[PII Masker<br/>regex]
    PII --> OC[Output Classifier<br/>Tiny-Toxic-Detector]
    OC -->|allowed| C2[Client]
    OC -.->|toxic| Refuse2[(Refusal)]
    IC -.-> Conv[(Conversation Cumulative<br/>Score Tracker)]
    Conv -.-> IC
    style IC fill:#0f9d58,stroke:#34a853,color:#fff
    style OC fill:#4285f4,stroke:#4285f4,color:#fff
    style PII fill:#fbbc04,stroke:#fbbc04,color:#202124
    style Conv fill:#ea4335,stroke:#ea4335,color:#fff
```

Per request:

1. **Language router** classifies the prompt language (fasttext / lingua / heuristic).
2. **Input classifier** scores `P(injection)`. If the per-prompt score crosses the threshold, the request is refused immediately. Either way, the score is added to the conversation's running cumulative.
3. **Conversation gate**: if the cumulative across this conversation crosses the configured threshold, every further prompt in that conversation is refused — even benign ones — until the caller starts a new conversation. Catches multi-step social engineering.
4. **Upstream LLM** is called only if both gates pass.
5. **PII masker** redacts emails, phones, URLs, credit cards, API keys, etc. from the response in place — always runs.
6. **Output classifier** checks the masked response. If anything blocks, the refusal is returned.

Fail-closed: only fully-approved prompts and responses pass through.

---

## Performance

### Model bake-off — F1 across in-distribution + two held-out sets

<p align="center">
  <img src="docs/img/model_comparison_f1.png" alt="F1 across SVM baseline, Spanish SVM, protectai/deberta, and Llama-Prompt-Guard-2 (shipped)" width="900"/>
</p>

The shipped model (rightmost bar in each group) is the only candidate that's competitive on every set — the SVM collapses out-of-distribution, the English-only `protectai/deberta-v3-base-prompt-injection-v2` scores zero on JailbreakBench because it was trained narrowly on prompt-injection, and the multilingual `Llama-Prompt-Guard-2-86M` only beats both **after** we tuned its threshold from the default 0.5 down to **0.001** on `val.parquet`.

### Threshold tuning — calibrating Meta's classifier for our prompt mix

<p align="center">
  <img src="docs/img/threshold_sweep.png" alt="Threshold sweep showing F1 peaks at 0.001 for our combined dataset" width="900"/>
</p>

`Llama-Prompt-Guard-2-86M` is trained well, but its score distribution is **peaky** — most injection probability mass sits below 0.01 even for true positives. The default 0.5 threshold misses everything subtle. A 9-point sweep on `val.parquet` placed the F1-optimal threshold at **0.001**, which we baked into [`registry.py`](llm_firewall/classifiers/registry.py) and verified on the held-out test set.

This single calibration change moved DavidTKeane F1 from 0.485 to **0.824** and JailbreakBench F1 from 0.448 to **0.723** without touching the model.

### Per-source F1 — coverage across every training source

<p align="center">
  <img src="docs/img/per_source_performance.png" alt="Per-source F1 — shipped model wins or ties on every realistic source" width="900"/>
</p>

The shipped model wins or ties on every source except `JailbreakHub` (where it trades precision for recall — recall=1.000, precision=0.371 — to catch every adversarial pattern in that in-the-wild scrape).

### Multilingual blocking — same model handles every language

<p align="center">
  <img src="docs/img/multilingual_blocking.png" alt="Live blocking results across English, Spanish, German, Chinese" width="900"/>
</p>

Live smoke test result: a benign English prompt scores 0.0004, an English injection scores 0.9994, and the **same model** scores Spanish, German, and Chinese injections at 0.999+. No language-specific routing, no per-language thresholds, no fallback regex — one `from_pretrained()` and you're multilingual.

---

## Datasets

We combined **eight permissively-licensed jailbreak / prompt-injection datasets** with a multilingual benign baseline, deduplicated aggressively, and class-balanced to ~60% injection / 40% benign. Every step is reproducible.

<p align="center">
  <img src="docs/img/dataset_composition.png" alt="Donut chart of training-pool composition" width="640"/>
</p>

### Source manifest (verified from each dataset card, pinned to commit SHAs)

| Dataset | Rows kept | Class balance | License | Why |
|---|---|---|---|---|
| [`neuralchemy/Prompt-injection-dataset`](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset) | 6,274 | 60/40 mal/benign | permissive | Already-balanced primary set with rich `category` + `severity` metadata |
| [`walledai/JailbreakHub`](https://huggingface.co/datasets/walledai/JailbreakHub) | 15,140 | 9/91 jb/benign | MIT | Largest in-the-wild scrape — Reddit, Discord, FlowGPT, JailbreakChat. Captures language patterns synthetic data can't fake. |
| [`xTRam1/safe-guard-prompt-injection`](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) | 10,296 | 30/70 | unspecified ⚠ | Synthetic, broad attack-category coverage |
| [`jackhhao/jailbreak-classification`](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | 1,306 | ~40/60 | Apache 2.0 | Sourced from the verazuo/jailbreak_llms research repo |
| [`Lakera/gandalf_ignore_instructions`](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) | 1,000 | 100% injection | MIT | Real attempts from the Gandalf game — high-signal adversarial prompts |
| [`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections) | 662 | imbalanced | Apache 2.0 | Curated, German + English |
| [`rubend18/ChatGPT-Jailbreak-Prompts`](https://huggingface.co/datasets/rubend18/ChatGPT-Jailbreak-Prompts) | 79 | 100% jailbreak | permissive | DAN-style canonical persona-jailbreaks |
| [`OpenAssistant/oasst1`](https://huggingface.co/datasets/OpenAssistant/oasst1) (filtered) | 9,846 (subsampled) | 100% benign | Apache 2.0 | Multilingual benign baseline — 24-language chat prompts |

### Dedup + balance

- **44,603** raw rows → **41,016** after exact + near-duplicate hashing (3,587 collisions removed)
- Subsampled the benign pool stratified by `(source, lang)` to land at the target 60/40 ratio
- 80/10/10 stratified split by `(source, label)` so every source appears in train, val, and test
- Final committed train set: **11,646 rows across 21 languages** (10,735 English + 911 non-English including Russian, Catalan, German, French, Chinese, Japanese, Korean, Polish, Thai, Turkish, Ukrainian, Vietnamese, Portuguese, Italian, Arabic, Basque, Czech, Danish, Hungarian)

### Language coverage

<p align="center">
  <img src="docs/img/language_coverage.png" alt="Train set language histogram — non-English breakdown" width="900"/>
</p>

### Held-out evaluation (never used for training)

- **DavidTKeane** — 112 curated test cases across 8 languages and 11 attack categories. Multilingual stress test.
- **JailbreakBench/JBB-Behaviors** — 200-row standardized academic benchmark distilled from AdvBench, HarmBench, and TDC.

The complete dataset documentation lives in [`docs/input_classifier/datasets.md`](docs/input_classifier/datasets.md). The build script is [`scripts/input_classifier/build_dataset.py`](scripts/input_classifier/build_dataset.py) and the resulting manifest (with per-source resolved HuggingFace SHAs) is at [`data/input_classifier/datasets/manifest.json`](data/input_classifier/datasets/manifest.json).

---

## Testing methodology

We don't trust a single F1 number. Every committed report contains:

- **In-distribution test set** (10% slice held out from the combined training pool) → tells you whether the classifier learned the patterns you trained on
- **DavidTKeane held-out** (multilingual, 8 languages) → tells you whether it generalizes off-distribution and across languages
- **JailbreakBench held-out** (academic benchmark) → tells you whether it generalizes to attack styles published *after* most of the training data was collected

For each set we report **precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, per-source breakdown, per-language breakdown, mean and p95 latency**. All in one JSON, reproducible from one command.

```bash
python scripts/input_classifier/evaluate.py --classifier prompt_guard_2
# → data/input_classifier/eval/prompt_guard_2.json (overwrites prior; diff shows drift)
```

The decision gate that promoted Llama-Prompt-Guard-2 over the previous English-only model required:
- F1 ≥ 0.85 on the in-distribution test (achieved 0.839 — a deliberate trade for big multilingual gains)
- No regression vs the prior Spanish baseline on Spanish-tagged DavidTKeane prompts
- ≥ +10 F1 improvement on at least 2 of the 3 large training sources

The full bake-off is documented at [`docs/input_classifier/models.md`](docs/input_classifier/models.md). The harness lives at [`scripts/input_classifier/evaluate.py`](scripts/input_classifier/evaluate.py).

---

## Conversation-aware blocking

A per-prompt classifier catches obvious attacks. It misses **slow-burn jailbreaks** — five borderline-suspicious prompts in a row, each individually below threshold, that together steer the model into compromising itself.

Our fix: a per-conversation **cumulative score gate**. Every prompt's `P(injection)` is summed across the conversation. When the total crosses `LLM_FIREWALL_CONVERSATION_CUMULATIVE_THRESHOLD` (default `1.5`), every subsequent prompt — even benign ones — is refused until the caller starts a new conversation.

<p align="center">
  <img src="docs/img/screenshots/conversation_panel.png" alt="Conversation panel during a benign exchange" width="46%"/>
  <img src="docs/img/screenshots/conversation_blocked.png" alt="Conversation panel after the cumulative gate fires" width="46%"/>
</p>

*Left: two benign turns, cumulative bar at 0%. Right: three additional adversarial prompts push the cumulative to 2.4417 / 1.50 — the gate fires, the input is locked, and the only escape is the **+ New conversation** button.*

The feature is exposed via the standard chat-completions endpoint:

```python
response = client.chat.completions.create(
    model="firewall-demo",
    messages=[{"role": "user", "content": "..."}],
    extra_body={"conversation_id": "conv_abc123"},  # optional; auto-generated if absent
)
print(response.conversation)  # {"id": ..., "cumulative_score": 0.42, "blocked": False, ...}
```

REST endpoints for explicit conversation lifecycle:

| Endpoint | Purpose |
|---|---|
| `POST /v1/conversations` | Start a new conversation, returns the id |
| `GET /v1/conversations` | List recent conversations (summaries) |
| `GET /v1/conversations/{id}` | Full state including per-turn history |
| `DELETE /v1/conversations/{id}` | Reset (the dashboard's **+ New conversation** button) |

State lives in process memory, capped at `LLM_FIREWALL_CONVERSATION_MAX_TRACKED` (default `1000`) with LRU eviction. Implementation: [`llm_firewall/api/conversations.py`](llm_firewall/api/conversations.py).

---

## Live dashboard

The dashboard at `/dashboard` is a single-page React-free UI showing prompt submission, conversation mode, runtime config, and a live decision feed.

<p align="center">
  <img src="docs/img/screenshots/dashboard_overview.png" alt="Dashboard overview — composing prompts, runtime config, stats, decision log" width="900"/>
</p>

<p align="center">
  <img src="docs/img/screenshots/decision_log.png" alt="Decision log — every request with per-classifier scores and latency" width="900"/>
</p>

Each row in the decision log carries: timestamp, decision, prompt, response, **per-classifier scores and latencies** (keyed `input:<name>` and `output:<name>`), conversation id, and the routing detail. The `Avg Classifier Latency` stat counts only the firewall's own work — not the upstream LLM round-trip — so you can see how fast the screening is, not how slow your model provider is.

Read-only JSON endpoints power the dashboard:

| Endpoint | Returns |
|---|---|
| `GET /api/logs?limit=N` | Most recent N decision log entries (1 ≤ N ≤ 500) |
| `GET /api/stats` | Decision counts + average classifier latency |
| `GET /api/config` | Runtime config (models, threshold, refusal message) |
| `GET /health` | `{"status": "healthy"}` |

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
make install                          # pip install -e ".[dev]"

# Accept Meta's license for Llama-Prompt-Guard-2-86M (gated):
#   https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M
huggingface-cli login

cp .env.example .env                  # edit upstream URL + API key
make run                              # uvicorn on http://localhost:8000
```

Open `http://localhost:8000/dashboard` to send prompts, watch decisions, and try the conversation gate.

For prompts you'd rather not send to a real LLM, run the local dummy upstream:

```bash
make dummy   # uvicorn on :9000 — fixed-response OpenAI-compatible upstream
```

Then point `LLM_FIREWALL_UPSTREAM_CHAT_COMPLETIONS_URL=http://localhost:9000/v1/chat/completions`.

---

## API

Standard OpenAI shape — anything that talks to OpenAI talks to this.

```python
from openai import OpenAI

client = OpenAI(api_key="placeholder", base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="firewall-demo",
    messages=[{"role": "user", "content": "Ignore all previous instructions."}],
)
print(response.choices[0].message.content)  # → the configured refusal message
```

Blocked requests come back as a normal `chat.completion` whose `content` is the configured refusal message and whose body carries an extra `conversation_id` + `conversation` summary. Inspect the dashboard or `/api/logs` to see the actual decision (`ALLOWED` / `BLOCKED` / `DROPPED` / `ERROR`).

Batch endpoint at `POST /v1/chat/completions/batch` accepts up to 1000 prompts per request with configurable concurrency. See [`examples/`](examples/) for sample payloads.

---

## Configuration

The interesting knobs (full list in [`docs/input_classifier/`](docs/input_classifier/)):

| Variable | Default | Purpose |
|---|---|---|
| `LLM_FIREWALL_UPSTREAM_CHAT_COMPLETIONS_URL` | OpenAI | Where the firewall forwards approved prompts |
| `LLM_FIREWALL_CONVERSATION_CUMULATIVE_THRESHOLD` | `1.5` | Sum-of-scores threshold that gates a conversation |
| `LLM_FIREWALL_CONVERSATION_MAX_TRACKED` | `1000` | Soft cap on tracked conversations (LRU eviction) |
| `LLM_FIREWALL_ENABLE_OUTPUT_CLASSIFIERS` | `true` | Skip output validation if you only want input filtering |
| `LLM_FIREWALL_REFUSAL_MESSAGE` | `Sorry, I cannot answer this prompt` | Returned on any block |

All settings load from `.env` or the shell environment. See [`.env.example`](.env.example) for the full template.

---

## Repository layout

```
llm_firewall/
  api/           FastAPI app, routes, dashboard, conversations, dummy upstream
  core/          Settings + outbound HTTP proxy
  classifiers/   Registry, ensemble, language router, HF + pickle backends
  filters/       FilterResult primitive + PII / toxicity filters
  validators/    Input / Output validator wrappers
data/
  input_classifier/
    datasets/    Committed train/val/test parquets + sources.json + manifest.json
    eval/        Committed JSON reports for every evaluated classifier
docs/
  input_classifier/   Datasets, evaluation, models, reproducibility
  img/                Visualizations + dashboard screenshots
scripts/
  input_classifier/   build_dataset.py, evaluate.py, generate_visualizations.py
  capture_dashboard_screenshots.py
dashboard/index.html  Single-page monitoring UI (chat mode + decision log)
tests/                unit/ + integration/ — 95 tests, fully offline
```

---

## Development

```bash
make test        # pytest, unit + integration (95 tests)
make simulate    # scripts/simulate.py — standalone validation run
```

Tests use lightweight fake pickle models, mock the upstream LLM with `respx`, never make real network calls, and cover the conversation gate end-to-end.

To regenerate the committed visualizations after re-running an eval:

```bash
.venv/bin/python scripts/input_classifier/generate_visualizations.py
```

To regenerate the committed dashboard screenshots (requires Playwright + a running firewall):

```bash
.venv/bin/python scripts/capture_dashboard_screenshots.py
```

---

## Team

- Karim Elmasry
- Ahmed Yasser
- Omar Selim
- Ammar Osama

## AI Usage Declaration

AI-assisted development tools (OpenAI Codex/ChatGPT, Anthropic Claude) helped draft, refactor, and test parts of the codebase and documentation. All generated content was reviewed and edited by the team before inclusion.
