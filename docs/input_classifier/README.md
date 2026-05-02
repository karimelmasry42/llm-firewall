# Input classifier

The input classifier inspects the user's prompt before it reaches the
upstream LLM. If the classifier blocks, the firewall returns a refusal
without making the upstream call.

This directory documents how the classifier was built, evaluated, and
chosen — at a level of detail that makes the pipeline reproducible and
the trade-offs auditable.

| Doc | Purpose |
|---|---|
| [`datasets.md`](datasets.md) | Every source dataset, the dedup + balance pipeline, the final manifest |
| [`evaluation.md`](evaluation.md) | The eval harness, every metric we report, the decision gate |
| [`models.md`](models.md) | Models tested, headline numbers, per-source breakdown, the chosen winner |
| [`reproducibility.md`](reproducibility.md) | Exact commands to regenerate every committed artifact from scratch |

## The simplified plan, in five steps

### 1. Choosing datasets → [`datasets.md`](datasets.md)

The previous classifier was a regex tag-substituter feeding a Linear SVM
trained on a dataset that no longer ships in the repo. To do better, we
combined eight permissively-licensed jailbreak / prompt-injection sources
on HuggingFace (neuralchemy, JailbreakHub, xTRam1 safe-guard, jackhhao,
Lakera Gandalf, deepset, rubend18) with a multilingual benign source
(OpenAssistant oasst1, filtered to initial human prompts), then deduped
and class-balanced to ~60% injection / 40% benign.

Two more datasets (DavidTKeane, JailbreakBench) are reserved as held-out
evaluation — never used for training.

The combined parquets (~3 MB total) are committed under
[`data/input_classifier/datasets/`](../../data/input_classifier/datasets/)
along with a [`manifest.json`](../../data/input_classifier/datasets/manifest.json)
recording the exact row counts, dedup collisions, and language histogram
of the build.

### 2. Choosing models → [`models.md`](models.md)

Rather than train from scratch immediately, we benchmarked off-the-shelf
prompt-injection classifiers against the SVM baseline. The current bake-off:

| Model | in-dist F1 | DavidTKeane F1 | JailbreakBench F1 |
|---|---|---|---|
| Old SVM (English) | 0.712 | 0.447 | 0.489 |
| Old SVM (Spanish) | 0.495 | 0.169 | 0.140 |
| `protectai/deberta-v3-base-prompt-injection-v2` (English-only) | **0.885** | 0.661 | 0.000 |
| **`meta-llama/Llama-Prompt-Guard-2-86M` @ tuned thr=0.001 (shipped)** | 0.839 | **0.824** | **0.723** |

The shipped model is genuinely multilingual — Meta trained it on 8
languages of adversarial data. At the default threshold (0.5) it scored
poorly because its score distribution is peaky (most injection mass
below 0.01); a sweep over `val.parquet` found `threshold=0.001` lifts F1
on every held-out set. The trade-off vs protectai/deberta is a 4.6-point
in-distribution loss (mostly English) for a 16.3-point DavidTKeane
(multilingual) gain and a 72.3-point JailbreakBench gain.

`protectai/deberta-v3-base-prompt-injection-v2` is retained in the eval
harness as a strong English baseline. `Llama-Prompt-Guard-2-86M` is
gated — requires accepting Meta's license and `huggingface-cli login`.
See [`models.md`](models.md) for the full bake-off and threshold-sweep
table.

### 3. Training (skipped for v1)

The off-the-shelf bake-off cleared the decision gate
([`evaluation.md`](evaluation.md#decision-gate-for-phase-3)), so we did not
need to train our own model. If a future bake-off falls short, the
fallback is a frozen embedder + sklearn head on the committed parquets —
the script exists at
[`scripts/input_classifier/train_embedding_head.py`](../../scripts/input_classifier/train_embedding_head.py)
when needed.

### 4. Testing → [`evaluation.md`](evaluation.md)

Every classifier runs through `scripts/input_classifier/evaluate.py`
which reports precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix,
per-source and per-language breakdown, and mean + p95 latency on each of:
the in-distribution test set, the DavidTKeane multilingual held-out, and
the JailbreakBench academic benchmark.

The reports are committed under [`data/input_classifier/eval/`](../../data/input_classifier/eval/)
so model comparisons are visible in `git diff`, not just in someone's
laptop.

### 5. Integration

The chosen model is registered in
[`llm_firewall/classifiers/registry.py`](../../llm_firewall/classifiers/registry.py)
behind the `huggingface_sequence` backend, served by `HFSequenceClassifier`
in [`llm_firewall/classifiers/huggingface.py`](../../llm_firewall/classifiers/huggingface.py).
The model file itself is never committed to the repo — `from_pretrained()`
lazy-downloads it from HuggingFace and caches under `~/.cache/huggingface/`,
which keeps the repo tiny and works the same way Llama-Prompt-Guard-2 will when
its auth is set up.

The language router still exists for output-side use, but for input
classification a single multilingual model handles every language —
`INPUT_CLASSIFIER_SPECS_BY_LANGUAGE` returns the same spec list for every
key.

## Status (v1)

- ✅ Dataset pipeline + committed parquets + manifest
- ✅ `HFSequenceClassifier` runtime (handles any `AutoModelForSequenceClassification` checkpoint)
- ✅ Eval harness + committed reports for SVM (English + Spanish), protectai/deberta, and Llama-Prompt-Guard-2-86M
- ✅ `meta-llama/Llama-Prompt-Guard-2-86M` (multilingual, threshold=0.001) shipped as v1 input classifier
- ✅ Threshold sweep documented and committed; protectai/deberta retained as English baseline
- 🔜 Per-deployment configurable threshold via env var (currently hardcoded in the spec)
- 🔜 Native-language Spanish adversarial training data (separate task)
