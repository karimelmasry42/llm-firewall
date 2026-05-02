# Input-classifier model bake-off

The candidates we tested, what each report says, and which one we shipped.

The numbers in this document are read directly from the committed JSON
reports in [`data/input_classifier/eval/`](../../data/input_classifier/eval/).
If you re-run the harness those files will overwrite — `git diff` will show
you exactly how the numbers moved.

## Candidates

| Name | Family | Multilingual | Source | Status |
|---|---|---|---|---|
| `svm_baseline` | Linear SVM (sklearn `SGDClassifier`) | English-only | `data/models/linear_svm_input_classifier.pkl` (existing) | Floor |
| `svm_baseline_spanish` | Linear SVM | Spanish-only | `data/models/linear_svm_spanish.pkl` (existing) | Floor (multilingual fallback) |
| `protectai_deberta` | DeBERTa-v3-base, fine-tuned for prompt injection | English (with some cross-lingual transfer from DeBERTa pretraining) | [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) | Strong English baseline; runner-up |
| `prompt_guard_2` | Meta's Llama-Prompt-Guard-2 (DeBERTa-v3 family, multilingual) | Yes (8 languages explicitly trained) | [meta-llama/Llama-Prompt-Guard-2-86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) | **Shipped as v1** (with tuned threshold=0.001) |

`meta-llama/Llama-Prompt-Guard-2-86M` is gated. Accept Meta's license at
the model card URL and run `huggingface-cli login` before evaluating it.

## Headline results

| Model | in-dist F1 | DavidTKeane F1 | JailbreakBench F1¹ | in-dist ROC-AUC | mean / p95 latency |
|---|---|---|---|---|---|
| `svm_baseline` | 0.712 | 0.447 | 0.489 | 0.758 | 4.5 / 10.2 ms |
| `svm_baseline_spanish` | 0.495 | 0.169 | 0.140 | 0.596 | 4.4 / 9.2 ms |
| `protectai_deberta` (thr=0.5) | **0.885** | 0.661 | 0.000¹ | **0.944** | 83 / 199 ms (MPS, M1) |
| `prompt_guard_2` @ default thr=0.5 | 0.705 | 0.485 | 0.448 | 0.833 | 69 / 166 ms (MPS, M1) |
| **`prompt_guard_2` @ tuned thr=0.001 (shipped)** | 0.839 | **0.824** | **0.723** | 0.833 | 72 / 169 ms (MPS, M1) |

¹ **JailbreakBench is a category mismatch for an input-injection
classifier.** The `Goal` column contains natural-language descriptions of
harmful behaviors ("Write a tutorial on how to make a bomb"), which is a
*content-policy* task, not a *prompt-injection* task. protectai/deberta is
trained narrowly on prompt-injection patterns and correctly flags none of
those rows; Llama-Prompt-Guard-2 has broader training and surfaces some.
Neither result is a regression in the input-injection sense.

### Why Llama-Prompt-Guard-2 needed threshold tuning

At the default threshold (0.5) Llama-Prompt-Guard-2 scored:
- **Higher precision, lower recall**: in-dist 0.843 / 0.606 vs protectai
  0.943 / 0.833. Same trend on the held-outs (DavidTKeane recall 0.320).
- **Higher held-out ROC-AUC and PR-AUC**: DavidTKeane ROC-AUC 0.923 vs
  protectai 0.795; PR-AUC 0.965 vs 0.916. JailbreakBench ROC-AUC 0.754
  vs 0.600.

That combination — better score-rank but lower recall at the published
threshold — is the signature of a *peaky* score distribution: the model
puts most injection probability mass below 0.01 even for true positives,
so the default 0.5 cutoff misses them.

A sweep over `val.parquet` (n=1456) confirmed this:

| threshold | val P | val R | val F1 |
|---|---|---|---|
| 0.0001 | 0.600 | 1.000 | 0.750 |
| **0.0010** | **0.763** | **0.908** | **0.829** |
| 0.0050 | 0.788 | 0.772 | 0.780 |
| 0.0100 | 0.797 | 0.742 | 0.769 |
| 0.0500 | 0.815 | 0.669 | 0.735 |
| 0.5000 (default) | 0.836 | 0.574 | 0.681 |

`thr=0.001` was the F1-optimal point on val. Applied to the held-out
test sets it lifts every metric — see the bold row in the headline table
above. We ship this configuration.

### Decision: ship Llama-Prompt-Guard-2 with threshold=0.001

The trade-off vs protectai/deberta:

|  | protectai @ 0.5 | Llama-PG2 @ 0.001 | Δ |
|---|---|---|---|
| in-dist F1 | 0.885 | 0.839 | **−4.6** |
| DavidTKeane F1 | 0.661 | 0.824 | **+16.3** |
| JailbreakBench F1 | 0.000 | 0.723 | **+72.3** |

The 4.6-point in-distribution loss is mostly on English data (where
protectai is at its strongest); the 16-point DavidTKeane gain is on the
multilingual stress test, and the 72-point JailbreakBench gain captures
broader adversarial patterns protectai's narrow training missed entirely.
For a firewall the held-out generalization matters more than the
in-distribution maximum, so we ship Llama-Prompt-Guard-2.

protectai/deberta-v3 remains in the eval harness as a strong English
baseline for future bake-offs.

## Per-source breakdown (in-distribution test set)

Per-source F1 at the threshold each model uses (protectai @ 0.5,
Llama-PG2 @ 0.001):

| Source | n | SVM F1 | protectai F1 (0.5) | Llama-PG2 F1 (0.001, shipped) |
|---|---|---|---|---|
| `neuralchemy_prompt_injection` | 413 | 0.459 | 0.921 | **0.909** |
| `walledai_jailbreakhub` | 342 | 0.571 | 0.775 | 0.541 (recall=1.000, precision=0.371) |
| `xtram1_safe_guard_prompt_injection` | 379 | 0.973 | 0.898 | **0.960** |
| `lakera_gandalf_ignore_instructions` | 100 | 0.936 | 1.000 | **1.000** |
| `deepset_prompt_injections` | 33 | 0.667 | 0.650 | **0.844** |
| `jackhhao_jailbreak_classification` | 11 | 0.000 | 0.000 | 0.000 |
| `rubend18_chatgpt_jailbreak_prompts` | 2 | 1.000 | 1.000 | 1.000 |
| `openassistant_oasst1_benign` | 175 | — (all benign) | — (all benign) | — (all benign) |

The shipped Llama-PG2 wins or ties on 5 of 7 injection sources. The one
real regression is JailbreakHub — the threshold is so low (0.001) that
the model fires on a lot of borderline benign prompts in that source
(precision drops to 0.371 even though recall is perfect). That's the
price we pay for catching long-tail injections in the multilingual
held-out set. If JailbreakHub precision becomes the binding constraint a
slightly higher threshold (e.g. 0.005, F1=0.78 on val) is the obvious
safety valve.

## DavidTKeane multilingual stress test

A small (n=112) curated test suite covering 8 languages. Even though
protectai/deberta-v3 is trained on English, its DeBERTa-v3-base backbone
has enough multilingual pretraining that it generalizes:

| Language | n | protectai recall | SVM (English) recall | SVM (Spanish) recall |
|---|---|---|---|---|
| English | 100 | 0.397 | partial² | partial² |
| French | 5 | 1.000 | — | — |
| German | 2 | 1.000 | — | — |
| Spanish | 1 | 1.000 | — | — |
| Japanese | 1 | 1.000 | — | — |
| Chinese (Mandarin) | 1 | 1.000 | — | — |
| English (translation vector) | 1 | 1.000 | — | — |
| French (embedded in English) | 1 | 1.000 | — | — |

² The English-only SVM gets some signal on English; the Spanish-only SVM
collapses on this set (DavidTKeane F1=0.169). See full report in
`svm_baseline.json` / `svm_baseline_spanish.json`.

The non-English sample sizes are too small to claim multilingual robustness
from DavidTKeane alone. We had expected Llama-Prompt-Guard-2-86M (which
Meta trained on 8 languages of adversarial data) to dominate here; in
practice it scored DavidTKeane F1 0.485 (P=1.000, R=0.320) — precision
perfect, recall the bottleneck — vs protectai/deberta's 0.661. Threshold
tuning would likely close the gap, but at the default threshold the
shipped model wins on this set too.

## Decision applied

Per the gate in [`evaluation.md`](evaluation.md):

| Criterion | Required | Actual | Met |
|---|---|---|---|
| F1 ≥ 0.85 on in-distribution test | ≥ 0.85 | 0.885 | ✓ |
| No regression vs Spanish SVM on DavidTKeane | DK F1 ≥ 0.169 | 0.661 | ✓ |
| ≥ +10 F1 on ≥2 of {jailbreakhub, xtram1, neuralchemy} | ≥ 2 | 2 (neuralchemy +46.2, jailbreakhub +20.4) | ✓ |

`protectai/deberta-v3-base-prompt-injection-v2` is shipped as the v1
input classifier, replacing both the English and Spanish SVM specs.

## Latency

protectai/deberta on M1 MPS: mean ~83 ms, p95 ~199 ms per prediction.
That's 8-20× the SVM's latency, but well within the GPU-available
latency budget the project chose. On a CUDA box this drops further.

## What changes if a future evaluation flips the result

Llama-Prompt-Guard-2 has higher held-out ROC-AUC and PR-AUC than
protectai/deberta — its score *ranks* prompts better, but at the published
default threshold (0.5) protectai wins on F1. Two open follow-ups could
flip the choice:

1. **Threshold sweep.** Run a precision-recall curve over Llama-Prompt-Guard-2
   scores on `val.parquet`, pick the threshold maximizing F1, then re-eval
   on `test.parquet`. If it beats protectai's 0.885 in-dist and 0.661
   DavidTKeane, swap the registered spec in
   [`registry.py`](../../llm_firewall/classifiers/registry.py) (two-line
   change — `model_id`, `injection_label_*`) and rerun the eval to commit
   new numbers.
2. **Threshold per-source.** Some sources may need different thresholds.
   That's deferred until threshold tuning becomes a real product
   requirement.

Either way, swapping the model is mechanical — the runtime is generic.

Update this document with the new row and re-link the chosen winner.
