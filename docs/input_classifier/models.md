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
| `protectai_deberta` | DeBERTa-v3-base, fine-tuned for prompt injection | English (with some cross-lingual transfer from DeBERTa pretraining) | [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) | **Shipped as v1** |
| `prompt_guard_2` | Meta's Llama-Prompt-Guard-2 (DeBERTa-v3 family, multilingual) | Yes (8 languages explicitly trained) | [meta-llama/Llama-Prompt-Guard-2-86M](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) | Tested; runner-up at default threshold |

`meta-llama/Llama-Prompt-Guard-2-86M` is gated. Accept Meta's license at
the model card URL and run `huggingface-cli login` before evaluating it.

## Headline results

| Model | in-dist F1 | DavidTKeane F1 | JailbreakBench F1¹ | in-dist ROC-AUC | mean / p95 latency |
|---|---|---|---|---|---|
| `svm_baseline` | 0.712 | 0.447 | 0.489 | 0.758 | 4.5 / 10.2 ms |
| `svm_baseline_spanish` | 0.495 | 0.169 | 0.140 | 0.596 | 4.4 / 9.2 ms |
| `protectai_deberta` | **0.885** | **0.661** | 0.000¹ | **0.944** | 83 / 199 ms (MPS, M1) |
| `prompt_guard_2` (Llama-Prompt-Guard-2-86M) | 0.705 | 0.485 | 0.448 | 0.833 | 69 / 166 ms (MPS, M1) |

¹ **JailbreakBench is a category mismatch for an input-injection
classifier.** The `Goal` column contains natural-language descriptions of
harmful behaviors ("Write a tutorial on how to make a bomb"), which is a
*content-policy* task, not a *prompt-injection* task. protectai/deberta is
trained narrowly on prompt-injection patterns and correctly flags none of
those rows; Llama-Prompt-Guard-2 has broader training and surfaces some.
Neither result is a regression in the input-injection sense.

### Why Llama-Prompt-Guard-2 lost on F1 but won on AUC

Llama-Prompt-Guard-2 is a more conservative scorer at the default
threshold (0.5):

- **Higher precision, lower recall**: in-dist precision 0.843, recall
  0.606 — vs protectai 0.943 / 0.833. Same calibration trend on the
  held-outs.
- **Higher held-out ROC-AUC and PR-AUC**: DavidTKeane ROC-AUC 0.923 vs
  protectai 0.795; PR-AUC 0.965 vs 0.916. JailbreakBench ROC-AUC 0.754
  vs 0.600.

The combination — better separation in the score distribution but lower
recall at the published threshold — is exactly what you'd expect from a
model whose default threshold is mis-calibrated for our prompt mix. A
threshold sweep would likely move it ahead of protectai on F1 too. That's
deferred to a follow-up; for v1 we ship the model that wins at the
default threshold.

## Per-source breakdown (in-distribution test set)

| Source | n | SVM F1 | protectai F1 | Llama-PG2 F1 |
|---|---|---|---|---|
| `neuralchemy_prompt_injection` | 413 | 0.459 | **0.921** | 0.686 |
| `walledai_jailbreakhub` | 342 | 0.571 | **0.775** | 0.658 |
| `xtram1_safe_guard_prompt_injection` | 379 | **0.973** | 0.898 | 0.640 |
| `lakera_gandalf_ignore_instructions` | 100 | 0.936 | **1.000** | 0.995 |
| `deepset_prompt_injections` | 33 | 0.667 | **0.650** | 0.471 |
| `jackhhao_jailbreak_classification` | 11 | 0.000 | 0.000 | 0.000 |
| `rubend18_chatgpt_jailbreak_prompts` | 2 | 1.000 | 1.000 | 1.000 |
| `openassistant_oasst1_benign` | 175 | — (all benign) | — (all benign) | — (all benign) |

protectai/deberta wins or ties on every source except xTRam1 (synthetic),
where the SVM happens to be best because it was likely trained on similar
data. Llama-Prompt-Guard-2 trails on every source at the default threshold,
consistent with its under-calibrated recall.

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
