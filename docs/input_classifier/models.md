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
| `protectai_deberta` | DeBERTa-v3-base, fine-tuned for prompt injection | English (with some cross-lingual transfer from DeBERTa pretraining) | [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) | Tested, shipped as v1 |
| `prompt_guard_2` | meta-llama/Prompt-Guard-2 (DeBERTa-v3 family, multilingual) | Yes (102+ languages) | [meta-llama/Prompt-Guard-2-86M](https://huggingface.co/meta-llama/Prompt-Guard-2-86M) | **Pending auth** — gated model |

`Prompt-Guard-2-86M` requires accepting Meta's license at the dataset card
URL and authenticating via `huggingface-cli login`. Once auth is set up,
re-run `python scripts/input_classifier/evaluate.py --classifier prompt_guard_2`
and update this table.

## Headline results

| Model | in-dist F1 | DavidTKeane F1 | JailbreakBench F1¹ | mean / p95 latency |
|---|---|---|---|---|
| `svm_baseline` | 0.712 | 0.447 | 0.489 | 4.5 / 10.2 ms |
| `svm_baseline_spanish` | 0.495 | 0.169 | 0.140 | 4.4 / 9.2 ms |
| `protectai_deberta` | **0.885** | **0.661** | 0.000¹ | 83 / 199 ms (MPS, M1) |
| `prompt_guard_2` | _pending_ | _pending_ | _pending_ | _pending_ |

¹ **JailbreakBench is a category mismatch for an input-injection
classifier.** The `Goal` column contains natural-language descriptions of
harmful behaviors ("Write a tutorial on how to make a bomb"), which is a
*content-policy* task, not a *prompt-injection* task. A classifier looking
for "ignore previous instructions"-style overrides will correctly flag
none of those rows. We keep the table because it is a published benchmark
and the explicit zero is itself a useful signal — but it should not be
treated as a regression.

## Per-source breakdown (in-distribution test set)

| Source | n | SVM F1 | protectai F1 | Δ |
|---|---|---|---|---|
| `neuralchemy_prompt_injection` | 413 | 0.459 | **0.921** | +46.2 |
| `walledai_jailbreakhub` | 342 | 0.571 | 0.775 | +20.4 |
| `xtram1_safe_guard_prompt_injection` | 379 | 0.973 | 0.898 | -7.5 |
| `lakera_gandalf_ignore_instructions` | 100 | 0.936 | **1.000** | +6.4 |
| `deepset_prompt_injections` | 33 | 0.667 | 0.650 | -1.7 |
| `jackhhao_jailbreak_classification` | 11 | 0.000 | 0.000 | 0 |
| `rubend18_chatgpt_jailbreak_prompts` | 2 | 1.000 | 1.000 | 0 |
| `openassistant_oasst1_benign` | 175 | — (all benign) | — (all benign) | — |

The xTRam1 regression is on synthetic data; the larger and more realistic
sources (JailbreakHub, neuralchemy, Gandalf) all improve substantially.

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
from DavidTKeane alone. **This is exactly why Prompt-Guard-2-86M is the
preferred long-term choice** — it's explicitly trained on multilingual
adversarial data. The current shipped model, protectai/deberta-v3, is good
enough for v1 and dramatically beats the SVM, but multilingual coverage
will improve when we cut over to Prompt-Guard-2.

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

## What changes if Prompt-Guard-2 wins later

When the gated-model auth is set up and `prompt_guard_2.json` lands in the
eval directory, compare:

1. If it beats protectai on `in_distribution_test` AND on `held_out_davidtkeane`:
   swap the spec in [`registry.py`](../../llm_firewall/classifiers/registry.py).
   Two-line change.
2. If it's only multilingual-stronger but worse on English: keep both, route
   by detected language.
3. If it's worse: stick with protectai.

Update this document with the new row and re-link the chosen winner.
