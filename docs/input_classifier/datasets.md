# Input-classifier datasets

This document lists every dataset combined into the input-classifier training
and evaluation sets, why each was chosen or excluded, the cleaning and
balancing pipeline, and exactly what ends up in the committed parquet files
under [`data/input_classifier/datasets/`](../../data/input_classifier/datasets/).

The build is fully reproducible: the [`sources.json`](../../data/input_classifier/datasets/sources.json)
file pins HuggingFace revisions, [`build_dataset.py`](../../scripts/input_classifier/build_dataset.py)
turns those into committed parquets, and the resulting
[`manifest.json`](../../data/input_classifier/datasets/manifest.json) records
the actual numbers (per-source row counts, dedup collisions, final balance,
language histogram, build timestamp, git SHA).

## Sources used for training

| Dataset | Rows | Class balance | License | Why we use it |
|---|---|---|---|---|
| [`neuralchemy/Prompt-injection-dataset`](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset) (`core` config) | 6,274 | ~60% malicious / 40% benign | permissive | Already-balanced primary training set with rich metadata (`category`, `severity`) and clean train/val/test splits. |
| [`walledai/JailbreakHub`](https://huggingface.co/datasets/walledai/JailbreakHub) | 15,140 | 9% jailbreak / 91% benign | MIT | Largest in-the-wild collection — Reddit, Discord, FlowGPT, JailbreakChat. Captures real adversarial language patterns the synthetic datasets miss. |
| [`xTRam1/safe-guard-prompt-injection`](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection) | 10,296 | 30% injection / 70% safe | **unspecified** ⚠ | Synthetic, large, broad attack-category coverage. License missing from the dataset card — verify before commercial deployment. |
| [`jackhhao/jailbreak-classification`](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | 1,306 | ~40/60 jb/benign | Apache 2.0 | Sourced from the well-known verazuo/jailbreak_llms repo on the injection side and OpenOrca/GPTeacher on the benign side. Useful for diversity. |
| [`Lakera/gandalf_ignore_instructions`](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) | 1,000 | 100% injection | MIT | Real attempts submitted to the Gandalf game by adversarial users — high signal, semantically diverse around "ignore previous instructions." |
| [`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections) | 662 | imbalanced | Apache 2.0 | Curated, widely used, includes German alongside English. |
| [`rubend18/ChatGPT-Jailbreak-Prompts`](https://huggingface.co/datasets/rubend18/ChatGPT-Jailbreak-Prompts) | 79 | 100% jailbreak | permissive | Tiny but DAN-style heavy. Adds coverage of canonical persona-jailbreaks. |
| [`OpenAssistant/oasst1`](https://huggingface.co/datasets/OpenAssistant/oasst1) (filtered to `role=='prompter' AND parent_id IS NULL`) | 88,800 (subsampled) | 100% benign | Apache 2.0 | Diverse multilingual benign prompts (24 languages). Heavily subsampled before joining — see balancing below. |

## Held-out evaluation sets

These are **never used for training**. They exist only so the eval harness can
report generalization to data we didn't combine into the training pool.

| Dataset | Rows | Why held out |
|---|---|---|
| [`DavidTKeane/ai-prompt-ai-injection-dataset`](https://huggingface.co/datasets/DavidTKeane/ai-prompt-ai-injection-dataset) | 112 | Curated test suite covering 8 languages and 11 attack categories. The single best multilingual stress test we have for v1, where we deliberately train on English-dominant data. |
| [`JailbreakBench/JBB-Behaviors`](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) (`behaviors` config, both `harmful` + `benign` splits) | 200 (100 harmful + 100 benign) | Standardized academic benchmark distilled from AdvBench, HarmBench, and TDC. Lets us compare against published numbers. |

## Excluded sources

| Dataset | Reason |
|---|---|
| `qualifire/Qualifire-prompt-injection-benchmark` | CC-BY-NC-4.0 (non-commercial). Cannot use it to train a deployed classifier. |

## Combination pipeline

Implemented in [`scripts/input_classifier/build_dataset.py`](../../scripts/input_classifier/build_dataset.py).
The script is deterministic given a fixed `--seed` (default 7).

1. **Load** each source via `datasets.load_dataset(..., revision=<pinned>)`,
   honoring per-source filters (oasst1 keeps only initial human prompts;
   DavidTKeane drops the `VARIES` rows; JailbreakBench loads its `harmful`
   and `benign` splits with explicit per-split labels).
2. **Normalize** to the common schema `(text, label, source, lang, length)`.
   `label` is `0` (benign) or `1` (injection). Source-specific label
   conventions are mapped via `sources.json` (`label_field`, `label_map`,
   `fixed_label`).
3. **Truncate** any row whose text exceeds `--max-text-chars` (default 10,000)
   to keep the parquet small and to drop the JailbreakHub long-tail outliers
   (one row reaches ~71k chars).
4. **Concatenate** every source into one pool.
5. **Deduplicate** in two passes:
   - **Exact**: drop any row whose `text` exactly matches an earlier row.
   - **Near-duplicate**: drop any row whose SHA-256 hash of
     `text[:300].lower().strip()` matches an earlier row. This catches
     trivial paraphrases and the platform-cross-posting that JailbreakHub is
     known to contain.
   Earlier sources win, so neuralchemy keeps its rows when they collide with
   later sources.
6. **Class-balance correction**. We target ~60% injection / 40% benign in the
   final dataset (matches neuralchemy's natural ratio). Without correction
   we'd land near 70% benign / 30% injection because oasst1 dominates the
   benign pool, biasing the model toward "is this normal-looking" rather
   than "is this adversarial." We compute `desired_benign = injection_count
   * (1 - 0.6) / 0.6` and subsample the benign rows down to that count,
   stratified by `(source, lang)` so no source or language is wiped out.
7. **Split** 80/10/10 stratified by `(source, label)`. Stratification
   guarantees every source and class appears in train, val, and test in
   proportion.
8. **Write** `train.parquet`, `val.parquet`, `test.parquet`, plus
   `eval_davidtkeane.parquet` and `eval_jailbreakbench.parquet` (held-out;
   not deduped against training, so the eval reflects the upstream verbatim).
9. **Emit** `manifest.json` with per-source row counts (pre and post dedup),
   final class balance, language histogram, dedup collision counts, build
   timestamp, and git SHA.

## Final dataset summary

The actual numbers depend on the most recent build — they live in
[`manifest.json`](../../data/input_classifier/datasets/manifest.json) so
they always match the committed parquets exactly. After the first run the
order of magnitude looks like:

- **Training pool (post-dedup, post-balance):** ~14k rows, ~60/40
  injection/benign, 7 source labels, 12+ languages.
- **Held-out evaluation:** 112 (DavidTKeane) + 200 (JailbreakBench) = 312 rows.

To regenerate from scratch:

```bash
python scripts/input_classifier/build_dataset.py
```

Output goes to `data/input_classifier/datasets/`. Re-running with the same
seed produces byte-identical parquets.

## Reproducibility caveats

- `sources.json` currently pins each dataset to `revision: "main"`. After
  a successful build you can replace `"main"` with the resolved commit SHA
  (the build doesn't yet write that into the manifest — a small follow-up).
- `xTRam1/safe-guard-prompt-injection` has no license declared on its
  dataset card. We're using it under fair-use/research assumptions; confirm
  with the dataset author before any commercial deployment.
- Native-language adversarial data (Spanish in particular) is deliberately
  out of scope for v1. We rely on the multilingual oasst1 prompts for benign
  multilingual coverage and on the multilingual model itself for cross-
  lingual transfer on the injection side. Adding native Spanish jailbreak
  data is a separate follow-up task.
