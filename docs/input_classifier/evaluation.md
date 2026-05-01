# Input-classifier evaluation

How we measure whether an input classifier is good enough to ship, and what
the numbers in the eval reports actually mean.

## What we evaluate

Every classifier is run against three tables. Each table produces its own row
in the JSON report under [`data/input_classifier/eval/`](../../data/input_classifier/eval/):

| Table | Source parquet | What it measures |
|---|---|---|
| `in_distribution_test` | [`test.parquet`](../../data/input_classifier/datasets/test.parquet) | The 10% test slice held out from the combined training pool. Same source distribution as training; closest to "how well did the classifier learn the data we showed it." |
| `held_out_davidtkeane` | [`eval_davidtkeane.parquet`](../../data/input_classifier/datasets/eval_davidtkeane.parquet) | 112-row curated test suite, 8 languages. The closest thing we have to a multilingual stress test in v1. |
| `held_out_jailbreakbench` | [`eval_jailbreakbench.parquet`](../../data/input_classifier/datasets/eval_jailbreakbench.parquet) | JailbreakBench academic benchmark (200 rows, balanced harmful + benign). Lets us compare against published numbers in literature. |

Held-out sets are never deduped against training. If a row leaks, that's
information about generalization worth seeing.

## Metrics

For each table the harness reports:

- **Confusion matrix**: TP / FP / TN / FN, where positive = injection.
- **Precision** = TP / (TP + FP). Of the prompts we flagged as injection,
  what fraction actually were.
- **Recall** = TP / (TP + FN). Of the actual injections, what fraction we
  caught. **For a security tool this is usually the metric that matters
  most** — a missed jailbreak is a security failure.
- **F1** = harmonic mean of precision and recall. Single summary number;
  useful for ranking models in the bake-off.
- **Accuracy** = (TP + TN) / total. Easy to report, but misleading on
  imbalanced data; prefer F1.
- **ROC-AUC**: area under the precision-recall ROC curve. Threshold-free
  measure of how well the score separates the two classes.
- **PR-AUC** (`average_precision_score`): area under precision-recall curve.
  More informative than ROC-AUC when the positive class is rare.
- **Latency**:
  - **`mean_latency_ms`**: average wall-clock time per `predict()` call.
  - **`p95_latency_ms`**: 95th percentile. Sort all measured latencies
    ascending; the value at the 95% position. Reported alongside the mean
    because **means hide the slow tail**, and tail latency is what a real
    user experiences when they're unlucky.
  - We also expose `p50` (median) implicitly through the mean comparison —
    if mean ≫ p50 you have a long tail.

Latency is measured per-row inside the harness using `perf_counter()`. It
covers preprocessing + tokenization + forward pass + result construction —
the full predict path. It does **not** include model load time; loading
happens once outside the measured loop.

## Per-source and per-language breakdowns

Every row in the parquet carries a `source` column (which dataset it came
from) and a `lang` column. The harness produces nested `by_source` and
`by_lang` tables alongside the headline numbers, so you can answer
questions like:

- "Does the new model regress on JailbreakHub specifically?"
- "Does it work on Spanish prompts?"
- "Which source contributes the most false positives?"

These breakdowns are what we use to gate Phase 3 vs Phase 4: an aggregate
F1 jump means little if it comes at the cost of one specific source
collapsing.

## How we read the numbers

When comparing two classifiers (e.g. SVM baseline vs Prompt-Guard-2):

1. **Look at recall on the in-distribution test set first.** If recall is
   below where the baseline already sits, the new model is worse at the
   core job and we stop reading.
2. **Then look at precision.** A model that catches everything (recall=1.0)
   but fires on every benign prompt is unusable. We want both above 0.85
   ideally.
3. **Look at the held-out tables.** A model that excels on the in-
   distribution set but tanks on JailbreakBench is overfit to our
   specific data combination. Held-out numbers should be in the same
   ballpark as in-distribution.
4. **Look at the per-source breakdown.** Particularly on JailbreakHub
   (the in-the-wild source) and Lakera/Gandalf (real adversarial
   attempts). A model that aces synthetic data but misses real attacks is
   a bad trade.
5. **Look at p95 latency.** With GPU at inference, p95 should be under
   100ms for a reasonable model. CPU can easily be 10× slower.

## Decision gate (for Phase 3)

We promote `Prompt-Guard-2-86M` (or any other off-the-shelf candidate) over
the SVM only if all of the following hold:

- **F1 ≥ 0.85** on `in_distribution_test`.
- **No regression** vs the Spanish SVM on Spanish-tagged DavidTKeane rows
  (recall must not drop on `lang == "es"`).
- **F1 improvement of at least +10 points** on at least two of the larger
  training sources (`walledai_jailbreakhub`,
  `xtram1_safe_guard_prompt_injection`, `neuralchemy_prompt_injection`)
  in the per-source breakdown.

If those don't hold, we move to Phase 4 (train a frozen-embedder + head).

## Reproducing the numbers

The eval reports under `data/input_classifier/eval/` are committed. To
regenerate any of them:

```bash
python scripts/input_classifier/evaluate.py --classifier svm_baseline
python scripts/input_classifier/evaluate.py --classifier prompt_guard_2
```

Each command writes `data/input_classifier/eval/<classifier>.json`
overwriting the previous version. Diff with `git diff` to see drift.

To smoke-test on a small sample:

```bash
python scripts/input_classifier/evaluate.py --classifier svm_baseline --limit 50
```

The limit is applied per parquet, so a `--limit 50` run produces a
50/50/50 row report.
