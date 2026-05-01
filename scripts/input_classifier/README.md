# `scripts/input_classifier/`

Offline tooling for the input-classifier overhaul. Nothing here ships at
runtime; everything is invoked manually to (re)build datasets and produce
evaluation reports.

Run them in order on a clean checkout:

```bash
# 1. Build the combined training/test parquets + held-out evaluation parquets.
#    Reads data/input_classifier/datasets/sources.json (pinned HF revisions).
#    Writes train/val/test/eval_*.parquet and manifest.json next to it.
python scripts/input_classifier/build_dataset.py

# 2. Evaluate any registered classifier against the committed parquets.
#    Writes data/input_classifier/eval/<classifier>.json.
python scripts/input_classifier/evaluate.py --classifier svm_baseline
python scripts/input_classifier/evaluate.py --classifier prompt_guard_2

# 3. (Phase 4, only if Phase 3 falls short — embedder + sklearn head training)
python scripts/input_classifier/train_embedding_head.py
```

Both scripts require the dev extras:

```bash
pip install -e '.[dev]'
```

See `docs/input_classifier/` for the narrative documentation: dataset
methodology, evaluation harness details, model bake-off results, and the
exact reproduction recipe.
