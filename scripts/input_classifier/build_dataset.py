"""
Combine, dedupe, balance, and split the input-classifier datasets.

Reads sources.json (pinned HF revisions per dataset), downloads each source
via the `datasets` library, normalizes them to a common schema, deduplicates
across all sources, applies class-balance correction, and writes the
training/validation/test parquets plus held-out evaluation parquets to
data/input_classifier/datasets/.

A manifest.json is also written recording per-source row counts (pre and post
dedup), final class balance, language histogram, dedup collision counts, build
timestamp, and the script's git SHA — making every committed dataset
artifact traceable back to the exact build that produced it.

Usage:
    python scripts/input_classifier/build_dataset.py
    python scripts/input_classifier/build_dataset.py --out data/input_classifier/datasets
    python scripts/input_classifier/build_dataset.py --target-ratio 0.6 --max-text-chars 10000

The script is idempotent and deterministic given a fixed `--seed` (default 7).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("build_dataset")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = REPO_ROOT / "data" / "input_classifier" / "datasets" / "sources.json"
DEFAULT_OUT = REPO_ROOT / "data" / "input_classifier" / "datasets"


@dataclass
class Row:
    text: str
    label: int
    source: str
    lang: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "source": self.source,
            "lang": self.lang,
            "length": len(self.text),
        }


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        )
        return bool(result.strip())
    except Exception:
        return False


def _resolve_hf_revision(repo_id: str, revision: str) -> str | None:
    """Resolve a branch/tag (e.g. 'main') to its underlying commit SHA on HF.

    Recorded in manifest.json so reviewers can verify they got byte-identical
    upstream data, even if the branch advances later. Returns None on failure
    rather than blocking the build — the parquet content is what matters; the
    SHA is metadata.
    """
    try:
        from huggingface_hub import HfApi  # type: ignore

        info = HfApi().dataset_info(repo_id, revision=revision)
        return getattr(info, "sha", None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not resolve revision for %s@%s: %s", repo_id, revision, exc)
        return None


def _ratio_argtype(value: str) -> float:
    """argparse type for --target-ratio: must lie strictly in (0, 1)."""
    try:
        v = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--target-ratio must be a float, got {value!r}"
        ) from exc
    if not (0.0 < v < 1.0):
        raise argparse.ArgumentTypeError(
            f"--target-ratio must be strictly between 0 and 1 (exclusive); got {v}"
        )
    return v


def _load_hf(spec: dict[str, Any]):
    """Load a HuggingFace dataset with the given spec; concatenate splits if needed."""
    from datasets import concatenate_datasets, load_dataset

    name = spec["hf_id"]
    config = spec.get("config")
    revision = spec.get("revision") or "main"
    split = spec.get("split")

    if "split_per_label" in spec:
        # JailbreakBench-style: per-split label assignment.
        loaded = []
        for split_name, label in spec["split_per_label"].items():
            ds = load_dataset(
                name, config, revision=revision, split=split_name, trust_remote_code=False
            )
            ds = ds.add_column("__forced_label", [label] * len(ds))
            loaded.append(ds)
        return concatenate_datasets(loaded)

    if split == "all" or split is None:
        ds = load_dataset(name, config, revision=revision, trust_remote_code=False)
        # ds is a DatasetDict; concat all splits.
        if hasattr(ds, "values"):
            return concatenate_datasets(list(ds.values()))
        return ds

    return load_dataset(name, config, revision=revision, split=split, trust_remote_code=False)


def _resolve_label(row: dict[str, Any], spec: dict[str, Any]) -> int | None:
    """Extract a 0/1 label from a row according to its spec."""
    if "__forced_label" in row:
        return int(row["__forced_label"])

    if spec.get("fixed_label") is not None:
        return int(spec["fixed_label"])

    field = spec.get("label_field")
    if field is None:
        return None

    raw = row.get(field)
    if raw is None:
        return None

    label_map = spec.get("label_map")
    if label_map and raw in label_map:
        return int(label_map[raw])

    if spec.get("drop_label_values") and raw in spec["drop_label_values"]:
        return None

    if isinstance(raw, bool):
        return int(raw)

    if isinstance(raw, (int, float)):
        return int(raw)

    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"1", "true", "injection", "jailbreak", "blocked", "malicious", "unsafe"}:
            return 1
        if s in {"0", "false", "benign", "pass", "safe"}:
            return 0

    return None


def _apply_filter(ds, spec: dict[str, Any]):
    flt = spec.get("filter")
    if not flt:
        return ds
    role = flt.get("role")
    parent_null = flt.get("parent_id_is_null", False)

    def keep(row: dict[str, Any]) -> bool:
        if role and row.get("role") != role:
            return False
        if parent_null and row.get("parent_id") is not None:
            return False
        return True

    return ds.filter(keep)


def _detect_lang(row: dict[str, Any], spec: dict[str, Any]) -> str:
    field = spec.get("language_field")
    if field and row.get(field):
        return str(row[field])
    declared = spec.get("language")
    if declared and declared not in {"multi", "mixed"}:
        return declared
    if "lang" in row and row["lang"]:
        return str(row["lang"])
    return "en"


def _load_source(spec: dict[str, Any], max_text_chars: int) -> tuple[list[Row], dict[str, Any]]:
    """Load and normalize one source. Returns rows + per-source stats."""
    logger.info("loading %s (%s)", spec["name"], spec["hf_id"])
    requested_revision = spec.get("revision") or "main"
    resolved_sha = _resolve_hf_revision(spec["hf_id"], requested_revision)
    ds = _load_hf(spec)
    ds = _apply_filter(ds, spec)

    rows: list[Row] = []
    label_counts: Counter[int] = Counter()
    skipped = 0
    text_field = spec["text_field"]

    for row in ds:
        text = row.get(text_field)
        if not isinstance(text, str):
            skipped += 1
            continue
        text = text.strip()
        if not text:
            skipped += 1
            continue
        if len(text) > max_text_chars:
            text = text[:max_text_chars]

        label = _resolve_label(row, spec)
        if label is None:
            skipped += 1
            continue

        lang = _detect_lang(row, spec)
        rows.append(Row(text=text, label=label, source=spec["name"], lang=lang))
        label_counts[label] += 1

    stats = {
        "raw_rows_after_filter": len(ds),
        "kept_rows": len(rows),
        "skipped_rows": skipped,
        "label_counts": {str(k): v for k, v in label_counts.items()},
        "hf_id": spec["hf_id"],
        "requested_revision": requested_revision,
        "resolved_revision": resolved_sha,
    }
    logger.info(
        "  %s: %d rows kept (skipped %d). labels=%s",
        spec["name"],
        len(rows),
        skipped,
        dict(label_counts),
    )
    return rows, stats


def _short_hash(text: str) -> str:
    norm = text[:300].lower().strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _dedupe(rows: list[Row]) -> tuple[list[Row], dict[str, int]]:
    """Two-pass dedupe: exact text + first-300-char hash. Earlier sources win."""
    seen_exact: set[str] = set()
    seen_short: set[str] = set()
    out: list[Row] = []
    exact_collisions = 0
    short_collisions = 0

    for row in rows:
        if row.text in seen_exact:
            exact_collisions += 1
            continue
        sh = _short_hash(row.text)
        if sh in seen_short:
            short_collisions += 1
            continue
        seen_exact.add(row.text)
        seen_short.add(sh)
        out.append(row)

    return out, {
        "exact_collisions": exact_collisions,
        "near_dup_collisions": short_collisions,
    }


def _balance(
    rows: list[Row], target_ratio: float, rng: random.Random
) -> tuple[list[Row], dict[str, Any]]:
    """Subsample the majority class, stratified by (source, lang).

    target_ratio is the desired fraction of label==1 (injection) rows.
    """
    inj = [r for r in rows if r.label == 1]
    ben = [r for r in rows if r.label == 0]

    if not inj or not ben:
        return rows, {
            "kept_inj": len(inj),
            "kept_ben": len(ben),
            "subsampled": False,
        }

    # Solve for target benign count: target_ratio = inj / (inj + ben).
    desired_ben = int(round(len(inj) * (1.0 - target_ratio) / target_ratio))

    if len(ben) <= desired_ben:
        # Already at or below target.
        return inj + ben, {
            "kept_inj": len(inj),
            "kept_ben": len(ben),
            "subsampled": False,
            "desired_ben": desired_ben,
        }

    # Stratified subsample by (source, lang).
    strata: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for r in ben:
        strata[(r.source, r.lang)].append(r)

    total_ben = len(ben)
    kept_ben: list[Row] = []
    for stratum, items in strata.items():
        share = len(items) / total_ben
        keep_n = max(1, int(round(desired_ben * share)))
        keep_n = min(keep_n, len(items))
        rng.shuffle(items)
        kept_ben.extend(items[:keep_n])

    # Trim or pad to exactly desired_ben if rounding overshot.
    if len(kept_ben) > desired_ben:
        rng.shuffle(kept_ben)
        kept_ben = kept_ben[:desired_ben]

    return inj + kept_ben, {
        "kept_inj": len(inj),
        "kept_ben": len(kept_ben),
        "subsampled": True,
        "desired_ben": desired_ben,
        "ben_strata": len(strata),
    }


def _split(
    rows: list[Row], rng: random.Random, ratios: tuple[float, float, float]
) -> tuple[list[Row], list[Row], list[Row]]:
    """80/10/10 split stratified by (source, label)."""
    train_r, val_r, test_r = ratios
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6

    strata: dict[tuple[str, int], list[Row]] = defaultdict(list)
    for r in rows:
        strata[(r.source, r.label)].append(r)

    train: list[Row] = []
    val: list[Row] = []
    test: list[Row] = []

    for items in strata.values():
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * train_r))
        n_val = int(round(n * val_r))
        # Whatever remains is test (ensures all rows assigned).
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _write_parquet(rows: list[Row], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        logger.warning("no rows to write to %s", path)
        return

    table = pa.Table.from_pylist([r.to_dict() for r in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    logger.info("wrote %s (%d rows, %.1f KB)", path, len(rows), path.stat().st_size / 1024)


def _summarize(rows: list[Row]) -> dict[str, Any]:
    sources = Counter(r.source for r in rows)
    langs = Counter(r.lang for r in rows)
    labels = Counter(r.label for r in rows)
    return {
        "total": len(rows),
        "by_source": dict(sources),
        "by_lang": dict(langs),
        "by_label": {str(k): v for k, v in labels.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-ratio", type=_ratio_argtype, default=0.6,
                        help="target fraction of label==1 rows; must be in (0, 1) (default 0.6)")
    parser.add_argument("--max-text-chars", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    rng = random.Random(args.seed)
    sources_doc = json.loads(args.sources.read_text())

    # ---- Training pool ------------------------------------------------------
    per_source_stats: dict[str, dict[str, Any]] = {}
    all_rows: list[Row] = []
    for spec in sources_doc["training_pool"]:
        rows, stats = _load_source(spec, args.max_text_chars)
        per_source_stats[spec["name"]] = stats
        all_rows.extend(rows)

    pre_dedup_total = len(all_rows)
    deduped, dedup_stats = _dedupe(all_rows)
    logger.info(
        "dedup: %d -> %d (exact=%d, near=%d)",
        pre_dedup_total,
        len(deduped),
        dedup_stats["exact_collisions"],
        dedup_stats["near_dup_collisions"],
    )

    balanced, balance_stats = _balance(deduped, args.target_ratio, rng)
    logger.info(
        "balance: kept %d injection / %d benign (target ratio %.2f)",
        balance_stats["kept_inj"],
        balance_stats["kept_ben"],
        args.target_ratio,
    )

    train, val, test = _split(balanced, rng, (0.8, 0.1, 0.1))
    logger.info("split: train=%d val=%d test=%d", len(train), len(val), len(test))

    out = args.out
    _write_parquet(train, out / "train.parquet")
    _write_parquet(val, out / "val.parquet")
    _write_parquet(test, out / "test.parquet")

    # ---- Held-out eval ------------------------------------------------------
    # Each held-out dataset gets its own parquet. We deliberately do NOT dedup
    # them against training data — a held-out set is supposed to be immutable
    # and reflect upstream verbatim; if there's overlap, that's information
    # about generalization worth seeing.
    out_eval_names = {
        "davidtkeane_prompt_injection": "eval_davidtkeane.parquet",
        "jailbreakbench_behaviors": "eval_jailbreakbench.parquet",
    }
    eval_stats: dict[str, dict[str, Any]] = {}
    for spec in sources_doc["held_out_evaluation"]:
        rows, stats = _load_source(spec, args.max_text_chars)
        eval_stats[spec["name"]] = stats
        target_name = out_eval_names.get(spec["name"], f"eval_{spec['name']}.parquet")
        _write_parquet(rows, out / target_name)

    # ---- Manifest -----------------------------------------------------------
    manifest = {
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "seed": args.seed,
        "target_ratio": args.target_ratio,
        "max_text_chars": args.max_text_chars,
        "training_pool": {
            "per_source": per_source_stats,
            "pre_dedup_total": pre_dedup_total,
            "post_dedup_total": len(deduped),
            "dedup": dedup_stats,
            "balance": balance_stats,
            "splits": {
                "train": _summarize(train),
                "val": _summarize(val),
                "test": _summarize(test),
            },
        },
        "held_out": eval_stats,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info("wrote manifest to %s", out / "manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
