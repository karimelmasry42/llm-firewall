"""
Run any configured classifier through the input-classifier eval harness.

Loads the committed test parquets (data/input_classifier/datasets/*.parquet),
runs the chosen classifier on each row, and reports precision / recall / F1 /
ROC-AUC / PR-AUC / confusion matrix / per-source / per-language breakdown
plus mean and p95 latency. The full report is written to
data/input_classifier/eval/<name>.json so it can be diffed across runs.

p95 latency is the 95th percentile of per-prediction wall-clock time. We
report it alongside the mean because means hide the slow tail and worst-case
latency is what users feel.

Three tables are produced in one report:
  - in-distribution test set (test.parquet)
  - DavidTKeane held-out (eval_davidtkeane.parquet)
  - JailbreakBench held-out (eval_jailbreakbench.parquet)

Usage:
    python scripts/input_classifier/evaluate.py --classifier svm_baseline
    python scripts/input_classifier/evaluate.py --classifier prompt_guard_2
    python scripts/input_classifier/evaluate.py --classifier <name> --limit 200
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Protocol

logger = logging.getLogger("evaluate")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "input_classifier"
DATASETS_DIR = DATA_DIR / "datasets"
REPORTS_DIR = DATA_DIR / "eval"


class ClassifierLike(Protocol):
    def predict(self, text: str) -> tuple[int, float]:
        """Return (predicted_label, blocking_score in [0,1])."""


# --------------------------------------------------------------------------- #
# Adapters that wrap firewall internals as ClassifierLike implementations.
# --------------------------------------------------------------------------- #
def _wrap_filter_result_classifier(inner) -> "ClassifierLike":
    """Wrap any firewall classifier (FilterResult-returning) as ClassifierLike.

    `FilterResult.confidence` from both `PickleClassifier` (when the underlying
    model exposes `predict_proba`) and `HFSequenceClassifier` is the blocking-
    label probability — i.e. P(injection) — regardless of whether the result
    passed or blocked. We pass it through unchanged. Inverting it for passed
    rows would corrupt ROC-AUC / PR-AUC because score then means P(injection)
    for positives and P(benign) for negatives.
    """
    class _Adapter:
        def predict(self, text: str) -> tuple[int, float]:
            r = inner.evaluate(text)
            label = 0 if r.passed else 1
            return label, float(r.confidence)

    return _Adapter()


def _build_svm_baseline() -> ClassifierLike:
    """Adapter for the legacy English Linear-SVM input classifier.

    Constructs the spec inline (the registry no longer registers the SVM as a
    runtime classifier — it's been replaced by the multilingual HF model).
    The pickle file stays in `data/models/` purely so the eval harness can
    keep reporting baseline numbers for comparison.
    """
    from llm_firewall.classifiers.pickle_classifier import PickleClassifier
    from llm_firewall.classifiers.registry import (
        ClassifierSpec,
        MODELS_DIR,
        preprocess_injection_text,
    )

    spec = ClassifierSpec(
        name="linear_svm_input_classifier",
        path=MODELS_DIR / "linear_svm_input_classifier.pkl",
        preprocess=preprocess_injection_text,
    )
    return _wrap_filter_result_classifier(PickleClassifier(spec))


def _build_svm_baseline_spanish() -> ClassifierLike:
    """Adapter for the legacy Spanish Linear-SVM. See `_build_svm_baseline`."""
    from llm_firewall.classifiers.pickle_classifier import PickleClassifier
    from llm_firewall.classifiers.registry import (
        ClassifierSpec,
        MODELS_DIR,
        normalize_whitespace,
    )

    spec = ClassifierSpec(
        name="linear_svm_spanish",
        path=MODELS_DIR / "linear_svm_spanish.pkl",
        preprocess=normalize_whitespace,
    )
    return _wrap_filter_result_classifier(PickleClassifier(spec))


def _build_prompt_guard_2() -> ClassifierLike:
    """Adapter for meta-llama/Llama-Prompt-Guard-2-86M via HFSequenceClassifier.

    NOTE: This model is gated. Before running you must accept the license at
    https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M and
    authenticate via `huggingface-cli login`.
    """
    from llm_firewall.classifiers.huggingface import HFSequenceClassifier
    from llm_firewall.classifiers.registry import ClassifierSpec, normalize_whitespace

    spec = ClassifierSpec(
        name="llama_prompt_guard_2_86m",
        display_name="Llama-Prompt-Guard-2-86M",
        backend="huggingface_sequence",
        model_id="meta-llama/Llama-Prompt-Guard-2-86M",
        preprocess=normalize_whitespace,
        injection_label_id=1,
        # Tuned on val.parquet (see docs/input_classifier/models.md).
        threshold=0.001,
        max_length=512,
    )
    return _wrap_filter_result_classifier(HFSequenceClassifier(spec))


def _build_protectai_deberta() -> ClassifierLike:
    """Adapter for protectai/deberta-v3-base-prompt-injection-v2.

    Not multilingual but unblocked, well-known, and useful as a sanity check
    that the HF runtime works against a real prompt-injection classifier.
    """
    from llm_firewall.classifiers.huggingface import HFSequenceClassifier
    from llm_firewall.classifiers.registry import ClassifierSpec, normalize_whitespace

    spec = ClassifierSpec(
        name="protectai_deberta_v3",
        display_name="protectai/deberta-v3-base-prompt-injection-v2",
        backend="huggingface_sequence",
        model_id="protectai/deberta-v3-base-prompt-injection-v2",
        preprocess=normalize_whitespace,
        injection_label_name="INJECTION",
        threshold=0.5,
        max_length=512,
    )
    return _wrap_filter_result_classifier(HFSequenceClassifier(spec))


CLASSIFIER_BUILDERS: dict[str, Callable[[], ClassifierLike]] = {
    "svm_baseline": _build_svm_baseline,
    "svm_baseline_spanish": _build_svm_baseline_spanish,
    "prompt_guard_2": _build_prompt_guard_2,
    "protectai_deberta": _build_protectai_deberta,
}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@dataclass
class TableMetrics:
    n: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    scores: list[tuple[int, float]] = field(default_factory=list)  # (true_label, score)
    by_source: dict[str, "TableMetrics"] = field(default_factory=dict)
    by_lang: dict[str, "TableMetrics"] = field(default_factory=dict)

    def update(self, y_true: int, y_pred: int, score: float, latency_ms: float) -> None:
        self.n += 1
        if y_true == 1 and y_pred == 1:
            self.tp += 1
        elif y_true == 0 and y_pred == 1:
            self.fp += 1
        elif y_true == 0 and y_pred == 0:
            self.tn += 1
        else:
            self.fn += 1
        self.latencies_ms.append(latency_ms)
        self.scores.append((y_true, score))

    def _safe_div(self, a: int, b: int) -> float:
        return a / b if b else 0.0

    @property
    def precision(self) -> float:
        return self._safe_div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return self._safe_div(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        return self._safe_div(self.tp + self.tn, self.n)

    @property
    def mean_latency_ms(self) -> float:
        return statistics.fmean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lats = sorted(self.latencies_ms)
        idx = max(0, min(len(sorted_lats) - 1, int(round(0.95 * (len(sorted_lats) - 1)))))
        return sorted_lats[idx]

    def auc_or_none(self) -> tuple[float | None, float | None]:
        """Compute ROC-AUC and PR-AUC if sklearn is available and data permits."""
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
        except ImportError:
            return None, None
        if len({y for y, _ in self.scores}) < 2:
            return None, None
        y = [s[0] for s in self.scores]
        p = [s[1] for s in self.scores]
        try:
            return float(roc_auc_score(y, p)), float(average_precision_score(y, p))
        except Exception:
            return None, None

    def to_dict(self, recurse: bool = True) -> dict[str, Any]:
        roc, pr = self.auc_or_none()
        out: dict[str, Any] = {
            "n": self.n,
            "confusion": {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn},
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "roc_auc": round(roc, 4) if roc is not None else None,
            "pr_auc": round(pr, 4) if pr is not None else None,
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
        }
        if recurse:
            if self.by_source:
                out["by_source"] = {
                    k: v.to_dict(recurse=False) for k, v in sorted(self.by_source.items())
                }
            if self.by_lang:
                out["by_lang"] = {
                    k: v.to_dict(recurse=False) for k, v in sorted(self.by_lang.items())
                }
        return out


def _evaluate_parquet(
    classifier: ClassifierLike,
    parquet: Path,
    limit: int | None,
    score_rows: list[dict] | None = None,
    score_table_name: str | None = None,
) -> TableMetrics:
    """Score every row of a parquet table; optionally append per-prompt
    rows to `score_rows` so callers can dump them to CSV for downstream
    visualization without re-running the classifier."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet).to_pylist()
    if limit:
        table = table[:limit]

    metrics = TableMetrics()
    metrics.by_source = defaultdict(TableMetrics)
    metrics.by_lang = defaultdict(TableMetrics)

    for row in table:
        text = row["text"]
        y_true = int(row["label"])
        source = row.get("source", "unknown")
        lang = row.get("lang", "unknown")

        t0 = perf_counter()
        try:
            y_pred, score = classifier.predict(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("predict() raised on row from %s: %s", source, exc)
            continue
        latency_ms = (perf_counter() - t0) * 1000.0

        metrics.update(y_true, y_pred, score, latency_ms)
        metrics.by_source[source].update(y_true, y_pred, score, latency_ms)
        metrics.by_lang[lang].update(y_true, y_pred, score, latency_ms)

        if score_rows is not None:
            score_rows.append({
                "table": score_table_name or parquet.stem,
                "text": text,
                "lang": lang,
                "source": source,
                "label": y_true,
                "predicted": y_pred,
                "score": float(score),
                "latency_ms": round(latency_ms, 3),
            })

    metrics.by_source = dict(metrics.by_source)
    metrics.by_lang = dict(metrics.by_lang)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classifier",
        required=True,
        choices=sorted(CLASSIFIER_BUILDERS),
        help="Which classifier to evaluate.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap rows per parquet (useful for smoke-testing).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output JSON report path. Defaults to data/input_classifier/eval/<classifier>.json",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    logger.info("loading classifier %s", args.classifier)
    classifier = CLASSIFIER_BUILDERS[args.classifier]()

    report: dict[str, Any] = {
        "classifier": args.classifier,
        "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "limit": args.limit,
        "tables": {},
    }

    parquets = {
        "in_distribution_test": DATASETS_DIR / "test.parquet",
        "held_out_davidtkeane": DATASETS_DIR / "eval_davidtkeane.parquet",
        "held_out_jailbreakbench": DATASETS_DIR / "eval_jailbreakbench.parquet",
    }

    score_rows: list[dict] = []
    for table_name, path in parquets.items():
        if not path.exists():
            logger.warning("missing %s; skipping", path)
            continue
        logger.info("evaluating %s on %s", args.classifier, path.name)
        metrics = _evaluate_parquet(
            classifier, path, args.limit,
            score_rows=score_rows, score_table_name=table_name,
        )
        report["tables"][table_name] = metrics.to_dict()
        logger.info(
            "  %s: n=%d f1=%.3f mean=%.1fms p95=%.1fms",
            table_name,
            metrics.n,
            metrics.f1,
            metrics.mean_latency_ms,
            metrics.p95_latency_ms,
        )

    out_path = args.out or REPORTS_DIR / f"{args.classifier}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    logger.info("wrote %s", out_path)

    # Dump per-prompt scores so visualizations (e.g. the multilingual
    # blocking strip plot) can read real numbers without re-running the
    # classifier. Parquet keeps the file small and types intact.
    if score_rows:
        import pyarrow as pa
        import pyarrow.parquet as pq
        scores_path = out_path.with_name(out_path.stem + "_scores.parquet")
        pq.write_table(pa.Table.from_pylist(score_rows), scores_path)
        logger.info("wrote %s (%d rows)", scores_path, len(score_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
