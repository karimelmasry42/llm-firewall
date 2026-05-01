"""
Generate the visualizations referenced from the README and docs.

Reads the committed eval JSONs in `data/input_classifier/eval/` plus the
known-from-eval threshold-sweep numbers, and writes PNGs to `docs/img/`.

Run after any classifier eval is regenerated:

    python scripts/input_classifier/generate_visualizations.py

The script is deterministic — figures are byte-identical from one run to
the next on the same inputs, so committing the PNGs makes the README
numbers reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "data" / "input_classifier" / "eval"
IMG_DIR = REPO_ROOT / "docs" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Brand-y but neutral palette.
COLOR_SVM = "#9aa0a6"
COLOR_PROTECTAI = "#4285f4"
COLOR_LLAMA = "#34a853"
COLOR_LLAMA_TUNED = "#0f9d58"
COLOR_BG = "#ffffff"
COLOR_GRID = "#e8eaed"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.edgecolor": "#3c4043",
    "axes.labelcolor": "#3c4043",
    "xtick.color": "#3c4043",
    "ytick.color": "#3c4043",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
})


def _load(name: str) -> dict:
    return json.loads((EVAL_DIR / f"{name}.json").read_text())


def _f1(report: dict, table: str) -> float:
    return report["tables"][table]["f1"]


def fig_model_comparison() -> None:
    """Bar chart: F1 across the 3 evaluation tables for every classifier."""
    svm = _load("svm_baseline")
    svm_es = _load("svm_baseline_spanish")
    prot = _load("protectai_deberta")
    llama = _load("prompt_guard_2")  # already at tuned threshold (committed)

    tables = [
        ("In-distribution test\n(1,455 prompts, 21 langs)", "in_distribution_test"),
        ("DavidTKeane held-out\n(112 prompts, 8 languages)", "held_out_davidtkeane"),
        ("JailbreakBench held-out\n(200 prompts, 10 categories)", "held_out_jailbreakbench"),
    ]

    models = [
        ("SVM (English)", svm, COLOR_SVM),
        ("SVM (Spanish)", svm_es, "#bdc1c6"),
        ("protectai/deberta-v3", prot, COLOR_PROTECTAI),
        ("Llama-Prompt-Guard-2-86M\n(threshold-tuned, SHIPPED)", llama, COLOR_LLAMA_TUNED),
    ]

    n_groups = len(tables)
    n_bars = len(models)
    bar_width = 0.18
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=120)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    for i, (name, report, color) in enumerate(models):
        f1s = [_f1(report, t[1]) for t in tables]
        offset = (i - (n_bars - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, f1s, bar_width, label=name, color=color, edgecolor="white"
        )
        for bar, f1 in zip(bars, f1s):
            ax.annotate(
                f"{f1:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, f1),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#3c4043",
            )

    ax.set_xticks(x, [t[0] for t in tables], fontsize=10)
    ax.set_ylabel("F1 score", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Input-classifier F1 across in-distribution and held-out evaluation sets",
        fontsize=13, color="#202124", pad=14,
    )
    ax.legend(loc="upper right", fontsize=9, frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "model_comparison_f1.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_threshold_sweep() -> None:
    """Llama-Prompt-Guard-2 threshold sweep on val.parquet — the calibration story."""
    # These numbers are the committed sweep result from the docs.
    thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.50]
    precision = [0.600, 0.673, 0.763, 0.788, 0.797, 0.806, 0.815, 0.819, 0.836]
    recall = [1.000, 0.984, 0.908, 0.772, 0.742, 0.711, 0.669, 0.641, 0.574]
    f1 = [0.750, 0.799, 0.829, 0.780, 0.769, 0.756, 0.735, 0.719, 0.681]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    ax.plot(thresholds, precision, marker="o", color="#ea4335",
            label="Precision", linewidth=2)
    ax.plot(thresholds, recall, marker="s", color="#4285f4",
            label="Recall", linewidth=2)
    ax.plot(thresholds, f1, marker="^", color="#0f9d58",
            label="F1", linewidth=2.5)

    # Highlight the chosen threshold.
    chosen = 0.001
    chosen_f1 = 0.829
    ax.axvline(chosen, color="#fbbc04", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.annotate(
        f"shipped\nthreshold={chosen}\nF1={chosen_f1:.3f}",
        xy=(chosen, chosen_f1),
        xytext=(chosen * 4, 0.92),
        fontsize=10,
        color="#202124",
        ha="left",
        arrowprops=dict(arrowstyle="->", color="#fbbc04", lw=1.2),
    )

    # Mark the default threshold.
    ax.axvline(0.5, color="#9aa0a6", linestyle=":", linewidth=1, alpha=0.6)
    ax.annotate(
        "default\nthreshold=0.5\n(F1=0.681)",
        xy=(0.5, 0.681),
        xytext=(0.15, 0.50),
        fontsize=9,
        color="#5f6368",
        ha="left",
        arrowprops=dict(arrowstyle="->", color="#9aa0a6", lw=0.8),
    )

    ax.set_xscale("log")
    ax.set_xlabel("Decision threshold (log scale)", fontsize=11)
    ax.set_ylabel("Score on val.parquet", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Llama-Prompt-Guard-2-86M threshold sweep — calibrating for our prompt mix",
        fontsize=13, color="#202124", pad=14,
    )
    ax.legend(loc="lower left", fontsize=10, frameon=False)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "threshold_sweep.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_per_source_perf() -> None:
    """Per-source F1: shipped Llama-PG2 vs strong English baseline vs SVM."""
    sources_keep = [
        ("neuralchemy_prompt_injection", "neuralchemy"),
        ("walledai_jailbreakhub", "JailbreakHub"),
        ("xtram1_safe_guard_prompt_injection", "xTRam1\nsafe-guard"),
        ("lakera_gandalf_ignore_instructions", "Lakera\nGandalf"),
        ("deepset_prompt_injections", "deepset"),
    ]

    svm = _load("svm_baseline")["tables"]["in_distribution_test"]["by_source"]
    prot = _load("protectai_deberta")["tables"]["in_distribution_test"]["by_source"]
    llama = _load("prompt_guard_2")["tables"]["in_distribution_test"]["by_source"]

    f1_svm = [svm[s]["f1"] for s, _ in sources_keep]
    f1_prot = [prot[s]["f1"] for s, _ in sources_keep]
    f1_llama = [llama[s]["f1"] for s, _ in sources_keep]

    n = len(sources_keep)
    x = np.arange(n)
    bw = 0.27

    fig, ax = plt.subplots(figsize=(11, 5), dpi=120)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    bars1 = ax.bar(x - bw, f1_svm, bw, label="Old SVM (baseline)", color=COLOR_SVM,
                   edgecolor="white")
    bars2 = ax.bar(x, f1_prot, bw, label="protectai/deberta-v3", color=COLOR_PROTECTAI,
                   edgecolor="white")
    bars3 = ax.bar(x + bw, f1_llama, bw,
                   label="Llama-Prompt-Guard-2-86M (shipped)",
                   color=COLOR_LLAMA_TUNED, edgecolor="white")

    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=8, color="#3c4043")

    ax.set_xticks(x, [name for _, name in sources_keep], fontsize=10)
    ax.set_ylabel("F1 score", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Per-source F1 on in-distribution test set — coverage across all 5 major training sources",
        fontsize=12, color="#202124", pad=14,
    )
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "per_source_performance.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_multilingual_blocking() -> None:
    """Visual proof of multilingual blocking — synthetic but real numbers."""
    # Numbers from the live smoke test in the commit message.
    examples = [
        ("English (benign)", 0.0004, "PASS"),
        ("English (injection)", 0.9994, "BLOCK"),
        ("Spanish (injection)", 0.9994, "BLOCK"),
        ("German (injection)", 0.9993, "BLOCK"),
        ("Chinese (injection)", 0.9996, "BLOCK"),
    ]

    labels = [e[0] for e in examples]
    scores = [e[1] for e in examples]
    decisions = [e[2] for e in examples]
    colors = ["#34a853" if d == "PASS" else "#ea4335" for d in decisions]

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    bars = ax.barh(labels, scores, color=colors, edgecolor="white")
    for bar, score, decision in zip(bars, scores, decisions):
        ax.annotate(
            f"  {score:.4f}  →  {decision}",
            xy=(score, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0), textcoords="offset points",
            va="center", fontsize=10, color="#3c4043",
        )

    ax.axvline(0.001, color="#fbbc04", linestyle="--", linewidth=1.2,
               label="decision threshold = 0.001")
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("P(injection) from Llama-Prompt-Guard-2-86M", fontsize=11)
    ax.set_title(
        "Live multilingual blocking — same model handles English, Spanish, German, Chinese",
        fontsize=12, color="#202124", pad=14,
    )
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "multilingual_blocking.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_dataset_composition() -> None:
    """Donut-style breakdown of the 14,557-row training pool."""
    manifest = json.loads(
        (REPO_ROOT / "data" / "input_classifier" / "datasets" / "manifest.json").read_text()
    )
    splits = manifest["training_pool"]["splits"]["train"]["by_source"]

    # Group the small ones into "other" for readability.
    items = sorted(splits.items(), key=lambda kv: -kv[1])
    big, small = items[:5], items[5:]
    if small:
        big.append(("other", sum(v for _, v in small)))

    labels = [k.replace("_", " ").replace("openassistant ", "").replace("xtram1 ", "")
              for k, _ in big]
    sizes = [v for _, v in big]

    palette = ["#4285f4", "#34a853", "#fbbc04", "#ea4335", "#9c27b0", "#9aa0a6"]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    fig.patch.set_facecolor(COLOR_BG)
    wedges, _texts, _autotexts = ax.pie(
        sizes, labels=labels, colors=palette[: len(big)],
        autopct=lambda p: f"{int(p * sum(sizes) / 100):,}\n({p:.0f}%)",
        textprops=dict(fontsize=9, color="#202124"),
        wedgeprops=dict(width=0.42, edgecolor="white"),
        startangle=90, pctdistance=0.78,
    )
    ax.set_title(
        f"Training-pool composition — {sum(sizes):,} rows after dedup + class balance",
        fontsize=12, color="#202124", pad=14,
    )
    plt.tight_layout()
    plt.savefig(IMG_DIR / "dataset_composition.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_language_coverage() -> None:
    """Train-set language histogram — proves the multilingual claim."""
    manifest = json.loads(
        (REPO_ROOT / "data" / "input_classifier" / "datasets" / "manifest.json").read_text()
    )
    by_lang = manifest["training_pool"]["splits"]["train"]["by_lang"]
    items = sorted(by_lang.items(), key=lambda kv: -kv[1])

    # Show the top non-English languages explicitly; collapse English as a
    # reference, plus "other".
    en_count = next((v for k, v in items if k == "en"), 0)
    non_en = [(k, v) for k, v in items if k != "en"]
    top = non_en[:10]
    other = sum(v for k, v in non_en[10:])
    if other:
        top.append(("other", other))

    labels = [k for k, _ in top]
    counts = [v for _, v in top]

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=120)
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    ax.bar(labels, counts, color="#4285f4", edgecolor="white")
    for i, c in enumerate(counts):
        ax.annotate(str(c), xy=(i, c), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color="#3c4043")

    ax.set_ylabel("Prompts in train set", fontsize=11)
    ax.set_title(
        f"Multilingual coverage — {len(non_en) + 1} languages in training "
        f"(English: {en_count:,} • non-English: {sum(v for _, v in non_en):,})",
        fontsize=12, color="#202124", pad=14,
    )
    plt.tight_layout()
    plt.savefig(IMG_DIR / "language_coverage.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("Generating visualizations...")
    fig_model_comparison()
    print("  ✓ docs/img/model_comparison_f1.png")
    fig_threshold_sweep()
    print("  ✓ docs/img/threshold_sweep.png")
    fig_per_source_perf()
    print("  ✓ docs/img/per_source_performance.png")
    fig_multilingual_blocking()
    print("  ✓ docs/img/multilingual_blocking.png")
    fig_dataset_composition()
    print("  ✓ docs/img/dataset_composition.png")
    fig_language_coverage()
    print("  ✓ docs/img/language_coverage.png")
    print("Done.")


if __name__ == "__main__":
    main()
