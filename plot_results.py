"""
Standalone plot script for Part 3 mitigation results.
Reads data/benchmark_results.json + data/mitigation_results.json; no model loading.

Usage:
    uv run python plot_results.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

BENCHMARK_PATH  = Path("data/benchmark_results.json")
MITIGATION_PATH = Path("data/mitigation_results.json")
FIGURES_DIR     = Path("data/figures/final")

ALL_MODELS = [
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "HuggingFaceTB/SmolVLM-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "google/gemma-4-E2B-it",
]

MITIGATIONS = ["few_shot", "explained_few_shot", "layout_few_shot", "multi_few_shot", "skeptical"]

MITIGATION_LABELS = {
    "few_shot":          "few-shot",
    "explained_few_shot": "expl. few-shot",
    "layout_few_shot":   "layout few-shot",
    "multi_few_shot":    "multi few-shot",
    "skeptical":         "skeptical",
}

CORRUPTION_TYPES = [
    "element",
    "element+layout",
    "layout",
    "nlp_entity",
    "nlp_entity+element",
    "nlp_entity+layout",
]

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────

with open(BENCHMARK_PATH) as f:
    benchmark = json.load(f)

with open(MITIGATION_PATH) as f:
    mitigation = json.load(f)

# Only plot models that have been benchmarked
MODELS = [m for m in ALL_MODELS if m in benchmark]

# ── Helpers ───────────────────────────────────────────────────────────────────

def short(model: str) -> str:
    return model.split("/")[-1]


def type_recall(per_sample: list, ctype: str) -> float:
    items = [p for p in per_sample if p["label"] == "UNANSWERABLE" and p["corruption_type"] == ctype]
    if not items:
        return float("nan")
    return sum(p["correct"] for p in items) / len(items)


def all_variants(model: str) -> dict:
    base = benchmark[model]
    mits = {m: mitigation[model][m] for m in MITIGATIONS if m in mitigation.get(model, {})}
    return {"baseline": base, **mits}


def variant_labels(variants: dict) -> list[str]:
    return ["baseline"] + [MITIGATION_LABELS[m] for m in MITIGATIONS if m in variants]


# ── Plot 1: Answerable vs Unanswerable recall per model ───────────────────────

fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS), 5), sharey=True)

for ax, model in zip(axes, MODELS):
    variants = all_variants(model)
    labels   = variant_labels(variants)
    x        = np.arange(len(labels))
    width    = 0.35

    ans_rec  = [variants[k]["answerable_recall"]   for k in variants]
    unans_rec = [variants[k]["unanswerable_recall"] for k in variants]

    ax.bar(x - width / 2, ans_rec,   width, label="Answerable",   color="#4C72B0")
    ax.bar(x + width / 2, unans_rec, width, label="Unanswerable", color="#DD8452")
    ax.axhline(variants["baseline"]["unanswerable_recall"],
               color="#DD8452", linestyle="--", linewidth=0.9, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Recall")
    ax.set_title(short(model))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Recall by class across mitigations", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "mitigation_recall.jpg", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {FIGURES_DIR}/mitigation_recall.jpg")

# ── Plot 2: Metrics line chart per model ─────────────────────────────────────

METRICS = {
    "macro_f1":           ("Macro F1",     "#2ca02c"),
    "unanswerable_f1":    ("Unansw. F1",   "#d62728"),
    "unanswerable_recall":("Unansw. Rec.", "#ff7f0e"),
}

fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS), 4.5), sharey=True)

for ax, model in zip(axes, MODELS):
    variants = all_variants(model)
    labels   = variant_labels(variants)

    for key, (label, color) in METRICS.items():
        values = [variants[k][key] for k in variants]
        ax.plot(labels, values, marker="o", label=label, color=color)

    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(short(model))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)

fig.suptitle("Key metrics across mitigations", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "mitigation_metrics.jpg", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {FIGURES_DIR}/mitigation_metrics.jpg")

# ── Plot 3: Per-type unanswerable recall heatmap per model ────────────────────

fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 5))

for ax, model in zip(axes, MODELS):
    variants = all_variants(model)
    row_labels = variant_labels(variants)

    matrix = [
        [type_recall(variants[k]["per_sample"], ct) for ct in CORRUPTION_TYPES]
        for k in variants
    ]

    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(range(len(CORRUPTION_TYPES)))
    ax.set_xticklabels(CORRUPTION_TYPES, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(len(row_labels)):
        for j, ct in enumerate(CORRUPTION_TYPES):
            v = matrix[i][j]
            text = f"{v:.2f}" if not math.isnan(v) else "—"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="black" if 0.3 < v < 0.8 else "white" if math.isnan(v) else "black")

    plt.colorbar(im, ax=ax, label="Unanswerable recall", fraction=0.046, pad=0.04)
    ax.set_title(short(model), fontsize=10)

fig.suptitle("Unanswerable recall by mitigation and corruption type", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "mitigation_by_type.jpg", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {FIGURES_DIR}/mitigation_by_type.jpg")

# ── Plot 4: Cross-model delta (Δ unanswerable recall vs baseline) ─────────────

x     = np.arange(len(MITIGATIONS))
width = 0.8 / len(MODELS)

fig, ax = plt.subplots(figsize=(9, 5))

for i, model in enumerate(MODELS):
    base_urec = benchmark[model]["unanswerable_recall"]
    deltas = [
        mitigation[model][m]["unanswerable_recall"] - base_urec
        if m in mitigation.get(model, {}) else float("nan")
        for m in MITIGATIONS
    ]
    offset = (i - len(MODELS) / 2 + 0.5) * width
    bars = ax.bar(x + offset, deltas, width, label=short(model))

    for bar, delta in zip(bars, deltas):
        if not math.isnan(delta):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.005 if delta >= 0 else -0.015),
                    f"{delta:+.3f}", ha="center", va="bottom", fontsize=7.5)

ax.axhline(0, color="black", linewidth=0.9)
ax.set_xticks(x)
ax.set_xticklabels([MITIGATION_LABELS[m] for m in MITIGATIONS], rotation=15, ha="right")
ax.set_ylabel("Δ Unanswerable Recall vs Baseline")
ax.set_title("Mitigation impact by model — cross-model comparison")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "mitigation_delta_crossmodel.jpg", dpi=150)
plt.show()
print(f"Saved → {FIGURES_DIR}/mitigation_delta_crossmodel.jpg")

# ── Summary table ─────────────────────────────────────────────────────────────

for model in MODELS:
    variants = all_variants(model)
    base_urec = variants["baseline"]["unanswerable_recall"]
    labels    = list(variants.keys())

    print(f"\n=== {short(model)} ===")
    print(f"{'variant':<22} {'acc':>6} {'macro-f1':>9} {'unansw-f1':>10} {'unansw-rec':>11}  {'delta':>8}")
    print("-" * 72)
    for k in labels:
        r = variants[k]
        delta = r["unanswerable_recall"] - base_urec if k != "baseline" else 0.0
        sign  = "+" if delta >= 0 else ""
        label = MITIGATION_LABELS.get(k, k)
        print(f"{label:<22} {r['accuracy']:>6.3f} {r['macro_f1']:>9.3f} "
              f"{r['unanswerable_f1']:>10.3f} {r['unanswerable_recall']:>11.3f}  "
              f"{sign}{delta:>7.3f}")

print(f"\nAll figures saved to {FIGURES_DIR}/")
