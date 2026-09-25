"""Figure 3 — test AUROC per training seed, every model and level (uncorrected test sets).

Source: results/accuracy_v2/cells.csv (view = uncorrected; one row per cell and seed).
"""
import numpy as np
import pandas as pd

from _style import (INK, LEVEL_NAMES, LEVELS, MODEL_COLORS, MODEL_MARKERS, MODEL_NAMES, MODEL_ORDER,
                    RESULTS, plt, save)

SRC = RESULTS / "accuracy_v2" / "cells.csv"


def build():
    df = pd.read_csv(SRC)
    df = df[df.view == "uncorrected"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [4, 2.2]}, sharey=True)
    for ax, dataset in zip(axes, ["davis", "kiba"]):
        sub = df[df.dataset == dataset]
        levels = [l for l in LEVELS if l in set(sub.level)]
        models = [m for m in MODEL_ORDER if m in set(sub.model)]
        width = 0.8 / len(models)
        for i, level in enumerate(levels):
            for j, model in enumerate(models):
                vals = sub[(sub.level == level) & (sub.model == model)].auroc.to_numpy()
                if len(vals) == 0:
                    continue
                x = i - 0.4 + width * (j + 0.5)
                ax.plot([x - width * 0.4, x + width * 0.4], [vals.mean()] * 2, color=INK, lw=1.2)
                ax.scatter(x + np.linspace(-width * 0.25, width * 0.25, len(vals)), vals, s=16, marker=MODEL_MARKERS[model],
                           color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5, zorder=3,
                           label=MODEL_NAMES[model] if i == 0 else None)
        ax.set_xticks(range(len(levels)), [LEVEL_NAMES[l] for l in levels])
        ax.set_title(dataset.upper())
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("test AUROC (one point per seed)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    return save(fig, "fig3_accuracy")


if __name__ == "__main__":
    print(*build(), sep="\n")
