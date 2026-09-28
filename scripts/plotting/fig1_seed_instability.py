"""Figure 1 — per-seed precision@10 on DAVIS against each level's chance level.

Source: results/effects_v2/seed_spread.csv (families P1 = UniProt, S3-D = KLIFS pocket; DAVIS).
One dot per training seed, a short bar at the seed mean, a dashed tick at the exact chance level.
Seeds of one recipe that fall on both sides of chance are the instability the paper reports.
"""
import numpy as np
import pandas as pd

from _style import (INK, LEVELS, LEVEL_NAMES, MODEL_COLORS, MODEL_MARKERS, MODEL_NAMES, REFERENCE,
                    RESULTS, plt, save)

SRC = RESULTS / "effects_v2" / "seed_spread.csv"
MODELS = ["moltrans", "hyperattentiondti", "drugban", "coldsite_dti"]
ROWS = [("P1", "UniProt residues"), ("S3-D", "KLIFS 85-residue pocket")]


def build():
    df = pd.read_csv(SRC)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex=True)
    for r, (family, gt_label) in enumerate(ROWS):
        sub = df[(df.family == family) & (df.dataset == "davis")]
        for c, model in enumerate(MODELS):
            ax = axes[r, c]
            cells = sub[sub.model == model].set_index("level")
            for x, level in enumerate(LEVELS):
                row = cells.loc[level]
                seeds = np.array([float(v) for v in str(row.per_seed_precision).split(";")])
                jitter = np.linspace(-0.12, 0.12, len(seeds))
                ax.plot([x - 0.3, x + 0.3], [row.chance_exact] * 2, ls="--", lw=1, color=REFERENCE)
                ax.plot([x - 0.18, x + 0.18], [seeds.mean()] * 2, lw=1.6, color=INK)
                ax.scatter(x + jitter, seeds, s=22, marker=MODEL_MARKERS[model],
                           color=MODEL_COLORS[model], edgecolor="white", linewidth=0.6, zorder=3)
            ax.set_xticks(range(len(LEVELS)), [LEVEL_NAMES[l] for l in LEVELS], rotation=40, ha="right")
            if r == 0:
                ax.set_title(MODEL_NAMES[model])
            if c == 0:
                ax.set_ylabel(f"precision@10\n{gt_label}")
    handles = [
        plt.Line2D([], [], ls="none", marker="o", color="#8a8984", label="one training seed"),
        plt.Line2D([], [], color=INK, lw=1.6, label="mean of 3 seeds"),
        plt.Line2D([], [], color=REFERENCE, ls="--", lw=1, label="chance level"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    return save(fig, "fig1_seed_instability")


if __name__ == "__main__":
    print(*build(), sep="\n")
