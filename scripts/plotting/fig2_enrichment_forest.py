"""Figure 2 — enrichment over chance (precision@10 / chance) with 95% bootstrap intervals.

Source: results/effects_v2_2d/enrichment.csv (two-way bootstrap: seeds and targets resampled; families P1, P2 = UniProt; S3-D, S3-K = KLIFS pocket).
Intervals resample seeds and targets (10,000 resamples; column n_resamples). 1 = chance.
"""
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from _style import (LEVEL_NAMES, LEVELS, MODEL_COLORS, MODEL_MARKERS, MODEL_NAMES, REFERENCE, RESULTS,
                    plt, save)

SRC = RESULTS / "effects_v2_2d" / "enrichment.csv"
PANELS = [("P1", "DAVIS · UniProt residues"), ("S3-D", "DAVIS · KLIFS pocket"),
          ("P2", "KIBA · UniProt residues"), ("S3-K", "KIBA · KLIFS pocket")]
TICKS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3]
MODELS = ["moltrans", "hyperattentiondti", "drugban", "coldsite_dti"]


def build():
    df = pd.read_csv(SRC)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), gridspec_kw={"height_ratios": [16, 6]})
    for ax, (family, title) in zip(axes.flat, PANELS):
        sub = df[df.family == family]
        labels, y = [], 0
        for level in LEVELS:
            for model in MODELS:
                row = sub[(sub.model == model) & (sub.level == level)]
                if row.empty:
                    continue
                row = row.iloc[0]
                ax.plot([row.enrichment_low, row.enrichment_high], [y, y], lw=1.4,
                        color=MODEL_COLORS[model], solid_capstyle="round")
                ax.scatter(row.enrichment, y, s=24, marker=MODEL_MARKERS[model],
                           color=MODEL_COLORS[model], edgecolor="white", linewidth=0.6, zorder=3)
                labels.append(f"{MODEL_NAMES[model]} · {LEVEL_NAMES[level]}")
                y += 1
        ax.axvline(1, color=REFERENCE, ls="--", lw=1)
        ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(TICKS))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(title)
        ax.set_xlabel("enrichment over chance (log scale)")
    fig.tight_layout()
    return save(fig, "fig2_enrichment_forest")


if __name__ == "__main__":
    print(*build(), sep="\n")
