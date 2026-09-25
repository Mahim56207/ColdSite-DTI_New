"""Figure 5 — faithfulness: attended masking minus size-matched random masking, 95% intervals.

Source: results/effects_v2/faithfulness_effects.csv (targets resampled, 10,000 resamples).
MolTrans is measured in token space and the others in residue space, so the panels do not share an axis.
"""
import pandas as pd

from _style import (LEVEL_NAMES, LEVELS, MODEL_COLORS, MODEL_MARKERS, MODEL_NAMES, REFERENCE, RESULTS,
                    plt, save)

SRC = RESULTS / "effects_v2" / "faithfulness_effects.csv"
MODELS = ["moltrans", "hyperattentiondti", "coldsite_dti"]


def build():
    df = pd.read_csv(SRC)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, model in zip(axes, MODELS):
        sub = df[df.model == model]
        labels, y = [], 0
        for dataset in ["davis", "kiba"]:
            for level in LEVELS:
                row = sub[(sub.dataset == dataset) & (sub.level == level)]
                if row.empty:
                    continue
                row = row.iloc[0]
                ax.plot([row.delta_low, row.delta_high], [y, y], lw=1.4, color=MODEL_COLORS[model])
                ax.scatter(row.delta, y, s=22, marker=MODEL_MARKERS[model], color=MODEL_COLORS[model],
                           edgecolor="white", linewidth=0.6, zorder=3)
                labels.append(f"{dataset.upper()} · {LEVEL_NAMES[level]}")
                y += 1
        ax.axvline(0, color=REFERENCE, ls="--", lw=1)
        ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_xlim(left=min(0, ax.get_xlim()[0]))
        space = "token" if (sub.kind == "token_faithfulness").all() else "residue"
        ax.set_title(f"{MODEL_NAMES[model]} ({space} space)")
        ax.set_xlabel("faithfulness delta")
    fig.tight_layout()
    return save(fig, "fig5_faithfulness")


if __name__ == "__main__":
    print(*build(), sep="\n")
