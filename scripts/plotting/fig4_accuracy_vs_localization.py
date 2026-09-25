"""Figure 4 — is a more accurate cell a better localiser? AUROC vs enrichment over chance.

Sources: results/accuracy_v2/localization_cells.csv (one point per cell and seed; UniProt ground truth;
`auroc` is scored on sequence-unseen targets for DAVIS cold-target/cold-pair, as the explanation metrics are)
and results/accuracy_v2/localization_spearman.csv (rho with 95% bootstrap interval, all attention models).
"""
import pandas as pd

from _style import (MODEL_COLORS, MODEL_MARKERS, MODEL_NAMES, MODEL_ORDER, REFERENCE, RESULTS, plt,
                    save)

SRC = RESULTS / "accuracy_v2" / "localization_cells.csv"
RHO = RESULTS / "accuracy_v2" / "localization_spearman.csv"


def build():
    df = pd.read_csv(SRC)
    rho = pd.read_csv(RHO)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, dataset in zip(axes, ["davis", "kiba"]):
        sub = df[df.dataset == dataset]
        for model in [m for m in MODEL_ORDER if m in set(sub.model)]:
            m = sub[sub.model == model]
            ax.scatter(m.auroc, m.enrichment, s=20, marker=MODEL_MARKERS[model],
                       color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5,
                       label=MODEL_NAMES[model], zorder=3)
        ax.axhline(1, color=REFERENCE, ls="--", lw=1)
        r = rho[(rho.dataset == dataset) & (rho.group == "all attention models") & (rho.y == "enrichment")].iloc[0]
        ax.set_title(f"{dataset.upper()}\nSpearman ρ = {r.rho:.2f} [{r.low:.2f}, {r.high:.2f}], "
                     f"n = {int(r.n_cells)} cells", fontsize=8)
        ax.set_xlabel("test AUROC (DAVIS cold-target/cold-pair:\nsequence-unseen targets only)"
                      if dataset == "davis" else "test AUROC")
    axes[0].set_ylabel("enrichment over chance\n(UniProt, precision@10 / chance)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    return save(fig, "fig4_accuracy_vs_localization")


if __name__ == "__main__":
    print(*build(), sep="\n")
