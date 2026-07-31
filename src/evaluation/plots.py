"""
Track C (124AD0067) — the paper's headline figure.

Plots explanation fidelity (precision@k) and prediction accuracy side by
side across the four difficulty levels. See docs/03_GUIDE_124AD0067.md Step 3.
"""
import matplotlib.pyplot as plt

LEVELS = ["Warm", "Cold-Drug", "Cold-Target", "Cold-Pair"]


def plot_degradation_curve(fidelity_scores: list, accuracy_scores: list,
                            title: str = "DRAFT — replace with real results",
                            save_path: str = "results/headline_figure.png"):
    """
    fidelity_scores:  list of 4 precision@k values, one per level in LEVELS order
    accuracy_scores:  list of 4 AUROC/AUPRC (or other accuracy metric) values
    """
    assert len(fidelity_scores) == 4 and len(accuracy_scores) == 4, \
        "Need exactly one score per difficulty level: warm, cold-drug, cold-target, cold-pair"

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(LEVELS, fidelity_scores, marker="o", linewidth=2, label="Explanation fidelity (precision@k)")
    ax.plot(LEVELS, accuracy_scores, marker="s", linewidth=2, label="Prediction accuracy")
    ax.set_ylabel("Score")
    ax.set_xlabel("Difficulty level (how unfamiliar is the drug/target to the model)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved figure to {save_path}")
    return fig


if __name__ == "__main__":
    # placeholder numbers only -- swap for real results once experiments are done
    fake_fidelity = [0.62, 0.51, 0.47, 0.35]
    fake_accuracy = [0.89, 0.81, 0.78, 0.69]
    plot_degradation_curve(fake_fidelity, fake_accuracy,
                            save_path="results/headline_figure_DRAFT.png")
