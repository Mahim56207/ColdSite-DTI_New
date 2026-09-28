"""Figure 6 — what DAVIS's sequence leakage is worth to the accuracy anchor (DeepDTA).

Source: results/leakage_retrain_davis.md, the per-arm tables (mean ± sd over 3 seeds). The per-seed
JSON beside it is git-ignored, so the committed Markdown table is parsed instead.
seqmatched − seqclean = the leak; original − seqmatched = the smaller training set.
"""
import re

from _style import INK, INK_2, RESULTS, plt, save

SRC = RESULTS / "leakage_retrain_davis.md"
ARMS = ["original", "seqmatched", "seqclean"]
SUBSETS = ["all rows", "leaked targets", "unleaked targets"]
ARM_COLORS = {"original": "#2a78d6", "seqmatched": "#eb6834", "seqclean": "#1baf7a"}
ARM_MARKERS = {"original": "o", "seqmatched": "s", "seqclean": "^"}


def parse(text: str) -> dict:
    """{level: {arm: [(mean, sd) for each subset]}} from the '| arm | train rows | …' tables."""
    out, level = {}, None
    for line in text.splitlines():
        h = re.match(r"^## (cold-target|cold-pair)", line)
        if h:
            level = h.group(1)
            out[level] = {}
            continue
        m = re.match(r"^\| (original|seqmatched|seqclean) \| \d+ \|(.*)\|$", line)
        if m and level and m.group(1) not in out[level]:
            vals = re.findall(r"([\d.]+) ± ([\d.]+)", m.group(2))
            out[level][m.group(1)] = [(float(a), float(b)) for a, b in vals]
    return out


def build():
    data = parse(SRC.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, level in zip(axes, ["cold-target", "cold-pair"]):
        for j, arm in enumerate(ARMS):
            for i, (mean, sd) in enumerate(data[level][arm]):
                x = i + (j - 1) * 0.22
                ax.errorbar(x, mean, yerr=sd, fmt=ARM_MARKERS[arm], ms=5, color=ARM_COLORS[arm],
                            ecolor=ARM_COLORS[arm], elinewidth=1.2, capsize=0,
                            label=arm if i == 0 else None)
        ax.set_xticks(range(len(SUBSETS)), SUBSETS)
        ax.set_title(f"DeepDTA, DAVIS {level}")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("test AUROC (mean ± sd, 3 seeds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, ["original training set", "seqmatched (leak kept, size matched)",
                         "seqclean (leak removed)"], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    return save(fig, "fig6_leakage")


if __name__ == "__main__":
    print(*build(), sep="\n")
