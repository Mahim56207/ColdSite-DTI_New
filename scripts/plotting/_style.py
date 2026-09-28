"""Shared constants and helpers for the manuscript figures (paper/v2/figures/).

Every figure is drawn from a file committed under results/; nothing is typed by hand. Output is
deterministic: PNG and PDF metadata that would carry a timestamp or software version is removed,
so rebuilding from the same inputs gives byte-identical files (checked by make_all.py --check).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "v2" / "figures"

# Display names follow the manuscript: the in-house model is "XAttn-Ref" (paper/v2/methods.md).
MODEL_NAMES = {
    "coldsite_dti": "XAttn-Ref",
    "moltrans": "MolTrans",
    "hyperattentiondti": "HyperAttentionDTI",
    "drugban": "DrugBAN",
    "deepdta": "DeepDTA",
}
MODEL_ORDER = ["moltrans", "hyperattentiondti", "drugban", "coldsite_dti", "deepdta"]

# Categorical slots in a fixed order per model (validated: dataviz validate_palette.js, light mode,
# 5 slots, all hard checks pass; three slots are below 3:1 contrast, so every figure also carries
# a legend and a marker shape per model).
MODEL_COLORS = {
    "moltrans": "#2a78d6",
    "hyperattentiondti": "#eb6834",
    "drugban": "#1baf7a",
    "coldsite_dti": "#eda100",
    "deepdta": "#e87ba4",
}
MODEL_MARKERS = {
    "moltrans": "o",
    "hyperattentiondti": "s",
    "drugban": "^",
    "coldsite_dti": "D",
    "deepdta": "v",
}

LEVELS = ["random", "cold_drug", "cold_target", "cold_pair"]
LEVEL_NAMES = {
    "random": "random",
    "cold_drug": "cold-drug",
    "cold_target": "cold-target",
    "cold_pair": "cold-pair",
}

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8a8984"  # chance / no-effect lines: neutral, never a series colour

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def save(fig, stem: str) -> list:
    """Write <stem>.png and <stem>.pdf deterministically; return the two paths."""
    OUT.mkdir(parents=True, exist_ok=True)
    png, pdf = OUT / f"{stem}.png", OUT / f"{stem}.pdf"
    fig.savefig(png, metadata={"Software": None}, bbox_inches="tight")
    fig.savefig(pdf, metadata={"Creator": None, "Producer": None, "CreationDate": None},
                bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]
