"""
Aggregation across seeds, and correction across the audit grid.

Two statistical problems the single-model version did not have.

1. Seeds. One training run gives one curve, and a curve from one seed is that
   run's idiosyncrasy, not a property of the model. A fidelity drop from 0.42 to
   0.31 across the ladder is only a finding if the seed-to-seed spread is small
   relative to it. Everything here reports mean +/- std over seeds and refuses
   to pretend a single run is an estimate.

2. Multiplicity. The audit grid is models x splits x datasets x k-values. At
   4 models x 4 splits x 2 datasets that is 32 significance tests; roughly 1.6
   of them clear p < 0.05 by chance alone. Quoting uncorrected per-cell
   p-values would let noise masquerade as the paper's central claim, so
   holm_bonferroni() runs over the whole family before anything is called
   significant.

Holm-Bonferroni rather than plain Bonferroni: it controls the same family-wise
error rate but is uniformly more powerful, which matters when the honest
expected result is a modest effect.
"""
import numpy as np

MIN_SEEDS_FOR_A_CLAIM = 3


def aggregate_seeds(values, label: str = "") -> dict:
    """Mean, spread and a CI over seeds, with an explicit reliability flag.

    `sufficient_seeds` is False below three runs. A two-seed "mean" has no
    usable spread estimate, and reporting one invites a reviewer to treat noise
    as signal.
    """
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)],
                        dtype=float)
    n = finite.size

    if n == 0:
        return {"label": label, "mean": float("nan"), "std": float("nan"),
                "n_seeds": 0, "sufficient_seeds": False,
                "ci95_low": float("nan"), "ci95_high": float("nan"),
                "values": []}

    mean = float(finite.mean())
    # ddof=1: sample std. ddof=0 understates the spread and is the wrong
    # estimator when reporting variability across a handful of runs.
    std = float(finite.std(ddof=1)) if n > 1 else float("nan")
    stderr = std / np.sqrt(n) if n > 1 else float("nan")

    return {
        "label": label,
        "mean": mean,
        "std": std,
        "n_seeds": n,
        "sufficient_seeds": n >= MIN_SEEDS_FOR_A_CLAIM,
        "ci95_low": mean - 1.96 * stderr if n > 1 else float("nan"),
        "ci95_high": mean + 1.96 * stderr if n > 1 else float("nan"),
        "values": finite.tolist(),
    }


def holm_bonferroni(p_values: dict, alpha: float = 0.05) -> dict:
    """Family-wise error control across the whole audit grid.

    Returns {key: {p_value, adjusted_alpha, significant}}. Sort ascending, test
    against alpha/(m - i), and stop at the first failure -- every remaining
    hypothesis is rejected too, which is what makes the procedure valid rather
    than a per-test threshold.
    """
    if not p_values:
        return {}

    usable = {k: v for k, v in p_values.items()
              if v is not None and np.isfinite(v)}
    m = len(usable)
    ordered = sorted(usable.items(), key=lambda kv: kv[1])

    results, still_rejecting = {}, True
    for i, (key, p) in enumerate(ordered):
        threshold = alpha / (m - i)
        if still_rejecting and p > threshold:
            still_rejecting = False
        results[key] = {
            "p_value": float(p),
            "adjusted_alpha": float(threshold),
            "rank": i + 1,
            "significant": bool(still_rejecting),
        }

    for key, p in p_values.items():
        if key not in results:
            results[key] = {"p_value": p, "adjusted_alpha": float("nan"),
                            "rank": None, "significant": False}
    return results


def degradation(level_means: dict, order=("random", "cold_drug",
                                          "cold_target", "cold_pair")) -> dict:
    """How much fidelity is lost from the warm split to the hardest one.

    The paper's headline number. `monotonic` is reported because the
    interesting claim is a monotone decline; a non-monotone ladder is a
    different and more complicated story that must not be described as
    "degradation".
    """
    present = [l for l in order if l in level_means
               and np.isfinite(level_means[l])]
    if len(present) < 2:
        return {"absolute_drop": float("nan"), "relative_drop": float("nan"),
                "monotonic": False, "levels_present": present}

    warm, coldest = level_means[present[0]], level_means[present[-1]]
    series = [level_means[l] for l in present]

    return {
        "warm": float(warm),
        "coldest": float(coldest),
        "coldest_level": present[-1],
        "absolute_drop": float(warm - coldest),
        "relative_drop": float((warm - coldest) / warm) if warm else float("nan"),
        "monotonic": all(a >= b for a, b in zip(series, series[1:])),
        "levels_present": present,
    }


def audit_table(grid: dict, metric: str = "precision_at_k") -> str:
    """Markdown table of the audit grid: one row per model, one column per level.

    Cells are `mean +/- std`. A cell built from fewer than three seeds is
    suffixed with `!` so an under-powered number cannot be quietly read as an
    estimate.
    """
    levels = ("random", "cold_drug", "cold_target", "cold_pair")
    labels = {"random": "Warm", "cold_drug": "Cold-Drug",
              "cold_target": "Cold-Target", "cold_pair": "Cold-Pair"}

    lines = ["| Model | " + " | ".join(labels[l] for l in levels) + " | Drop |",
             "|---" * (len(levels) + 2) + "|"]

    for model_name in sorted(grid):
        cells, means = [], {}
        for level in levels:
            entry = grid[model_name].get(level, {}).get(metric)
            if entry is None or not np.isfinite(entry.get("mean", float("nan"))):
                cells.append("n/a")
                continue
            means[level] = entry["mean"]
            spread = "" if not np.isfinite(entry.get("std", float("nan"))) \
                else f" ± {entry['std']:.3f}"
            flag = "" if entry.get("sufficient_seeds") else " !"
            cells.append(f"{entry['mean']:.3f}{spread}{flag}")

        drop = degradation(means)
        drop_cell = "n/a" if not np.isfinite(drop["absolute_drop"]) \
            else f"{drop['absolute_drop']:+.3f}"
        lines.append(f"| {model_name} | " + " | ".join(cells) + f" | {drop_cell} |")

    lines.append("")
    lines.append("`±` is the standard deviation over seeds. "
                 "`!` marks a cell with fewer than "
                 f"{MIN_SEEDS_FOR_A_CLAIM} seeds — not a usable estimate.")
    return "\n".join(lines)
