"""
Track C (124AD0067) — the core explanation-quality measurement.

Checks whether a model's attention weights actually point at real,
annotated binding-site residues. This is the metric computed at all
four difficulty levels (warm, cold-drug, cold-target, cold-pair) to
produce the paper's headline result. See docs/03_GUIDE_124AD0067.md Step 1.
"""
import numpy as np


def precision_at_k(attention_weights: np.ndarray, true_binding_sites: set, k: int = 10) -> float:
    """
    attention_weights: 1D array, one score per protein position
    true_binding_sites: set of position indices that are real binding sites
                         (build this from src/data/binding_sites.py's output)
    k: how many top-attended positions to check

    Returns the fraction of the top-k attended positions that are real
    binding sites.
    """
    top_k_positions = np.argsort(attention_weights)[-k:]
    hits = sum(1 for pos in top_k_positions if pos in true_binding_sites)
    return hits / k


def precision_at_k_curve(attention_weights: np.ndarray, true_binding_sites: set,
                          k_values=(5, 10, 20)) -> dict:
    """Report precision@k at several k values -- don't rely on just one."""
    return {k: precision_at_k(attention_weights, true_binding_sites, k) for k in k_values}


def batch_precision_at_k(all_attention_weights: list, all_true_sites: list, k: int = 10) -> float:
    """
    Average precision@k across a whole test set (a whole split).
    all_attention_weights: list of 1D arrays, one per protein in the batch/split
    all_true_sites: list of sets, one per protein, aligned with all_attention_weights
    """
    scores = [
        precision_at_k(attn, sites, k)
        for attn, sites in zip(all_attention_weights, all_true_sites)
        if len(sites) > 0  # skip proteins with no annotated sites
    ]
    return float(np.mean(scores)) if scores else float("nan")


if __name__ == "__main__":
    # smoke test with fake data before real model outputs exist
    rng = np.random.default_rng(0)
    fake_attention = rng.random(300)               # pretend protein of length 300
    fake_binding_sites = {41, 42, 43, 143, 144, 145}

    print("precision@10:", precision_at_k(fake_attention, fake_binding_sites, k=10))
    print("curve:", precision_at_k_curve(fake_attention, fake_binding_sites))
