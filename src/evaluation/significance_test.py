"""
Track C (124AD0067) — is the precision@k score better than chance?

A permutation test: compares the observed precision@k against many random
selections of k positions, to get a p-value. See docs/03_GUIDE_124AD0067.md
Step 2.
"""
import numpy as np
from src.evaluation.precision_at_k import precision_at_k


def permutation_test(true_binding_sites: set, protein_length: int, k: int = 10,
                      n_trials: int = 1000, observed_precision: float = None, seed: int = 0):
    """
    Returns (mean_random_precision, p_value).
    p_value = fraction of random trials that scored >= the observed precision.
    A small p-value (e.g. < 0.05) means the model's attention is doing
    meaningfully better than picking k random positions.
    """
    rng = np.random.default_rng(seed)
    random_scores = []
    for _ in range(n_trials):
        random_top_k = rng.choice(protein_length, size=k, replace=False)
        hits = sum(1 for pos in random_top_k if pos in true_binding_sites)
        random_scores.append(hits / k)
    random_scores = np.array(random_scores)

    p_value = float(np.mean(random_scores >= observed_precision))
    return float(random_scores.mean()), p_value


def evaluate_with_significance(attention_weights: np.ndarray, true_binding_sites: set,
                                k: int = 10, n_trials: int = 1000) -> dict:
    """Convenience wrapper: computes precision@k AND its significance in one call."""
    observed = precision_at_k(attention_weights, true_binding_sites, k)
    chance_mean, p_value = permutation_test(
        true_binding_sites, len(attention_weights), k, n_trials, observed
    )
    return {
        "precision_at_k": observed,
        "chance_level": chance_mean,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fake_attention = rng.random(300)
    fake_binding_sites = {41, 42, 43, 143, 144, 145}

    result = evaluate_with_significance(fake_attention, fake_binding_sites, k=10)
    print(result)
