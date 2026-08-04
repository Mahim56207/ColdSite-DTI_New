"""
Track C (124AD0067) — is the precision@k score better than chance?

A permutation test compares the observed precision@k against many random
selections of k positions, giving a p-value. See docs/03_GUIDE_124AD0067.md
Step 2.

Two levels are provided:

  permutation_test        one protein  -- "is this explanation better than chance?"
  permutation_test_batch  one split    -- "is the MEAN over this split better than chance?"

The split-level test is the one the paper needs. Per-protein p-values across
400 proteins are 400 hypothesis tests, and roughly 20 of them will land under
0.05 by luck alone; quoting those as evidence would be p-hacking. Test the
mean, once, per split.
"""
import numpy as np

from src.evaluation.precision_at_k import precision_at_k


def _chance_precision_sample(rng, true_binding_sites: set, protein_length: int,
                             k: int) -> float:
    """One draw of precision@k from k uniformly random positions."""
    random_top_k = rng.choice(protein_length, size=k, replace=False)
    hits = sum(1 for pos in random_top_k if int(pos) in true_binding_sites)
    return hits / k


def permutation_test(true_binding_sites: set, protein_length: int,
                     observed_precision: float, k: int = 10,
                     n_trials: int = 1000, seed: int = 0) -> dict:
    """Is one protein's precision@k better than picking k positions at random?

    `observed_precision` is required, not optional. The previous signature
    defaulted it to None, and `np.mean(scores >= None)` raises deep inside the
    call rather than at the point the caller forgot the argument.

    The p-value uses the add-one estimator (1 + hits) / (1 + n_trials) rather
    than a bare mean. A bare mean can return exactly 0.0, which claims infinite
    confidence from a finite number of draws; with 1000 trials the smallest
    honest statement is p < 0.001.
    """
    if k > protein_length:
        raise ValueError(
            f"k={k} exceeds protein length {protein_length}; cannot sample "
            f"{k} distinct positions"
        )
    if not 0.0 <= observed_precision <= 1.0:
        raise ValueError(
            f"observed_precision must be in [0, 1], got {observed_precision}"
        )

    rng = np.random.default_rng(seed)
    random_scores = np.array([
        _chance_precision_sample(rng, true_binding_sites, protein_length, k)
        for _ in range(n_trials)
    ])

    n_at_least = int(np.sum(random_scores >= observed_precision))
    p_value = (1 + n_at_least) / (1 + n_trials)

    return {
        "observed": float(observed_precision),
        "chance_mean": float(random_scores.mean()),
        "chance_std": float(random_scores.std()),
        "p_value": float(p_value),
        "n_trials": n_trials,
        "significant": bool(p_value < 0.05),
    }


def evaluate_with_significance(attention_weights, true_binding_sites: set,
                               k: int = 10, n_trials: int = 1000,
                               seed: int = 0) -> dict:
    """Convenience wrapper: precision@k AND its significance for one protein."""
    attention = np.asarray(attention_weights, dtype=float)
    observed = precision_at_k(attention, true_binding_sites, k,
                              rng=np.random.default_rng(seed))
    result = permutation_test(true_binding_sites, attention.size, observed,
                              k=k, n_trials=n_trials, seed=seed)
    result["precision_at_k"] = observed
    return result


def permutation_test_batch(all_attention_weights: list, all_true_sites: list,
                           k: int = 10, n_trials: int = 1000,
                           seed: int = 0) -> dict:
    """Is the MEAN precision@k over a whole split better than chance?

    Builds the null by drawing k random positions for every protein in the
    split and taking the mean, n_trials times. This preserves the split's own
    mix of protein lengths and site counts, which a single pooled null would
    wash out -- short proteins with many annotated sites have a much higher
    chance level than long ones with two, and the mean of those is not the
    chance level of any single protein.
    """
    rng = np.random.default_rng(seed)

    usable = [
        (np.asarray(a, dtype=float), s)
        for a, s in zip(all_attention_weights, all_true_sites)
        if len(s) > 0 and k <= np.asarray(a).size
    ]
    if not usable:
        return {"observed_mean": float("nan"), "p_value": float("nan"),
                "n_proteins": 0, "significant": False}

    observed = float(np.mean([
        precision_at_k(a, s, k, rng=rng) for a, s in usable
    ]))

    null_means = np.array([
        np.mean([_chance_precision_sample(rng, s, a.size, k) for a, s in usable])
        for _ in range(n_trials)
    ])

    n_at_least = int(np.sum(null_means >= observed))
    p_value = (1 + n_at_least) / (1 + n_trials)

    return {
        "observed_mean": observed,
        "chance_mean": float(null_means.mean()),
        "chance_std": float(null_means.std()),
        "p_value": float(p_value),
        "n_proteins": len(usable),
        "n_trials": n_trials,
        "significant": bool(p_value < 0.05),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sites = {41, 42, 43, 143, 144, 145}

    random_attention = rng.random(300)
    print("random attention: ", evaluate_with_significance(random_attention, sites, k=10))

    perfect = np.zeros(300)
    perfect[sorted(sites)] = 1.0
    print("perfect attention:", evaluate_with_significance(perfect, sites, k=10))
