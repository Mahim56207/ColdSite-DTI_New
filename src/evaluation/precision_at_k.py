"""
Track C (124AD0067) — the core explanation-quality measurement.

Checks whether a model's attention weights actually point at real, annotated
binding-site residues. This is the metric computed at all four difficulty
levels (warm, cold-drug, cold-target, cold-pair) to produce the paper's
headline result. See docs/03_GUIDE_124AD0067.md Step 1.

IMPORTANT -- coordinate system
------------------------------
`true_binding_sites` must be a set of 0-INDEXED array positions. UniProt
annotations are 1-indexed inclusive ranges and must be converted first. Always
build the set with `src.data.ground_truth.load_site_sets`; reading the
ground-truth JSON directly costs about a third of the score on a model with
perfect attention, silently.
"""
import numpy as np

DEFAULT_K_VALUES = (5, 10, 20)


def _validate(attention_weights, k: int) -> np.ndarray:
    """Reject the inputs that would otherwise return a quietly wrong number."""
    attention = np.asarray(attention_weights, dtype=float)
    if attention.ndim != 1:
        raise ValueError(
            f"attention_weights must be 1D (one score per residue), got shape "
            f"{attention.shape}. If this is a (1, seq_len) cross-attention map, "
            f"squeeze the query axis first."
        )
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    if k > attention.size:
        # np.argsort(x)[-k:] returns only len(x) indices when k > len(x), so the
        # hit count gets divided by a k that was never actually retrieved. That
        # silently deflates the score instead of failing.
        raise ValueError(
            f"k={k} exceeds protein length {attention.size}; precision@k is "
            f"undefined. Use a smaller k, or skip this protein."
        )
    if not np.all(np.isfinite(attention)):
        raise ValueError(
            "attention_weights contains NaN or inf. Check for an all-padding "
            "row or an untrained model before evaluating."
        )
    return attention


def top_k_positions(attention_weights, k: int = 10, rng=None) -> np.ndarray:
    """The k highest-attention positions, with ties broken at random.

    Tie-breaking is not cosmetic here. `np.argsort` is stable, so on any tied
    block it returns the LOWEST indices first -- and `[-k:]` then takes the
    highest of those, biasing selection toward the C-terminus of the protein.
    Attention maps tie constantly (exact zeros on masked positions, saturated
    softmax on an untrained model), so a stable sort turns a positional
    artefact into apparent explanation quality. A random tie-break makes the
    tied case behave like the chance baseline the permutation test compares
    against, which is what we actually want.
    """
    attention = _validate(attention_weights, k)
    rng = np.random.default_rng() if rng is None else rng
    jitter = rng.random(attention.size)
    # lexsort orders by the LAST key first: attention descending, ties by noise
    order = np.lexsort((jitter, -attention))
    return order[:k]


def precision_at_k(attention_weights, true_binding_sites: set,
                   k: int = 10, rng=None) -> float:
    """
    attention_weights:  1D array, one score per protein position
    true_binding_sites: set of 0-INDEXED positions that are real binding sites
                        (build with src.data.ground_truth.load_site_sets)
    k:                  how many top-attended positions to check

    Returns the fraction of the top-k attended positions that are real binding
    sites. Note the ceiling: with fewer than k annotated sites the best
    achievable score is |sites| / k, not 1.0 -- see `achievable_ceiling`.
    """
    top_k = top_k_positions(attention_weights, k, rng=rng)
    hits = sum(1 for pos in top_k if int(pos) in true_binding_sites)
    return hits / k


def achievable_ceiling(true_binding_sites: set, k: int = 10) -> float:
    """The best precision@k a perfect explanation could possibly reach.

    A protein with 6 annotated positions caps out at 6/20 = 0.30 at k=20. A raw
    precision@20 of 0.28 is therefore near-perfect, not poor. Reporting raw
    precision across proteins with different site counts compares numbers that
    have different maxima, so report this alongside it.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    return min(len(true_binding_sites), k) / k


def normalised_precision_at_k(attention_weights, true_binding_sites: set,
                              k: int = 10, rng=None) -> float:
    """precision@k rescaled by its achievable ceiling, so 1.0 means perfect.

    Comparable across proteins with different numbers of annotated sites.
    Returns NaN when there are no sites to find.
    """
    ceiling = achievable_ceiling(true_binding_sites, k)
    if ceiling == 0:
        return float("nan")
    return precision_at_k(attention_weights, true_binding_sites, k, rng=rng) / ceiling


def precision_at_k_curve(attention_weights, true_binding_sites: set,
                         k_values=DEFAULT_K_VALUES, rng=None) -> dict:
    """Report precision@k at several k values -- don't rely on just one.

    k values larger than the protein are skipped rather than raising, so a
    short protein doesn't abort a whole sweep.
    """
    attention = np.asarray(attention_weights, dtype=float)
    return {
        k: precision_at_k(attention, true_binding_sites, k, rng=rng)
        for k in k_values if k <= attention.size
    }


def batch_precision_at_k(all_attention_weights: list, all_true_sites: list,
                         k: int = 10, rng=None) -> dict:
    """Average precision@k across a whole split.

    all_attention_weights: list of 1D arrays, one per protein
    all_true_sites:        list of sets, one per protein, aligned to the above

    Returns a dict rather than a bare float, because the count of proteins that
    were skipped is part of the result. A mean over 12 usable proteins out of
    400 is not the same finding as a mean over 400, and a bare float hides
    which one you have.
    """
    if len(all_attention_weights) != len(all_true_sites):
        raise ValueError(
            f"got {len(all_attention_weights)} attention arrays but "
            f"{len(all_true_sites)} site sets -- these must be aligned"
        )

    rng = np.random.default_rng() if rng is None else rng
    scores, normalised, ceilings = [], [], []
    skipped_no_sites = skipped_too_short = 0

    for attention, sites in zip(all_attention_weights, all_true_sites):
        attention = np.asarray(attention, dtype=float)
        if len(sites) == 0:
            skipped_no_sites += 1
            continue
        if k > attention.size:
            skipped_too_short += 1
            continue
        scores.append(precision_at_k(attention, sites, k, rng=rng))
        ceilings.append(achievable_ceiling(sites, k))
        normalised.append(scores[-1] / ceilings[-1])

    return {
        "k": k,
        "mean_precision_at_k": float(np.mean(scores)) if scores else float("nan"),
        "mean_normalised": float(np.mean(normalised)) if normalised else float("nan"),
        "mean_ceiling": float(np.mean(ceilings)) if ceilings else float("nan"),
        "n_evaluated": len(scores),
        "n_skipped_no_sites": skipped_no_sites,
        "n_skipped_too_short": skipped_too_short,
        "per_protein": scores,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fake_attention = rng.random(300)
    fake_binding_sites = {41, 42, 43, 143, 144, 145}

    print("precision@10:", precision_at_k(fake_attention, fake_binding_sites, k=10))
    print("curve:", precision_at_k_curve(fake_attention, fake_binding_sites))
    print("ceiling@20:", achievable_ceiling(fake_binding_sites, 20))

    perfect = np.zeros(300)
    perfect[sorted(fake_binding_sites)] = 1.0
    print("perfect attention, precision@10:",
          precision_at_k(perfect, fake_binding_sites, 10))
    print("perfect attention, normalised@10:",
          normalised_precision_at_k(perfect, fake_binding_sites, 10))
