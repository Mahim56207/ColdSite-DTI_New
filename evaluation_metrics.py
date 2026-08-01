import numpy as np

def precision_at_k(attention_weights, true_binding_sites, k=10):
    """
    Calculates the precision@k for model attention weights against known binding sites.

    Args:
        attention_weights (np.array or list): 1D array of attention scores per protein position.
        true_binding_sites (set or list): Indices of the real binding sites.
        k (int): The number of top-attended positions to check.

    Returns:
        float: The precision@k score (hits / k). Returns 0.0 if ground truth is empty.
    """
    # Edge case 1: No true binding sites exist
    if not true_binding_sites:
        return 0.0

    # Edge case 2: k is larger than the sequence length
    k = min(k, len(attention_weights))

    # Convert attention weights to a numpy array if it isn't already
    attention_weights = np.array(attention_weights)

    # Get the indices of the top k attention scores. 
    # np.argsort sorts ascending, so we take the last 'k' elements and reverse them.
    top_k_indices = np.argsort(attention_weights)[-k:][::-1]

    # Count how many of our top_k_indices are in the true_binding_sites set
    hits = sum(1 for pos in top_k_indices if pos in true_binding_sites)

    # Calculate precision
    return hits / k

# ==========================================
# Dummy Data Testing (Step 1 Requirement)
# ==========================================
if __name__ == "__main__":
    # Pretend protein of length 300
    protein_length = 300

    # Generate random fake attention weights
    fake_attention = np.random.rand(protein_length)

    # Let's artificially boost a few spots so our dummy test finds some hits
    fake_attention[42] = 0.99
    fake_attention[144] = 0.95
    fake_attention[250] = 0.90

    # Fake ground truth binding sites (using a set for faster lookups)
    fake_true_sites = {41, 42, 43, 143, 144, 145}

    print("Testing precision@k with dummy data:")
    print("-" * 35)

    # Test at multiple values of k as requested by the guide
    for test_k in [5, 10, 20]:
        score = precision_at_k(fake_attention, fake_true_sites, k=test_k)
        print(f"Precision@{test_k:<2}: {score:.2f} ({int(score * test_k)} hits out of {test_k})")
def permutation_test(true_binding_sites, protein_length, actual_hits, k=10, n_trials=1000):
    """
    Performs a permutation test to calculate the statistical significance of the model's hits.

    Args:
        true_binding_sites (set or list): Indices of the real binding sites.
        protein_length (int): Total length of the protein sequence.
        actual_hits (int): The number of correct hits our model achieved.
        k (int): The number of top positions selected.
        n_trials (int): How many random trials to run (default 1000).

    Returns:
        float: The p-value (fraction of random trials that did as well or better than the model).
    """
    if not true_binding_sites or protein_length == 0:
        return 1.0  # If there's no ground truth, the result isn't significant

    random_hits = []

    # Run the random trials
    for _ in range(n_trials):
        # Pick 'k' random unique positions from the protein
        random_top_k = np.random.choice(protein_length, size=k, replace=False)

        # Count how many of these random positions hit a true binding site
        hits = sum(1 for pos in random_top_k if pos in true_binding_sites)
        random_hits.append(hits)

    # Calculate p-value: how many random trials got >= actual_hits?
    better_or_equal = sum(1 for h in random_hits if h >= actual_hits)
    p_value = better_or_equal / n_trials

    return p_value

# ==========================================
# Dummy Data Testing (Step 2 Requirement)
# ==========================================
if __name__ == "__main__":
    # Continuing from our Step 1 dummy data...
    protein_length = 300
    fake_true_sites = {41, 42, 43, 143, 144, 145}

    print("\nTesting permutation_test with dummy data:")
    print("-" * 40)

    test_k = 10
    # Let's pretend our model got 3 hits out of 10
    pretend_actual_hits = 3 

    p_val = permutation_test(fake_true_sites, protein_length, pretend_actual_hits, k=test_k)

    print(f"Model Hits: {pretend_actual_hits} out of {test_k}")
    print(f"P-value: {p_val:.4f}")
    if p_val < 0.05:
        print("Result: Statistically significant (p < 0.05)! Our model beats random chance.")
    else:
        print("Result: Not significant. Pure luck could achieve this score.")
