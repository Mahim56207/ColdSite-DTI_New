"""
Faithfulness — does the model actually USE the residues it points at?

This is now a core measurement, not the optional stretch layer the original
guides described. The reason is in the literature: attention-as-explanation has
been contested since Jain & Wallace (2019) and Serrano & Smith (2019), who
showed high attention weights do not necessarily indicate high influence on the
prediction. Wiegreffe & Pinter (2019) pushed back, but the dispute is live, and
a 2026 paper measuring only whether attention *looks* biologically sensible is
measuring plausibility, not explanation quality.

Plausibility and faithfulness are independent axes:

                    faithful          not faithful
    plausible       the good case     a convincing lie  <-- the dangerous one
    not plausible   honest oddity     noise

precision@k measures the ROWS. This module measures the COLUMNS. The paper's
claim needs both, because "plausible but not faithful" is exactly the failure
mode a scientist cannot detect by eye.

Metrics
-------
comprehensiveness  remove the top-k attended residues. A faithful explanation
                   should cause a LARGE prediction change. Higher is better.
sufficiency        keep ONLY the top-k attended residues. A faithful
                   explanation should cause a SMALL change. Lower is better.
AOPC               area over the perturbation curve: remove residues
                   progressively and track the drop.

The random-masking control
--------------------------
Every metric here is reported against a random-residue baseline, and that is
not optional politeness. Masking anything at all shifts the input off the
training distribution, and the prediction moves for reasons that have nothing
to do with the explanation (this is the ROAR critique). A comprehensiveness of
0.4 means nothing on its own; it means something only next to a random-masking
comprehensiveness of 0.05. Report the delta, never the raw number.
"""
import numpy as np
import torch

from src.evaluation.precision_at_k import top_k_positions

# Residues are replaced with the UNK token rather than PAD. PAD is excluded from
# attention by the key_padding_mask, so masking with it would silently shorten
# the protein and change the model's view of sequence length -- conflating
# "these residues mattered" with "the protein got shorter".
MASK_TOKEN = 1


def _as_batch(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.dim() == 2 else tensor.unsqueeze(0)


@torch.no_grad()
def _predict(model, drug: torch.Tensor, protein: torch.Tensor) -> float:
    """One scalar prediction, from either a raw model or a registry adapter.

    ColdSite-DTI's `forward` returns `(pred, attention)`, which is what this
    module was written against. The audited baselines do not share that
    signature -- they arrive wrapped in an `ExplainableDTIModel` adapter whose
    contract is `predict(drug, protein) -> float`. Without this branch,
    faithfulness could only ever be measured on our own model, which is the one
    model whose result matters least under the audit framing.

    Duck-typed rather than imported, to keep model_registry out of this
    module's import graph.
    """
    if hasattr(model, "predict"):
        return float(model.predict(_as_batch(drug), _as_batch(protein)))
    model.eval()
    pred, _attn = model(_as_batch(drug), _as_batch(protein))
    return float(pred.squeeze().item())


def mask_positions(protein: torch.Tensor, positions, keep: bool = False,
                   mask_token: int = MASK_TOKEN) -> torch.Tensor:
    """Replace residues with the mask token.

    keep=False  mask the listed positions   (comprehensiveness)
    keep=True   mask everything EXCEPT them (sufficiency)
    """
    masked = protein.clone()
    flat = masked if masked.dim() == 1 else masked[0]
    positions = [int(p) for p in positions if 0 <= int(p) < flat.numel()]

    if keep:
        selector = torch.ones(flat.numel(), dtype=torch.bool)
        selector[positions] = False
        # never mask padding into UNK -- that would invent residues
        selector &= flat != 0
        flat[selector] = mask_token
    else:
        for p in positions:
            if flat[p] != 0:
                flat[p] = mask_token
    return masked


def comprehensiveness(model, drug, protein, attention, k: int = 10,
                      rng=None) -> float:
    """Prediction change when the top-k attended residues are removed.

    A faithful explanation points at residues the model depends on, so deleting
    them should move the prediction. Near zero means the model reached its
    answer some other way and the highlighted residues were decoration.
    """
    baseline = _predict(model, drug, protein)
    top_k = top_k_positions(attention, k, rng=rng)
    ablated = _predict(model, drug, mask_positions(protein, top_k, keep=False))
    return abs(baseline - ablated)


def sufficiency(model, drug, protein, attention, k: int = 10, rng=None) -> float:
    """Prediction change when ONLY the top-k attended residues are kept.

    Low is good: if those residues carry the signal, discarding everything else
    should barely matter.
    """
    baseline = _predict(model, drug, protein)
    top_k = top_k_positions(attention, k, rng=rng)
    reduced = _predict(model, drug, mask_positions(protein, top_k, keep=True))
    return abs(baseline - reduced)


def random_control(model, drug, protein, k: int = 10, mode: str = "comprehensiveness",
                   n_trials: int = 10, rng=None) -> float:
    """The same intervention on k RANDOM residues.

    This is the number that makes the others interpretable. Masking always
    perturbs the prediction somewhat because the masked input is off the
    training distribution; the question is whether masking the *attended*
    residues perturbs it more than masking arbitrary ones.
    """
    rng = np.random.default_rng() if rng is None else rng
    flat = protein if protein.dim() == 1 else protein[0]
    length = int((flat != 0).sum().item())
    if length < k:
        return float("nan")

    baseline = _predict(model, drug, protein)
    deltas = []
    for _ in range(n_trials):
        positions = rng.choice(length, size=k, replace=False)
        keep = mode == "sufficiency"
        perturbed = _predict(model, drug,
                             mask_positions(protein, positions, keep=keep))
        deltas.append(abs(baseline - perturbed))
    return float(np.mean(deltas))


def aopc(model, drug, protein, attention, k_values=(1, 5, 10, 20, 50),
         rng=None) -> dict:
    """Area over the perturbation curve: remove progressively more residues.

    A faithful ranking degrades the prediction fastest at the top of the list.
    A flat curve means the ordering carries no information even if the top-k
    happens to overlap a real pocket.
    """
    attention = np.asarray(attention, dtype=float)
    baseline = _predict(model, drug, protein)
    usable = [k for k in k_values if k <= attention.size]

    curve = {}
    for k in usable:
        top_k = top_k_positions(attention, k, rng=rng)
        ablated = _predict(model, drug, mask_positions(protein, top_k))
        curve[k] = abs(baseline - ablated)

    return {
        "curve": curve,
        "aopc": float(np.mean(list(curve.values()))) if curve else float("nan"),
        "baseline_prediction": baseline,
    }


def evaluate_faithfulness(model, drug, protein, attention, k: int = 10,
                          n_random_trials: int = 10, seed: int = 0) -> dict:
    """All faithfulness metrics for one drug-protein pair, with controls.

    `comprehensiveness_delta` is the headline: observed minus random. A value
    at or below zero means the explanation is no more load-bearing than an
    arbitrary set of residues, however biologically plausible it looked.
    """
    rng = np.random.default_rng(seed)
    attention = np.asarray(attention, dtype=float)

    comp = comprehensiveness(model, drug, protein, attention, k, rng=rng)
    suff = sufficiency(model, drug, protein, attention, k, rng=rng)
    comp_random = random_control(model, drug, protein, k, "comprehensiveness",
                                 n_random_trials, rng=rng)
    suff_random = random_control(model, drug, protein, k, "sufficiency",
                                 n_random_trials, rng=rng)

    return {
        "comprehensiveness": comp,
        "comprehensiveness_random": comp_random,
        "comprehensiveness_delta": comp - comp_random,
        "sufficiency": suff,
        "sufficiency_random": suff_random,
        "sufficiency_delta": suff - suff_random,
        "aopc": aopc(model, drug, protein, attention, rng=rng)["aopc"],
        "k": k,
    }


def batch_faithfulness(model, drug_batch, protein_batch, attentions,
                       k: int = 10, n_random_trials: int = 5,
                       seed: int = 0, max_pairs: int = None) -> dict:
    """Faithfulness averaged over a split.

    Expensive: every pair costs (2 + 2*n_random_trials + len(k_values)) forward
    passes. Use max_pairs to subsample -- the number that matters is the mean
    over a few hundred pairs, not over all of them.
    """
    results = []
    n = len(attentions) if max_pairs is None else min(max_pairs, len(attentions))

    for i in range(n):
        results.append(evaluate_faithfulness(
            model, drug_batch[i], protein_batch[i], attentions[i],
            k=k, n_random_trials=n_random_trials, seed=seed + i))

    def mean(key):
        values = [r[key] for r in results if np.isfinite(r[key])]
        return float(np.mean(values)) if values else float("nan")

    summary = {key: mean(key) for key in (
        "comprehensiveness", "comprehensiveness_random", "comprehensiveness_delta",
        "sufficiency", "sufficiency_random", "sufficiency_delta", "aopc")}
    summary["n_pairs"] = len(results)
    summary["k"] = k
    # the one-line verdict the paper needs per split
    summary["explanation_is_load_bearing"] = bool(
        np.isfinite(summary["comprehensiveness_delta"])
        and summary["comprehensiveness_delta"] > 0
    )
    return summary
