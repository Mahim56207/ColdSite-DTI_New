"""
Faithfulness for a sub-word model, measured in the space the model actually reads.

Why this exists
---------------
The residue-space test masks k residues, re-tokenises, and compares that with masking k
random residues. For MolTrans the two are not the same intervention: replacing a residue
with `X` re-segments the protein, and masking its ten most-attended residues changes 48%
of its tokens where ten random residues change 95% (`src/evaluation/mask_comparability.py`).
Subtracting a larger intervention from a smaller one produced a negative delta in 11 of
12 cells, which says nothing about the attention.

Here the intervention is a **token**, which is what MolTrans reads:

* the explanation's arm removes the tokens carrying its top-k attention,
* the control removes **the same number** of tokens, chosen at random,
* so both arms change an identical amount of the model's input, by construction --
  no rejection sampling, no re-encoding, and nothing to match after the fact.

"Remove" means clearing the token's entry in the attention mask, so the transformer cannot
see it. That is a real deletion rather than a substitution: replacing a token id with
another sub-word would ask a different question ("what if this were a different peptide")
instead of ("what if this were not there").

Attention is taken per token, before the projection onto residues, so the readout choice
in `attention_projection` plays no part either.

What it cannot say
------------------
This measures MolTrans against the residue-level models only in the loose sense that both
report "the explanation moves the prediction more than a size-matched control does". The
units differ -- tokens rather than residues -- so the deltas are not numerically
comparable across models, and Methods has to say so. Within a model, across levels and
seeds, it is exactly comparable, which is what the audit uses it for.

    python -m src.evaluation.token_faithfulness --model moltrans --dataset davis --seed 1
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

K = 10
N_RANDOM_TRIALS = 5
LEVELS = ("random", "cold_drug", "cold_target", "cold_pair")


def token_attention(adapter, smiles: str, sequence: str):
    """(encoded tensors, one attention weight per protein token).

    The weights are the adapter's own, read before `project_token_attention` spreads them
    over residues.
    """
    from src.evaluation.attention_projection import capture_moltrans_attention

    from src.evaluation.baseline_adapters import _as_batch

    drug, drug_mask, protein, protein_mask, tokens = type(adapter).encode(smiles, sequence)
    # the encoder returns one pair unbatched; the vendored forward indexes size(1)
    drug, drug_mask = _as_batch(drug), _as_batch(drug_mask)
    protein, protein_mask = _as_batch(protein), _as_batch(protein_mask)
    layers = adapter.model.p_encoder.layer
    target = layers[adapter.attention_layer].attention.self
    store, handle = capture_moltrans_attention(target)
    try:
        adapter.model.eval()
        adapter._fit_batch_size(drug.shape[0])
        with torch.no_grad():
            adapter.model(drug.to(adapter.device), protein.to(adapter.device),
                          drug_mask.to(adapter.device), protein_mask.to(adapter.device))
    finally:
        handle.remove()
    probs = store.get("probs")
    if probs is None:
        raise RuntimeError("the attention hook captured nothing -- check the vendored "
                           "module layout before trusting any number")
    heads = probs[0]
    reduced = (heads.mean(dim=0) if adapter.head_reduce == "mean"
               else heads.max(dim=0).values)
    weights = reduced.mean(dim=0).cpu().numpy()
    n_real = int(np.asarray(protein_mask).reshape(-1).sum())
    return {"drug": drug, "drug_mask": drug_mask, "protein": protein,
            "protein_mask": protein_mask, "tokens": tokens,
            "n_real": n_real}, np.asarray(weights, dtype=float)[:n_real]


def remove_tokens(protein_mask: torch.Tensor, positions, keep: bool = False) -> torch.Tensor:
    """The attention mask with `positions` cleared (or everything but them, if `keep`).

    Padding stays padding either way: a cleared entry means "the model cannot see this
    token", and inventing visibility for padding would be a different intervention.
    """
    mask = protein_mask.clone()
    flat = mask if mask.dim() == 1 else mask[0]
    real = flat != 0
    positions = [int(p) for p in positions if 0 <= int(p) < flat.numel()]
    if keep:
        selector = torch.ones(flat.numel(), dtype=torch.bool)
        selector[positions] = False
        flat[selector & real] = 0
    else:
        for position in positions:
            flat[position] = 0
    return mask


def _predict(adapter, encoded, protein_mask) -> float:
    return float(adapter.predict(encoded["drug"], encoded["protein"],
                                 drug_mask=encoded["drug_mask"],
                                 protein_mask=protein_mask))


def pair_faithfulness(adapter, encoded, weights, k: int = K,
                      n_random_trials: int = N_RANDOM_TRIALS, seed: int = 0) -> dict:
    """One pair: the explanation's tokens against the same number of random ones."""
    rng = np.random.default_rng(seed)
    n_real = encoded["n_real"]
    k = min(k, n_real)
    baseline = _predict(adapter, encoded, encoded["protein_mask"])
    top = np.argsort(-weights[:n_real])[:k]

    comp = abs(baseline - _predict(adapter, encoded,
                                   remove_tokens(encoded["protein_mask"], top)))
    suff = abs(baseline - _predict(adapter, encoded,
                                   remove_tokens(encoded["protein_mask"], top, keep=True)))
    comp_random, suff_random = [], []
    for _trial in range(n_random_trials):
        positions = rng.choice(n_real, size=k, replace=False)
        comp_random.append(abs(baseline - _predict(
            adapter, encoded, remove_tokens(encoded["protein_mask"], positions))))
        suff_random.append(abs(baseline - _predict(
            adapter, encoded, remove_tokens(encoded["protein_mask"], positions, keep=True))))
    return {"comprehensiveness": comp, "comprehensiveness_random": float(np.mean(comp_random)),
            "comprehensiveness_delta": comp - float(np.mean(comp_random)),
            "sufficiency": suff, "sufficiency_random": float(np.mean(suff_random)),
            "sufficiency_delta": suff - float(np.mean(suff_random)),
            "k": k, "n_tokens": n_real}


def level(adapter, rows, k: int = K, n_random_trials: int = N_RANDOM_TRIALS,
          seed: int = 0, max_pairs: int | None = None) -> dict:
    records = []
    for index, (_i, row) in enumerate(rows.iterrows()):
        if max_pairs and index >= max_pairs:
            break
        encoded, weights = token_attention(adapter, str(row["Drug"]),
                                           str(row["Target"])[:1000])
        records.append(pair_faithfulness(adapter, encoded, weights, k,
                                         n_random_trials, seed + index))
    if not records:
        return {"n": 0}
    out = {"n": len(records), "k": k}
    for field in ("comprehensiveness", "comprehensiveness_random", "comprehensiveness_delta",
                  "sufficiency", "sufficiency_random", "sufficiency_delta"):
        out[field] = float(np.mean([r[field] for r in records]))
    out["load_bearing"] = out["comprehensiveness_delta"] > 0
    out["median_tokens"] = int(np.median([r["n_tokens"] for r in records]))
    return out


def report(results: dict, model: str, dataset: str, seed: int) -> str:
    lines = [f"# Faithfulness in token space — {model}, {dataset}, seed {seed}", "",
             "The intervention is a token, which is what this model reads, and the control "
             "removes the same number of tokens as the explanation does -- so both arms "
             "change an identical amount of the input "
             "(`src/evaluation/token_faithfulness.py`). The residue-space test could not "
             "do that: masking its top-10 residues changes 48% of its tokens where 10 "
             "random residues change 95%.", "",
             "| Level | comp. | random control | **delta** | suff. | suff. random | "
             "tokens | n | load-bearing? |", "|---|---|---|---|---|---|---|---|---|"]
    for name in LEVELS:
        entry = results.get(name)
        if not entry or not entry.get("n"):
            lines.append(f"| {name.replace('_', '-')} | — | — | — | — | — | — | 0 | — |")
            continue
        lines.append(
            f"| {name.replace('_', '-')} | {entry['comprehensiveness']:.4f} | "
            f"{entry['comprehensiveness_random']:.4f} | "
            f"**{entry['comprehensiveness_delta']:+.4f}** | {entry['sufficiency']:.4f} | "
            f"{entry['sufficiency_random']:.4f} | {entry['median_tokens']} | "
            f"{entry['n']} | {'yes' if entry['load_bearing'] else 'no'} |")
    lines += ["", "`delta` is the only column that is a result. A negative delta here "
              "means the explanation's tokens matter less than an equal number of "
              "arbitrary ones -- which, unlike the residue-space version, is a statement "
              "about the attention rather than about the intervention.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model", default="moltrans", choices=("moltrans",))
    parser.add_argument("--dataset", default="davis")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--task", default="binary")
    parser.add_argument("--split-root", default="data/splits")
    parser.add_argument("--checkpoint-dir", default="results")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--n-random-trials", type=int, default=N_RANDOM_TRIALS)
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-sequence-policy", action="store_true")
    args = parser.parse_args()

    import pandas as pd

    from src.evaluation.collect import _read_test_rows
    from src.evaluation.model_registry import model_class
    from src.model.checkpoint_naming import checkpoint_path

    checkpoint = checkpoint_path(args.checkpoint_dir, args.dataset, LEVELS[0], args.task,
                                 args.seed, model=args.model)
    results = {}
    for name in LEVELS:
        split_dir = os.path.join(args.split_root, args.dataset, name)
        checkpoint = checkpoint_path(args.checkpoint_dir, args.dataset, name, args.task,
                                     args.seed, model=args.model)
        if not os.path.exists(checkpoint):
            print(f"   {name}: no checkpoint at {checkpoint}")
            continue
        adapter = model_class(args.model)(checkpoint_path=checkpoint, device=args.device)
        triples = _read_test_rows(split_dir, 1, policy=not args.no_sequence_policy)
        rows = pd.DataFrame(triples, columns=["Target_ID", "Drug", "Target"])
        print(f"{name}: {min(len(rows), args.max_pairs)} pairs, "
              f"{2 + 2 * args.n_random_trials} forward passes each")
        results[name] = level(adapter, rows, args.k, args.n_random_trials,
                              args.seed, args.max_pairs)
    if not results:
        raise SystemExit("no level had a checkpoint -- check --checkpoint-dir")
    text = report(results, args.model, args.dataset, args.seed)
    print("\n" + text)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir,
                        f"token_faithfulness_{args.model}_{args.dataset}_seed{args.seed}")
    with open(stem + ".json", "w") as handle:
        json.dump(results, handle, indent=1)
    with open(stem + ".md", "w") as handle:
        handle.write(text)
    print(f"Saved -> {stem}.json and {stem}.md")


if __name__ == "__main__":
    main()
