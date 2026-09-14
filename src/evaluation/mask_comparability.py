"""
Is a masking-faithfulness test comparing like with like?

Faithfulness here is comprehensiveness minus a random-masking control: mask the k residues
the explanation points at, mask k random residues, and call the difference evidence that
the explanation is load-bearing. That subtraction assumes **both arms are the same size of
intervention**. For a model that reads residues, they are: masking k residues changes
exactly k input positions either way.

For MolTrans they are not. It reads ESPF sub-word tokens, so replacing a residue with `X`
re-segments the protein, and how much of the token sequence changes depends on *where* the
masked residues sit. Measured on DAVIS (2026-09-14): masking MolTrans's ten most-attended
residues changes **43%** of its tokens, while masking ten random residues changes **93%**
-- and for some proteins the attended set changes under 2%. Its faithfulness delta came out
negative in 11 of 12 cells, which would read as "its attention points at residues that
matter less than arbitrary ones". That conclusion is not available from this measurement:
the random arm is simply a bigger intervention.

This module measures the asymmetry so it can be reported, and provides the fair control:

    token_matched_control   random residue sets resampled until the fraction of tokens
                            they change matches the explanation's, within a tolerance.

`asymmetry` is the diagnostic; `residue_level_models_are_unaffected` is the check that the
other models need no correction (one token per residue, so k residues is k tokens by
construction).

    python -m src.evaluation.mask_comparability --model moltrans --n-proteins 20
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

MAX_TRIES = 200
DEFAULT_K = 10
TOLERANCE = 0.10          # |attended - control| fraction of tokens changed


def token_change_fraction(encode, sequence: str, positions, smiles: str = "C") -> float:
    """Fraction of protein tokens that differ once `positions` are masked to X.

    Counts positions that differ over the common length plus any change in token count,
    divided by the original token count -- so a re-segmentation that shortens the
    sequence is not free.
    """
    masked = list(sequence)
    for index in positions:
        masked[int(index)] = "X"
    _d, _dm, before, before_mask, _t = encode(smiles, sequence)
    _d2, _dm2, after, after_mask, _t2 = encode(smiles, "".join(masked))
    n_before, n_after = int(before_mask.sum()), int(after_mask.sum())
    common = min(n_before, n_after)
    differing = int((np.asarray(before).reshape(-1)[:common]
                     != np.asarray(after).reshape(-1)[:common]).sum())
    return (differing + abs(n_before - n_after)) / max(n_before, 1)


def token_matched_control(encode, sequence: str, k: int, target: float, rng,
                          smiles: str = "C", tolerance: float = TOLERANCE,
                          max_tries: int = MAX_TRIES):
    """k random residues whose masking changes about as many tokens as the explanation's.

    Returns (positions, fraction, tries). The closest draw is kept if nothing lands inside
    the tolerance, and `tries` says so -- a control that never matched is a fact about the
    protein, not something to hide.
    """
    best, best_gap, best_fraction = None, np.inf, np.nan
    for attempt in range(1, max_tries + 1):
        positions = rng.choice(len(sequence), size=min(k, len(sequence)), replace=False)
        fraction = token_change_fraction(encode, sequence, positions, smiles)
        gap = abs(fraction - target)
        if gap < best_gap:
            best, best_gap, best_fraction = positions, gap, fraction
        if gap <= tolerance:
            return positions, fraction, attempt
    return best, best_fraction, max_tries


def asymmetry(rows, adapter, k: int = DEFAULT_K, seed: int = 0,
              matched: bool = True) -> list[dict]:
    """Per protein: how big each arm's intervention really is."""
    encode = type(adapter).encode
    rng = np.random.default_rng(seed)
    out = []
    for _index, row in rows.iterrows():
        sequence = str(row["Target"])[:1000]
        smiles = str(row["Drug"])
        drug, drug_mask, protein, protein_mask, tokens = encode(smiles, sequence)
        weights = np.asarray(adapter.explain(drug, protein, protein_tokens=tokens,
                                             drug_mask=drug_mask,
                                             protein_mask=protein_mask), dtype=float)
        top = np.argsort(-weights[:len(sequence)])[:k]
        attended = token_change_fraction(encode, sequence, top, smiles)
        scattered = float(np.mean([
            token_change_fraction(encode, sequence,
                                  rng.choice(len(sequence), k, replace=False), smiles)
            for _ in range(3)]))
        record = {"target": str(row["Target_ID"]), "tokens": int(protein_mask.sum()),
                  "attended": attended, "random": scattered,
                  "ratio": scattered / attended if attended else np.inf}
        if matched:
            _positions, fraction, tries = token_matched_control(
                encode, sequence, k, attended, rng, smiles)
            record.update(matched=fraction, matched_tries=tries)
        out.append(record)
    return out


def residue_level_models_are_unaffected(sequence: str, k: int = DEFAULT_K) -> dict:
    """ColdSite-DTI and HyperAttentionDTI read one token per residue, so k residues is k
    tokens in both arms by construction. Checked rather than asserted."""
    from src.evaluation.residue_space import encode_residues
    before = encode_residues(sequence)
    masked = list(sequence)
    for index in range(k):
        masked[index] = "X"
    after = encode_residues("".join(masked))
    return {"same_length": before.shape == after.shape,
            "positions_changed": int((before != after).sum())}


def report(records: list[dict], model: str, k: int = DEFAULT_K) -> str:
    attended = np.mean([r["attended"] for r in records])
    random_arm = np.mean([r["random"] for r in records])
    lines = [f"# Are the two masking arms the same size of intervention? — {model}", "",
             f"Masking the top-{k} attended residues versus {k} random ones, measured as "
             "the fraction of the model's protein tokens that change "
             "(`src/evaluation/mask_comparability.py`, "
             f"{len(records)} DAVIS proteins).", "",
             f"- explanation's residues: **{attended:.1%}** of tokens change",
             f"- random residues: **{random_arm:.1%}**",
             f"- the random arm is therefore **{random_arm / attended:.1f}x** the "
             "intervention, not the same one", ""]
    if any("matched" in r for r in records):
        matched = np.mean([r["matched"] for r in records if "matched" in r])
        tries = np.median([r["matched_tries"] for r in records if "matched_tries" in r])
        lines += [f"- a token-matched control lands at **{matched:.1%}** "
                  f"(median {int(tries)} draws), which is the comparison the faithfulness "
                  "delta needs", ""]
    lines += ["| protein | tokens | attended | random | ratio |", "|---|---|---|---|---|"]
    for r in sorted(records, key=lambda r: r["ratio"], reverse=True):
        lines.append(f"| {r['target']} | {r['tokens']} | {r['attended']:.1%} | "
                     f"{r['random']:.1%} | {r['ratio']:.1f}x |")
    lines += ["", "**What this means for the audit.** A faithfulness delta computed "
              "against the scattered-random arm is not evidence about the explanation "
              "for a sub-word model: the arms differ in how much of the input they "
              "actually change. MolTrans's negative deltas are reported as this "
              "artefact, not as an anti-faithful explanation. Models that read one token "
              "per residue are unaffected, which "
              "`residue_level_models_are_unaffected` checks.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model", default="moltrans")
    parser.add_argument("--dataset", default="davis")
    parser.add_argument("--split", default="random")
    parser.add_argument("--n-proteins", type=int, default=20)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from src.evaluation.model_registry import model_class
    rows = (pd.read_csv(f"data/splits/{args.dataset}/{args.split}/test.csv")
            .drop_duplicates("Target_ID").head(args.n_proteins))
    adapter = model_class(args.model)(checkpoint_path=None, device=args.device)
    records = asymmetry(rows, adapter, args.k, args.seed)
    text = report(records, args.model, args.k)
    print(text)
    out = args.out or f"results/mask_comparability_{args.dataset}.md"
    with open(out, "w") as handle:
        handle.write(text)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
