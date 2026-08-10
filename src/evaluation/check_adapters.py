"""
Smoke-check every baseline adapter against the registry contract.

Run this before any baseline number is produced, and again after every
checkpoint is trained. It takes seconds and it is the only thing standing
between a wrong adapter and a wrong precision@k that looks reasonable.

    python -m src.evaluation.check_adapters
    python -m src.evaluation.check_adapters --checkpoints results/

What it actually checks
-----------------------
`validate_adapter` verifies the contract that matters: `predict()` returns a
finite scalar, and `explain()` returns a 1D non-negative array **of exactly the
right length**. That last one is the whole point. An adapter returning 979
convolution positions, or 545 ESPF tokens, instead of the protein's real residue
count does not crash — it silently shifts every ground-truth index and produces
a precision@k that is measuring the wrong thing.

Without `--checkpoints`, models are randomly initialised. That is fine and
deliberate: shapes, tokenisation, the attention hook and the residue projection
are all exercised by a random model exactly as they are by a trained one. What a
random model cannot tell you is whether the *numbers* are meaningful — for that,
`tests/test_integration.py` has a sanity floor asserting an untrained model
scores around chance, and if that ever fires the metric is measuring an artefact.
"""
import argparse
import os

# A real human kinase (ABL1 catalytic domain region) and a real inhibitor
# (imatinib). Real inputs, not random tokens: the ESPF tokeniser and the
# charset tables both behave differently on strings that are not valid.
DEMO_SEQUENCE = (
    "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNH"
    "NGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSES"
    "RFNTLAELVHHHSTVADGLITTLHYPAPKRNKPTVYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLK"
    "EAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVA"
    "DFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRAC"
    "WQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGK"
)
DEMO_SMILES = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"

MAX_PROTEIN_LEN = 1000


def _expected_length(sequence: str, cap: int = MAX_PROTEIN_LEN) -> int:
    """What explain() must return: residues the model actually saw."""
    return min(len(sequence), cap)


def check_deepdta(checkpoint: str = None) -> dict:
    from src.evaluation.baseline_adapters import DeepDTAAdapter
    from src.model.deepdta_torch import encode_protein, encode_smiles

    import torch

    adapter = DeepDTAAdapter(checkpoint_path=checkpoint)
    drug = torch.from_numpy(encode_smiles(DEMO_SMILES))
    protein = torch.from_numpy(encode_protein(DEMO_SEQUENCE))

    value = adapter.predict(drug, protein)
    result = {"adapter": "deepdta", "valid": True, "problems": [],
              "n_weights": 0, "prediction": value}
    if not isinstance(value, float) or value != value:
        result["valid"] = False
        result["problems"].append("predict() did not return a finite float")

    # explain() MUST raise -- DeepDTA has no attention, and an adapter that
    # quietly returned something here would put a fabricated explanation into
    # the audit under DeepDTA's name.
    try:
        adapter.explain(drug, protein)
        result["valid"] = False
        result["problems"].append(
            "explain() returned instead of raising -- DeepDTA has no attention")
    except NotImplementedError:
        pass
    return result


def check_hyperattentiondti(checkpoint: str = None) -> dict:
    from src.evaluation.baseline_adapters import HyperAttentionDTIAdapter
    from src.evaluation.model_registry import validate_adapter

    adapter = HyperAttentionDTIAdapter(checkpoint_path=checkpoint)
    drug, protein = HyperAttentionDTIAdapter.encode(DEMO_SMILES, DEMO_SEQUENCE)
    return validate_adapter(adapter, drug, protein,
                            expected_length=_expected_length(DEMO_SEQUENCE))


def check_moltrans(checkpoint: str = None) -> dict:
    """MolTrans needs its own path: explain() takes the ESPF tokens too."""
    import numpy as np

    from src.evaluation.attention_projection import moltrans_covered_residues
    from src.evaluation.baseline_adapters import MolTransAdapter

    adapter = MolTransAdapter(checkpoint_path=checkpoint)
    drug, drug_mask, protein, protein_mask, tokens = MolTransAdapter.encode(
        DEMO_SMILES, DEMO_SEQUENCE)

    result = {"adapter": "moltrans", "valid": True, "problems": [],
              "n_weights": 0}
    try:
        value = adapter.predict(drug, protein, drug_mask, protein_mask)
        result["prediction"] = value
        if not np.isfinite(value):
            result["problems"].append("predict() returned NaN or inf")
    except Exception as exc:
        result["problems"].append(f"predict() raised {type(exc).__name__}: {exc}")

    try:
        weights = np.asarray(adapter.explain(drug, protein, protein_tokens=tokens,
                                             drug_mask=drug_mask,
                                             protein_mask=protein_mask))
        result["n_weights"] = int(weights.size)
        expected = moltrans_covered_residues(tokens)
        if weights.ndim != 1:
            result["problems"].append(f"explain() returned shape {weights.shape}")
        if weights.size != expected:
            result["problems"].append(
                f"explain() returned {weights.size} weights, expected "
                f"{expected} covered residues")
        if weights.size and weights.min() < 0:
            result["problems"].append("explain() returned negative weights")
        if not np.all(np.isfinite(weights)):
            result["problems"].append("explain() returned NaN or inf")
    except Exception as exc:
        result["problems"].append(f"explain() raised {type(exc).__name__}: {exc}")

    result["valid"] = not result["problems"]
    return result


CHECKS = {
    "deepdta": check_deepdta,
    "hyperattentiondti": check_hyperattentiondti,
    "moltrans": check_moltrans,
}


def main():
    parser = argparse.ArgumentParser(description="Validate the baseline adapters")
    parser.add_argument("--checkpoints", help="directory of trained checkpoints; "
                                              "without it, models are randomly "
                                              "initialised (shapes still checked)")
    parser.add_argument("--only", choices=sorted(CHECKS), help="check one model")
    args = parser.parse_args()

    names = [args.only] if args.only else sorted(CHECKS)
    print(f"expected explain() length for the demo protein: "
          f"{_expected_length(DEMO_SEQUENCE)} residues "
          f"(sequence is {len(DEMO_SEQUENCE)})\n")

    failures = 0
    for name in names:
        checkpoint = None
        if args.checkpoints:
            for candidate in (f"{name}.pt", f"{name}_best.pt", f"{name}.pth"):
                path = os.path.join(args.checkpoints, candidate)
                if os.path.exists(path):
                    checkpoint = path
                    break

        try:
            result = CHECKS[name](checkpoint)
        except Exception as exc:
            result = {"adapter": name, "valid": False, "n_weights": 0,
                      "problems": [f"{type(exc).__name__}: {exc}"]}

        mark = "PASS" if result["valid"] else "FAIL"
        source = os.path.basename(checkpoint) if checkpoint else "random init"
        print(f"[{mark}] {name:20s} weights={result['n_weights']:>5}  ({source})")
        for problem in result.get("problems", []):
            print(f"         {problem}")
        if not result["valid"]:
            failures += 1

    print()
    if failures:
        print(f"{failures} adapter(s) failed. Do not produce baseline numbers "
              f"until every one passes -- a wrong-length explain() does not "
              f"crash, it misaligns every ground-truth index.")
        raise SystemExit(1)
    print("All adapters honour the contract.")
    print("Shapes and projections are verified; whether the numbers are "
          "meaningful is a separate question that needs trained checkpoints.")


if __name__ == "__main__":
    main()
