"""
Positive control -- can the audit's measurements recognise a good explanation?

Every audited model so far scores close to chance on plausibility. That result
is only worth reporting if the same pipeline, given an explanation that really
does point at the binding sites, says so. Otherwise a reviewer can fairly ask
whether the near-chance numbers describe the models or an insensitive test.
This module answers that on the real data: the real test rows, the real
ground-truth file and the real 1,000-residue window, scored by the same
functions the ladder and the faithfulness runner use.

Explanations of known quality
-----------------------------
`dosed_explanation(length, sites, dose)` builds an explanation whose quality
is set by one number. Every residue gets a random weight in [0, 1); each
annotated site is then, with probability `dose`, lifted above every
non-site. So:

    dose 1.0   the oracle: every site ranked first. precision@k must equal
               the achievable ceiling for every protein.
    dose 0.0   pure noise. precision@k should sit at chance.
    between    a known fraction of sites is visible to the explanation.

Two measurements, each at every dose
------------------------------------
Plausibility  precision@k, its ceiling, chance and the split-level
              permutation p-value -- `run_ladder.evaluate_level`, unchanged.

Faithfulness  `batch_faithfulness`, unchanged, on a planted model: its
              prediction is a weighted sum over the residues at the
              protein's annotated sites, and nothing else. By construction
              the sites are what it uses, so an explanation pointing at them
              must be load-bearing and a random one must not.

What counts as a pass
---------------------
Hard checks (the run exits non-zero if one fails):

  * no annotated site lies outside the sequence the model is shown -- the
    ground-truth / sequence / truncation mismatch the DAVIS re-numbering
    fixed. Counted directly (`sites_outside_sequence`): the oracle's score
    alone cannot detect it on a protein with more than k sites;
  * the oracle's normalised precision@k is exactly 1.0 at every level -- the
    scoring code agrees with its own ceiling;
  * the oracle is significant (p < 0.05) at every level;
  * on the planted model, the oracle explanation is load-bearing
    (comprehensiveness delta > 0) at every level.

Reported, not enforced: dose 0 against chance, the planted model's dose-0
delta (should be ~0), and the **smallest dose the permutation test detects**
at each level's real protein count. That last number is the test's
resolution. Beside an audited model's precision@k it says what fraction of
sites that model's attention is effectively finding, and whether "at chance"
could have been told apart from "slightly better than chance".

One row per protein
-------------------
An explanation built from a protein's sites is the same for every drug that
protein was measured against, so extra pairs would only repeat it. The unit
here is the protein (`pairs_per_target=1`), and it is the same set of proteins
the ladder and the audit table score: all three keep each protein's first test
pair in file order. A ladder run with `--pairs-per-target 0` (every pair, as
ladders before 2026-09-12 did) weights proteins by their pair counts, and an
equivalent dose read against it is only approximate.

Usage
-----
    python -m src.evaluation.positive_control --dataset davis
    python -m src.evaluation.positive_control --dataset kiba
    python -m src.evaluation.positive_control --dataset davis \\
        --compare coldsite_dti=results/ladder_davis_seed1.json \\
        --compare hyperattentiondti=results/ladder_hyperattentiondti_davis_seed1.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

LEVELS = ("random", "cold_drug", "cold_target", "cold_pair")
DOSES = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
ALPHA = 0.05


# --------------------------------------------------------------------------
# explanations of known quality
# --------------------------------------------------------------------------

def dosed_explanation(length: int, sites, dose: float, rng) -> np.ndarray:
    """Random weights, with each site lifted above every non-site w.p. `dose`."""
    if not 0.0 <= dose <= 1.0:
        raise ValueError(f"dose must be in [0, 1], got {dose}")
    weights = rng.random(length)
    for site in sites:
        if 0 <= site < length and rng.random() < dose:
            weights[site] = 1.0 + rng.random()
    return weights


def collect_rows(dataset: str, level: str, site_sets: dict,
                 split_root: str = "data/splits", max_len: int = 1000,
                 pairs_per_target: int = 1) -> list:
    """[(target_id, sequence, positions)] for the level's test rows with usable sites.

    The same join `collect_cell` makes: Target_ID from the split, sites from
    the ground-truth file, the sequence the split holds.
    """
    from src.evaluation.collect import _read_test_rows

    split_dir = os.path.join(split_root, dataset, level)
    rows = []
    for target_id, _smiles, sequence in _read_test_rows(split_dir, pairs_per_target):
        site_set = site_sets.get(target_id)
        if site_set is None or not site_set.usable:
            continue
        rows.append((target_id, sequence, set(site_set.positions)))
    return rows


def _length(sequence: str, max_len: int) -> int:
    """Residues the model sees: the sequence, cut at the window."""
    return min(len(sequence), max_len)


def sites_outside_sequence(rows: list, max_len: int = 1000) -> int:
    """Annotated sites (inside the window) at positions the sequence does not have.

    Counted directly, because the oracle's score cannot see them: a protein
    with more than k sites fills its top k with real ones whatever else is
    annotated, so a misnumbered site only lowers the oracle when a protein has
    fewer than k. Every such site silently lowers the achievable precision of
    every model scored against it.
    """
    return sum(1 for _t, sequence, sites in rows for s in sites
               if s >= _length(sequence, max_len))


# --------------------------------------------------------------------------
# plausibility
# --------------------------------------------------------------------------

def plausibility_curve(rows: list, doses=DOSES, k: int = 10, n_trials: int = 1000,
                       max_len: int = 1000, seed: int = 0) -> dict:
    """{dose: the ladder's entry at k} for explanations of each dose."""
    from src.evaluation.run_ladder import evaluate_level

    curve = {}
    for dose in doses:
        rng = np.random.default_rng([seed, int(round(dose * 1e6))])
        weights = [dosed_explanation(_length(seq, max_len), sites, dose, rng)
                   for _t, seq, sites in rows]
        sites = [s for _t, _seq, s in rows]
        entry = evaluate_level(weights, sites, k_values=(k,), n_trials=n_trials,
                               seed=seed)["by_k"][k]
        curve[dose] = entry
    return curve


def minimum_detectable_dose(curve: dict) -> float | None:
    """Smallest dose from which every larger dose is also significant.

    Not simply the smallest significant dose. Each dose is one random draw,
    and below ~2% of sites the draw dominates: on KIBA cold-target a 0.5%
    dose came out significant and a 1% dose did not. Quoting the lucky 0.5%
    as the test's resolution would overstate it.
    """
    detected = None
    for dose in sorted((d for d in curve if d > 0), reverse=True):
        p = curve[dose].get("p_value")
        if p is None or p >= ALPHA:
            break
        detected = dose
    return detected


def equivalent_dose(curve: dict, observed_precision: float) -> float | None:
    """The dose whose expected precision@k matches an observed one.

    Interpolated along the measured curve. None if the curve is not monotone
    (too few proteins to trust it) or the value is outside its range.
    """
    return equivalent_dose_with_reason(curve, observed_precision)[0]


def equivalent_dose_with_reason(curve: dict, observed_precision: float) -> tuple:
    """(dose or None, why). A missing dose has three different meanings, and a
    report that calls all of them "at chance" misreads two of them."""
    doses = sorted(curve)
    values = [curve[d]["precision_at_k"] for d in doses]
    if any(b < a - 1e-9 for a, b in zip(values, values[1:])):
        return None, "curve not monotone (too few proteins to read)"
    if observed_precision < values[0]:
        return None, "at or below chance"
    if observed_precision > values[-1]:
        return None, "above the oracle"
    return float(np.interp(observed_precision, values, doses)), "ok"


# --------------------------------------------------------------------------
# faithfulness, on a model whose dependence is known
# --------------------------------------------------------------------------

class PlantedModel:
    """Prediction = weighted sum of the residues at a protein's annotated sites.

    Works on residue-code tensors (`residue_space`: 0 pad, 1 = X). Each amino
    acid gets a fixed weight in [0.5, 1.5); X and pad weigh 0, so masking a
    site always moves the prediction and masking any other residue never
    does. The drug tensor carries the pair's index, as in ResidueSpaceModel.
    """

    def __init__(self, seed: int = 0):
        from src.evaluation.residue_space import MASK_CODE, PAD_CODE

        rng = np.random.default_rng(seed)
        self.weight = rng.uniform(0.5, 1.5, size=32)
        self.weight[[PAD_CODE, MASK_CODE]] = 0.0
        self.sites: list[set] = []

    def add_pair(self, sequence: str, sites: set, max_len: int):
        import torch

        from src.evaluation.residue_space import encode_residues

        length = _length(sequence, max_len)
        index = len(self.sites)
        self.sites.append({s for s in sites if s < length})
        return (torch.tensor([[index]], dtype=torch.long),
                encode_residues(sequence[:length]))

    def predict(self, drug, protein) -> float:
        import torch

        index = int(torch.as_tensor(drug).reshape(-1)[0])
        codes = torch.as_tensor(protein).reshape(-1)
        return float(sum(self.weight[int(codes[s])] for s in self.sites[index]
                         if s < codes.numel()))


def faithfulness_curve(rows: list, doses=DOSES, k: int = 10,
                       n_random_trials: int = 5, max_len: int = 1000,
                       seed: int = 0) -> dict:
    """{dose: batch_faithfulness summary} on the planted model."""
    from src.evaluation.faithfulness import batch_faithfulness

    model = PlantedModel(seed=seed)
    drugs, proteins, lengths = [], [], []
    for _target, sequence, sites in rows:
        drug, protein = model.add_pair(sequence, sites, max_len)
        drugs.append(drug)
        proteins.append(protein)
        lengths.append(protein.shape[1])

    curve = {}
    for dose in doses:
        rng = np.random.default_rng([seed, 7, int(round(dose * 1e6))])
        attentions = [dosed_explanation(n, model.sites[i], dose, rng)
                      for i, n in enumerate(lengths)]
        curve[dose] = batch_faithfulness(model, drugs, proteins, attentions, k=k,
                                         n_random_trials=n_random_trials, seed=seed)
    return curve


# --------------------------------------------------------------------------
# verdicts and report
# --------------------------------------------------------------------------

def verdicts(results: dict) -> list:
    """[(passed, is_hard_check, message)] for the whole run."""
    out = []
    for level, entry in results["levels"].items():
        if "sites_outside_sequence" in entry:
            outside = entry["sites_outside_sequence"]
            out.append((outside == 0, True,
                        f"{level}: {outside} annotated site(s) lie outside the "
                        f"sequence the model sees (must be 0)"))
        oracle = entry["plausibility"].get(1.0)
        if oracle is None:
            continue
        normalised = oracle["normalised"]
        out.append((normalised >= 0.999, True,
                    f"{level}: oracle normalised precision@k = {normalised:.4f} "
                    f"(must be 1.0)"))
        out.append((oracle["p_value"] is not None and oracle["p_value"] < ALPHA, True,
                    f"{level}: oracle p = {oracle['p_value']:.4g} (must be < {ALPHA})"))
        planted = entry["faithfulness"].get(1.0)
        if planted is not None:
            out.append((bool(planted["explanation_is_load_bearing"]), True,
                        f"{level}: planted-model oracle comprehensiveness delta = "
                        f"{planted['comprehensiveness_delta']:+.4f} (must be > 0)"))
    return out


def _fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def report(results: dict) -> str:
    k = results["k"]
    lines = [f"# Positive control — {results['dataset']}", "",
             f"Explanations of known quality, scored by the audit's own functions "
             f"(k = {k}, {results['n_trials']} permutation trials, one test row "
             f"per protein, {results['max_len']}-residue window). See "
             f"`src/evaluation/positive_control.py`.", ""]

    lines += ["## Plausibility: precision@k by dose", "",
              "| level | n | " + " | ".join(f"dose {d:g}" for d in results["doses"])
              + " | ceiling | chance | smallest detectable dose |",
              "|---|---|" + "---|" * len(results["doses"]) + "---|---|---|"]
    for level, entry in results["levels"].items():
        curve = entry["plausibility"]
        cells = []
        for d in results["doses"]:
            mark = "*" if curve[d]["p_value"] is not None and curve[d]["p_value"] < ALPHA else ""
            cells.append(f"{_fmt(curve[d]['precision_at_k'])}{mark}")
        any_dose = curve[results["doses"][-1]]
        detectable = entry["minimum_detectable_dose"]
        lines.append(f"| {level} | {entry['n_proteins']} | " + " | ".join(cells)
                     + f" | {_fmt(any_dose['ceiling'])} | {_fmt(any_dose['chance'])} | "
                     + (f"{detectable:g}" if detectable is not None else "none") + " |")
    lines += ["", "`*` = significantly above chance (split-level permutation test, "
              f"p < {ALPHA}).", ""]

    lines += ["## Faithfulness on a planted model: comprehensiveness delta by dose", "",
              "| level | " + " | ".join(f"dose {d:g}" for d in results["doses"]) + " |",
              "|---|" + "---|" * len(results["doses"])]
    for level, entry in results["levels"].items():
        curve = entry["faithfulness"]
        lines.append(f"| {level} | " + " | ".join(
            _fmt(curve[d]["comprehensiveness_delta"], 4) for d in results["doses"]) + " |")
    lines += ["", "The planted model's prediction depends only on the annotated "
              "sites, so the oracle (dose 1) must be load-bearing and dose 0 "
              "should sit near zero.", ""]

    if results.get("compare"):
        lines += ["## Audited models, read against the curve", "",
                  "| model | level | precision@k | equivalent dose |", "|---|---|---|---|"]
        for model, per_level in results["compare"].items():
            for level, item in per_level.items():
                eq = item["equivalent_dose"]
                lines.append(f"| {model} | {level} | {_fmt(item['precision_at_k'])} | "
                             + (f"{eq:.3f}" if eq is not None
                                else item.get("reason", "outside the curve"))
                             + " |")
        lines += ["", "Equivalent dose: the fraction of sites a dosed explanation "
                  "must rank first to match the model's precision@k. Like for like "
                  "when the ladder scored one pair per protein (its default); a ladder "
                  "run over every pair is only approximately comparable.", ""]

    lines += ["## Checks", ""]
    for passed, hard, message in results["verdicts"]:
        tag = "PASS" if passed else ("FAIL" if hard else "note")
        lines.append(f"- **{tag}** {message}")
    return "\n".join(lines) + "\n"


def _read_ladder(path: str, k: int) -> dict:
    """{level: precision@k} from a run_ladder JSON (k keys are strings on disk)."""
    data = json.load(open(path))
    out = {}
    for level, entry in data.items():
        by_k = entry.get("by_k", {})
        at_k = by_k.get(str(k), by_k.get(k))
        if at_k is not None:
            out[level] = at_k["precision_at_k"]
    return out


def run(dataset: str, ground_truth: str, split_root: str = "data/splits",
        levels=LEVELS, doses=DOSES, k: int = 10, n_trials: int = 1000,
        max_len: int = 1000, seed: int = 0, compare: dict | None = None,
        verbose: bool = True) -> dict:
    from src.data.ground_truth import load_site_sets

    site_sets = load_site_sets(ground_truth, max_len=max_len)
    results = {"dataset": dataset, "ground_truth": ground_truth, "k": k,
               "n_trials": n_trials, "max_len": max_len, "seed": seed,
               "doses": list(doses), "levels": {}}

    for level in levels:
        if not os.path.isdir(os.path.join(split_root, dataset, level)):
            if verbose:
                print(f"[skip] {level}: no split at {split_root}/{dataset}/{level}")
            continue
        rows = collect_rows(dataset, level, site_sets, split_root, max_len)
        if verbose:
            print(f"{level}: {len(rows)} proteins with usable sites", flush=True)
        plausibility = plausibility_curve(rows, doses, k, n_trials, max_len, seed)
        results["levels"][level] = {
            "n_proteins": len(rows),
            "sites_outside_sequence": sites_outside_sequence(rows, max_len),
            "plausibility": plausibility,
            "minimum_detectable_dose": minimum_detectable_dose(plausibility),
            "faithfulness": faithfulness_curve(rows, doses, k, max_len=max_len,
                                               seed=seed),
        }

    if compare:
        results["compare"] = {}
        for model, path in compare.items():
            observed = _read_ladder(path, k)
            results["compare"][model] = {}
            for level, value in observed.items():
                if level not in results["levels"]:
                    continue
                dose, why = equivalent_dose_with_reason(
                    results["levels"][level]["plausibility"], value)
                results["compare"][model][level] = {
                    "precision_at_k": value, "equivalent_dose": dose, "reason": why}

    results["verdicts"] = verdicts(results)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Positive control: does the audit recognise a good explanation?")
    parser.add_argument("--dataset", default="davis", choices=["davis", "kiba"])
    parser.add_argument("--ground-truth",
                        help="default: data/<dataset>_ground_truth_sites.json")
    parser.add_argument("--split-root", default="data/splits")
    parser.add_argument("--levels", default=",".join(LEVELS))
    parser.add_argument("--doses", default=",".join(f"{d:g}" for d in DOSES))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--max-protein-len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare", action="append", default=[],
                        metavar="MODEL=LADDER_JSON",
                        help="read a run_ladder output against the curve; repeatable")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    doses = sorted({float(d) for d in args.doses.split(",") if d.strip()})
    if 0.0 not in doses or 1.0 not in doses:
        parser.error("--doses must include 0 (noise) and 1 (the oracle)")
    compare = dict(item.split("=", 1) for item in args.compare)

    results = run(args.dataset,
                  args.ground_truth or f"data/{args.dataset}_ground_truth_sites.json",
                  args.split_root,
                  [lv.strip() for lv in args.levels.split(",") if lv.strip()],
                  doses, args.k, args.n_trials, args.max_protein_len, args.seed,
                  compare)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"positive_control_{args.dataset}")
    with open(f"{stem}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    text = report(results)
    with open(f"{stem}.md", "w") as f:
        f.write(text)
    print("\n" + text)
    print(f"Saved -> {stem}.json\nSaved -> {stem}.md")

    if any(hard and not passed for passed, hard, _ in results["verdicts"]):
        raise SystemExit("A hard check failed -- see Checks above. Do not report "
                         "audit numbers until it passes.")


if __name__ == "__main__":
    main()
