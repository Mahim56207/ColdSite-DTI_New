"""
T05 -- effect sizes and confidence intervals for the committed ladders.

    python -m src.evaluation.effects_v2 --out-dir results/effects_v2

What it adds to the audit's verdict counts (`docs/PROTOCOL_AMENDMENT_v2.md` section 5):

* **enrichment over chance** -- mean precision@10 divided by mean chance, both taken over
  the SAME resampled proteins, with a 95% percentile interval (10,000 resamples, unit =
  target, each protein carried in with all its seeds averaged, `default_rng(0)`: the
  resampling `bootstrap_ci.bootstrap_mean` already does, so the precision interval here
  equals the committed `results/ci_davis.json` interval to the last digit -- checked);
* **seed spread against distance from chance** for every cell, on the same rule as
  `seed_agreement.py` (spread exceeds signal iff max - min of the seeds' precision is larger
  than |mean - chance|).

Chance per protein is the exact expectation of k uniformly random positions: the number of
annotated sites inside the scored window divided by the window length
(`significance_test._chance_precision_sample` draws exactly that). The ladder stores only
the cell mean of the Monte-Carlo estimate, so the per-protein values are rebuilt from the
protein's sequence and site set and CHECKED against the recorded cell chance: a cell whose
rebuilt mean is farther from the recorded one than five Monte-Carlo standard errors is
reported, never quietly kept.

Nothing here scores a model: it reads ladder files that already exist and writes to a new
directory.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import warnings

import numpy as np

from src.evaluation.bootstrap_ci import CONFIDENCE, N_RESAMPLES, percentile_ci

LEVELS = ("random", "cold_drug", "cold_target", "cold_pair")
K = "10"
WINDOW = 1000
# the smallest permutation count any committed ladder can have been run with (audit default
# before T02); a larger count only shrinks the Monte-Carlo error, so this is conservative
MIN_RECORDED_TRIALS = 500
MC_SIGMAS = 5.0
TAG = re.compile(r"^ladder_(?:(?P<model>.+)_)?(?P<dataset>davis|kiba)_seed(?P<seed>\d+)\.json$")


# --------------------------------------------------------------------------
# chance, per protein
# --------------------------------------------------------------------------

def per_protein_chance(n_sites_in_window: int, window_length: int) -> float:
    """Expected precision@k of k random positions: sites in the window / window length."""
    if window_length <= 0:
        return float("nan")
    return n_sites_in_window / window_length


def chance_variance(n_sites: int, window_length: int, k: int) -> float:
    """Variance of one protein's random precision@k (hypergeometric, k of N without replacement)."""
    if window_length <= 1:
        return 0.0
    p = n_sites / window_length
    return p * (1.0 - p) / k * (window_length - k) / (window_length - 1)


def mc_standard_error(variances, n_trials: int = MIN_RECORDED_TRIALS) -> float:
    """SE of a recorded null mean: the null mean over n proteins has variance sum(var)/n^2."""
    variances = np.asarray(variances, dtype=float)
    if variances.size == 0:
        return float("nan")
    return float(np.sqrt(variances.sum()) / variances.size / np.sqrt(n_trials))


def load_lengths(dataset: str, level: str, split_root: str = "data/splits") -> dict:
    """{target id: window length} -- min(sequence length, 1000) -- from the level's test set."""
    import pandas as pd

    frame = pd.read_csv(os.path.join(split_root, dataset, level, "test.csv"),
                        usecols=["Target_ID", "Target"])
    out = {}
    for target_id, sequence in zip(frame["Target_ID"].astype(str), frame["Target"].astype(str)):
        out.setdefault(target_id, min(len(sequence), WINDOW))
    return out


def protein_chances(ids, site_sets: dict, lengths: dict) -> dict:
    """{id: (chance, variance, n_sites_in_window, window)} for the proteins of one cell."""
    out = {}
    for target_id in ids:
        window = lengths[target_id]
        positions = {p for p in site_sets[target_id].positions if 0 <= p < window}
        out[target_id] = (per_protein_chance(len(positions), window),
                          chance_variance(len(positions), window, int(K)),
                          len(positions), window)
    return out


# --------------------------------------------------------------------------
# the bootstrap
# --------------------------------------------------------------------------

def bootstrap_effect(precision: dict, chance: dict, n_resamples: int = N_RESAMPLES,
                     seed: int = 0, confidence: float = CONFIDENCE) -> dict:
    """Point values and percentile intervals for one cell.

    `precision` is {protein: [precision per seed]}, `chance` {protein: expected precision}.
    The draws are generated exactly as `bootstrap_ci.bootstrap_mean` generates them, so the
    precision interval is that function's. Enrichment is mean precision / mean chance over
    the same draws; a resample whose chance mean is 0 gives no ratio (nan, dropped from the
    percentile), and a cell whose overall chance mean is 0 is given none at all.
    """
    ids = sorted(precision)
    n = len(ids)
    if n == 0:
        return {"n": 0}
    prec = np.array([float(np.mean(precision[i])) for i in ids])
    chan = np.array([float(chance[i]) for i in ids])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_resamples, n))
    prec_means = prec[draws].mean(axis=1)
    chan_means = chan[draws].mean(axis=1)
    p_low, p_high = percentile_ci(prec_means, confidence)
    row = {"n": n, "precision": float(prec.mean()), "precision_low": p_low,
           "precision_high": p_high, "chance": float(chan.mean()),
           "n_resamples": n_resamples}
    diff = prec_means - chan_means
    d_low, d_high = percentile_ci(diff, confidence)
    row.update({"excess": float(prec.mean() - chan.mean()),
                "excess_low": d_low, "excess_high": d_high})
    if chan.mean() > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(chan_means > 0, prec_means / chan_means, np.nan)
        ratios = ratios[np.isfinite(ratios)]
        e_low, e_high = percentile_ci(ratios, confidence)
        row.update({"enrichment": float(prec.mean() / chan.mean()),
                    "enrichment_low": e_low, "enrichment_high": e_high,
                    "n_resamples_with_ratio": int(ratios.size)})
    else:
        row.update({"enrichment": float("nan"), "enrichment_low": float("nan"),
                    "enrichment_high": float("nan"), "n_resamples_with_ratio": 0})
    return row


RESAMPLING = ("targets", "seeds_and_targets")


def _seed_matrix(by_seed: dict) -> tuple:
    """(ids, seeds, matrix) with matrix[i, j] = protein i's value under seed j, nan if absent."""
    ids = sorted(by_seed)
    seeds = sorted({s for values in by_seed.values() for s in values})
    matrix = np.full((len(ids), len(seeds)), np.nan)
    for i, protein in enumerate(ids):
        for j, s in enumerate(seeds):
            if s in by_seed[protein]:
                matrix[i, j] = by_seed[protein][s]
    return ids, seeds, matrix


def _two_way_means(matrix, target_draws, seed_draws) -> np.ndarray:
    """Per-resample mean over the drawn targets of each target's mean over the drawn seeds.

    `target_draws` is (B, n), `seed_draws` (B, S). A seed a protein lacks is skipped (nanmean),
    as the target-only bootstrap averages each protein over the seeds it has.
    """
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)              # all-nan slices -> nan
        per_target = np.nanmean(matrix[:, seed_draws], axis=2).T          # (B, n)
        picked = np.take_along_axis(per_target, target_draws, axis=1)      # (B, n)
        return np.nanmean(picked, axis=1)


def bootstrap_effect_2d(by_seed: dict, chance: dict, n_resamples: int = N_RESAMPLES,
                        seed: int = 0, confidence: float = CONFIDENCE) -> dict:
    """`bootstrap_effect` with seeds resampled as well as targets (a two-way bootstrap).

    `by_seed` is {protein: {seed: precision}}. Each resample draws n targets with replacement
    AND S seeds with replacement from the cell's S seeds, independently, and averages over the
    drawn (target, seed) grid; chance, which does not depend on the seed, is averaged over the
    same drawn targets. The target draws are generated first and exactly as in
    `bootstrap_effect`, so a cell whose seeds all agree gets the same interval from both.
    The point values are unchanged: mean over proteins of each protein's mean over its seeds.
    """
    ids, seeds, matrix = _seed_matrix(by_seed)
    n = len(ids)
    if n == 0:
        return {"n": 0}
    prec = np.nanmean(matrix, axis=1)
    chan = np.array([float(chance[i]) for i in ids])
    rng = np.random.default_rng(seed)
    target_draws = rng.integers(0, n, size=(n_resamples, n))
    seed_draws = rng.integers(0, len(seeds), size=(n_resamples, len(seeds)))
    prec_means = _two_way_means(matrix, target_draws, seed_draws)
    chan_means = chan[target_draws].mean(axis=1)
    keep = np.isfinite(prec_means)
    p_low, p_high = percentile_ci(prec_means[keep], confidence)
    row = {"n": n, "precision": float(prec.mean()), "precision_low": p_low,
           "precision_high": p_high, "chance": float(chan.mean()),
           "n_resamples": n_resamples, "resampling": "seeds_and_targets",
           "n_seeds_resampled": len(seeds)}
    d_low, d_high = percentile_ci((prec_means - chan_means)[keep], confidence)
    row.update({"excess": float(prec.mean() - chan.mean()),
                "excess_low": d_low, "excess_high": d_high})
    if chan.mean() > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(chan_means > 0, prec_means / chan_means, np.nan)
        ratios = ratios[np.isfinite(ratios)]
        e_low, e_high = percentile_ci(ratios, confidence)
        row.update({"enrichment": float(prec.mean() / chan.mean()),
                    "enrichment_low": e_low, "enrichment_high": e_high,
                    "n_resamples_with_ratio": int(ratios.size)})
    else:
        row.update({"enrichment": float("nan"), "enrichment_low": float("nan"),
                    "enrichment_high": float("nan"), "n_resamples_with_ratio": 0})
    return row


def seed_spread(precisions: list, chance: float) -> dict:
    """Rule 1.13 for one cell: does the spread across seeds exceed the distance from chance?"""
    mean = float(np.mean(precisions))
    spread = float(max(precisions) - min(precisions))
    distance = abs(mean - chance)
    return {"n_seeds": len(precisions), "mean": mean, "spread": spread,
            "sd": float(np.std(precisions, ddof=1)) if len(precisions) > 1 else float("nan"),
            "distance_from_chance": distance,
            "spread_exceeds_signal": bool(spread > distance)}


# --------------------------------------------------------------------------
# reading the ladders
# --------------------------------------------------------------------------

def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def discover(folder: str) -> dict:
    """{(model, dataset): {seed: path}} for the ladder JSONs in one folder."""
    found: dict = {}
    for path in sorted(glob.glob(os.path.join(folder, "ladder_*.json"))):
        match = TAG.match(os.path.basename(path))
        if not match:
            continue
        model = match.group("model") or "coldsite_dti"
        found.setdefault((model, match.group("dataset")), {})[int(match.group("seed"))] = path
    return found


def cell_rows(folder: str, ground_truth: str, family: str, ground_truth_label: str,
              split_root: str = "data/splits", n_resamples: int = N_RESAMPLES,
              resampling: str = "targets") -> tuple:
    """(enrichment rows, seed rows, input hashes, problems) for every ladder in `folder`."""
    from src.data.ground_truth import load_site_sets

    site_sets = load_site_sets(ground_truth, max_len=WINDOW)
    enrichment, spreads, hashes, problems = [], [], {}, []
    lengths_cache: dict = {}
    for (model, dataset), seeds in sorted(discover(folder).items()):
        payloads = {}
        for seed, path in sorted(seeds.items()):
            payloads[seed] = json.load(open(path))
            hashes[path] = sha256(path)
        for level in LEVELS:
            entries = {s: p[level] for s, p in payloads.items() if p.get(level)}
            if not entries:
                continue
            precision: dict = {}
            by_seed: dict = {}
            recorded, ceilings, per_seed = [], [], []
            ids_seen: set = set()
            for seed, entry in sorted(entries.items()):
                cell = entry.get("by_k", {}).get(K)
                ids = entry.get("ids")
                scores = (cell or {}).get("per_protein")
                if cell is None or ids is None or scores is None or len(ids) != len(scores):
                    problems.append(f"{family} {model} {dataset} {level} seed {seed}: "
                                    "no per-protein scores with ids")
                    continue
                for protein, score in zip(ids, scores):
                    precision.setdefault(protein, []).append(float(score))
                    by_seed.setdefault(protein, {})[seed] = float(score)
                ids_seen |= set(ids)
                recorded.append(cell["chance"])
                ceilings.append(cell["ceiling"])
                per_seed.append(cell["precision_at_k"])
            if not precision:
                continue
            key = (dataset, level)
            if key not in lengths_cache:
                lengths_cache[key] = load_lengths(dataset, level, split_root)
            missing = sorted(i for i in ids_seen if i not in site_sets or i not in lengths_cache[key])
            if missing:
                problems.append(f"{family} {model} {dataset} {level}: {len(missing)} ids without "
                                f"sites or a sequence, e.g. {missing[:3]}")
                continue
            chances = protein_chances(sorted(ids_seen), site_sets, lengths_cache[key])
            chance = {i: v[0] for i, v in chances.items()}
            se = mc_standard_error([v[1] for v in chances.values()])
            exact = float(np.mean(list(chance.values())))
            check = max(abs(exact - r) for r in recorded)
            row = {"family": family, "ground_truth": ground_truth_label, "dataset": dataset,
                   "model": model, "level": level, "seeds": len(entries),
                   "ceiling": float(np.mean(ceilings)),
                   "recorded_chance_mean": float(np.mean(recorded)),
                   "chance_check_max_abs_diff": float(check),
                   "chance_check_mc_se": se,
                   "chance_check_ok": bool(check <= MC_SIGMAS * se)}
            row.update(bootstrap_effect(precision, chance, n_resamples) if resampling == "targets"
                       else bootstrap_effect_2d(by_seed, chance, n_resamples))
            enrichment.append(row)
            # rule 1.13 reads the first seed's recorded chance (`seed_agreement.cells`)
            spread = seed_spread(per_seed, recorded[0])
            spread_exact = seed_spread(per_seed, exact)
            spreads.append({"family": family, "ground_truth": ground_truth_label,
                            "dataset": dataset, "model": model, "level": level,
                            "per_seed_precision": ";".join(f"{v:.6f}" for v in per_seed),
                            "chance_recorded": recorded[0],
                            "chance_exact": exact,
                            **{k: v for k, v in spread.items()},
                            "spread_exceeds_signal_exact_chance": spread_exact["spread_exceeds_signal"],
                            "enrichment": row.get("enrichment"),
                            "enrichment_low": row.get("enrichment_low"),
                            "enrichment_high": row.get("enrichment_high")})
    return enrichment, spreads, hashes, problems


# --------------------------------------------------------------------------
# faithfulness delta, resampled by target
# --------------------------------------------------------------------------

FAITH = re.compile(r"^(?P<kind>token_faithfulness|faithfulness)_(?:(?P<model>.+)_)?"
                   r"(?P<dataset>davis|kiba)_seed(?P<seed>\d+)\.json$")


def bootstrap_delta(per_target: dict, n_resamples: int = N_RESAMPLES, seed: int = 0,
                    confidence: float = CONFIDENCE) -> dict:
    """Mean faithfulness delta over targets, with a percentile interval and the CI verdict.

    `per_target` is {target: [delta per seed]} (each seed's value already the mean of that
    target's pairs). "Load-bearing by interval" means the lower bound is above 0; the sign
    verdict (rule 1.10) is reported next to it, never replaced by it.
    """
    ids = sorted(per_target)
    if not ids:
        return {"n_targets": 0}
    values = np.array([float(np.mean(per_target[i])) for i in ids])
    draws = np.random.default_rng(seed).integers(0, len(values), size=(n_resamples, len(values)))
    low, high = percentile_ci(values[draws].mean(axis=1), confidence)
    mean = float(values.mean())
    return {"n_targets": len(values), "delta": mean, "delta_low": low, "delta_high": high,
            "load_bearing_sign": bool(mean > 0), "load_bearing_ci": bool(low > 0),
            "n_resamples": n_resamples}


def bootstrap_delta_2d(by_seed: dict, n_resamples: int = N_RESAMPLES, seed: int = 0,
                       confidence: float = CONFIDENCE) -> dict:
    """`bootstrap_delta` with seeds resampled as well as targets (see `bootstrap_effect_2d`).

    `by_seed` is {target: {seed: delta}} (each value already the mean of that target's pairs
    in that seed). The point value is unchanged; only the interval widens by the seed term.
    """
    ids, seeds, matrix = _seed_matrix(by_seed)
    if not ids:
        return {"n_targets": 0}
    rng = np.random.default_rng(seed)
    target_draws = rng.integers(0, len(ids), size=(n_resamples, len(ids)))
    seed_draws = rng.integers(0, len(seeds), size=(n_resamples, len(seeds)))
    means = _two_way_means(matrix, target_draws, seed_draws)
    low, high = percentile_ci(means[np.isfinite(means)], confidence)
    mean = float(np.nanmean(matrix, axis=1).mean())
    return {"n_targets": len(ids), "delta": mean, "delta_low": low, "delta_high": high,
            "load_bearing_sign": bool(mean > 0), "load_bearing_ci": bool(low > 0),
            "n_resamples": n_resamples, "resampling": "seeds_and_targets",
            "n_seeds_resampled": len(seeds)}


def faithfulness_cells(folder: str, n_resamples: int = N_RESAMPLES,
                       committed_dirs: tuple = (),
                       resampling: str = "targets") -> tuple:
    """(rows, hashes, problems) for the per-pair faithfulness files of one folder.

    Reads `faithfulness_*` and `token_faithfulness_*` JSONs written with `--record-pairs`.
    Each seed's per-pair deltas are grouped by target (mean within target), then targets are
    resampled with all their seeds. The pair-level mean is reported too: it is the number the
    committed verdict uses.

    `committed_dirs` are the original analysis folders: the same-named file there holds the
    committed per-level delta, and the largest difference to the re-run is recorded per cell
    (`max_abs_diff_vs_committed`), so a re-run that did not reproduce the original is visible.
    """
    def committed_delta(name: str, level: str):
        for directory in committed_dirs:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                entry = (json.load(open(path)).get("levels", None) or json.load(open(path))).get(level)
                if entry:
                    return entry.get("comprehensiveness_delta")
        return None

    files: dict = {}
    for path in sorted(glob.glob(os.path.join(folder, "*faithfulness_*_seed*.json"))):
        match = FAITH.match(os.path.basename(path))
        if match:
            model = match.group("model") or "coldsite_dti"
            files.setdefault((match.group("kind"), model, match.group("dataset")), {})[
                int(match.group("seed"))] = path
    rows, hashes, problems = [], {}, []
    for (kind, model, dataset), seeds in sorted(files.items()):
        payloads = {}
        for seed, path in seeds.items():
            payloads[seed] = json.load(open(path))
            hashes[path] = sha256(path)
        for level in LEVELS:
            per_target: dict = {}
            by_seed: dict = {}
            pair_means, sign_flags, n_pairs, versus_committed = [], [], [], []
            for seed, payload in sorted(payloads.items()):
                entry = (payload.get("levels", payload)).get(level)
                if not entry:
                    continue
                pairs = entry.get("per_pair")
                if pairs is None:
                    problems.append(f"{kind} {model} {dataset} {level} seed {seed}: "
                                    "no per_pair (run with --record-pairs)")
                    continue
                grouped: dict = {}
                for pair in pairs:
                    value = pair["comprehensiveness_delta"]
                    if value is not None and pair.get("id") is not None:
                        grouped.setdefault(pair["id"], []).append(value)
                for target, values in grouped.items():
                    per_target.setdefault(target, []).append(float(np.mean(values)))
                    by_seed.setdefault(target, {})[seed] = float(np.mean(values))
                finite = [v for g in grouped.values() for v in g]
                pair_means.append(float(np.mean(finite)) if finite else float("nan"))
                n_pairs.append(len(finite))
                recorded = entry.get("comprehensiveness_delta")
                sign_flags.append((pair_means[-1], recorded))
                original = committed_delta(os.path.basename(seeds[seed]), level)
                if original is not None and recorded is not None:
                    versus_committed.append(abs(recorded - original))
            if not per_target:
                continue
            row = {"kind": kind, "model": model, "dataset": dataset, "level": level,
                   "seeds": len(pair_means), "pairs_per_seed": int(np.median(n_pairs)),
                   "pair_mean_delta": float(np.mean(pair_means)),
                   "per_seed_pair_mean": ";".join(f"{v:.6f}" for v in pair_means),
                   "max_abs_diff_vs_summary": float(max(
                       abs(a - b) for a, b in sign_flags if b is not None and np.isfinite(b)))
                   if any(b is not None for _a, b in sign_flags) else float("nan"),
                   "max_abs_diff_vs_committed": max(versus_committed) if versus_committed
                   else float("nan")}
            row.update(bootstrap_delta(per_target, n_resamples) if resampling == "targets"
                       else bootstrap_delta_2d(by_seed, n_resamples))
            rows.append(row)
    return rows, hashes, problems


def faithfulness_report(rows: list, resampling: str = "targets") -> str:
    out = ["# Faithfulness delta with confidence intervals", "",
           "Mean `comprehensiveness_delta` (attended masking minus the size-matched random "
           f"control), pairs grouped by target, targets resampled ({N_RESAMPLES:,} resamples, "
           "95% percentile, `default_rng(0)`, each target carried in with all its seeds). "
           "`sign` is the original verdict (mean > 0); `CI` is the added one (lower bound > 0). "
           "Reported beside each other, neither replaces the other. MolTrans is in token space "
           "(size-matched arms), as in the paper.", "",
           "| model | space | dataset | level | targets | pairs/seed | mean delta (pairs) | "
           "mean delta (targets) | 95% CI | sign | CI | max diff vs committed |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    if resampling != "targets":
        out[2] = ("Mean `comprehensiveness_delta` (attended masking minus the size-matched random "
                  "control), pairs grouped by target within each seed. " + TWO_WAY_NOTE +
                  " `sign` is the original verdict (mean > 0); `CI` is the added one (lower bound > 0). "
                  "MolTrans is in token space (size-matched arms), as in the paper.")
    for r in rows:
        space = "token" if r["kind"] == "token_faithfulness" else "residue"
        out.append(f"| {r['model']} | {space} | {r['dataset'].upper()} | {r['level'].replace('_', '-')} | "
                   f"{r['n_targets']} | {r['pairs_per_seed']} | {fmt(r['pair_mean_delta'], 4)} | "
                   f"{fmt(r['delta'], 4)} | {fmt(r['delta_low'], 4)}–{fmt(r['delta_high'], 4)} | "
                   f"{'yes' if r['load_bearing_sign'] else 'no'} | "
                   f"{'yes' if r['load_bearing_ci'] else 'no'} | "
                   f"{fmt(r['max_abs_diff_vs_committed'], 6)} |")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# do the original verdicts come back from the original outputs?
# --------------------------------------------------------------------------

def reproduce_seed_agreement(spread_rows: list, davis_dir: str, kiba_dir: str,
                             committed: str = "results/seed_agreement.md") -> dict:
    """Regenerate `results/seed_agreement.md` byte for byte, and match its spread count."""
    from src.evaluation import seed_agreement

    table = seed_agreement.cells(seed_agreement.default_sources(davis_dir, kiba_dir))
    text = seed_agreement.report(table)
    stats = seed_agreement.summarise(table)
    mine = [r for r in spread_rows if r["family"] in ("P1", "P2")]
    return {"check": "seed_agreement.md regenerated from the ladders",
            "identical_to_committed": text == open(committed).read(),
            "cells": stats["cells"], "seeds_disagree": stats["seeds_disagree"],
            "spread_exceeds_signal": stats["spread_exceeds_signal"],
            "effects_v2_cells": len(mine),
            "effects_v2_spread_exceeds_signal": sum(r["spread_exceeds_signal"] for r in mine),
            "ok": text == open(committed).read() and len(mine) == stats["cells"]
            and sum(r["spread_exceeds_signal"] for r in mine) == stats["spread_exceeds_signal"]}


def reproduce_audit(path: str) -> dict:
    """Apply Holm to the recorded raw p-values; compare with the recorded verdicts."""
    from src.evaluation.aggregate import holm_bonferroni

    payload = json.load(open(path))
    redone = holm_bonferroni(payload["p_values_raw"])
    recorded = payload["p_values_corrected"]
    same = set(redone) == set(recorded) and all(
        redone[k]["significant"] == recorded[k]["significant"]
        and abs(redone[k]["adjusted_alpha"] - recorded[k]["adjusted_alpha"]) < 1e-15
        for k in redone)
    return {"check": f"Holm over {os.path.basename(path)}", "cells": len(redone),
            "significant_recorded": sorted(k for k, v in recorded.items() if v["significant"]),
            "significant_redone": sorted(k for k, v in redone.items() if v["significant"]),
            "ok": same}


def reproduce_faithfulness(folders: list) -> dict:
    """Every recorded `explanation_is_load_bearing` equals the sign of its own delta."""
    checked = flipped = 0
    positive = 0
    for folder in folders:
        for path in sorted(glob.glob(os.path.join(folder, "*faithfulness_*_seed*.json"))):
            if os.path.basename(path).startswith("accuracy"):
                continue
            payload = json.load(open(path))
            levels = payload.get("levels", payload)
            for name, entry in levels.items():
                if not isinstance(entry, dict) or "comprehensiveness_delta" not in entry:
                    continue
                flag = entry.get("explanation_is_load_bearing", entry.get("load_bearing"))
                delta = entry["comprehensiveness_delta"]
                checked += 1
                positive += bool(flag)
                flipped += bool(flag) != bool(np.isfinite(delta) and delta > 0)
    return {"check": "faithfulness verdict = sign of the recorded delta",
            "level_summaries": checked, "load_bearing": positive, "disagree": flipped,
            "ok": checked > 0 and flipped == 0}


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def sources(ig_root: str) -> list:
    """(folder, ground-truth file, family, label). Family names are the amendment's."""
    davis_gt, kiba_gt = "data/davis_ground_truth_sites.json", "data/kiba_ground_truth_sites.json"
    davis_pocket, kiba_pocket = "data/davis_klifs_pocket_sites.json", "data/kiba_klifs_pocket_sites.json"
    return [
        ("results/analysis_davis_policyA", davis_gt, "P1", "UniProt"),
        ("results/analysis_kiba_policyA", kiba_gt, "P2", "UniProt"),
        ("results/analysis_davis_policyA_klifs", davis_pocket, "S3-D", "KLIFS pocket"),
        ("results/analysis_kiba_policyA_klifs", kiba_pocket, "S3-K", "KLIFS pocket"),
        (os.path.join(ig_root, "ig_davis"), davis_gt, "S1", "UniProt"),
        (os.path.join(ig_root, "ig_davis_klifs"), davis_pocket, "S1-klifs", "KLIFS pocket"),
        (os.path.join(os.path.dirname(ig_root), "ig_kiba"), kiba_gt, "S2", "UniProt"),
        (os.path.join(os.path.dirname(ig_root), "ig_kiba_klifs"), kiba_pocket, "S2-klifs", "KLIFS pocket"),
    ]


def fmt(value, digits=3):
    return "—" if value is None or (isinstance(value, float) and not np.isfinite(value)) \
        else f"{value:.{digits}f}"


TWO_WAY_NOTE = ("TWO-WAY bootstrap (`--resample seeds_and_targets`), " + f"{N_RESAMPLES:,}" +
                " resamples, 95% percentile intervals, `default_rng(0)`: each resample draws the targets "
                "with replacement and, independently, the seeds with replacement from the cell's seeds, "
                "and averages over the drawn (target, seed) grid, so an interval carries seed-to-seed "
                "variance as well as target sampling. Point values are identical to the target-only "
                "tables of `results/effects_v2/` (amendment section 5); only the intervals differ.")


def enrichment_report(rows: list, resampling: str = "targets") -> str:
    out = ["# Enrichment over chance, with confidence intervals", "",
           "Mean precision@10 divided by mean chance over the same resampled proteins; 95% "
           f"percentile intervals, {N_RESAMPLES:,} resamples, unit = target, each protein carried in "
           "with all its seeds averaged, `default_rng(0)` (`src/evaluation/effects_v2.py`, "
           "amendment section 5). Effect sizes only: no threshold is applied. `excess` is precision "
           "minus chance. Ceiling is the recorded mean ceiling.", "",
           "| family | ground truth | dataset | model | level | n | precision@10 | 95% CI | chance | "
           "ceiling | enrichment | 95% CI | excess | 95% CI |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    if resampling != "targets":
        out[2] = ("Mean precision@10 divided by mean chance over the same resampled targets. "
                  + TWO_WAY_NOTE + " Effect sizes only: no threshold is applied. `excess` is precision "
                  "minus chance. Ceiling is the recorded mean ceiling.")
    for r in rows:
        out.append(
            f"| {r['family']} | {r['ground_truth']} | {r['dataset'].upper()} | {r['model']} | "
            f"{r['level'].replace('_', '-')} | {r['n']} | {fmt(r['precision'])} | "
            f"{fmt(r['precision_low'])}–{fmt(r['precision_high'])} | {fmt(r['chance'], 4)} | "
            f"{fmt(r['ceiling'])} | {fmt(r['enrichment'], 2)} | "
            f"{fmt(r['enrichment_low'], 2)}–{fmt(r['enrichment_high'], 2)} | {fmt(r['excess'], 4)} | "
            f"{fmt(r['excess_low'], 4)}–{fmt(r['excess_high'], 4)} |")
    return "\n".join(out) + "\n"


def spread_report(rows: list) -> str:
    out = ["# Seed spread against distance from chance", "",
           "Rule 1.13 of the amendment (`seed_agreement.py`): the spread exceeds the signal iff "
           "max − min of the seeds' precision@10 is larger than |mean − chance|; chance is the "
           "cell's recorded chance. `exact` repeats the test with the rebuilt per-protein chance.", "",
           "| family | ground truth | dataset | model | level | per-seed precision@10 | spread | SD | "
           "|mean − chance| | spread > signal | same, exact chance | enrichment (95% CI) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['family']} | {r['ground_truth']} | {r['dataset'].upper()} | {r['model']} | "
            f"{r['level'].replace('_', '-')} | {r['per_seed_precision'].replace(';', ' / ')} | "
            f"{fmt(r['spread'], 4)} | {fmt(r['sd'], 4)} | {fmt(r['distance_from_chance'], 4)} | "
            f"{'yes' if r['spread_exceeds_signal'] else 'no'} | "
            f"{'yes' if r['spread_exceeds_signal_exact_chance'] else 'no'} | "
            f"{fmt(r['enrichment'], 2)} ({fmt(r['enrichment_low'], 2)}–{fmt(r['enrichment_high'], 2)}) |")
    return "\n".join(out) + "\n"


def write_csv(path: str, rows: list):
    fields = list(rows[0])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out-dir", default="results/effects_v2")
    parser.add_argument("--ig-root", default=os.path.expanduser(
        "~/ColdSite-results/integrated_gradients"))
    parser.add_argument("--split-root", default="data/splits")
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--faithfulness-dir", default=None,
                        help="folder of faithfulness JSONs written with --record-pairs "
                             "(default: results/effects_v2/faithfulness if it exists)")
    parser.add_argument("--families", default=None,
                        help="comma-separated subset of family ids (default: all)")
    parser.add_argument("--resample", choices=RESAMPLING, default="targets",
                        help="'targets' (amendment section 5; each target carried in with its seeds "
                             "averaged) or 'seeds_and_targets' (two-way bootstrap: targets and seeds "
                             "drawn with replacement, independently). The two-way mode must write to "
                             "its own --out-dir, never over the committed target-only tables.")
    args = parser.parse_args()
    if args.resample != "targets" and os.path.abspath(args.out_dir) == os.path.abspath("results/effects_v2"):
        parser.error("--resample seeds_and_targets must not write into results/effects_v2")
    if args.n_resamples < N_RESAMPLES:
        parser.error(f"amendment section 5 fixes {N_RESAMPLES} resamples")
    os.makedirs(args.out_dir, exist_ok=True)
    wanted = set(args.families.split(",")) if args.families else None

    enrichment, spreads, hashes, problems = [], [], {}, []
    for folder, ground_truth, family, label in sources(args.ig_root):
        if wanted and family not in wanted:
            continue
        if not os.path.isdir(folder):
            problems.append(f"{family}: folder {folder} not found")
            continue
        e, s, h, p = cell_rows(folder, ground_truth, family, label, args.split_root,
                               args.n_resamples, args.resample)
        print(f"{family:9s} {folder}: {len(e)} cells")
        enrichment += e
        spreads += s
        hashes.update(h)
        problems += p
    if not enrichment:
        raise SystemExit("no ladder cells found")
    faith_dir = args.faithfulness_dir or os.path.join(args.out_dir, "faithfulness")
    if os.path.isdir(faith_dir):
        f_rows, f_hashes, f_problems = faithfulness_cells(
            faith_dir, args.n_resamples,
            ("results/analysis_davis_policyA", "results/analysis_kiba_policyA"), args.resample)
        if f_rows:
            write_csv(os.path.join(args.out_dir, "faithfulness_effects.csv"), f_rows)
            open(os.path.join(args.out_dir, "faithfulness_effects.md"), "w").write(
                faithfulness_report(f_rows, args.resample))
        hashes.update(f_hashes)
        problems += f_problems
        print(f"faithfulness: {len(f_rows)} cells from {faith_dir}")
    with open(os.path.join(args.out_dir, "problems.txt"), "w") as handle:
        handle.write("\n".join(problems) + ("\n" if problems else ""))
    write_csv(os.path.join(args.out_dir, "enrichment.csv"), enrichment)
    write_csv(os.path.join(args.out_dir, "seed_spread.csv"), spreads)
    open(os.path.join(args.out_dir, "enrichment.md"), "w").write(enrichment_report(enrichment, args.resample))
    open(os.path.join(args.out_dir, "seed_spread.md"), "w").write(spread_report(spreads))
    with open(os.path.join(args.out_dir, "inputs_sha256.json"), "w") as handle:
        json.dump(dict(sorted(hashes.items())), handle, indent=1)
    davis, kiba = "results/analysis_davis_policyA", "results/analysis_kiba_policyA"
    checks = [reproduce_seed_agreement(spreads, davis, kiba)]
    for folder in (davis, kiba):
        for path in sorted(glob.glob(os.path.join(folder, "audit_*_binary*.json"))):
            if "superseded" in path:
                continue
            checks.append(reproduce_audit(path))
    checks.append(reproduce_faithfulness([davis, kiba]))
    with open(os.path.join(args.out_dir, "reproduction.json"), "w") as handle:
        json.dump(checks, handle, indent=1)
    for check in checks:
        print(("OK   " if check["ok"] else "FAIL ") + check["check"])
    bad = [r for r in enrichment if not r["chance_check_ok"]]
    print(f"{len(enrichment)} cells; chance rebuild within {MC_SIGMAS:g} MC SE for "
          f"{len(enrichment) - len(bad)}; outside for {len(bad)}; {len(problems)} problems")
    for line in problems:
        print("  problem:", line)
    for r in bad:
        print(f"  chance check: {r['family']} {r['model']} {r['dataset']} {r['level']}: "
              f"rebuilt {r['chance']:.5f} vs recorded {r['recorded_chance_mean']:.5f}")


if __name__ == "__main__":
    main()
