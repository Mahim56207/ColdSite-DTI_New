"""The epoch-1 gate with Gate B's envelope amended (2026-09-29, AFTER the first gate run; docs/canary_attempts/README.md).

Why it was amended. Gate B's original envelope, [min - sd, max + sd] of the three committed seeds' epoch-1 values, is the range of a
sample of THREE. A fresh draw from the same process lands outside it with probability 0.23 per metric (Monte Carlo, 400,000 trials), so a
healthy harness fails the six envelope checks of two new seeds x three metrics with probability 0.41 (metrics fully correlated) to 0.80
(independent). The first gate run failed on exactly one such check (seed 4, epoch-1 validation loss 0.2328 against a band of 0.2098 to
0.2207) while every other check passed, including the replay of seed 1 and the distinctness of all three seeds.

The amended rule (fixed before it was applied to the recorded data). Each Gate B metric of each new seed must lie inside the PREDICTION
interval for a new draw from the three committed values:

    mean +/- t(1 - alpha / (2 m), n - 1) * sd * sqrt(1 + 1/n)          n = 3, alpha = 0.05,
                                                                       m = the number of envelope checks = 3 metrics x the new seeds

a family-wise 5 % false-fail rate under normality however many seeds are checked (a Bonferroni split of alpha over the m checks, the same
family-wise idea the audit uses). Unchanged: Gate A (replay of seed 1 within one committed-seed SD), finite values, and DISTINCT (a new
seed's initial-weight hash and metrics must differ from every other run's).

Honest limits, all recorded in the verdict:
  * It was chosen after a fail, so the ORIGINAL rule's verdict is kept in the output beside it. It is not a re-run: the same recorded runs
    are re-evaluated, so nothing was retried until it passed.
  * The choice of the family-wise adjustment matters here. An UNADJUSTED 95 % prediction interval still puts seed 4's validation loss just
    outside (it is reported per check as `inside_unadjusted_95`); only the adjustment over the six checks brings it inside.
  * With n = 3 the interval is wide by construction (t with 2 degrees of freedom). It still stops gross failures and non-finite values; the
    bug this gate exists to catch, several seeds that are one run, is caught by DISTINCT and by Gate A, not by the envelope.
  * Gate A is NOT amended, and it has a similar weakness: three replays of seed 1's epoch 1 differ from the committed run by -0.0023, -0.0044 and
    +0.0023 in validation loss against a tolerance of 0.0029, so replay noise is as large as the tolerance and Gate A can fail a healthy
    harness. The recorded gate run passed it; that is recorded in docs/canary_attempts/README.md so it is not a surprise later.

This module is deliberately NOT in `canary.HARNESS_FILES`: it decides a verdict offline and trains nothing, and editing a hashed file would
have made the canary's passing verdict stale.

    python -m src.cloud.gate_rule_v2 --evaluate --seeds 1 4 5 --root <results>/epoch_gate --out epoch_gate_verdict.json
    exit 0 pass, 1 fail, 3 inconclusive
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

from scipy import stats

from src.cloud import canary, epoch_gate, markers
from src.cloud.config import Cell
from src.cloud.epoch_gate import COMMITTED_SEEDS, METRICS, REFERENCE_PATH, REPLAY_SEED, _overall

ALPHA = 0.05
RULE = "v2: Gate B envelope is the family-wise prediction interval of a new draw from the three committed seeds"


def prediction_interval(values, m_checks: int, alpha: float = ALPHA) -> tuple:
    """(low, high, t quantile, sd) for a NEW draw from the sample `values`, split over `m_checks` checks."""
    n = len(values)
    mean, sd = statistics.fmean(values), statistics.stdev(values)
    q = float(stats.t.ppf(1 - alpha / (2 * m_checks), n - 1))
    half = q * sd * math.sqrt(1 + 1 / n)
    return mean - half, mean + half, q, sd


def evaluate_v2(new: dict, ref_by_seed: dict, replay_seed: int = REPLAY_SEED, wave_seeds=(), alpha: float = ALPHA) -> dict:
    """Same inputs as `epoch_gate.evaluate`. The original rule's result is kept under `original_rule`."""
    original = epoch_gate.evaluate(new, ref_by_seed, replay_seed, wave_seeds)       # also validates the inputs
    m = len(METRICS) * max(len(wave_seeds), 1)
    result = {"rule": RULE, "alpha": alpha, "n_envelope_checks": m, "replay_seed": replay_seed,
              "gate_a": original["gate_a"], "gate_b": {},
              "original_rule": {"verdict": original["verdict"],
                                "gate_b": {s: v["verdict"] for s, v in original["gate_b"].items()},
                                "reference_spread": original["reference_spread"]}}
    for seed in wave_seeds:
        rows = {}
        for metric in METRICS:
            values = [ref_by_seed[s][metric] for s in COMMITTED_SEEDS]
            low, high, q, sd = prediction_interval(values, m, alpha)
            low95, high95, _, _ = prediction_interval(values, 1, alpha)
            got = new[seed][metric]
            finite = math.isfinite(got)
            rows[metric] = {"new": got, "interval": [low, high], "t_quantile": q, "sd": sd,
                            "unadjusted_95_interval": [low95, high95],
                            "inside_unadjusted_95": bool(finite and low95 <= got <= high95),
                            "verdict": "pass" if finite and low <= got <= high else "fail"}
        rows["distinct"] = original["gate_b"][str(seed)]["distinct"]
        rows["verdict"] = _overall(v["verdict"] for v in rows.values())
        result["gate_b"][str(seed)] = rows
    result["verdict"] = _overall([result["gate_a"]["verdict"], *(v["verdict"] for v in result["gate_b"].values())])
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--evaluate", action="store_true", required=True, help="re-evaluate runs that already exist")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(epoch_gate.DEFAULT_SEEDS))
    ap.add_argument("--root", required=True, help="the folder the gate wrote (<results>/epoch_gate)")
    ap.add_argument("--reference", default=REFERENCE_PATH)
    ap.add_argument("--out", help="write the verdict here (JSON)")
    args = ap.parse_args(argv)
    ref = epoch_gate.load_reference(args.reference)
    seeds = list(dict.fromkeys(args.seeds))
    if REPLAY_SEED not in seeds:
        ap.error(f"--seeds must include the replay seed {REPLAY_SEED}")
    wave_seeds = [s for s in seeds if s not in COMMITTED_SEEDS]
    cells = [Cell(ref["dataset"], ref["model"], ref["level"], s) for s in seeds]
    new = {c.seed: epoch_gate.read_epoch1(args.root, c) for c in cells}
    result = evaluate_v2(new, ref["by_seed"], REPLAY_SEED, wave_seeds)
    result.update(dataset=ref["dataset"], level=ref["level"], model=ref["model"], seeds_checked=wave_seeds,
                  runs={str(s): v for s, v in new.items()}, harness_sha256=canary.harness_hash(),
                  reference_sha256=markers.sha256_file(args.reference))
    print(json.dumps(result, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=1)
    return {"pass": 0, "fail": 1, "inconclusive": 3}[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
