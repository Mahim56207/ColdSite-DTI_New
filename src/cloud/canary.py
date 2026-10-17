"""The canary: does the new harness reproduce a committed cell? (amendment §6, D7)

One existing DAVIS cell of the cheapest attention model is retrained with the harness, on the
same seed. It passes iff BOTH test AUROC and test AUPRC lie within TOLERANCE_SDS sample standard
deviations (ddof = 1) of the committed value, that SD being taken over the three committed seeds of
the same model, dataset and level. If that SD is 0 the result is inconclusive and goes back to the
user; no floor is invented.

Rule history (kept here because this file is hashed into every canary verdict):
  v1, as first written (amendment section 6): TOLERANCE_SDS = 1.
  v2, declared 2026-10-17 BEFORE the canary was re-run (the user chose it): TOLERANCE_SDS = 4. Reason: two runs of the identical cell (same seed,
  same initial-weight hash, same torch version) gave test AUROC 0.8868 and 0.8494 against the committed 0.8501, a difference of 3.4 committed-seed
  SDs (docs/canary_attempts/README.md), so a one-SD gate fails a healthy harness most of the time. The value 4 was chosen with those two runs in view;
  that is disclosed there and in the paper's limitations. It still rejects a broken harness (a wrong split, a NaN, a collapsed model).

    python -m src.cloud.canary --write-reference --committed ~/ColdSite-results --level cold_target
    python -m src.cloud.canary --new <harness results dir>            # exit 0 pass, 1 fail, 3 inconclusive

A passing verdict, copied to `config/canary_verdict.json` and committed, is what unlocks every non-canary wave: the
pre-flight refuses to train unless it says "pass" for the canaried cell and its `harness_sha256` equals the current
harness files' (`HARNESS_FILES`).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

from src.model.checkpoint_naming import results_path, run_tag

METRICS = ("auroc", "auprc")
COMMITTED_SEEDS = (1, 2, 3)
TOLERANCE_SDS = 4          # rule v2, declared before the re-run; v1 was 1 (see the module docstring)


def committed_value(committed_dir: str, dataset: str, level: str, model: str, seed: int) -> dict:
    path = results_path(os.path.join(committed_dir, f"{dataset}_binary"),
                        run_tag(dataset, level, "binary", seed), model=model)
    with open(path) as handle:
        return json.load(handle)["test_metrics"]


def evaluate(new_metrics: dict, committed_by_seed: dict, seed: int) -> dict:
    """committed_by_seed: {seed: {metric: value}} for the three committed seeds."""
    if set(committed_by_seed) != set(COMMITTED_SEEDS):
        raise ValueError(f"need the three committed seeds {COMMITTED_SEEDS}, "
                         f"got {sorted(committed_by_seed)}")
    rows, verdicts = {}, []
    for metric in METRICS:
        values = [committed_by_seed[s][metric] for s in COMMITTED_SEEDS]
        sd = statistics.stdev(values)                       # sample SD, ddof = 1
        reference = committed_by_seed[seed][metric]
        new = new_metrics[metric]
        if sd == 0:
            verdict = "inconclusive"
        else:
            verdict = "pass" if abs(new - reference) <= TOLERANCE_SDS * sd else "fail"
        rows[metric] = {"committed": reference, "new": new, "difference": new - reference,
                        "tolerance_sd": sd, "tolerance_multiplier": TOLERANCE_SDS,
                        "tolerance": TOLERANCE_SDS * sd, "verdict": verdict}
        verdicts.append(verdict)
    overall = ("inconclusive" if "inconclusive" in verdicts
               else "pass" if all(v == "pass" for v in verdicts) else "fail")
    return {"seed": seed, "metrics": rows, "verdict": overall}


REFERENCE_PATH = os.path.join("config", "canary_reference.json")
VERDICT_PATH = os.path.join("config", "canary_verdict.json")
# The files whose behaviour the canary vouches for. A verdict is only good for a checkout in which every
# one of them is byte-identical to the one that was canaried; a change to any of them needs a new canary.
HARNESS_FILES = ("src/cloud/__init__.py", "src/cloud/config.py", "src/cloud/markers.py",
                 "src/cloud/preflight.py", "src/cloud/recipes.py", "src/cloud/restore.py",
                 "src/cloud/runner.py", "src/cloud/canary.py", "src/cloud/quota.py",
                 "src/cloud/epoch_gate.py", "src/model/resume.py")


def harness_hash(root: str = ".") -> str:
    from src.cloud.markers import sha256_file
    import hashlib
    digest = hashlib.sha256()
    for relative in HARNESS_FILES:
        digest.update(relative.encode())
        digest.update(sha256_file(os.path.join(root, relative)).encode())
    return digest.hexdigest()


def write_reference(committed_dir: str, dataset: str, level: str, model: str,
                    path: str = REFERENCE_PATH) -> dict:
    """Freeze the committed values the canary is judged against, with the files they came from,
    so a Kaggle run (which has no ~/ColdSite-results) can evaluate itself."""
    from src.cloud.markers import sha256_file
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; delete it deliberately to re-freeze the reference")
    seeds = {}
    for seed in COMMITTED_SEEDS:
        source = results_path(os.path.join(committed_dir, f"{dataset}_binary"),
                              run_tag(dataset, level, "binary", seed), model=model)
        metrics = committed_value(committed_dir, dataset, level, model, seed)
        seeds[str(seed)] = {**{m: metrics[m] for m in METRICS},
                            "source_file": os.path.basename(source),
                            "source_sha256": sha256_file(source)}
    reference = {"dataset": dataset, "level": level, "model": model, "seeds": seeds,
                 "rule": "docs/PROTOCOL_AMENDMENT_v2.md section 6"}
    with open(path, "w") as handle:
        json.dump(reference, handle, indent=1)
        handle.write("\n")
    return reference


def load_reference(path: str = REFERENCE_PATH) -> dict:
    with open(path) as handle:
        ref = json.load(handle)
    ref["by_seed"] = {int(s): {m: v[m] for m in METRICS} for s, v in ref["seeds"].items()}
    return ref


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new", help="results root written by the harness")
    ap.add_argument("--committed", help="~/ColdSite-results (only to write the reference)")
    ap.add_argument("--reference", default=REFERENCE_PATH)
    ap.add_argument("--write-reference", action="store_true")
    ap.add_argument("--dataset", default="davis")
    ap.add_argument("--level")
    ap.add_argument("--model", default="coldsite_dti")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", help="write the verdict here (JSON)")
    args = ap.parse_args(argv)
    if args.write_reference:
        if not (args.committed and args.level):
            ap.error("--write-reference needs --committed and --level")
        ref = write_reference(args.committed, args.dataset, args.level, args.model, args.reference)
        print(f"wrote {args.reference}: {args.model} {args.dataset} {args.level}, "
              f"seeds {sorted(ref['seeds'])}")
        return 0
    if not args.new:
        ap.error("--new is required")
    ref = load_reference(args.reference)
    new = committed_value(args.new, ref["dataset"], ref["level"], ref["model"], args.seed)
    result = evaluate(new, ref["by_seed"], args.seed)
    result.update(dataset=ref["dataset"], level=ref["level"], model=ref["model"],
                  harness_sha256=harness_hash())
    print(json.dumps(result, indent=1))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=1)
    return {"pass": 0, "fail": 1, "inconclusive": 3}[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
