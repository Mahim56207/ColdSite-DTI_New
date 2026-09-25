"""Pre-flight: refuse to start (non-zero exit, no GPU work) unless every check passes.

Checks, in the order the plan lists them (docs/REMEDIATION_PLAN.md, cloud_rules (a)):

  gpus            the declared number of GPUs is visible and each is the declared card (T4)
  canary          a non-canary wave needs a passing canary verdict for this exact harness (T11 gate)
  splits          data/splits/**, the ground truths and the panel match data/splits/MANIFEST.json
  raw_dataset     the raw DAVIS / KIBA files match config/cloud_manifest.json
  vendored        each vendored baseline's content hash matches config/cloud_manifest.json
  rng_import      importing each vendored trainer module leaves torch/numpy/python RNG unchanged
  disk            enough free space for the planned checkpoints plus a declared reserve
  cells           no planned cell is in an unexplained state; complete cells are skipped and
                  partial ones marked for resume
  wave            the wave manifest exists, lists this account, and lists no cell twice

`--dry-run` runs everything except the checks that need the cloud machine itself: the GPU check
is reported as SKIPPED (never silently passed), and a DrugBAN RNG check that needs DGL is
skipped only when DGL is absent. A skipped check is printed as such and counted separately.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

from src.cloud import markers
from src.cloud.config import (CLOUD_MANIFEST_PATH, Cell, load_harness, load_json, load_waves)
from src.cloud.recipes import cell_dirs

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VENDORED = {"MolTrans": "baselines/MolTrans", "HyperAttentionDTI": "baselines/HpyerAttentionDTI",
            "DrugBAN": "baselines/DrugBAN", "DeepDTA": "baselines/DeepDTA"}
_SKIP_DIRS = {"__pycache__", "datasets", "image", ".git", ".ipynb_checkpoints"}
RAW_FILES = tuple(f"src/data/baselines/deepdta/data/{ds}/{name}"
                  for ds in ("davis", "kiba") for name in ("ligands_can.txt", "proteins.txt", "Y"))

# each vendored trainer module, imported the way the trainers do (tests/test_integrity_rng_seeds.py)
RNG_IMPORTS = {
    "moltrans": "from src.model.train_moltrans import _import_vendored as f; f()",
    "hyperattentiondti": "from src.model.train_hyperattentiondti import _import_vendored as f; f()",
    "drugban": ("from src.evaluation.baseline_adapters import _vendored; "
                "_vendored('DrugBAN', 'vendored DrugBAN'); from models import DrugBAN"),
}


@dataclass
class Check:
    name: str
    status: str            # "pass" | "fail" | "skipped"
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status != "fail"


# -- hashes -----------------------------------------------------------------------------

def _sha256(path: str) -> str:
    return markers.sha256_file(path)


def tree_files(directory: str) -> list:
    """Relative paths of the files a vendored hash covers, sorted."""
    found = []
    for folder, dirs, names in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in names:
            if name.endswith((".pyc", ".pyo")) or name == ".DS_Store":
                continue
            found.append(os.path.relpath(os.path.join(folder, name), directory).replace(os.sep, "/"))
    return sorted(found)


def tree_hash(directory: str) -> str:
    """SHA-256 over every file's relative path and content hash in sorted path order;
    independent of git and of directory-walk order."""
    digest = hashlib.sha256()
    for relative in tree_files(directory):
        digest.update(relative.encode())
        digest.update(_sha256(os.path.join(directory, relative)).encode())
    return digest.hexdigest()


def vendored_hashes(root: str = ROOT) -> dict:
    return {name: tree_hash(os.path.join(root, rel)) for name, rel in VENDORED.items()
            if os.path.isdir(os.path.join(root, rel))}


def raw_hashes(root: str = ROOT) -> dict:
    return {rel: _sha256(os.path.join(root, rel)) for rel in RAW_FILES
            if os.path.exists(os.path.join(root, rel))}


def write_cloud_manifest(root: str = ROOT, path: str = CLOUD_MANIFEST_PATH) -> dict:
    full = os.path.join(root, path)
    if os.path.exists(full):
        raise SystemExit(f"{path} exists. Overwriting it would re-bless whatever is on disk; "
                         f"delete it deliberately if that is the intent.")
    manifest = {"note": ("Content hashes of the vendored baselines and the raw dataset files, "
                         "written by `python -m src.cloud.preflight --write-manifest`. Like "
                         "data/splits/MANIFEST.json it fixes the files as they were then; it "
                         "cannot show they are what the 84 existing cells trained on."),
                "vendored": vendored_hashes(root), "raw_dataset": raw_hashes(root)}
    with open(full, "w") as handle:
        json.dump(manifest, handle, indent=1)
        handle.write("\n")
    return manifest


# -- the checks -------------------------------------------------------------------------

def probe_gpus() -> list:
    """Names of the visible CUDA devices, from nvidia-smi so that this process never opens
    a CUDA context of its own (each trainer is a separate process on its own GPU)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def check_gpus(cfg: dict, dry_run: bool, probe=probe_gpus) -> Check:
    names = probe()
    want, card = cfg["required_gpus"], cfg["gpu_name_contains"]
    if dry_run:
        return Check("gpus", "skipped", f"dry run: needs {want} x {card}; saw {names or 'none'}")
    if len(names) != want:
        return Check("gpus", "fail", f"{len(names)} GPU(s) visible, need {want}: {names}")
    wrong = [n for n in names if card not in n]
    if wrong:
        return Check("gpus", "fail", f"not a {card}: {wrong}. The recipes' batch sizes and every "
                                     f"measured epoch time are for a {card}")
    return Check("gpus", "pass", ", ".join(names))


def check_splits(root: str = ROOT, cells: list | None = None) -> Check:
    """Every ground-truth and panel file, and the split files of the cells to be trained,
    against data/splits/MANIFEST.json. Derived splits (`*_seqclean`, `*_seqmatched`) are built
    by a separate step and are checked only if a planned cell uses them."""
    from src.data import manifest
    try:
        recorded = manifest.load(root)
    except FileNotFoundError as exc:
        return Check("splits", "fail", f"no manifest: {exc}")
    wanted = None
    if cells is not None:
        wanted = tuple(f"data/splits/{c.dataset}/{c.level}/" for c in cells)
    files = {path: entry for path, entry in recorded["files"].items()
             if entry["section"] != "splits" or wanted is None or path.startswith(wanted)}
    result = manifest.compare({**recorded, "files": files}, root)
    if wanted is not None:                     # `unlisted` is about files outside the scope
        result["unlisted"] = [u for u in result["unlisted"] if u.startswith(wanted)
                              or not u.startswith("data/splits/")]
    bad = {k: v for k, v in result.items() if v}
    if bad:
        return Check("splits", "fail", "; ".join(f"{k}: {len(v)} (e.g. {v[0]})"
                                                 for k, v in bad.items()))
    return Check("splits", "pass", f"{len(files)} files match data/splits/MANIFEST.json")


def _compare_section(name: str, recorded: dict, current: dict) -> Check:
    if not recorded:
        return Check(name, "fail", f"config/cloud_manifest.json has no {name} entries")
    missing = sorted(set(recorded) - set(current))
    changed = sorted(k for k in recorded if k in current and recorded[k] != current[k])
    if missing or changed:
        return Check(name, "fail", f"missing: {missing}; changed: {changed}")
    return Check(name, "pass", f"{len(recorded)} match")


def check_raw_dataset(root: str = ROOT, manifest_path: str = CLOUD_MANIFEST_PATH) -> Check:
    try:
        recorded = load_json(os.path.join(root, manifest_path))["raw_dataset"]
    except (FileNotFoundError, KeyError) as exc:
        return Check("raw_dataset", "fail", f"cloud manifest unreadable: {exc!r}")
    return _compare_section("raw_dataset", recorded, raw_hashes(root))


def check_vendored(models: set, root: str = ROOT, manifest_path: str = CLOUD_MANIFEST_PATH) -> Check:
    try:
        recorded = load_json(os.path.join(root, manifest_path))["vendored"]
    except (FileNotFoundError, KeyError) as exc:
        return Check("vendored", "fail", f"cloud manifest unreadable: {exc!r}")
    needed = {"moltrans": "MolTrans", "hyperattentiondti": "HyperAttentionDTI",
              "drugban": "DrugBAN", "deepdta": "DeepDTA"}
    names = {needed[m] for m in models if m in needed}
    if not names:
        return Check("vendored", "pass", "no vendored code in this plan (ColdSite-DTI is ours)")
    current = vendored_hashes(root)
    return _compare_section("vendored", {n: recorded.get(n) for n in names if n in recorded}
                            or {n: None for n in names},
                            {n: current.get(n) for n in names})


def check_rng_import(models: set, dry_run: bool, root: str = ROOT, python: str | None = None) -> Check:
    """Import each vendored trainer module in a fresh process; no generator may move."""
    from src.model.integrity import rng_delta  # noqa: F401  (keeps the snippet's import honest)
    problems, skipped, ran = [], [], []
    for model in sorted(m for m in models if m in RNG_IMPORTS):
        code = ("from src.model.integrity import rng_state, rng_delta\n"
                "b = rng_state()\n" + RNG_IMPORTS[model] + "\n"
                "print('MOVED', rng_delta(b, rng_state()))")
        out = subprocess.run([python or sys.executable, "-c", code], capture_output=True,
                             text=True, cwd=root, env={**os.environ, "PYTHONPATH": root})
        if out.returncode != 0:
            tail = (out.stderr or "").strip().splitlines()[-1:] or ["?"]
            if model == "drugban" and "dgl" in (out.stderr or "").lower() and dry_run:
                skipped.append("drugban (DGL not installed here)")
                continue
            problems.append(f"{model}: import failed: {tail[0]}")
            continue
        moved = out.stdout.strip().splitlines()[-1]
        if moved != "MOVED []":
            problems.append(f"{model}: importing it moved the RNG {moved}")
        else:
            ran.append(model)
    if problems:
        return Check("rng_import", "fail", "; ".join(problems))
    detail = f"unchanged after importing: {', '.join(ran) or 'nothing to import'}"
    if skipped:
        return Check("rng_import", "skipped" if not ran else "pass", detail + f"; skipped {skipped}")
    return Check("rng_import", "pass", detail)


def planned_bytes(cells: list, cfg: dict) -> int:
    factor = cfg.get("planned_bytes_per_cell_factor", 4)
    return sum(int(cfg["checkpoint_bytes"][c.model]) * factor for c in cells)


def check_disk(cells: list, results_root: str, cfg: dict, free_bytes=None) -> Check:
    os.makedirs(results_root, exist_ok=True)
    free = free_bytes() if free_bytes else shutil.disk_usage(results_root).free
    need = planned_bytes(cells, cfg) + int(cfg["disk_reserve_gb"] * 1e9)
    detail = f"free {free / 1e9:.1f} GB, need {need / 1e9:.1f} GB ({len(cells)} cell(s) + reserve)"
    return Check("disk", "pass" if free >= need else "fail", detail)


def check_canary(root: str, dry_run: bool, wave: str | None) -> Check:
    """A non-canary wave trains only if the canary passed on THIS harness (plan T11: else BLOCKED)."""
    from src.cloud import canary
    if wave == "canary":
        return Check("canary", "pass", "this is the canary wave")
    path = os.path.join(root, canary.VERDICT_PATH)
    try:
        verdict = load_json(path)
        ref = canary.load_reference(os.path.join(root, canary.REFERENCE_PATH))
        problems = []
        if verdict.get("verdict") != "pass":
            problems.append(f"verdict is {verdict.get('verdict')!r}, not 'pass'")
        if (verdict.get("dataset"), verdict.get("level"), verdict.get("model")) != (
                ref["dataset"], ref["level"], ref["model"]):
            problems.append("verdict is for a different cell from the canary reference")
        if verdict.get("harness_sha256") != canary.harness_hash(root):
            problems.append("the harness files changed since the canary ran (harness_sha256 differs)")
        detail = "; ".join(problems) or "canary passed on this harness"
    except FileNotFoundError:
        problems, detail = ["missing"], (f"no {canary.VERDICT_PATH}: the canary has not passed. Run "
                                         f"notebooks/kaggle_canary.ipynb, copy its canary_verdict.json here and commit it")
    except (ValueError, KeyError) as exc:
        problems, detail = ["unreadable"], f"canary verdict unreadable: {exc!r}"
    if not problems:
        return Check("canary", "pass", detail)
    return Check("canary", "skipped" if dry_run else "fail",
                 (f"dry run; a real run would be REFUSED: " if dry_run else "") + detail)


def check_cells(cells: list, results_root: str) -> tuple:
    """(Check, plan) where plan maps cell id -> 'run' | 'resume' | 'skip'."""
    plan, problems = {}, []
    for cell in cells:
        st, detail = markers.state(results_root, cell)
        if st == "inconsistent":
            problems.append(f"{cell.id}: {detail}")
        plan[cell.id] = {"fresh": "run", "partial": "resume", "complete": "skip",
                         "needs_finalize": "finalize"}.get(st, "stop")
    counts = {k: list(plan.values()).count(k) for k in ("run", "resume", "skip", "finalize")}
    if problems:
        return Check("cells", "fail", "; ".join(problems)), plan
    return Check("cells", "pass", f"{counts}"), plan


def check_wave(waves_path: str, account: str) -> tuple:
    try:
        waves = load_waves(waves_path)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return Check("wave", "fail", f"{waves_path}: {exc}"), None
    if account not in waves["accounts"]:
        return Check("wave", "fail", f"account {account!r} is not in {waves_path} "
                                     f"(has {sorted(waves['accounts'])})"), None
    n = sum(len(q) for q in waves["accounts"][account].values())
    return Check("wave", "pass", f"{waves['wave']}: {n} cell(s) for {account}, none listed twice"), waves


def run_preflight(account: str, waves_path: str, results_root: str, dry_run: bool,
                  cfg: dict | None = None, root: str = ROOT, gpu_probe=probe_gpus,
                  free_bytes=None) -> dict:
    cfg = cfg or load_harness(os.path.join(root, "config", "harness.json"))
    checks = []
    wave_check, waves = check_wave(waves_path, account)
    checks.append(wave_check)
    cells = ([c for q in waves["accounts"][account].values() for c in q] if waves else [])
    models = {c.model for c in cells}
    plan = {}
    checks.append(check_gpus(cfg, dry_run, gpu_probe))
    checks.append(check_canary(root, dry_run, waves["wave"] if waves else None))
    checks.append(check_splits(root, cells if cells else None))
    checks.append(check_raw_dataset(root))
    checks.append(check_vendored(models, root))
    checks.append(check_rng_import(models, dry_run, root))
    cell_check, plan = check_cells(cells, results_root) if cells else (
        Check("cells", "fail", "no cells planned for this account"), {})
    checks.append(cell_check)
    to_train = [c for c in cells if plan.get(c.id) in ("run", "resume")]
    checks.append(check_disk(to_train, results_root, cfg, free_bytes))
    return {"account": account, "dry_run": dry_run, "checks": checks, "plan": plan,
            "cells": cells, "ok": all(c.ok for c in checks)}


def format_report(report: dict) -> str:
    lines = [f"PRE-FLIGHT for {report['account']}" + ("  [dry run]" if report["dry_run"] else "")]
    for c in report["checks"]:
        lines.append(f"  {c.status.upper():8s} {c.name:12s} {c.detail}")
    counts = {k: list(report["plan"].values()).count(k)
              for k in ("run", "resume", "finalize", "skip", "stop")}
    lines.append(f"  plan: {counts}")
    skipped = [c.name for c in report["checks"] if c.status == "skipped"]
    lines.append("  RESULT: " + ("OK" if report["ok"] else "REFUSED")
                 + (f" (skipped in dry run: {', '.join(skipped)})" if skipped else ""))
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--account")
    ap.add_argument("--waves", default="config/waves.json")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.write_manifest:
        manifest = write_cloud_manifest()
        print(f"wrote {CLOUD_MANIFEST_PATH}: vendored {sorted(manifest['vendored'])}, "
              f"{len(manifest['raw_dataset'])} raw dataset files")
        return 0
    if not args.account:
        ap.error("--account is required")
    report = run_preflight(args.account, args.waves, args.results_root, args.dry_run)
    print(format_report(report))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
