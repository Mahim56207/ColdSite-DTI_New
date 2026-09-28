"""Per-cell state on disk: fresh, partial (resumable) or complete, and the complete-marker.

A cell is complete only when its checkpoint, its `_results.json` and its complete-marker all
exist and agree (marker hashes match the files). A checkpoint with a `_resume.pt` beside it is
a partial cell and continues; a checkpoint alone, or results with no marker, is neither — the
runner reports it and refuses to skip or overwrite it (an unexplained file is a decision for a
person, not something to retrain over).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys

from src.cloud.config import Cell
from src.cloud.recipes import cell_dirs
from src.model.checkpoint_naming import checkpoint_path, results_path, run_tag
from src.model.resume import START_SUFFIX, resume_path

MARKER_SUFFIX = "_complete.json"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def paths(results_root: str, cell: Cell) -> dict:
    out = cell_dirs(results_root, cell)
    ckpt = checkpoint_path(out, cell.dataset, cell.level, "binary", cell.seed, model=cell.model)
    stem = ckpt[: -len(".pt")]
    return {"dir": out, "checkpoint": ckpt, "resume": resume_path(ckpt),
            "results": results_path(out, run_tag(cell.dataset, cell.level, "binary", cell.seed),
                                    model=cell.model),
            "history": stem + "_history.json", "start": stem + START_SUFFIX,
            "marker": stem + MARKER_SUFFIX, "status": stem + "_status.json"}


def state(results_root: str, cell: Cell) -> tuple:
    """(state, detail). state in {'complete', 'partial', 'fresh', 'needs_finalize', 'inconsistent'}."""
    p = paths(results_root, cell)
    have = {k: os.path.exists(v) for k, v in p.items() if k != "dir"}
    if have["marker"]:
        problems = verify_marker(p)
        return ("complete", "") if not problems else ("inconsistent", "; ".join(problems))
    if have["results"] and have["checkpoint"] and not have["resume"] and have["start"]:
        return ("needs_finalize", "trained by this harness; predictions hash and complete-marker "
                                  "not written yet (run with --finalize)")
    if have["results"]:
        return ("inconsistent", "results file without a complete-marker (not written by this "
                                "harness); refusing to skip or overwrite it")
    if have["checkpoint"] and have["resume"]:
        return "partial", "checkpoint and resume file present"
    if have["resume"]:
        return "partial", "resume file present (no best checkpoint yet)"
    if have["checkpoint"]:
        return ("inconsistent", "checkpoint without a resume file or results: an interrupted "
                                "cell that cannot continue exactly; refusing to guess")
    return "fresh", ""


def environment() -> dict:
    import torch
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "cudnn": (torch.backends.cudnn.version() if torch.backends.cudnn.is_available()
                     else None),
           "cudnn_benchmark": torch.backends.cudnn.benchmark,
           "cudnn_deterministic": torch.backends.cudnn.deterministic}
    try:
        from src.cloud.preflight import vendored_hashes
        env["vendored_sha256"] = vendored_hashes()
    except Exception as exc:                     # the marker must still be written
        env["vendored_sha256"] = f"unavailable: {type(exc).__name__}: {exc}"
    return env


def write_marker(results_root: str, cell: Cell, predictions_sha256: str | None,
                 predictions_note: str, account: str, gpu: str, extra_env: dict | None = None,
                 ) -> dict:
    p = paths(results_root, cell)
    for key in ("checkpoint", "results"):
        if not os.path.exists(p[key]):
            raise FileNotFoundError(f"cannot mark {cell.id} complete: no {p[key]}")
    if os.path.exists(p["resume"]):
        raise RuntimeError(f"cannot mark {cell.id} complete: its resume file still exists")
    marker = {
        "cell": cell.id, "account": account, "gpu": gpu, "dataset": cell.dataset,
        "model": cell.model, "level": cell.level, "seed": cell.seed,
        "checkpoint": {"file": os.path.basename(p["checkpoint"]),
                       "sha256": sha256_file(p["checkpoint"])},
        "results": {"file": os.path.basename(p["results"]),
                    "sha256": sha256_file(p["results"])},
        "start_record": ({"file": os.path.basename(p["start"]),
                          "sha256": sha256_file(p["start"])}
                         if os.path.exists(p["start"]) else None),
        "predictions_sha256": predictions_sha256, "predictions_note": predictions_note,
        "environment": {**environment(), **(extra_env or {})},
    }
    tmp = p["marker"] + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(marker, handle, indent=1)
    os.replace(tmp, p["marker"])
    return marker


def verify_marker(p: dict) -> list:
    """Problems, empty if the marker agrees with the files it names."""
    with open(p["marker"]) as handle:
        marker = json.load(handle)
    problems = []
    for key, name in (("checkpoint", "checkpoint"), ("results", "results")):
        if not os.path.exists(p[key]):
            problems.append(f"{name} file missing")
        elif sha256_file(p[key]) != marker[key]["sha256"]:
            problems.append(f"{name} hash differs from the marker")
    if os.path.exists(p["resume"]):
        problems.append("a resume file exists beside a complete cell")
    return problems
