"""Restore a previous notebook version's output (attached as an input dataset).

Only the files of the cells this account is planned to train are copied, into the layout
`<results_root>/<dataset>_binary/`, and never over a file already there. Files are found by
name anywhere under `input_root` (Kaggle does not always mount a dataset where its name says:
2026-09-12, `ledger`/CLAUDE.md), and two different files with one name are an error, not a
choice. A complete cell's marker is verified against the restored files; a mismatch removes
the copies and reports it, so a corrupt or partial restore cannot pass for a finished cell.
"""
from __future__ import annotations

import os
import shutil

from src.cloud import markers
from src.cloud.config import Cell

WANTED = ("checkpoint", "resume", "results", "history", "start", "marker", "status")


def index_inputs(input_root: str) -> dict:
    """basename -> [paths]"""
    found: dict = {}
    for folder, _dirs, names in os.walk(input_root):
        for name in names:
            found.setdefault(name, []).append(os.path.join(folder, name))
    return found


def restore(cells: list, results_root: str, input_root: str) -> dict:
    """Returns {'copied': [...], 'present': [...], 'removed': [...], 'problems': [...]}."""
    report = {"copied": [], "present": [], "removed": [], "problems": []}
    if not os.path.isdir(input_root):
        report["problems"].append(f"restore source is not a directory: {input_root}")
        return report
    index = index_inputs(input_root)
    for cell in cells:
        p = markers.paths(results_root, cell)
        copied_here = []
        for key in WANTED:
            name = os.path.basename(p[key])
            hits = index.get(name, [])
            if not hits:
                continue
            if len({markers.sha256_file(h) for h in hits}) > 1:
                report["problems"].append(f"{name}: {len(hits)} different files of that name "
                                          f"under {input_root}")
                continue
            if os.path.exists(p[key]):
                report["present"].append(p[key])
                continue
            os.makedirs(p["dir"], exist_ok=True)
            shutil.copy2(hits[0], p[key])
            copied_here.append(p[key])
            report["copied"].append(p[key])
        if os.path.exists(p["marker"]):
            bad = markers.verify_marker(p)
            if bad:
                report["problems"].append(f"{cell.id}: restored marker disagrees with its files "
                                          f"({'; '.join(bad)})")
                for path in copied_here:
                    os.remove(path)
                    report["removed"].append(path)
    return report
