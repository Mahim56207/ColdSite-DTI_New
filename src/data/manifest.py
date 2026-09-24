"""
SHA-256 manifest of the files every result rests on: the split CSVs, the ground truths, and the
non-kinase panel.

    python -m src.data.manifest --write     # (re)create data/splits/MANIFEST.json
    python -m src.data.manifest --check     # exit 1 if any file differs, is missing, or is new

Why this exists. `data/splits/` is gitignored, so git cannot say whether the splits on this
machine are the ones a cell was trained on, and a cloud run that regenerates them would not
notice a difference. The manifest is the record a pre-flight can check against.

What it does and does not prove. It fixes the files *as they were when it was written*. It
cannot show that they are the files the 84 existing cells trained on -- nothing recorded that at
the time. (Corroboration exists but is indirect: each `_results.json` records `n_train_rows`,
which matched its split's row count in every cell checked.) From now on, the manifest turns a
silent change to any covered file into a failing test.

Deliberately not covered: `data/klifs_structures.json` and `data/klifs_ligands.json`, raw KLIFS
API caches that are gitignored and re-fetchable (the per-pair provenance that matters is in
`data/davis_drug_sites_report.json`, which is covered).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

MANIFEST_PATH = os.path.join("data", "splits", "MANIFEST.json")

# (section, directory, suffixes, recurse)
SECTIONS = (
    ("splits", os.path.join("data", "splits"), (".csv",), True),
    ("ground_truth", "data", (".json",), False),
    ("processed", os.path.join("data", "processed"), (".csv",), False),
)
EXCLUDED = {
    os.path.join("data", "klifs_structures.json"): "raw KLIFS API cache; gitignored, re-fetchable",
    os.path.join("data", "klifs_ligands.json"): "raw KLIFS API cache; gitignored, re-fetchable",
}


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _discover(root: str = ".") -> dict:
    found = {}
    for section, directory, suffixes, recurse in SECTIONS:
        base = os.path.join(root, directory)
        if not os.path.isdir(base):
            continue
        walker = os.walk(base) if recurse else [(base, [], sorted(os.listdir(base)))]
        for folder, _dirs, names in walker:
            for name in sorted(names):
                path = os.path.join(folder, name)
                relative = os.path.relpath(path, root)
                if (name.endswith(suffixes) and os.path.isfile(path)
                        and relative not in EXCLUDED):
                    found[relative.replace(os.sep, "/")] = section
    return dict(sorted(found.items()))


def build(root: str = ".") -> dict:
    files = {}
    for relative, section in _discover(root).items():
        path = os.path.join(root, relative)
        files[relative] = {"section": section, "sha256": sha256_of(path),
                           "bytes": os.path.getsize(path)}
    return {
        "note": ("SHA-256 of the local files at the time this was written (remediation task "
                 "T02). It cannot show these are the files the pre-existing cells trained on; "
                 "see src/data/manifest.py."),
        "excluded": {path.replace(os.sep, "/"): why for path, why in EXCLUDED.items()},
        "files": files,
    }


def compare(manifest: dict, root: str = ".") -> dict:
    """{'changed': [...], 'missing': [...], 'unlisted': [...]} against the files on disk."""
    recorded = manifest["files"]
    present = _discover(root)
    changed, missing = [], []
    for relative, entry in recorded.items():
        path = os.path.join(root, relative)
        if not os.path.isfile(path):
            missing.append(relative)
        elif sha256_of(path) != entry["sha256"]:
            changed.append(relative)
    unlisted = sorted(set(present) - set(recorded))
    return {"changed": changed, "missing": missing, "unlisted": unlisted}


def load(root: str = ".") -> dict:
    with open(os.path.join(root, MANIFEST_PATH)) as handle:
        return json.load(handle)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if args.write:
        if os.path.exists(MANIFEST_PATH):
            raise SystemExit(
                f"{MANIFEST_PATH} exists. Overwriting it would re-bless whatever is on disk; "
                f"delete it deliberately if that is the intent.")
        manifest = build()
        with open(MANIFEST_PATH, "w") as handle:
            json.dump(manifest, handle, indent=1, sort_keys=False)
            handle.write("\n")
        counts = {}
        for entry in manifest["files"].values():
            counts[entry["section"]] = counts.get(entry["section"], 0) + 1
        print(f"wrote {MANIFEST_PATH}: {len(manifest['files'])} files {counts}")
        return 0

    result = compare(load())
    bad = {kind: names for kind, names in result.items() if names}
    for kind, names in bad.items():
        print(f"{kind}: {len(names)}")
        for name in names[:10]:
            print(f"   {name}")
    print("manifest OK" if not bad else "manifest MISMATCH")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
