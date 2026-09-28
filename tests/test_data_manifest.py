"""T02: the split and ground-truth files must hash to what the manifest recorded.

`data/splits/` is gitignored, so git cannot notice a changed split. The manifest can.
"""
import json
import os

import pytest

from src.data import manifest as manifest_module
from src.data.manifest import MANIFEST_PATH, build, compare, load, sha256_of


@pytest.fixture(scope="module")
def recorded():
    if not os.path.exists(MANIFEST_PATH):
        pytest.fail(f"{MANIFEST_PATH} is missing. Create it with "
                    f"`python -m src.data.manifest --write` on a machine holding the "
                    f"original splits, and commit it.")
    return load()


def test_the_manifest_is_committed_not_ignored():
    import subprocess
    out = subprocess.run(["git", "check-ignore", MANIFEST_PATH], capture_output=True, text=True)
    assert out.returncode == 1, f"{MANIFEST_PATH} is gitignored: {out.stdout.strip()}"


def test_the_manifest_has_the_expected_shape(recorded):
    files = recorded["files"]
    assert files, "empty manifest"
    for path, entry in files.items():
        assert len(entry["sha256"]) == 64 and entry["bytes"] > 0, path
        assert entry["section"] in {"splits", "ground_truth", "processed"}
    sections = {e["section"] for e in files.values()}
    assert sections == {"splits", "ground_truth", "processed"}


def test_it_covers_every_split_level_that_exists(recorded):
    files = recorded["files"]
    for dataset in ("davis", "kiba"):
        for level in ("random", "cold_drug", "cold_target", "cold_pair"):
            for part in ("train", "valid", "test"):
                assert f"data/splits/{dataset}/{level}/{part}.csv" in files
    # the leakage arms trained in leakage_davis/ are covered too
    assert "data/splits/davis/cold_target_seqclean/test.csv" in files


def test_the_raw_api_caches_are_excluded_on_purpose(recorded):
    assert "data/klifs_structures.json" in recorded["excluded"]
    assert "data/klifs_structures.json" not in recorded["files"]


def test_ground_truth_files_match_their_recorded_hashes(recorded):
    """These are tracked by git, so they exist in any clone and must match."""
    result = compare(recorded)
    ground_truth = {p for p, e in recorded["files"].items() if e["section"] != "splits"}
    assert not [p for p in result["changed"] if p in ground_truth], result["changed"]
    assert not [p for p in result["missing"] if p in ground_truth], result["missing"]


def test_split_files_match_their_recorded_hashes(recorded):
    """Splits are gitignored. Where they are present they must match; where the directory
    is absent (a fresh clone that has not run build_splits) there is nothing to compare."""
    if not os.path.isdir("data/splits/davis"):
        pytest.skip("data/splits not present in this checkout")
    result = compare(recorded)
    assert not result["changed"], f"changed since the manifest was written: {result['changed']}"
    assert not result["missing"], f"listed but missing: {result['missing']}"


def test_no_split_or_ground_truth_file_is_missing_from_the_manifest(recorded):
    """A new split or ground truth must be added to the manifest, or it is unguarded."""
    if not os.path.isdir("data/splits/davis"):
        pytest.skip("data/splits not present in this checkout")
    assert compare(recorded)["unlisted"] == []


def test_compare_detects_a_changed_a_missing_and_an_unlisted_file(tmp_path):
    (tmp_path / "data" / "splits" / "toy" / "random").mkdir(parents=True)
    keep = tmp_path / "data" / "splits" / "toy" / "random" / "train.csv"
    gone = tmp_path / "data" / "splits" / "toy" / "random" / "test.csv"
    keep.write_text("a,b\n1,2\n")
    gone.write_text("a,b\n3,4\n")
    made = build(str(tmp_path))
    assert compare(made, str(tmp_path)) == {"changed": [], "missing": [], "unlisted": []}

    keep.write_text("a,b\n1,999\n")                              # one value edited
    gone.unlink()                                                 # one file removed
    (tmp_path / "data" / "splits" / "toy" / "random" / "valid.csv").write_text("a\n")
    result = compare(made, str(tmp_path))
    assert result["changed"] == ["data/splits/toy/random/train.csv"]
    assert result["missing"] == ["data/splits/toy/random/test.csv"]
    assert result["unlisted"] == ["data/splits/toy/random/valid.csv"]


def test_write_refuses_to_re_bless_an_existing_manifest(tmp_path, monkeypatch):
    """Regenerating would accept whatever is on disk, which defeats the point."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "splits").mkdir(parents=True)
    (tmp_path / MANIFEST_PATH).write_text("{}")
    with pytest.raises(SystemExit, match="exists"):
        manifest_module.main(["--write"])


def test_the_hash_is_the_files_sha256(tmp_path):
    path = tmp_path / "x"
    path.write_bytes(b"abc")
    assert sha256_of(str(path)) == \
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
