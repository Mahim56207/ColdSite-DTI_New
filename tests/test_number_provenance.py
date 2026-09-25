"""Planted cases for scripts/check_number_provenance.py (plan T21/P29): each rule must catch the
failure it exists for, not only pass on the real manuscript."""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "check_number_provenance", os.path.join(ROOT, "scripts", "check_number_provenance.py"))
cnp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cnp)


@pytest.fixture
def src(tmp_path):
    (tmp_path / "t.csv").write_text("model,level,auroc,cell\nA,random,0.92412,0.5 ± 0.1\nB,random,0.7,x\nB,random,0.8,y\n")
    (tmp_path / "t.md").write_text("line one\n12 of 22 cells disagree; -0.019 mean\n")
    (tmp_path / "t.json").write_text('{"a": {"b": [0.1, 0.25]}}')
    return tmp_path


def run(tmp_path, text):
    f = tmp_path / "doc.md"
    f.write_text(text)
    return cnp.check_file(str(f))["failures"]


def test_matching_csv_tag_passes(src):
    assert run(src, f"AUROC was 0.924. <!-- src: {src}/t.csv#model=A&level=random->auroc = 0.924 -->\n") == []


def test_wrong_value_fails(src):
    fails = run(src, f"AUROC was 0.931. <!-- src: {src}/t.csv#model=A&level=random->auroc = 0.931 -->\n")
    assert any("stated 0.931" in f for f in fails)


def test_value_rounds_at_the_stated_precision_only(src):
    assert run(src, f"AUROC 0.92. <!-- src: {src}/t.csv#model=A&level=random->auroc = 0.92 -->\n") == []
    assert run(src, f"AUROC 0.9242. <!-- src: {src}/t.csv#model=A&level=random->auroc = 0.9242 -->\n") != []


def test_untagged_number_fails(src):
    fails = run(src, "We saw 17 cells.\n")
    assert any("untagged number '17'" in f for f in fails)


def test_a_tag_in_another_paragraph_does_not_cover_a_number(src):
    text = f"AUROC 0.924.\n\nOther. <!-- src: {src}/t.csv#model=A&level=random->auroc = 0.924 -->\n"
    assert any("untagged number '0.924'" in f for f in run(src, text))


def test_ambiguous_or_missing_row_fails(src):
    assert any("2 rows match" in f for f in run(src, f"0.7 <!-- src: {src}/t.csv#model=B->auroc = 0.7 -->\n"))
    assert any("0 rows match" in f for f in run(src, f"0.7 <!-- src: {src}/t.csv#model=Z->auroc = 0.7 -->\n"))


def test_missing_file_fails(src):
    assert any("file not found" in f for f in run(src, "0.7 <!-- src: results/no_such_file.csv#a=b->c = 0.7 -->\n"))


def test_formatted_mean_sd_cell_uses_its_leading_number(src):
    assert run(src, f"0.5 <!-- src: {src}/t.csv#model=A->cell = 0.5 -->\n") == []


def test_text_line_tag(src):
    assert run(src, f"12 of 22, and −0.019. <!-- src: {src}/t.md:2 = 12, 22, -0.019 -->\n") == []
    assert any("not on" in f for f in run(src, f"13 <!-- src: {src}/t.md:2 = 13 -->\n"))


def test_json_tag(src):
    assert run(src, f"0.25 <!-- src: {src}/t.json#$.a.b[1] = 0.25 -->\n") == []
    assert run(src, f"0.4 <!-- src: {src}/t.json#$.a.b[1] = 0.4 -->\n") != []


def test_derived_value_needs_sourced_operands_and_correct_arithmetic(src):
    tag = f"<!-- src: {src}/t.csv#model=A&level=random->auroc = 0.924 -->"
    assert run(src, f"0.924 is 1.85 times 0.5. {tag} <!-- src: {src}/t.csv#model=A->cell = 0.5 --> "
                    "<!-- src: derived: 0.924 / 0.5 = 1.85 -->\n") == []
    wrong = run(src, f"0.924 and 2.0. {tag} <!-- src: derived: 0.924 / 0.5 = 2.0 -->\n")
    assert any("not a sourced value" in f for f in wrong)          # 0.5 is not sourced in this block
    assert any("stated 2.0" in f for f in wrong)
    assert any("characters other" in f for f in run(src, "3 <!-- src: derived: __import__('os') = 3 -->\n"))


def test_exemptions(src):
    text = ("MolTrans (2021) reports precision@10 on T4 GPUs for P1 and S3-D (Table 2, §4, Figure 3); "
            "see `run_all.py:62`.\n\n1. a numbered item\n\n# 5 Heading\n")
    assert run(src, text) == []


def test_ranges_and_signs_parse(src):
    assert cnp.prose_numbers("0.030–0.038 and 0.030-0.038 and [−0.276, 0.322] and 1,000") == [
        "0.030", "0.038", "0.030", "0.038", "-0.276", "0.322", "1000"]


def test_blockquote_list_markers_are_exempt(src):
    assert run(src, "> **Box 1.**\n>\n> 1. first\n> 2. second\n") == []


def test_the_manuscript_passes():
    external = os.path.expanduser("~/ColdSite-results/readouts/readout_comparison.csv")
    if not os.path.exists(external):
        pytest.skip("the manuscript cites a checkpoint-side file outside the repository")
    assert cnp.main([]) == 0
