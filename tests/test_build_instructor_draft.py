"""The committed INSTRUCTOR_DRAFT.md is the concatenation of the paper/v2 section files."""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_instructor_draft as bid  # noqa: E402


def test_committed_draft_is_up_to_date():
    assert bid.main(["--check"]) == 0


def test_structure_and_figures():
    text = bid.build()
    assert len(re.findall(r"^# ", text, flags=re.M)) == 1          # the title only
    order = ["## Abstract", "## Key Points", "## 1 Introduction", "## 2 Methods", "## 3 Results",
             "## 4 Discussion", "## Figures", "## Supplementary Methods"]
    positions = [text.index(h) for h in order]
    assert positions == sorted(positions)
    images = re.findall(r"!\[Figure (\d+)\]\((figures/\w+\.png)\)", text)
    assert [n for n, _ in images] == ["1", "2", "3", "4", "5", "6"]
    for _, path in images:
        assert os.path.isfile(os.path.join(bid.V2, path))
    assert "draft, 2026" not in text                               # drafting notes dropped
    assert text.count("<!-- src:") == sum(
        bid.read(f).count("<!-- src:") for f in ["abstract.md", *bid.SECTIONS, bid.SUPPLEMENT]
    ) + len(re.findall(r"<!-- src:", text[text.index("## Figures"):text.index("## Supplementary Methods")]))
