"""Manuscript figure scripts: the leakage-table parser and deterministic output."""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "plotting"))

import fig6_leakage  # noqa: E402


def test_leakage_parser_reads_every_arm_and_subset():
    data = fig6_leakage.parse(fig6_leakage.SRC.read_text())
    assert set(data) == {"cold-target", "cold-pair"}
    for level in data.values():
        assert set(level) == {"original", "seqmatched", "seqclean"}
        assert all(len(v) == 3 for v in level.values())


def test_leakage_parser_on_a_planted_table():
    text = ("## cold-target\n\n| arm | train rows | all rows | leaked targets | unleaked targets |\n"
            "|---|---|---|---|---|\n| original | 10 | 0.900 ± 0.010 | 0.950 ± 0.020 | 0.850 ± 0.030 |\n"
            "\n| seed | all | leaked | unleaked |\n| original | 10 | 0.1 ± 0.1 | 0.1 ± 0.1 | 0.1 ± 0.1 |\n")
    assert fig6_leakage.parse(text) == {"cold-target": {"original": [(0.9, 0.01), (0.95, 0.02), (0.85, 0.03)]}}


def test_a_figure_rebuilds_byte_identically(tmp_path, monkeypatch):
    import _style
    monkeypatch.setattr(_style, "OUT", tmp_path)
    digests = []
    for _ in range(2):
        paths = fig6_leakage.build()
        digests.append([hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths])
    assert digests[0] == digests[1]
