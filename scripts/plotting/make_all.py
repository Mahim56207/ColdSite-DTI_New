"""Build every manuscript figure into paper/v2/figures/.

    python scripts/plotting/make_all.py           # build
    python scripts/plotting/make_all.py --check   # build twice, fail unless the hashes are identical
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fig1_seed_instability  # noqa: E402
import fig2_enrichment_forest  # noqa: E402
import fig3_accuracy  # noqa: E402
import fig4_accuracy_vs_localization  # noqa: E402
import fig5_faithfulness  # noqa: E402
import fig6_leakage  # noqa: E402
from _style import rel  # noqa: E402

FIGURES = [fig1_seed_instability, fig2_enrichment_forest, fig3_accuracy,
           fig4_accuracy_vs_localization, fig5_faithfulness, fig6_leakage]


def build_all() -> dict:
    hashes = {}
    for module in FIGURES:
        for path in module.build():
            hashes[rel(path)] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return hashes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true", help="rebuild and compare hashes")
    args = ap.parse_args()
    first = build_all()
    for path, digest in first.items():
        print(f"{digest[:16]}  {path}")
    if args.check:
        second = build_all()
        diff = [p for p in first if first[p] != second[p]]
        print("byte-identical on rebuild:", "yes" if not diff else f"NO ({diff})")
        return 1 if diff else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
