"""
Target-family stratification — the audit's main confound control.

The problem this exists to solve
-------------------------------
DAVIS and KIBA are kinase panels. A "cold-target" kinase is not an unfamiliar
protein: it shares the ATP-binding pocket architecture of the several hundred
kinases already in training, and UniProt annotates that pocket in nearly all of
them. A model can therefore score well on cold-target precision@k by learning
"ATP pockets look roughly like this" -- which is exactly the generic
pattern-matching the paper claims to be testing against.

Left uncontrolled, the headline ladder may be measuring kinase-family
similarity rather than explanation robustness, and that is the objection most
likely to sink the paper in review.

The control
-----------
Stratify every evaluated target into KINASE / NON_KINASE / UNKNOWN and report
the ladder separately for each. The antiviral subset (HIV-1 protease, HIV-1 RT,
SARS-CoV-2 Mpro and RdRp, influenza neuraminidase) is entirely non-kinase, which
is what turns it from a decorative case study into the control arm.

Interpreting the result
-----------------------
  fidelity holds on kinases, collapses on non-kinases
      -> the kinase result was family similarity. Report both; this IS the
         finding, and it is a stronger one than the original framing.
  fidelity degrades similarly on both
      -> the degradation is real and not a family artefact. The claim survives.
  fidelity holds on both
      -> an honest negative result. Report it.

All three outcomes are publishable. What is not publishable is reporting the
kinase number alone and hoping nobody asks.
"""
import re

KINASE = "kinase"
NON_KINASE = "non_kinase"
UNKNOWN = "unknown"

# DAVIS/KIBA target IDs are gene symbols. Kinase gene families follow strong
# naming conventions, which covers most of the panel without a lookup table.
_KINASE_NAME = re.compile(
    r"^(ABL|AKT|ALK|AMPK|AURK|AXL|BRAF|BTK|CAMK|CDK|CHEK|CSNK|CSF1R|DDR|"
    r"EGFR|EPH|ERBB|ERK|FGFR|FLT|FYN|GSK3|HCK|IGF1R|IKB|INSR|IRAK|ITK|JAK|"
    r"KDR|KIT|LCK|LYN|MAP\d?K|MAPK|MELK|MET|MTOR|MYLK|NEK|NTRK|PAK|PDGFR|"
    r"PDPK|PIK3|PIM|PKN|PLK|PRKA|PRKC|PRKD|PTK|RAF1|RET|ROCK|ROS1|RPS6K|"
    r"SGK|SRC|STK|SYK|TEK|TGFBR|TIE|TNK|TYK|VEGFR|WEE1|YES1|ZAP70)",
    re.IGNORECASE)

_KINASE_WORD = re.compile(r"kinase|\bCDK\b|\bMAPK\b", re.IGNORECASE)

# The antiviral control arm. None of these are kinases.
_NON_KINASE_WORD = re.compile(
    r"protease|proteinase|reverse transcriptase|neuraminidase|polymerase|"
    r"nsp5|nsp12|rdrp|mpro|3c[- ]?like|integrase|hydrolase|isomerase|"
    r"carbonic anhydrase|secretase",
    re.IGNORECASE)


def classify_target(target_id: str, description: str = "") -> str:
    """Target identifier -> KINASE / NON_KINASE / UNKNOWN.

    Non-kinase patterns are checked FIRST. Several viral polymerases and
    proteases carry gene-symbol prefixes that collide with kinase families, and
    a false KINASE label would contaminate the control arm -- the one place in
    the audit where contamination is unrecoverable.
    """
    text = f"{target_id} {description}".strip()
    if not text:
        return UNKNOWN

    if _NON_KINASE_WORD.search(text):
        return NON_KINASE
    if _KINASE_WORD.search(text):
        return KINASE

    base = str(target_id).split("(")[0].strip()
    if _KINASE_NAME.match(base):
        return KINASE
    return UNKNOWN


def stratify(target_ids, descriptions: dict = None) -> dict:
    """Group target IDs by family. Returns {family: [target_id, ...]}."""
    descriptions = descriptions or {}
    groups = {KINASE: [], NON_KINASE: [], UNKNOWN: []}
    for target_id in target_ids:
        groups[classify_target(target_id, descriptions.get(target_id, ""))].append(
            target_id)
    return groups


def stratified_indices(target_ids) -> dict:
    """{family: [positions in the input list]} -- for slicing aligned results."""
    out = {KINASE: [], NON_KINASE: [], UNKNOWN: []}
    for i, target_id in enumerate(target_ids):
        out[classify_target(target_id)].append(i)
    return out


def confound_report(target_ids) -> dict:
    """Is there enough non-kinase data for the control to mean anything?

    `control_is_usable` is the gate. Below ~20 distinct non-kinase targets the
    comparison has no power, and the honest move is to state the limitation
    rather than report a stratified number a reviewer will not believe.
    """
    groups = stratify(target_ids)
    total = len(target_ids)
    n_non_kinase = len(set(groups[NON_KINASE]))

    return {
        "n_total": total,
        "n_kinase": len(groups[KINASE]),
        "n_non_kinase": len(groups[NON_KINASE]),
        "n_unknown": len(groups[UNKNOWN]),
        "distinct_non_kinase": n_non_kinase,
        "kinase_fraction": len(groups[KINASE]) / total if total else float("nan"),
        "control_is_usable": n_non_kinase >= 20,
        "note": (
            "Enough non-kinase targets for a stratified comparison."
            if n_non_kinase >= 20 else
            "Too few non-kinase targets to stratify. Either enlarge the "
            "antiviral/BindingDB subset or state the kinase confound as an "
            "explicit limitation in the Discussion -- do not report the "
            "unstratified ladder as if the confound were absent."
        ),
    }


if __name__ == "__main__":
    import json

    for path in ("data/davis_ground_truth_sites.json",
                 "data/kiba_ground_truth_sites.json"):
        with open(path) as f:
            targets = list(json.load(f))
        report = confound_report(targets)
        print(f"\n{path}")
        for key, value in report.items():
            print(f"  {key:24s} {value}")
