"""
DAVIS target name -> (UniProt gene symbol, organism taxon).

Why this exists
---------------
DAVIS names its targets the way a kinase-panel assay sheet names them, not the
way UniProt does. 33 of 442 targets fail to resolve for three distinct reasons,
and all three are naming, not biology:

1. **Lab shorthand.** 'AMPK-alpha1', 'p38-alpha' and 'IKK-beta' are how
   pharmacologists say PRKAA1, MAPK14 and IKBKB. UniProt files them under the
   HGNC symbol and does not index the shorthand.

2. **A phospho marker with no mutation bracket.** `normalise_target_id` strips a
   trailing 'p' only when it has just removed a `(mutation)` bracket, which
   protects accessions ending in 'p' from being mangled. 'ABL1p' has the marker
   but no bracket, so nothing is stripped and UniProt is asked for a gene called
   "ABL1p". Handling it here rather than loosening that regex keeps the
   protection intact and makes the exception explicit.

3. **An organism in the brackets.** 'PFCDPK1(Pfalciparum)' and
   'PKNB(Mtuberculosis)' look like mutation brackets to the normaliser. Stripping
   them leaves a bare symbol that is then searched against *human* UniProt, where
   a malaria and a tuberculosis kinase will never be found.

Why symbols and not accessions
------------------------------
Every value below is a gene symbol handed to UniProt's own search, not a
hardcoded accession. That is deliberate. A wrong symbol returns no results and
the target stays in the not-found list where it is visible. A wrong accession
would return *some* protein's binding sites, attach them to this target, and
never raise anything -- the same silent-wrong-number failure shape as the
coordinate bug in Section 6 of the handover. The whole point of the ground truth
is that it is ground truth.

Verify, do not trust
--------------------
These mappings are standard nomenclature, but they are a human's reading of a
naming convention and belong on a spot-check list, not on a faith list. The
fetcher records `resolution: "alias"` plus the symbol it used in
`*_provenance.json` for exactly this reason:

    python -c "import json; p=json.load(open('data/davis_ground_truth_sites_provenance.json')); print({k:v['uniprot_accession'] for k,v in p.items() if v['resolution'].startswith('alias')})"

Check that list against uniprot.org before the numbers go in the paper. Anything
you cannot confirm should be moved to `data/davis_target_overrides.json` and
resolved by hand instead (see `src/data/resolve_unmapped.py`).
"""

HUMAN = 9606
P_FALCIPARUM = 5833
M_TUBERCULOSIS = 1773

# DAVIS target ID -> (gene symbol to search, organism taxon id)
DAVIS_TARGET_ALIASES = {
    # -- phospho marker without a mutation bracket -------------------------
    "ABL1p": ("ABL1", HUMAN),

    # -- lab shorthand for a subunit ---------------------------------------
    "AMPK-alpha1": ("PRKAA1", HUMAN),
    "AMPK-alpha2": ("PRKAA2", HUMAN),
    "PKAC-alpha": ("PRKACA", HUMAN),
    "PKAC-beta": ("PRKACB", HUMAN),

    # -- lab shorthand for the whole kinase --------------------------------
    "DLK": ("MAP3K12", HUMAN),          # dual leucine zipper kinase
    "IKK-alpha": ("CHUK", HUMAN),
    "IKK-beta": ("IKBKB", HUMAN),
    "IKK-epsilon": ("IKBKE", HUMAN),
    "MRCKA": ("CDC42BPA", HUMAN),       # myotonic dystrophy-related Cdc42-binding kinase alpha
    "MRCKB": ("CDC42BPB", HUMAN),
    "PFTAIRE2": ("CDK15", HUMAN),       # PFTAIRE1 is CDK14, PFTAIRE2 is CDK15
    # PRP4K, not PRPF4B. HGNC renamed this gene; "PRPF4B" survives only as a
    # synonym, and UniProt's gene search matches the current symbol, so the
    # obvious-looking alias returned nothing at all. Confirmed against Q13523,
    # "Serine/threonine-protein kinase PRP4 homolog", whose synonym list is
    # KIAA0536 / PRP4 / PRP4B / PRP4H / PRPF4K.
    "PRP4": ("PRP4K", HUMAN),
    "S6K1": ("RPS6KB1", HUMAN),

    # NOTE: 'OSR1' is genuinely ambiguous. It is also the symbol for the
    # transcription factor "odd-skipped related 1", which is not a kinase at
    # all. In a kinase panel it is the oxidative-stress-responsive kinase,
    # OXSR1. This one is worth confirming by hand before it ships.
    "OSR1": ("OXSR1", HUMAN),

    # 'MST1' is the same kind of collision, and it resolved to the wrong protein
    # in a real fetch before this entry existed. UniProt files MST1 as
    # macrophage-stimulating 1, a hepatocyte growth factor-like protein that is
    # not a kinase at all. The kinase every kinase-panel means by "MST1" is
    # STK4. The wrong entry came back with zero binding sites, so the target
    # silently contributed nothing to any average rather than erroring.
    #
    # Note MST1R is a different thing again and resolves correctly on its own:
    # it is the RON receptor tyrosine kinase, so it must NOT be aliased here.
    "MST1": ("STK4", HUMAN),

    # -- p38 MAP kinase family, named by Greek letter in the assay ---------
    "p38-alpha": ("MAPK14", HUMAN),
    "p38-beta": ("MAPK11", HUMAN),
    "p38-gamma": ("MAPK12", HUMAN),
    "p38-delta": ("MAPK13", HUMAN),

    # -- renamed since the DAVIS panel was published -----------------------
    "PIK4CB": ("PI4KB", HUMAN),
    "PIP5K2C": ("PIP4K2C", HUMAN),

    # -- a complex; the kinase subunit is what carries the ATP pocket ------
    # Both entries resolve to CDK4 and therefore inherit an identical site
    # list. That is the same wild-type approximation already applied to every
    # ABL1 point mutant, and it belongs in the Methods section for the same
    # reason: two DAVIS rows that differ biologically get one annotation.
    "CDK4-cyclinD1": ("CDK4", HUMAN),
    "CDK4-cyclinD3": ("CDK4", HUMAN),

    # -- non-human: the bracket is an organism, not a mutation -------------
    "PFCDPK1(Pfalciparum)": ("CDPK1", P_FALCIPARUM),
    # "PfPK5" is only an alternative *protein* name. The gene symbol is CRK2.
    # Note the entry sits under taxon 36329 (isolate 3D7), not the species id
    # 5833 -- which is exactly why the organism_id -> taxonomy_id change in the
    # fetcher was needed: taxonomy_id matches the subtree, organism_id does not.
    # Two reviewed strain entries exist (P61075 3D7, Q07785 K1) with identical
    # 288-residue sequences, so this target is also pinned to P61075 in
    # data/davis_target_overrides.json to keep the choice reproducible rather
    # than dependent on UniProt's result ordering.
    "PFPK5(Pfalciparum)": ("CRK2", P_FALCIPARUM),
    "PKNB(Mtuberculosis)": ("pknB", M_TUBERCULOSIS),
}


def resolve_alias(target_id: str):
    """DAVIS target ID -> (gene symbol, taxon) or None if there is no alias.

    Exact match only. Fuzzy matching here would be a way to silently attach one
    protein's binding sites to another target, which is the one error this
    module is built to avoid.
    """
    return DAVIS_TARGET_ALIASES.get(str(target_id).strip())


def alias_count() -> int:
    return len(DAVIS_TARGET_ALIASES)
