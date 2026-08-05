# Part 2 Guide — 124AD0008
## Track A: Data, Ground Truth, and the Confound Control

Read `docs/00_MASTER_PLAN_V2.md` first. Your track changed the most, and it is
now on the critical path for everyone else.

---

## What changed for you

Under v1 you were building comparison numbers. Under the audit framing, **the
three baselines are the paper's subjects**, so your reproduction work is now the
spine of the results section rather than a table beside it.

You also inherit the paper's biggest scientific risk: the kinase confound. The
antiviral subset is no longer a case study — it is the control arm that decides
whether our headline result means anything.

---

## Priority 1 — Rebuild the antiviral subset (BLOCKING, ~1 day)

`data/processed/antiviral_clean.csv` currently contains **614 rows, all HIV-1
protease**. Four of five required targets are missing, so the control arm does
not exist.

```bash
# ~3GB, from https://www.bindingdb.org/bind/downloads
# put it at data/raw/BindingDB_All.tsv
python -m src.data.extract_antiviral --source data/raw/BindingDB_All.tsv
```

The rewritten script matches every naming variant BindingDB actually uses
(Mpro appears as "3C-like proteinase", "main protease", "nsp5"), converts
affinities to p-scale to match DAVIS's pKd, keeps the measurement type instead
of pooling IC50/Ki/Kd/EC50, and **refuses to write a file unless all five
targets are present**.

If it refuses, do not use `--allow-partial` to get past it. Open the target-name
column, find the spelling your BindingDB release uses, and add it to
`TARGET_PATTERNS`. Then add that spelling to `tests/test_antiviral.py`.

**Definition of done:** ≥20 distinct non-kinase targets, verified by

```bash
python -c "
import pandas as pd
from src.evaluation.target_family import confound_report
df = pd.read_csv('data/processed/antiviral_clean.csv')
print(confound_report(df['Target_ID'].tolist()))"
```

You want `control_is_usable: True`.

## Priority 2 — Re-fetch the ground truth (~30 min)

The committed files were built by a fetcher that collected UniProt's `Site`
catch-all — protease cleavage points and chromosomal breakpoints, scored as
correct binding-site answers. Roughly 136 DAVIS and 68 KIBA annotations.

```bash
python -m src.data.fetch_binding_sites --dataset davis
python -m src.data.fetch_binding_sites --dataset kiba
python -m src.data.ground_truth        # verify: dropped_feature_type should be > 0
```

The new fetcher records `type` on every record and writes a
`*_provenance.json` saying which accession each target resolved to and how.

## Priority 3 — KIBA gene-name mapping (~2h)

KIBA uses UniProt accessions, so `target_family.py` classifies **all 224 targets
as UNKNOWN** and the confound control cannot run on KIBA at all.

Build `data/kiba_uniprot_to_gene.json` mapping accession → gene symbol. The
provenance file from Priority 2 gives you most of it; UniProt's ID-mapping
endpoint gives the rest. Then confirm:

```bash
python -m src.evaluation.target_family   # KIBA n_kinase should no longer be 0
```

## Priority 4 — The 33 unmapped DAVIS targets (~2h)

DAVIS has 442 unique targets; ground truth covers 409. Check the provenance
file for `not_found` entries and resolve by hand.

## Priority 5 — Baselines as audit subjects (the long pole)

This is now your main deliverable. For each of DeepDTA, HyperAttentionDTI and
MolTrans you need **two** things, not one:

1. Accuracy on all four splits, both datasets, three seeds
2. **An adapter** exposing its attention so Track C can measure its
   explanations

For (2), write a subclass in `src/evaluation/model_registry.py`:

```python
@register("deepdta")
class DeepDTAAdapter(ExplainableDTIModel):
    provides_attention = False       # DeepDTA has no attention -- that's fine,
    citation = "Öztürk et al. 2018"  # it anchors the accuracy axis

    def predict(self, drug, protein): ...
    def explain(self, drug, protein):
        raise NotImplementedError("DeepDTA exposes no attention")
```

HyperAttentionDTI and MolTrans **do** have attention, so their adapters must
return real per-residue weights. Validate every adapter before use:

```python
from src.evaluation.model_registry import validate_adapter
print(validate_adapter(my_adapter, drug, protein, expected_length=len(sequence)))
```

That call catches the one bug that matters: an adapter returning weights of the
wrong length does not crash, it produces a wrong precision@k that looks
plausible.

## Definition of done

- [ ] Antiviral subset with all 5 targets, `control_is_usable: True`
- [ ] Ground truth re-fetched with `type` recorded, for both datasets
- [ ] KIBA accession → gene mapping, confound control runs on KIBA
- [ ] 33 unmapped DAVIS targets resolved or documented as unresolvable
- [ ] Splits built for both datasets, three seeds each, leakage check passing
- [ ] Three baseline adapters registered and passing `validate_adapter`
- [ ] Accuracy table: 3 models × 4 splits × 2 datasets × 3 seeds

## Common mistakes

- **Using `--allow-partial` to get past the antiviral check.** That flag exists
  for a deliberate partial run you are going to document, not for making a red
  message go away.
- **Building splits with one seed.** Track C needs three per cell or nothing can
  be called significant.
- **Assuming an adapter works because it returns an array.** Run
  `validate_adapter` with `expected_length`.

## Where to get help

If a BindingDB target name will not match, paste the raw string to the team
rather than loosening the regex until it does — a pattern loose enough to catch
everything will also catch non-antiviral proteins and contaminate the control
arm, which is the one place in this project where contamination cannot be
undone later.
