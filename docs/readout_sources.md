# Readout sources (T07) — closed by user decision with placeholder figure references

Plan rule (`docs/REMEDIATION_PLAN.md`, T07 / P16): for **every** model the primary readout is the map its own paper displays; the user
supplies the figure or section. Nothing below is a paper figure reference unless it is marked *user-supplied*. As of 2026-09-25 **none
has been supplied**. Amendment §3 row E8 requires this document plus its family (cells, m) to be sealed as addendum A3 before any
readout variant not already run is scored.

| model | readout scored in the primary families P1 / P2 (`"published"` in `readout_comparison.csv`) | what the repo says the paper displays | user-supplied figure / section | status |
|---|---|---|---|---|
| MolTrans | protein-encoder self-attention, last layer, head mean (`baseline_adapters.py:289–331`) | the drug × protein interaction map — `src/evaluation/readout_variants.py` docstring: "the interaction map MolTrans's own paper visualises (Fig. 3)"; `baseline_adapters.py:485` quotes the paper | **missing** (the "Fig. 3" in the docstring is the previous author's note, not verified against the paper here) | **the primary readout would change** to the registered `moltrans_interaction`; that is a new family (E8), not a rewrite of P1 |
| HyperAttentionDTI | channel-mean, centre-placed attention (`baseline_adapters.py:160–243`) | not stated in the repo | **missing** | cannot decide |
| DrugBAN | head-mean, atom-sum marginal over the bilinear map (`drugban_adapter.py:14–33`) | the 2-D atom × residue map ("which is what the paper visualises", `drugban_adapter.py:14–18`); the per-residue reduction is the audit's own | **missing** | cannot decide; the reduction to one weight per residue is the audit's choice either way, so it must be declared |
| ColdSite-DTI | drug-conditioned cross-attention | authors' own model, no external paper | n/a — declare in Methods | primary by construction |
| DeepDTA | none | no attention; accuracy anchor only | n/a | not audited |

## Readouts already run (reuse; do not re-implement)

`~/ColdSite-results/readouts/readout_comparison.csv` (64 rows: ColdSite-DTI published/selfattn, HyperAttentionDTI
published/maxchannel/receptive, MolTrans published/maxhead/firstlayer; UniProt and KLIFS × four levels), plus
`results/readouts_moltrans_interaction/` and `results/readouts_drugban_davis/`.

Registered but **never run**: `drugban_maxhead`, `moltrans_interaction_sum` (`readout_variants.py`).

## What the user must supply to finish T07

One line per audited model (MolTrans, HyperAttentionDTI, DrugBAN): the paper, and the figure or section that shows its binding-site /
interaction map, so the readout can be confirmed or switched and the verdict reported under primary vs alternatives.

## User decision (2026-09-25) — T07 closed with placeholders

The user instructed: use the text placeholders `[MOLTRANS_FIG_X]`, `[HYPERATTENTION_FIG_X]` and `[DRUGBAN_FIG_X]` in place of
the figure references, do not wait for the papers' figures, and mark T07 DONE.

| model | figure reference (placeholder, to be replaced by the paper's figure/section) |
|---|---|
| MolTrans | `[MOLTRANS_FIG_X]` |
| HyperAttentionDTI | `[HYPERATTENTION_FIG_X]` |
| DrugBAN | `[DRUGBAN_FIG_X]` |

What this closure does **not** do (recorded so nobody reads it as done):

* No readout was confirmed against a paper figure, and no readout was switched. The primary readouts remain those scored in
  P1 / P2 (table above). The manuscript must describe them as "the readout scored in the primary analysis", not as "the map
  each paper displays", until the placeholders are replaced by verified references.
* No "verdict under primary vs alternative readouts" analysis was run under this task; the existing readout comparison
  (`readout_comparison.csv`, listed above) is the only readout-sensitivity evidence.
* E8 (amendment §3) stays unsealed: addendum A3 was not written, so no new readout variant may be scored.
