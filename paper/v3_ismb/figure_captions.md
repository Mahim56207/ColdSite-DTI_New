# Figure captions (v3)

<!-- v3 step 9, 2026-10-17. Adapted from paper/v2/figures/CAPTIONS.md (internal task IDs removed). Figures are built by `python scripts/plotting/make_all.py` from committed results; `--check`
rebuilt all twelve files byte-identically on 2026-10-17. Image files stay in paper/v2/figures/ (not copied, so they cannot drift from the script). Figures 1-4 are in the main text (files fig1, fig2, fig5, fig6);
Figures 3 and 4 are Supplementary Figures S1 and S2. -->

**Figure 1. The same recipe, three seeds, different verdicts.** Precision@10 of each training seed (points) on DAVIS at four levels of shift, against UniProt residues (top) and the KLIFS 85-residue pocket (bottom). Black bar: mean of the three seeds; dashed tick: that level's exact chance level. Seeds of one recipe that fall on both sides of chance are the instability reported here. The y-axes differ between panels.
<!-- src: src/data/klifs_pocket.py:54 = 85 -->

**Figure 2. Enrichment over chance with 95% bootstrap intervals.** Precision@10 divided by chance for every model and level, DAVIS (top) and KIBA (bottom), against UniProt residues (left) and the KLIFS pocket (right). Intervals are 95% percentile intervals from a two-way bootstrap that resamples both proteins and seeds (10,000 resamples); log scale; the dashed line is chance (an enrichment of one). Pocket-level enrichment is not controlled for residue conservation.
<!-- src: results/effects_v2_2d/enrichment.md:3 = 95, 10000 -->

**Figure 3. Masking the attended residues matters more than masking random ones.** Faithfulness delta (attended masking minus a size-matched random-masking control), mean over proteins, with 95% two-way bootstrap intervals (proteins and seeds resampled, 10,000 resamples). MolTrans is measured in its own token space and the others in residue space, so the three panels do not share a scale. DrugBAN is not drawn: its per-seed deltas are reported in the text but carry no interval, because its per-pair values were not recorded. No KIBA cell exists for XAttn-Ref.
<!-- src: results/effects_v2_2d/faithfulness_effects.md:3 = 95, 10000 -->

**Figure 4. What DAVIS's sequence leakage is worth to the accuracy anchor.** DeepDTA test AUROC (mean ± SD over three seeds) at cold-target and cold-pair, retrained on three training sets: the original split, "seqclean" (every seen-by-sequence target removed) and "seqmatched" (the leaked targets kept, at the size of "seqclean"). "seqmatched" minus "seqclean" isolates the leak; "original" minus "seqmatched" is the cost of fewer rows. Scored on all test rows and separately on leaked and unleaked targets.

**Supplementary Figure S1. Predictive accuracy per training seed.** Test AUROC of every cell (points, one per seed; bar = mean) on the uncorrected test sets. DeepDTA has no attention and anchors accuracy. KIBA has two levels (random, cold-drug) and no DrugBAN cells.

**Supplementary Figure S2. Accuracy does not predict localisation.** One point per attention-model cell and seed: test AUROC against enrichment over chance (UniProt, precision@10 divided by chance). For DAVIS cold-target and cold-pair, AUROC is scored on sequence-unseen targets, the same policy the explanation metrics use. Panel titles give Spearman's ρ with a 95% bootstrap interval over cells.
<!-- src: docs/PROTOCOL_AMENDMENT_v2.md:190 = 95 -->
