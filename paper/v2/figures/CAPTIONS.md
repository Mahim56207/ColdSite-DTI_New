# Figures — captions and sources (T22, 2026-09-25)

Built by `python scripts/plotting/make_all.py` from files committed under `results/`; no value is typed
by hand. `--check` rebuilds and confirms byte-identical PNG/PDF output. PDF for the manuscript, PNG for
drafts. XAttn-Ref is the in-house reference model (`coldsite_dti` in file names).

**Figure 1 — `fig1_seed_instability`. The same recipe, three seeds, different verdicts.**
precision@10 of each training seed (points) on DAVIS, at four levels of shift, against UniProt residues
(top) and the KLIFS 85-residue pocket (bottom). Black bar: mean of the three seeds; dashed tick: that
level's exact chance level. Seeds of one recipe that fall on both sides of chance are the instability
the paper reports. y-axes differ between panels. *Source:* `results/effects_v2/seed_spread.csv`
(families P1 and S3-D, column `per_seed_precision`, `chance_exact`). *Script:* `scripts/plotting/fig1_seed_instability.py`.
<!-- src: src/data/klifs_pocket.py:54 = 85 -->

**Figure 2 — `fig2_enrichment_forest`. Enrichment over chance with 95% bootstrap intervals.**
precision@10 ÷ chance for every model and level, DAVIS (top) and KIBA (bottom), against UniProt residues
(left) and the KLIFS pocket (right). Intervals resample targets (10,000 resamples); log scale; dashed line
= chance (an enrichment of one). Pocket-level enrichment is not controlled for residue conservation (T06 deferred).
*Source:* `results/effects_v2/enrichment.csv` (families P1, S3-D, P2, S3-K). *Script:* `scripts/plotting/fig2_enrichment_forest.py`.
<!-- src: results/effects_v2/enrichment.md:3 = 95, 10000 -->

**Figure 3 — `fig3_accuracy`. Predictive accuracy per training seed.**
Test AUROC of every cell (points, one per seed; bar = mean) on the uncorrected test sets. DeepDTA has no
attention and anchors accuracy. KIBA has two levels (random, cold-drug) and no DrugBAN cells.
*Source:* `results/accuracy_v2/cells.csv` (`view = uncorrected`). *Script:* `scripts/plotting/fig3_accuracy.py`.

**Figure 4 — `fig4_accuracy_vs_localization`. Accuracy does not predict localisation.**
One point per attention-model cell and seed: test AUROC against enrichment over chance (UniProt,
precision@10 ÷ chance). For DAVIS cold-target and cold-pair, AUROC is scored on sequence-unseen targets,
the same policy the explanation metrics use. Panel titles give Spearman ρ with a 95% bootstrap interval
over cells. *Sources:* `results/accuracy_v2/localization_cells.csv`, `results/accuracy_v2/localization_spearman.csv`
(group "all attention models", y = enrichment). *Script:* `scripts/plotting/fig4_accuracy_vs_localization.py`.
<!-- src: docs/PROTOCOL_AMENDMENT_v2.md:190 = 95 -->

**Figure 5 — `fig5_faithfulness`. Masking the attended residues matters more than masking random ones.**
Faithfulness delta (attended masking minus a size-matched random-masking control), mean over targets
with 95% bootstrap intervals (targets resampled, 10,000 resamples). MolTrans is measured in its own
token space, the others in residue space, so the three panels do not share a scale. No DrugBAN or KIBA
XAttn-Ref rows exist (deferred; see Discussion). *Source:* `results/effects_v2/faithfulness_effects.csv`.
*Script:* `scripts/plotting/fig5_faithfulness.py`.
<!-- src: results/effects_v2/faithfulness_effects.md:3 = 95, 10000 -->

**Figure 6 — `fig6_leakage`. What DAVIS's sequence leakage is worth to the accuracy anchor.**
DeepDTA test AUROC (mean ± sd over three seeds) at cold-target and cold-pair, retrained on three training
sets: the original split, `seqclean` (every seen-by-sequence target removed) and `seqmatched` (the leak
kept at `seqclean`'s size). `seqmatched` − `seqclean` isolates the leak; `original` − `seqmatched` is the
cost of fewer rows. Scored on all test rows and separately on leaked and unleaked targets.
*Source:* `results/leakage_retrain_davis.md` (per-arm tables; the per-seed JSON beside it is not
committed). *Script:* `scripts/plotting/fig6_leakage.py`.

## Figures in the plan not produced (declared)

* **Method-panel agreement** (plan T22): no E3 cell has been scored (T08: addendum A2 unsigned, compute not run).
* **Readout sensitivity** (plan T22): its source, `~/ColdSite-results/readouts/readout_comparison.csv`, is outside the
  repository; figures are built from committed files only (the plan's rule). Enters with the release (T23).
* **Seed-verdict forest, primary vs extension**: the extension (seeds 4–5, Wave A) has not been run, so Figure 1 shows
  the primary seeds 1–3 only.
