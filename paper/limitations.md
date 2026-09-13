# Discussion — Limitations (draft)

Draft for the Limitations part of the Discussion. Every limitation here is a property of
the design or the data, known before the results; none depends on what the grid shows.
Items marked *[PENDING]* depend on a decision not yet taken. Numbers come from
`paper/methods_data_and_evaluation.md`, where each is sourced.

---

**Kinase-only training data.** DAVIS and KIBA are kinase panels (DAVIS: 429 kinase,
0 non-kinase, 13 unclassified targets; KIBA: 227, 0, 2), so no within-dataset comparison
can separate "attention finds binding sites" from "attention finds the ATP pocket every
training protein shares". The non-kinase control addresses this by transfer to 60
BindingDB proteins that no model has seen, which is a strictly harder condition than
cold-target: protein, family and drugs all change at once. A difference between the
kinase and non-kinase arms therefore bounds the family effect rather than isolating it,
and rests on 60 proteins whose affinities come from different assays.

**Protein-level, not drug-level, ground truth.** The binding sites are UniProt
annotations of the protein (binding sites, active sites, nucleotide-binding regions), so
every drug measured against a protein is scored against the same residues. A compound
that binds outside the annotated pocket, an allosteric kinase inhibitor for example, is
scored as if it bound the ATP site. Annotation is also incomplete: a residue without an
annotation is not known to be uninvolved, so precision@k is a lower bound on how often
attention falls on functionally relevant residues. Mutant and phosphorylated DAVIS
variants (63 identifiers) are scored against the sites of their wild-type entry. The
second ground truth, the KLIFS ATP pocket (Methods §3.6), is structure-derived and uniform
across kinases, which answers UniProt's sparsity and unevenness, but it is still defined
per protein rather than per ligand, and it exists only for kinases, so the non-kinase
control is scored against UniProt alone.

**One split per level, three training seeds.** Each level has a single fixed split; the
three seeds vary initialisation and batch order only, so reported spreads exclude
split-selection variance. Training is not bit-reproducible on GPU (cuDNN's LSTM kernels
are nondeterministic; an identical re-run moved single seeds by up to 0.058 CI), which is
why only split means with their spread are reported.

**Small held-out sets on the cold levels.** DAVIS has 68 drugs, so its cold-drug level
holds out only 13. After the sequence policy (below), its cold-target and cold-pair test
sets contain 68 and 72 distinct proteins with usable sites (KIBA: 42 and 41), against 349
(KIBA: 212) on the random level, so the cold levels' plausibility estimates are the least
precise in the grid. The positive control
shows the permutation test still detects an explanation that ranks 2% of true sites
first at these sizes, so a result at chance there is a null rather than a lack of power;
it does not make the estimates as precise as the random level's.

**DAVIS's sequence file.** DeepDTA's DAVIS protein file, used here as in most DTI
benchmarks, gives all 54 variants that have a wild-type entry exactly the wild-type
sequence, so the 442 targets are 379 distinct sequences and a mutant cannot be told from
its wild type by any sequence model; ten targets' sequences hold few or none of the ATP
pocket's residues (RET and its three mutants are its extracellular residues 1–430). As a
result 13.6% of cold-target and 12.5% of cold-pair test rows are proteins seen in
training under another name. We score the cold levels on targets unseen by sequence,
count one protein per sequence and drop the pocketless targets (Methods §2.4); the
all-rows accuracy is reported beside it. Two effects cannot be removed without
retraining: 13.6% of cold-pair's validation rows are seen by sequence, which influenced
which epoch was kept, and the models learned from duplicated sequences. Results on the
same file elsewhere in the literature carry the same leakage. KIBA has none of these
properties.

**The 1,000-residue window.** Sequences are truncated to 1,000 residues and sites beyond
the window are excluded; 16 DAVIS and 9 KIBA targets lose every site this way and leave
the evaluation. They are systematically the longest proteins (median final annotated
residue 1,320 against 312 for retained DAVIS targets), mostly large multidomain receptor
kinases, so the results should not be extended to proteins much longer than the window.
MolTrans reads beyond it on long proteins; its explanation is cut to the same window for
comparability, which scores the model on less than it saw.

**A binary task for every model.** Two of the audited models are classifiers in their
published form, so all models are compared on a binary task at DeepDTA's published
thresholds (DAVIS pKd ≥ 7.0, KIBA score ≥ 12.1). ColdSite-DTI and DeepDTA were designed for
regression; their binary results are not their regression results, and a different
threshold would change the class balance and every AUPRC.

**Published models retrained under a shared protocol.** Each published model is trained
with its authors' optimiser, learning rate, batch size and tokeniser, but checkpoints are
selected by one rule for all (lowest validation loss after a 10-epoch floor, patience 15;
DeepDTA 10). MolTrans's published script trains a fixed number of epochs and keeps the
best validation AUROC; HyperAttentionDTI's published class weights are not used. These
choices make the subjects comparable with one another; they also mean each is the
published architecture and recipe under our protocol, not the published checkpoint.
MolTrans's published code keeps dropout active at inference, and its accuracy is
reported with that noise, as published. DeepDTA runs as a PyTorch port of the original
Keras implementation.

**Explanation extraction involves choices.** HyperAttentionDTI attends over convolution
positions, averaged over channels and projected to each window's central residue;
MolTrans over subword tokens, taken from the protein encoder's last layer, averaged over
heads and over query tokens, with each token's weight given undivided to every residue it
spans. Each is a defensible reading of the published attention and each is
recorded; a different projection or reduction would change the numbers, though not, we
expect, the comparison between levels. *[Check against the results: if a model sits
near a threshold, report the alternative projection as a sensitivity analysis.]*

**What faithfulness can and cannot say.** Comprehensiveness replaces the top-*k*
attended residues with an unknown amino acid and measures the change in prediction. The
masked input is off the training distribution, which moves predictions for reasons
unrelated to the explanation; the random-masking control removes that effect on average
but not per pair. For MolTrans, masking a residue also changes how its neighbours are
tokenised. Faithfulness is measured on the first 200 test pairs of each level at *k* = 10,
and establishes whether the attended residues are load-bearing for the prediction, not
that they are the model's full reason for it.

**Scope of the audit.** Three attention-based models are audited (HyperAttentionDTI,
MolTrans and our own ColdSite-DTI), with DeepDTA as an accuracy anchor. Attention is the
only explanation method examined; gradient- and perturbation-based attributions, which
the same models could be given, are outside this paper's question, which concerns the
claims the published models make for their own attention.

**What the non-kinase control can separate.** The control arm's binding sites differ from
kinase ATP sites in composition as well as family: they are histidine-rich (8.8× the
background), many of them metal sites, where kinase sites are glycine-, aspartate- and
lysine-rich. ColdSite-DTI's attention prefers histidine (3–15× enriched), and most of its
above-chance non-kinase precision is recovered by shuffling attention among residues of
the same amino acid. A kinase–non-kinase gap therefore mixes the protein family with the
sites' amino-acid composition; the same-residue null is reported beside it for every
model so the two can be told apart.

**Compute-driven choices on KIBA.** All four levels of KIBA are trained, but in mixed
precision (float16 autocast with loss scaling), which ran 1.3–2.3× faster on the T4s
available. On DAVIS HyperAttentionDTI cold-pair, three seeds each, mixed precision moved
test AUROC by −0.013 and precision@10 by +0.001, both within the full-precision seed
spread (0.038 and 0.012; `results/amp_validation_davis.md`); three seeds rule out only a
large effect, and the two arms also differed in PyTorch version. Every KIBA cell uses
mixed precision, so KIBA's models are compared under one protocol; comparisons between
DAVIS (full precision) and KIBA carry the caveat. KIBA cells longer than one 11-hour
compute session continue from their last finished epoch, restoring model, optimiser,
scheduler, loss scaler and random-number state.
