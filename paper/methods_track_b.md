# Methods — Track B (124AD0015)

Draft for the paper's Methods section. Covers the model under audit
(ColdSite-DTI), the training protocol, the faithfulness measurement, and the
two methodological decisions that belong to this track.

Every architectural statement below is taken from the implementation in
`src/model/`. Numbers marked *(pending)* require Track A's split files and are
deliberately left unfilled rather than estimated.

Track A's data, splits and ground truth, and Track C's plausibility metric,
significance testing and audit grid, are documented separately.

---

## 1. Model architecture

ColdSite-DTI is a two-branch encoder with a single cross-attention fusion step.
It is deliberately a conventional, mid-sized architecture: under the audit
framing it is one subject among several, and its role is to be a solid,
comparable predictor whose explanation mechanism is the standard one the
literature makes claims about. It is not tuned for competitive accuracy.

Default configuration: `hidden_dim = 128`, `n_heads = 8`, `dropout = 0.2`,
**618,273 parameters** (with a 70-token SMILES vocabulary and the 28-token
protein vocabulary).

### 1.1 Inputs and representation

Both inputs are character-level integer sequences. Index 0 is padding and index
1 is unknown throughout.

**Drug.** SMILES strings are tokenised per character against a vocabulary built
**from the training split only** (`build_smiles_vocab`) and truncated to 100
characters. Training-only vocabulary construction is required rather than
tidy: the cold-drug and cold-pair splits place unseen molecules in test, so
building the vocabulary over all splits would leak exactly the information
those splits exist to withhold. Unseen characters map to the unknown index.

**Protein.** Sequences are mapped per residue over the 20 standard amino acids
plus the ambiguity codes X, B, Z, J, U and O (`build_protein_vocab`, 26 symbols
at indices 2–27), and truncated to **1,000 residues** (Section 4.1).

Sequences are not padded at encoding time. `collate_batch` pads each batch to
its own longest drug and longest protein, so a short batch does not pay the
cost of the longest molecule in the dataset.

### 1.2 Drug encoder

Embedding (128-d) followed by three 1-D convolutions with widening kernels —
4 → 6 → 8 channels 64 → 96 → 128 — each with ReLU and `padding="same"`, then a
masked max-pool over positions, LayerNorm and dropout. Output: one 128-d vector
per molecule.

Two choices are load-bearing rather than incidental. `padding="same"` is used
because valid padding gives a zero-length feature map for any SMILES shorter
than the summed kernel widths, and such fragments occur in DAVIS and KIBA.
Activations at padded positions are re-zeroed after **every** convolution: the
convolution bias otherwise makes the padded region non-zero, the next layer's
window reads that at the sequence boundary, and a molecule's encoding then
changes depending on how much padding its batch-mates imposed. The pooling
step masks padding explicitly and maps all-padding rows to zero rather than
−inf. Padding invariance is asserted in the test suite.

### 1.3 Protein encoder

The protein encoder mirrors the PCLAtt-style architecture: a **parallel** CNN
and BiLSTM over the same embedding, concatenated, then multi-head
self-attention.

- Embedding (128-d, `padding_idx=0`)
- *Branch 1* — Conv1d, kernel 8, `padding="same"`, ReLU, output 128 channels;
  padded positions re-zeroed
- *Branch 2* — bidirectional LSTM with hidden size 64 per direction (128
  concatenated), run over a **packed** sequence
- Concatenation (256-d) → Linear → ReLU → LayerNorm (128-d)
- Multi-head self-attention (8 heads) with `key_padding_mask`, residual
  connection, dropout, LayerNorm; padded positions re-zeroed

The two branches run in parallel rather than in series so that neither filters
what the other may see: the convolution captures short local motifs, the
recurrent branch long-range order. Packing the LSTM input is a correctness
requirement, not an optimisation — unpacked, the backward direction begins
inside the padding and every residue representation in a padded batch is
contaminated.

Output: `(batch, seq_len, 128)` contextualised residue features, plus the
self-attention map `(batch, seq_len, seq_len)`.

### 1.4 Cross-attention fusion and the explanation

The drug vector is used as a **single query** against the protein residue
features as keys and values, through an 8-head `MultiheadAttention` layer with
`key_padding_mask` derived from the protein token ids.

Using one query is the design decision that makes the model auditable at all:
the resulting attention map has shape `(batch, 1, protein_seq_len)` — exactly
one interpretable weight per residue, which is the form the plausibility metric
consumes. A multi-query formulation would require an additional reduction step,
and that reduction — not the model — would then determine the reported
explanation.

Attention weights at padded keys are already ≈0 from the masked softmax; they
are additionally `masked_fill`ed to **exact** zero so that a stored weight
vector can be sliced by residue index without surprises. Over real residues the
weights sum to 1 (verified in eval mode; attention dropout perturbs them during
training).

### 1.5 Prediction head

The raw drug vector is **concatenated** with the attended protein context
(2 × 128 = 256-d) and passed through Linear(256→128) → ReLU → Dropout →
Linear(128→64) → ReLU → Dropout → Linear(64→1).

The raw drug vector is carried forward alongside the context because context
alone leaves the head blind to which molecule it is looking at whenever
attention collapses, which it tends to do in early epochs on the cold splits.

The output is a single unbounded score: a binding affinity under regression, or
a **logit** under binary framing. No sigmoid is applied inside the model;
`BCEWithLogitsLoss` applies it internally and is numerically stabler.

### 1.6 Explanation extraction

`ColdSiteDTI.explain()` returns, for each protein in the batch, a list of
weights of **exactly the length of that protein's real sequence**, 0-indexed.

Two details matter for the correctness of every number in this paper.

**Length is measured to the last non-pad position** (`real_lengths`), not by
counting non-pad tokens. The two disagree for any sequence containing an
interior pad id, and the counting version returns a shorter array than the
residues it covers, misaligning every ground-truth index past that point — a
silent error in precisely the quantity being reported.

**`explain()` forces eval mode.** Attention dropout is active in training mode
and perturbs the weights, so an explanation extracted mid-training is not the
model's explanation.

This is also the contract the audit's model registry imposes on every subject
model (`ExplainableDTIModel`): `predict()` returning a scalar, and `explain()`
returning one non-negative weight per real residue. `validate_adapter()` checks
it, because an adapter returning the wrong length does not crash — it produces
a plausible-looking but wrong plausibility score.

---

## 2. Training protocol

Adam, learning rate 1e-3, batch size 64, up to **100 epochs**, with
`ReduceLROnPlateau` (factor 0.5, patience 5) on validation loss and early
stopping after 15 epochs without improvement. Loss is MSE under regression and
`BCEWithLogitsLoss` under binary framing. Gradients are clipped at max-norm 5.0;
the BiLSTM-plus-attention combination occasionally spikes the gradient norm in
the first few hundred steps, and clipping prevents a long run silently becoming
NaN. The best checkpoint by validation loss is retained and reloaded before test
evaluation — the final epoch is usually not the best one.

Reported test metrics under regression are MSE, RMSE, Pearson correlation and
the concordance index (CI). **CI is used as the accuracy axis** of the
fidelity-versus-accuracy figure: it is bounded [0, 1] and higher-is-better, so
it shares an axis sensibly with precision@k, whereas MSE would invert the
reading of the figure.

### 2.1 The grid

**Two datasets × four splits × three training seeds = 24 runs.**

The three seeds are **training** seeds — weight initialisation and batch order
— on one fixed split per cell. Consequently the reported standard deviation
across seeds measures initialisation variance, **not** split-selection variance.
This should be stated wherever error bars appear; a reader will otherwise
assume the intervals cover split noise, which they do not.

Every run's outputs are named with dataset, split, task and seed:

```
results/coldsite_dti_{dataset}_{split}_{task}_seed{N}.pt
results/{dataset}_{split}_{task}_seed{N}_results.json
```

The seed is mandatory in the name. Without it the three runs per cell overwrite
each other and the loss is invisible: one checkpoint remains, the results file
is rewritten three times, and a "three-seed mean" is reported from a single run.
Writer and reader construct these names from one shared function
(`src/model/checkpoint_naming.py`) so the two ends cannot drift apart.

---

## 3. Faithfulness

Plausibility (does attention fall on annotated binding sites?) and faithfulness
(does the model actually use the residues it attends to?) are independent axes.
The dangerous cell is *plausible but not faithful* — an explanation that looks
biologically sensible and carries no causal weight — and it cannot be detected
by inspection. Attention-as-explanation has been contested since Jain & Wallace
(2019) and Serrano & Smith (2019); Wiegreffe & Pinter (2019) pushed back, and
the dispute remains live. Measuring only plausibility would measure only half
of it.

For each drug–protein pair we compute **comprehensiveness**: the absolute change
in prediction when the top-*k* attended residues are replaced with the unknown
token. Masking uses the unknown token rather than padding, because padding is
excluded from attention by the key-padding mask, so masking with it would
shorten the protein as far as the model is concerned and conflate "these
residues mattered" with "the protein got shorter". Padding is never converted
into a residue.

**Every metric is reported against a random-masking control**: the same
intervention applied to *k* randomly chosen residues, averaged over
`n_random_trials`. This is not a courtesy. Masking anything moves the input off
the training distribution and the prediction changes for reasons unrelated to
the explanation — the ROAR critique. A raw comprehensiveness of 0.4 is
uninterpretable; it means something only beside a random-masking value of 0.05.

> **The reported quantity is `comprehensiveness_delta` = observed − random.**
> Raw comprehensiveness is not reported as a result.

Sufficiency (prediction change when only the top-*k* residues are kept) and
AOPC (progressive ablation) are reported with their own controls alongside.

Cost is `2 + 2·n_random_trials + |k_values|` forward passes per pair, so each
split is measured over a bounded sample (`max_pairs`) rather than every pair.

**Interpretation.** A `comprehensiveness_delta` at or below zero means masking
the attended residues perturbs the prediction no more than masking arbitrary
ones: at that difficulty level the attention is decoration, whatever it looks
like. This is a result, not a failed run, and it is reported plainly. It is the
"plausible but not faithful" cell, and finding it is a large part of why this
audit is worth running.

---

## 4. Two methodological decisions

### 4.1 Protein truncation

Sequences are truncated to the first **1,000 residues**, and annotated
binding-site positions beyond that window are **excluded** from the ground truth
(`truncation="exclude"`). Targets retaining no annotated site within the window
are removed from the evaluation population rather than scored zero.

This affects **24 of 409 DAVIS targets** (283 annotated positions; 15 targets
removed) and **14 of 224 KIBA targets** (165 positions; 8 removed). A further
two KIBA targets are unusable for reasons unrelated to truncation.

The alternative — retaining out-of-window sites — leaves those targets in the
average with a structurally guaranteed precision@k of 0. That deflates each
split mean by approximately **3.7%** for reasons unrelated to explanation
quality, and by a *level-dependent* amount, since the proportion of affected
targets varies with each split's test set. A constant offset would be harmless
to a claim about the shape of a degradation curve; a varying one is not. We
prefer a stated selection bias to a moving one.

The bias must be stated: excluded targets are systematically the longest
proteins — median final annotated residue **1,204 versus 332** for retained
targets — and are predominantly large multidomain receptor kinases (ALK, MET,
IGF1R, ROS1, MTOR, LRRK2, MST1R). Results should not be extrapolated to
proteins substantially longer than the input window.

The mean achievable ceiling differs between policies by under 1.2% at every *k*
tested, so the ceiling is not a reason to prefer either.

### 4.2 Cold-pair training volume

The cold-pair split withholds both drug and target identities and therefore
discards every pair in which exactly one entity is held out. Its training set
contains approximately **71%** of the pairs available to the other three levels,
and it uses roughly **54%** of all measured pairs once discarded rows are
counted. *(Exact counts pending Track A's split files; the figures above are
derived from the split code at the published dataset shapes.)*

Note that the 54% figure is the proportion of pairs *used at all*, not the
training-set ratio. Quoting it as the training ratio overstates the confound by
roughly a factor of two.

We **report** these counts rather than subsampling the other three splits to
match, so that each level is trained on all data legitimately available to it.
Subsampling would discard roughly 30% of the training data across 18 of 24 runs
to control a confound on the accuracy axis, which is not this paper's
contribution. The cold-pair accuracy drop should therefore be read as
difficulty *and* reduced training volume, and we bound the latter with a single
**volume-matched control run**: the warm split retrained on a random subsample
of its training rows matched to cold-pair's volume, same seed, same test set.
*(Pending split files.)*

A second and larger comparability gap is inherent to the design and is not
fixable by subsampling. Because plausibility is averaged **per target**, warm
and cold-drug fidelity is averaged over all 442 (DAVIS) / 229 (KIBA) targets,
whereas cold-target and cold-pair are averaged over only the 88 / 45 held-out
targets. Wider seed-to-seed variance in the cold cells is expected on that basis
alone and should not be read as instability.

---

## 5. Models under audit — attention extraction

Every subject model must satisfy the same contract: one weight per **real**
residue, 0-indexed. Neither attention baseline produces that natively, and both
projections are implemented and tested in
`src/evaluation/attention_projection.py`.

**HyperAttentionDTI** computes attention over **convolution positions**, not
residues. Its protein stack of kernels [4, 8, 12] with unit stride and no
padding maps 1,000 residues to 979 positions, each summarising a 22-residue
window (offset 3 + 7 + 11 = 21). Convolution position *j* is projected to its
central residue *j* + 10; positions whose receptive field reaches into padding
are discarded, since a weight computed over padding is not evidence about any
residue. The centre projection is preferred over spreading each weight across
its full 22-residue window because the latter blurs a localised signal and
would mechanically lower precision@10 — manufacturing the degradation this
audit exists to test for. Note also that the published implementation averages
the drug axis away inside `forward`, so the axis remaining at the protein
attention gate is channels, not drug positions.

**MolTrans** attends over **ESPF subword tokens**. Its 545-token limit is a
token count, not a residue count, and one token spans several residues; the
number of residues the model saw is therefore protein-specific. Because the BPE
is constructed with an empty separator, tokens concatenate back to the original
sequence exactly, which permits an exact span mapping. Each token's weight is
assigned to every residue in its span **undivided**: dividing by span length
would make top-*k* selection a function of ESPF token length rather than of
attention.

Baseline adapter implementation is owned by 124AD0008; the extraction and
projection machinery above is contributed by this track.
