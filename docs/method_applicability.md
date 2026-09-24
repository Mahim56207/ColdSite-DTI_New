# Explanation-method applicability (T08, amendment family E3)

Committed **before any T08 score is computed**, as amendment §3 row E3 requires. A method is applicable to a model only if it is
defined for that architecture without changing the computation the audited checkpoint performs. Where it is not, it is **not
implemented** and the reason is written here; nothing substitutes for it. The machine-readable twin is
`src/evaluation/explanation_methods.py::APPLICABILITY` (a test keeps the two in step).

Methods: **occlusion** (leave a window of residues out, read the drop in the model's own score), **attention × gradient**
(Chefer et al. 2021: the readout's attention tensor times the positive part of ∂score/∂attention, reduced like the plain readout),
**rollout** (Abnar & Zuidema 2020: layer attentions plus residual, multiplied through the stack). Existing methods — the plain
attention readout and integrated gradients (family S1) — are unchanged and only appear in the agreement analysis.

| method | ColdSite-DTI | HyperAttentionDTI | MolTrans | DrugBAN |
|---|---|---|---|---|
| occlusion | `coldsite_dti_occlusion` — applicable; masks to UNK (token 1), batched | `hyperattentiondti_occlusion` — applicable; masks to its own `X` code, batched | `moltrans_occlusion` — applicable; masks **residues** and re-tokenises (as `residue_space.py`), one fixed dropout draw per pass | `drugban_occlusion` — applicable; **written but not run on the machine that wrote it (no DGL)**; one pass per window |
| attention × gradient | **N/A** — the cross-attention it returns is a head-averaged copy that never feeds the prediction (`nn.MultiheadAttention` computes its output from the per-head probabilities internally), so ∂score/∂(returned weights) does not exist; a gradient would require re-implementing the attention forward, a different computation | `hyperattentiondti_attngrad` — applicable; the gate is the second call of the model's `sigmoid` module (drug first, protein second), verified against the plain readout's tensor and by finite difference | `moltrans_attngrad` — applicable; last protein-encoder layer (the plain readout's layer), grabbed as the input of that layer's dropout module, verified against the plain readout's tensor and by finite difference | **N/A for now** — not implemented: needs DGL to inspect the bilinear map's path to the score; deferred, not judged impossible |
| rollout | **N/A** — one cross-attention read by a single query, and the protein tower's self-attention is one layer: nothing to compose | **N/A** — no stacked softmax attention; a single sigmoid channel gate | `moltrans_rollout` — applicable; the protein encoder is a stack of self-attention layers with residual connections | **N/A** — one bilinear attention map, no stack of attention layers |

Applicable (method, model) pairs: **7** (occlusion × 4, attention × gradient × 2, rollout × 1).

## Declared free parameters (fixed here, not tuned afterwards)

* occlusion window 5 residues, stride 2, last window flush with the end (`WINDOW`, `STRIDE`); a residue's weight is the mean drop over
  the windows covering it; the weight is the **magnitude** of the drop (the sign-free convention integrated gradients already uses;
  `signed=True` exists for anyone who wants direction);
* the drop is `score(original) − score(masked)` in the model's own score: ColdSite-DTI's output, HyperAttentionDTI's log-odds
  (logit₁ − logit₀), MolTrans's output, DrugBAN's logit — the quantities their `predict` reports;
* MolTrans: every forward pass, including the two gradient methods and rollout, runs under one fixed RNG seed (`PREDICT_SEED = 0`),
  because its vendored forward leaves dropout live at inference (`baselines/MolTrans/models.py:103`); a masked input the ESPF tokeniser
  cannot encode (it returns one token for the whole protein) is left out of the average rather than scored;
* attention × gradient and rollout reduce heads/channels (mean) and query rows (mean) exactly as the plain readout does, so a
  difference from it is the method, not the reduction; rollout uses `0.5·A + 0.5·I` with rows re-normalised;
* ground truth, pair selection (one pair per protein, seen-by-sequence and pocketless targets excluded), k = 10, 10,000 permutations
  and the Holm scope are those of the primary families (amendment §2–§4).

## Family sizes if scored (for addendum A2; nothing has been scored)

E3-D: 7 pairs × 4 DAVIS levels = **28 cells**, seeds 1–3, one joint Holm family (decision D8). E3-K: on KIBA the models are
ColdSite-DTI, HyperAttentionDTI, MolTrans (no DrugBAN cells): occlusion × 3 + attention × gradient × 2 + rollout × 1 = 6 pairs × 2
levels = **12 cells**, its own family. A cell that cannot be produced is analysed at the design *m* and counted a non-rejection
(amendment §3).

## Top-k IoU agreement

Descriptive only (no test, no Holm; amendment §3): mean top-10 IoU between every pair of methods for one model, over the same
proteins, beside the exact chance IoU for those proteins' lengths (`explanation_agreement.py`).
