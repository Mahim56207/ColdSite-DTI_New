# CLAUDE.md — ColdSite-DTI (session checkpoint, 2026-09-12)

Source of truth for status: `STATUS.md` § *2026-09-12* (top). Older tables in it are
marked superseded — **check dates before trusting any "what is left" list**.

## 1. Thesis
**Do published interpretability claims in drug–target interaction (DTI) prediction
survive realistic evaluation?** An *audit*, not a model paper (`docs/00_MASTER_PLAN_V2.md`).
Attention-based DTI models claim their attention marks binding sites. We test that across
four split difficulties (random, cold-drug, cold-target, cold-pair) on DAVIS, measuring
**plausibility** (precision@k vs UniProt binding sites, against chance and ceiling) and
**faithfulness** (masking attended residues vs a random-masking control).
Subjects: HyperAttentionDTI (published), ColdSite-DTI (ours), MolTrans (published;
trainer built 2026-09-12, now training — see §4); DeepDTA = accuracy anchor only
(no attention, never audited).
Early signal (dry run: 1 seed, regression checkpoints): ColdSite-DTI's attention is
**faithful but not plausible** — load-bearing at every level, at best ~2× chance at
hitting sites (precision@10 0.040 vs 0.020; ceiling 0.99); cold-target is the most
accurate level and sits at chance.
Team 124AD0008 (data) · 124AD0015 (model) · 124AD0067 (evaluation); supervisor
Dr. Chandra Mohan Dasari. Venue: Bioinformatics / Briefings in Bioinformatics / ISMB.
**Draft due 15 Nov 2026.**

## 2. Paper structure (planned) → where material lives
| Section | Source | State |
|---|---|---|
| Abstract, Introduction | — | not written |
| Related Work (DTI claims; OOD explainability; attention-as-explanation) | `literature_differentiation.md` | drafted |
| Methods: data, splits, ground truth, family control | `results/split_summary.md`, `data/GROUND_TRUTH_README.md` | material only |
| Methods: models, training, faithfulness, attention extraction | `paper/methods_track_b.md` | drafted |
| Methods: metrics & statistics | `src/evaluation/*` docstrings | not written |
| Results: audit grid, kinase control, antiviral case study | — | awaiting 36-run grid |
| Discussion / Limitations | — | not written |

## 3. Done vs. to do
**Done**
- Data: splits (verified on 3 machines), antiviral subset (3 targets), non-kinase panel
  (60 targets), gene maps. Ground truth re-numbered to DAVIS sequences
  (`src/data/align_ground_truth.py`); 4 wrong-protein targets corrected via overrides.
- Training: DeepDTA regression DAVIS+KIBA 24/24 (`results.md`); ColdSite-DTI regression
  DAVIS 12/12 + replication (`results/`); DeepDTA **binary** DAVIS 12/12 — AUROC
  random 0.929, cold-target 0.908, cold-pair 0.728, cold-drug 0.692.
- Code: all trainers, analysis pipeline verified on real checkpoints; 599 tests pass.
  `src/model/train_moltrans.py` added 2026-09-12 (MolTrans binary trainer — see §4);
  tested locally against real DAVIS rows end to end (encodes, trains, early-stops, writes
  a checkpoint and results JSON in the same shape as the other two baseline trainers).
- Checkpoint backup: all 12 DAVIS regression checkpoints (`.pt` + `_results.json` +
  `_history.json`) verified byte-for-byte in Drive folder `coldsite-grid24-kaggle`
  (2026-09-12) — local `results/` and `~/Downloads/results` are no longer the only copy.

**To do**
- ColdSite-DTI + HyperAttentionDTI binary on DAVIS (running on Kaggle, account 1).
- MolTrans binary on DAVIS, then KIBA (running on Kaggle, account 2 — see §4 for the
  split with account 1's KIBA share).
- Once every cell above is trained, per seed: `run_faithfulness` → `run_ladder`; then
  `run_audit` (Holm); `run_control` ± `--exclude-cotransport-ions`. Automatic in the DAVIS
  36-grid's own §11 for its three models only — MolTrans and KIBA need the same three
  commands run by hand once trained (see §4's "later" list for why that matters more than
  it sounds).
- Volume-matched control: `notebooks/colab_volume_control.ipynb` running now (seed 1,
  started 2026-09-12).
- Write every section marked above; fill Related Work DOIs; checklist at its end.

## 4. Active goals — next steps, in order

**Two Kaggle accounts train in parallel now. They must never train the same
(dataset, model, split, seed) cell — see the KIBA split below before starting either
account on KIBA.**

- **Account 1 — DAVIS 36-grid:** `notebooks/kaggle_davis_binary_grid36.ipynb`, Kaggle
  notebook `mahim5/notebooka7e4de1f63`, version 1 started 2026-09-12 ~10:40 (gate passed,
  stage 2 running). Each commit starts empty and self-stops at 11 h; expect 2–3 commits.
  When v1 finishes: Output → download `grid36_results.zip` → Kaggle Dataset (private) →
  **re-import the notebook from GitHub** (gets the per-cell log prefixes and STATUS
  lines) → Add Input → set `RESTORE_FROM` (its own §5) → check quota ≥ 11.5 h, else lower
  the `11` in `DEADLINE` (its own §6) → Save & Run All. Confirm its §1's `torch` version
  and Environment column match v1. Repeat until its own §10 shows 36/36.
- **Account 2 — MolTrans, DAVIS first:** new notebook `notebooks/kaggle_binary_grid.ipynb`
  (built and tested locally 2026-09-12; trainer is `src/model/train_moltrans.py`). Settings
  cell: `DATASET = 'davis'`, `MODELS = ['moltrans']`, `SPLIT_SUBSET` = all four splits,
  `SEEDS = [1, 2, 3]` — 12 cells, nothing account 1 is training. Same restore loop as
  account 1 (this notebook's own §6 restore cell, §9 "what landed" table), until 12/12.
  **MolTrans checkpoints are ~250 MB each** (vs ~2.5 MB for the other three models) — size
  the restore dataset and the eventual Drive backup for ~3 GB, not ~30 MB.
  `train_moltrans.py` must be pushed to `origin/main` before account 2's first commit, or
  the clone cell's assertion stops it.
- **After account 1's DAVIS grid reaches 36/36 — both accounts move to KIBA**, splitting
  the four split types so no cell trains twice. Pick the actual split with account 1 once
  DAVIS is close to done (KIBA is ~4x DAVIS by row count, so get this right before
  launching either side — see "splitting KIBA" below). From this point account 1 also uses
  `kaggle_binary_grid.ipynb` (the 36-grid notebook is DAVIS-only): `DATASET = 'kiba'`,
  `MODELS = ['deepdta', 'coldsite_dti', 'hyperattentiondti']`, its half of `SPLIT_SUBSET`.
  Account 2 continues with `MODELS = ['moltrans']` on the other half.
- **Send the first ColdSite-DTI binary cell's Test metrics** (and account 1's §10 table)
  to check, once account 1 produces one.
- **Colab volume control**: running now (seed 1, ~1.5 min/epoch, started 2026-09-12). If
  Colab shows "Monaco: unable to load", reload, or turn off Brave Shields for
  colab.research.google.com.
- Once every cell above is trained, per seed: `run_faithfulness` → `run_ladder`; then
  `run_audit` (Holm); `run_control` ± `--exclude-cotransport-ions`. Automatic in the DAVIS
  36-grid's own §11 for its three models — MolTrans and KIBA need the same three commands
  run by hand once trained, since neither notebook's analysis cell knows about them.
- Optional CPU task: check KIBA ground truth for the same sequence/protein mismatches.

**Later, to strengthen the paper (after the grids, before the draft is due 15 Nov 2026):**
- **Run the non-kinase control properly.** `run_control` already runs automatically
  inside the DAVIS 36-grid (its own §11, seed 1 only, `coldsite_dti` and
  `hyperattentiondti`). Re-run it across all 3 seeds, and extend it to MolTrans and to
  KIBA once each is trained — the panel (60 non-kinase targets, already built) is what
  tells us whether the plausibility gap is a kinase-domain artifact or general to the
  model, and right now it only ever runs on a third of the seeds and none of the subjects
  that will exist once the grids finish.
- **Add a positive control for the metric itself.** Every number so far shows the audited
  models scoring near chance at plausibility. Nothing yet proves the precision@k /
  faithfulness pipeline *would* score a genuinely good explanation as good — the
  `achievable_ceiling` already reported alongside precision@k is a mathematical best case
  (site count vs k), not a check that the measurement works. Construct one (e.g. an
  adapter or synthetic attention that returns the ground-truth sites themselves) and
  confirm it scores near that ceiling. Without it, a reviewer can reasonably ask whether
  the near-chance numbers reflect the models or an insensitive test.

## 5. Citation & reporting rules
- **No citation style chosen yet.** Drafts cite by model name + year (e.g. "CS-DTA
  (2026, *Frontiers in Chemistry*)"); authors, venues, DOIs still to add. Fix the
  style when the venue is chosen and follow that journal's author guidelines.
- Report split means ± spread over **3 seeds**, never one seed (same seed varies
  up to 0.058 CI — cuDNN nondeterminism).
- Holm–Bonferroni once across the whole family before calling anything significant.
- Faithfulness = delta over the random-masking control; the raw score means nothing alone.
- precision@k always beside chance and ceiling; at n > 1,000 report effect size, not p.
- State in Methods: binary threshold (DAVIS pKd ≥ 7.0, one shared constant), truncation
  (`exclude`, 1,000 residues), cold-pair volume (15,190 vs 21,039 rows), mutants
  mapped to wild-type sites, cotransport-ion choice, ground-truth re-numbering.
- Ladder is **not** monotonic on DAVIS: treat levels as categories, not severity.

## 6. Working rules
- Repo: fork `Mahim56207/ColdSite-DTI_New`, branch `main`; `upstream` = udayraj1238.
  Ignore `project-completion` (stale). Run sessions **locally** (Colab/Kaggle/`gh` need it):
  Claude Code → New → Local · folder ColdSite-DTI · branch **main** · worktree off.
- Confirm with the user before any `git push`.
- Do not change training code while a grid is mid-run (cells must share one code state).
- DAVIS ground truth: fetch/overrides write `data/davis_ground_truth_sites_uniprot.json`;
  only `align_ground_truth` writes `data/davis_ground_truth_sites.json`. Re-align after either.
- Explain in plain language; the user prefers step-by-step instructions.
