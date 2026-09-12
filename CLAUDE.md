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
accurate level and sits at chance. Counted one pair per protein (2026-09-12), cold-pair is
at chance too (0.024, p = 0.19; it was "significant" only when 79 proteins were counted as
1,027 pairs) — `results/ladder_dryrun_regression_davis_seed1.md`.
Team 124AD0008 (data) · 124AD0015 (model) · 124AD0067 (evaluation); supervisor
Dr. Chandra Mohan Dasari. Venue: Bioinformatics / Briefings in Bioinformatics / ISMB.
**Draft due 15 Nov 2026.**

## 2. Paper structure (planned) → where material lives
| Section | Source | State |
|---|---|---|
| Abstract | — | not written (last, from the results) |
| Introduction | `paper/introduction.md` | skeleton with draft prose, results as placeholders (2026-09-12) |
| Related Work (DTI claims; OOD explainability; attention-as-explanation) | `literature_differentiation.md` | drafted; citations verified 2026-09-12 (`paper/references.md`); **two factual errors corrected** (DMFF-DTA does test unseen drugs/targets; GPS-DTI does claim interpretability); SAE "faithfulness gap" claim removed, no source; full-text checks listed in its checklist |
| Discussion — Limitations | `paper/limitations.md` | drafted 2026-09-12; KIBA scope/AMP pending |
| Methods: data, splits, ground truth, family control | `paper/methods_data_and_evaluation.md` §1–4 | drafted 2026-09-12 |
| Methods: ColdSite-DTI architecture & training, attention extraction | `paper/methods_track_b.md` | drafted (stale placeholders filled 2026-09-12) |
| Methods: baselines' training, metrics & statistics, baseline faithfulness, positive control | `paper/methods_data_and_evaluation.md` §5–9 | drafted 2026-09-12; §10 lists 5 open decisions |
| Results: audit grid, kinase control, antiviral case study | — | awaiting 36-run grid |
| Discussion / Limitations | — | not written |

## 3. Done vs. to do
**Done**
- Data: splits (verified on 3 machines), antiviral subset (3 targets), non-kinase panel
  (60 targets), gene maps. Ground truth re-numbered to DAVIS sequences
  (`src/data/align_ground_truth.py`); 4 wrong-protein targets corrected via overrides.
  KIBA re-numbered the same way (2026-09-12, `--dataset kiba`): 212 of 221 identical,
  PIM1 and SGK2 were isoforms (24 site residues moved), 0 dropped.
- Training: DeepDTA regression DAVIS+KIBA 24/24 (`results.md`); ColdSite-DTI regression
  DAVIS 12/12 + replication (`results/`); DeepDTA **binary** DAVIS 12/12 — AUROC
  random 0.929, cold-target 0.908, cold-pair 0.728, cold-drug 0.692.
- Code: all trainers, analysis pipeline verified on real checkpoints; 599 tests pass.
  `src/model/train_moltrans.py` added 2026-09-12 (MolTrans binary trainer — see §4);
  tested locally against real DAVIS rows end to end (encodes, trains, early-stops, writes
  a checkpoint and results JSON in the same shape as the other two baseline trainers).
- Analysis for the baselines (2026-09-12): `run_faithfulness` and `run_ladder` now take
  `--model`; verified end to end on real (tiny) HyperAttentionDTI and MolTrans checkpoints.
  Masking for the baselines goes through `src/evaluation/residue_space.py`: a masked
  residue becomes `X` and is re-tokenised by the model's own tokeniser. That is the same
  intervention ColdSite-DTI gets (its UNK token), where the old code would have mutated
  HyperAttentionDTI residues to alanine. `HyperAttentionDTIAdapter.predict` now returns
  log-odds (logit1 − logit0), not the positive logit. `check_adapters --checkpoints` now
  actually finds trained checkpoints (before, it silently fell back to random weights).
- Checkpoint backup: all 12 DAVIS regression checkpoints (`.pt` + `_results.json` +
  `_history.json`) verified byte-for-byte in Drive folder `coldsite-grid24-kaggle`
  (2026-09-12) — local `results/` and `~/Downloads/results` are no longer the only copy.

**To do**
- ColdSite-DTI + HyperAttentionDTI binary on DAVIS (running on Kaggle, account 1).
- MolTrans binary on DAVIS (running on Kaggle, account 2). KIBA: six accounts, see §4 for the
  `KIBA_RUN` split.
- Once every cell above is trained, per seed: `run_faithfulness` → `run_ladder`; then
  `run_audit` (Holm); `run_control` ± `--exclude-cotransport-ions`. Automatic in the DAVIS
  36-grid's own §11 for its three models only — MolTrans and KIBA need the same three
  commands run by hand once trained (see §4's "later" list for why that matters more than
  it sounds).
- Volume-matched control **done 2026-09-12** — record in `results/volume_control_davis.md`
  (Colab; Drive folder `coldsite-volume-control`,
  3 seeds × `.pt`/results/history, renamed `_trainsub15190` — never copy into a grid
  folder). ColdSite-DTI binary, DAVIS `random` trained on cold-pair's 15,190 rows, full
  test set: AUROC 0.884 / 0.906 / 0.871 → **0.887 ± 0.018** (AUPRC 0.472 / 0.578 / 0.449;
  seed 2 ran to epoch 38, the others 14). Interpret once account 1 supplies ColdSite-DTI's
  full `random` and `cold_pair` binary cells: full − 0.887 = cost of fewer rows;
  0.887 − cold_pair = genuine cold-pair difficulty. A gap under the 0.018 spread is not a finding.
- Write every section marked above. Related Work citations are verified
  (`paper/references.md`); its checklist still lists full-text checks and newer papers.

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
  (built and tested locally 2026-09-12; trainer is `src/model/train_moltrans.py`), Kaggle
  notebook `mahim234/notebook51d23b99dd`, version 1 (scriptVersionId 349227830) started
  2026-09-12 on commit `e252048`. Settings
  cell: `DATASET = 'davis'`, `MODELS = ['moltrans']`, `SPLIT_SUBSET` = all four splits,
  `SEEDS = [1, 2, 3]` — 12 cells, nothing account 1 is training. Same restore loop as
  account 1 (this notebook's own §6 restore cell, §9 "what landed" table), until 12/12.
  **MolTrans checkpoints are ~250 MB each** (vs ~2.5 MB for the other three models) — size
  the restore dataset and the eventual Drive backup for ~3 GB, not ~30 MB.
  Next: the user sends the first two `STATUS` lines (to estimate epochs/hour and whether
  12 cells need one commit or two) and the first `✓` test AUROC (believable DAVIS
  `random` ≈ 0.85–0.93; ≥ 0.98 means leakage, ≈ 0.5 means it isn't learning).
- **KIBA plan — six accounts (2026-09-12; supersedes the two-account split).** In
  `kaggle_binary_grid.ipynb` on branch `kiba-resume` (commit `03ab140`), each account sets
  only `KIBA_RUN` = one of `K1`…`K6`, which fills in `DATASET='kiba'`, all 4 splits,
  `BRANCH='kiba-resume'`, `AMP=True`, and:
  - K1 / K2 / K3: MolTrans, seed 1 / 2 / 3 — ~19 h on the slower T4 (13–26 h at 25–50 epochs).
  - K4 / K5 / K6: DeepDTA + ColdSite-DTI + HyperAttentionDTI, seed 1 / 2 / 3 — ~24 h (17–34 h).
  - Together exactly the 48 cells, each once (tested). 2–3 commits per account; K4–K6 at
    50 epochs would pass a week's ~30 h quota. Each account has its own restore dataset;
    **never two accounts on the same K-run.** Accounts 1 and 2 can take K-runs once
    their DAVIS work is finished. Start only after the AMP validation passes.
  - Kaggle's terms allow one account per person — each account must be a different team
    member's own.
  - Cost per cell on a T4 (large split / cold-pair), hours at 25 (min) and 36 (typical)
    epochs: HyperAttentionDTI 9.5/6.3 and 13.8/9.1; MolTrans 8.2/5.5 and 11.8/7.9;
    ColdSite-DTI 3.4/2.3 and 4.8/3.2; DeepDTA ~0.6–0.8. Total ~240–340 GPU-h ≈ 2–3 weeks of
    both accounts at ~30 h/week each (verify quota). ETA ~6–10 Oct; Colab (with resume)
    is the overflow.
  - **Mixed precision, measured on a T4 (2026-09-12, `results/speed_test_kiba_t4.md`):**
    HyperAttentionDTI 2.0×, MolTrans 1.3×, ColdSite-DTI 1.15×, DeepDTA 2.3×; no non-finite
    steps; outputs shift ~0.02–0.04% (untrained weights). cudnn autotuning alone: nothing.
    KIBA per cell at 36 epochs: HyperAttentionDTI 14.8 → 7.3 h, MolTrans 12.4 → 9.4 h,
    ColdSite-DTI 5.2 → 4.5 h, DeepDTA ~0.1–0.3 h. Totals: full KIBA ~165–235 GPU-h with
    AMP (vs ~250–360); random + cold-drug ~90–130. Proposed: `--amp` opt-in flag (DAVIS
    stays fp32), used for every KIBA cell of every model; **validate first** on DAVIS
    HyperAttentionDTI cold-pair × 3 seeds under AMP on Colab vs the fp32 grid cells' spread
    (AUROC and precision@10). Rejected as protocol changes: shorter patience/epochs, larger
    batches, fewer rows/seeds.
  - **KIBA scope still open** (full 48 cells vs random + cold-drug 24): check with supervisor.
  - **Epoch-level resume + `--amp`: built 2026-09-12 on branch `kiba-resume`** (commit
    `0306e44`; worktree `../ColdSite-DTI-kiba-resume`, not on `main`). All four trainers
    write `<checkpoint>_resume.pt` each epoch (atomic; best checkpoint written after it, so
    the 11 h kill can land anywhere), continue on restart, refuse changed settings, delete
    it when the cell finishes. `--amp` off by default. Verified on real DAVIS rows: resumed
    == uninterrupted and branch (amp off) == `main`, bit for bit on CPU; 689 tests pass.
    `kaggle_binary_grid.ipynb` on the branch has `BRANCH`/`AMP` settings (KIBA:
    `BRANCH='kiba-resume'`, `AMP=True`; six-account `KIBA_RUN` presets above). **Merge into `main` only after both DAVIS runs
    finish.** Next: run `notebooks/colab_amp_validation.ipynb` (branch; DAVIS
    HyperAttentionDTI cold-pair × 3 seeds with `--amp`, set `FP32_DIR` to account 1's
    fp32 cells) — verdict = AMP mean within the fp32 seed spread for AUROC and precision@10.
- **Send the first ColdSite-DTI binary cell's Test metrics** (and account 1's §10 table)
  to check, once account 1 produces one.
- **Colab**: free since the volume control finished (2026-09-12; numbers in §3). If Colab
  shows "Monaco: unable to load", reload, or turn off Brave Shields for
  colab.research.google.com.
- **Results day = one command** (2026-09-12): `python -m src.evaluation.run_all --dataset
  davis --checkpoint-dir <merged results> [--volume-control-dir ...]` runs faithfulness and
  the ladder per model and seed (each ladder fed its own model's accuracy file), the audit
  (Holm), the non-kinase control both ion settings, the positive control, and writes
  `analysis_summary_<dataset>.md`. Resumable (skips finished outputs); `--dry-run` shows the
  plan. Run it on Colab with `notebooks/colab_analysis.ipynb` (merges result folders from
  Drive, refuses `_trainsub` files, writes outputs to Drive). The 36-grid's own §11 still
  runs the old ColdSite-only commands; `run_all` supersedes it. Verified end to end on tiny
  checkpoints of all four models — which caught the non-kinase control crashing for
  HyperAttentionDTI on BindingDB's extended SMILES (fixed in `collect.py`).
- ~~Check KIBA ground truth for sequence/protein mismatches~~ — done 2026-09-12 (see §3).

**Measured speeds (Kaggle T4, 2026-09-12 logs) and the KIBA risk they expose:**
- Account 1 at 6.5 h: GPU1 already on HyperAttentionDTI (~0.5 s/batch at 32, ~6 min/epoch
  on DAVIS); GPU0 on ColdSite-DTI cold-drug s1 (~2 min 15 s/epoch; val AUROC 0.53 at the
  epoch-10 floor, watch its test AUROC). v1 predates the per-cell log prefixes.
- Account 2 at 2.9 h: MolTrans ~0.21 s/batch at 16, ~5 min/epoch on DAVIS, train loss ~0.10
  (guessing the base rate ≈ 0.29). Every cell runs ≥ 25 epochs (floor 10 + patience 15), so
  ≥ ~2 h/cell → **2 commits** for 12 cells.
- Both show "Latest Container Image": pin the environment to v1's for later commits.
- **KIBA (~3.9× DAVIS rows) ⇒ ~20 min/epoch MolTrans, ~23 min/epoch HyperAttentionDTI ⇒
  ≥ 8–10 h per cell; any cell needing > ~25 epochs exceeds the 11 h commit and, with no
  mid-cell resume, restarts from scratch forever.** Needs epoch-level resume in the trainers
  before KIBA — a training-code change, so only after both DAVIS runs finish (or on a branch).

**Later, to strengthen the paper (after the grids, before the draft is due 15 Nov 2026):**
- **Run the non-kinase control properly.** `run_control` already runs automatically
  inside the DAVIS 36-grid (its own §11, seed 1 only, `coldsite_dti` and
  `hyperattentiondti`). Re-run it across all 3 seeds, and extend it to MolTrans and to
  KIBA once each is trained — the panel (60 non-kinase targets, already built) is what
  tells us whether the plausibility gap is a kinase-domain artifact or general to the
  model, and right now it only ever runs on a third of the seeds and none of the subjects
  that will exist once the grids finish.
- ~~Add a positive control for the metric itself~~ — **done 2026-09-12**:
  `python -m src.evaluation.positive_control --dataset davis|kiba` → `results/positive_control_<dataset>.md`.
  Explanations of known quality ("dose" = fraction of true sites ranked first) scored by
  the audit's own functions on the real splits and ground truth. All hard checks pass on
  both datasets: 0 sites outside the seen sequence, oracle = ceiling, oracle p = 0.001,
  oracle load-bearing on a planted model. It **fails** on DAVIS's pre-renumbering sites
  (146 outside the sequence), so it catches that class of bug. The test reliably detects
  a 2% dose at every level of both datasets (cold-target: 79 DAVIS / 42 KIBA proteins).
  Read against it, ColdSite-DTI's dry-run precision@10 (regression, seed 1) ≈ 1.7% of
  sites at warm, 2.1% cold-pair, 0.5% cold-drug (below reliable resolution), and
  cold-target at chance — a real null, not a power problem. For Results:
  `--compare <model>=<ladder json>` does this for every real ladder once the grids finish.

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
- Unit of averaging (settled 2026-09-12): the ladder and the audit table both score **one
  test pair per protein**, the first in file order (`run_ladder --pairs-per-target 1`, the
  default; `0` = every pair, the old behaviour). The dry-run ladder numbers in §1 and
  STATUS.md were computed per pair.
- Also state: faithfulness masks a residue as `X` through each model's own tokeniser;
  HyperAttentionDTI's prediction is its log-odds. Also, vendored MolTrans keeps dropout
  on at inference (`F.dropout` without `training=`), so its test AUROC includes that
  noise as published; faithfulness holds the RNG fixed per forward pass.
- Decided 2026-09-12 (`paper/methods_data_and_evaluation.md` §10): every model's
  explanation is scored over the same first 1,000 residues as the ground truth — MolTrans's
  is cut there (`collect.py`, and `residue_space` for faithfulness); DeepDTA keeps
  patience 10 on both datasets, stated not aligned (patience pinned per model in
  `kaggle_binary_grid.ipynb`); the **primary non-kinase result excludes cotransport
  ions** (`run_control --exclude-cotransport-ions`, `_noions` outputs), all ligands as
  sensitivity; truncation deflation recomputed (3.8% DAVIS / 4.1% KIBA, 2.3–7.1% by level).
- Ladder is **not** monotonic on DAVIS: treat levels as categories, not severity.

## 6. Working rules
- Repo: fork `Mahim56207/ColdSite-DTI_New`, branch `main`; `upstream` = udayraj1238.
  Ignore `project-completion` (stale). Run sessions **locally** (Colab/Kaggle/`gh` need it):
  Claude Code → New → Local · folder ColdSite-DTI · branch **main** · worktree off.
- Confirm with the user before any `git push`.
- Do not change training code while a grid is mid-run (cells must share one code state).
- Ground truth (DAVIS and KIBA): fetch/overrides write `data/<dataset>_ground_truth_sites_uniprot.json`;
  only `align_ground_truth --dataset <dataset>` writes `data/<dataset>_ground_truth_sites.json`.
  Re-align after either.
- Explain in plain language; the user prefers step-by-step instructions.
