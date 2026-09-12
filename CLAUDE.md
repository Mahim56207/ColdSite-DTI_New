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
Subjects: HyperAttentionDTI (published), ColdSite-DTI (ours), MolTrans (deferred);
DeepDTA = accuracy anchor only (no attention, never audited).
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

**To do**
- ColdSite-DTI + HyperAttentionDTI binary on DAVIS (running on Kaggle).
- Then per seed: `run_faithfulness` → `run_ladder`; then `run_audit` (Holm);
  `run_control` ± `--exclude-cotransport-ions`. All automatic in notebook §11.
- Volume-matched control: `notebooks/colab_volume_control.ipynb` (ready, not run).
- Write every section marked above; fill Related Work DOIs; checklist at its end.
- Deferred (write up as limitations unless time allows): **MolTrans** (~3–12 GPU-h,
  preferred next), **KIBA** (~2–5 weeks of quota).

## 4. Active goals this session
- Kaggle `notebooks/kaggle_davis_binary_grid36.ipynb` (notebook `mahim5/notebooka7e4de1f63`,
  version 1 running). Each commit starts empty: download `results` → Kaggle Dataset →
  set `RESTORE_FROM` → commit again. Self-stops at 11 h. Expect 2–3 commits.
- Ladder re-run on the aligned ground truth (seed 1) → add before/after to STATUS.md.
- Decide MolTrans protocol: published 13 epochs vs our early stopping (lean: ours).
- Optional CPU task: check KIBA ground truth for the same sequence/protein mismatches.

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
  Ignore `project-completion` (stale). Run sessions **locally** (Colab/Kaggle/`gh` need it).
- Confirm with the user before any `git push`.
- Do not change training code while a grid is mid-run (cells must share one code state).
- DAVIS ground truth: fetch/overrides write `data/davis_ground_truth_sites_uniprot.json`;
  only `align_ground_truth` writes `data/davis_ground_truth_sites.json`. Re-align after either.
- Explain in plain language; the user prefers step-by-step instructions.
