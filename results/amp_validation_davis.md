# Mixed-precision validation — HyperAttentionDTI, DAVIS cold-pair

Does `--amp` (float16 autocast + GradScaler, `src/model/precision.py`) change the two
numbers the paper reports? Decided in advance (`CLAUDE.md` §4, 2026-09-12): use `--amp`
for KIBA if, for both test AUROC and precision@10, the mixed-precision mean lies within
the full-precision seed-to-seed spread (sample sd over three seeds).

| | seed 1 | seed 2 | seed 3 | mean ± sd |
|---|---|---|---|---|
| **test AUROC**, full precision (Kaggle T4, torch 2.10) | 0.6964 | 0.6552 | 0.7301 | **0.694 ± 0.038** |
| **test AUROC**, mixed precision (Colab T4, torch 2.11) | 0.6030 | 0.7099 | 0.7303 | **0.681 ± 0.068** |
| **precision@10**, full precision | 0.011 | 0.022 | 0.034 | **0.022 ± 0.012** |
| **precision@10**, mixed precision | 0.009 | 0.018 | 0.043 | **0.023 ± 0.018** |

Chance precision@10 is 0.019 (79 proteins, one test pair each). Best epochs: 11 / 11 / 11
full precision, 10 / 11 / 10 mixed.

**Verdict: within.** AUROC differs by −0.013 against a full-precision spread of 0.038;
precision@10 by +0.001 against 0.012. `--amp` is used for every KIBA cell of every model.

**What this does and does not show.** Three seeds per arm detect only a gross effect: a
mixed-precision mean outside the full-precision spread. They cannot show equivalence, and
seed 1 under mixed precision (0.603) sits below every full-precision seed, which three
seeds cannot tell from seed-to-seed noise on the noisiest cell of the grid (264 validation
pairs). The arms also differ in more than precision: torch 2.10 (Kaggle) vs 2.11 (Colab),
and precision@10 was scored on a CPU for full precision (`run_ladder`, 2026-09-13) and on
the Colab GPU for mixed precision, both with the ladder code as it stood before the
sequence policy of 2026-09-13 (`--no-sequence-policy`), so the two are scored alike.
Because every KIBA cell uses `--amp`, KIBA's models are compared under one protocol; only
DAVIS-versus-KIBA comparisons carry the caveat, which Methods states.

Sources: full precision — account 1, `kaggle_davis_binary_grid36.ipynb` v2 output;
mixed precision — `notebooks/colab_amp_validation.ipynb` (branch `kiba-resume`), outputs
renamed `_ampcheck` in the Colab account's Drive folder `coldsite-amp-validation`. Seed 3
of the mixed-precision run was interrupted and resumed after epoch 13 — the first
epoch-level resume on a GPU (after the `bf2d068` fix).
