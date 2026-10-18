# Wave A, ACC3: DrugBAN on KIBA, 6 cells (2026-10-18)

Kaggle account `mahim123agarwal`, harness hash `52bd3335…`, notebook pinned to `5d55c63`, one session of 8.82 h on two T4 GPUs, 17.30 GPU-hours in the harness ledger (the budget estimated 8.8 h and 17.4). **Full precision** (`amp = false`: under float16 DrugBAN's first batch is NaN, `results/drugban_smoke_run_kiba_t4.md`); **torch 2.6.0+cu124** (pinned for DGL, as designed). Files: `~/ColdSite-results/wave_a_kiba_drugban` (outside the repository; SHA-256 manifest there). All six cells pass the marker check and every predictions file matches its marker.

| level | seed | test AUROC | test AUPRC | best epoch | torch | checkpoint sha256 | predictions sha256 |
|---|---|---|---|---|---|---|---|
| random | 1 | 0.9037 | 0.7610 | 20 | 2.6.0+cu124 | ef953273990e… | 96aa6ca940b4… |
| random | 2 | 0.9034 | 0.7588 | 15 | 2.6.0+cu124 | ef304db92c69… | 6f3f03a65957… |
| random | 3 | 0.8998 | 0.7475 | 18 | 2.6.0+cu124 | 7a35dceab16f… | 576d4872ee03… |
| cold-drug | 1 | 0.8460 | 0.6622 | 14 | 2.6.0+cu124 | 72651f242e45… | 0263cb1d8e94… |
| cold-drug | 2 | 0.8429 | 0.6451 | 12 | 2.6.0+cu124 | 1d6e460c7699… | 97982950f5fe… |
| cold-drug | 3 | 0.8438 | 0.6593 | 13 | 2.6.0+cu124 | 79254f86a8a5… | b711878ba6f9… |

Not yet analysed: DrugBAN's attention against UniProt on KIBA (declared family E2, Holm over its own two cells) needs DGL, which the analysis Mac lacks.
