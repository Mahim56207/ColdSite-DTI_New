# Speed test — KIBA random, Tesla T4 (Colab, 2026-09-12)

`python -m src.model.benchmark_speed --dataset kiba --split random --steps 50 --warmup 10`
(`notebooks/colab_speed_test.ipynb`); torch 2.11.0+cu128; grid batch sizes; nothing
trained for real. The fp32 timings agree with the Kaggle grid's own logs
(HyperAttentionDTI ~0.50 s/batch, MolTrans ~0.21 s/batch on DAVIS), so the projections
below rest on the same hardware the grid uses. Copied from Drive
`coldsite-speed-test/speed_kiba.{md,json}`.

| model | setting | s/batch (train) | speed-up | peak GB | non-finite steps |
|---|---|---|---|---|---|
| deepdta | fp32 | 0.083 | 1.00× | 0.6 | 0 |
| deepdta | fp32+benchmark | 0.105 | 0.79× | 0.6 | 0 |
| deepdta | amp+benchmark | 0.036 | 2.29× | 5.1 | 0 |
| coldsite_dti | fp32 | 0.387 | 1.00× | 8.2 | 0 |
| coldsite_dti | fp32+benchmark | 0.372 | 1.04× | 8.2 | 0 |
| coldsite_dti | amp+benchmark | 0.336 | 1.15× | 7.0 | 0 |
| hyperattentiondti | fp32 | 0.552 | 1.00× | 6.6 | 0 |
| hyperattentiondti | fp32+benchmark | 0.545 | 1.01× | 6.6 | 0 |
| hyperattentiondti | amp+benchmark | 0.273 | 2.02× | 6.5 | 0 |
| moltrans | fp32 | 0.228 | 1.00× | 5.5 | 0 |
| moltrans | fp32+benchmark | 0.242 | 0.94× | 5.5 | 0 |
| moltrans | amp+benchmark | 0.175 | 1.30× | 5.1 | 0 |

cudnn autotuning alone gives nothing (0.79–1.04×); the amp+benchmark gains are mixed
precision's.

## How far float16 moves the outputs (same weights, at initialisation)

| model | max abs diff | mean relative diff |
|---|---|---|
| deepdta | 0.0007253 | 0.0002004 |
| coldsite_dti | 0.0005164 | 0.0002047 |
| hyperattentiondti | 0.0005569 | 0.0002379 |
| moltrans | 0.0005535 | 0.0003079 |

Measured on untrained weights; a trained-cell check is still required before use.

## KIBA hours per cell, projected

| model | setting | random: min/epoch | random: h at 25 / 36 ep | cold_pair: h at 25 / 36 ep |
|---|---|---|---|---|
| deepdta | fp32 | 0.5 | 0.2 / 0.3 | 0.1 / 0.2 |
| deepdta | amp+benchmark | 0.2 | 0.1 / 0.1 | 0.1 / 0.1 |
| coldsite_dti | fp32 | 8.7 | 3.6 / 5.2 | 2.5 / 3.5 |
| coldsite_dti | amp+benchmark | 7.5 | 3.1 / 4.5 | 2.1 / 3.1 |
| hyperattentiondti | fp32 | 24.7 | 10.3 / 14.8 | 7.0 / 10.1 |
| hyperattentiondti | amp+benchmark | 12.1 | 5.0 / 7.3 | 3.5 / 5.0 |
| moltrans | fp32 | 20.7 | 8.6 / 12.4 | 5.8 / 8.3 |
| moltrans | amp+benchmark | 15.7 | 6.5 / 9.4 | 4.4 / 6.4 |

25 epochs is the minimum the early-stopping rule allows; 36 the median of the DAVIS
training histories (range 26–94). A cell over ~10.5 h cannot finish inside one 11-hour
Kaggle commit without resume: with mixed precision that is MolTrans past ~40 epochs and
HyperAttentionDTI past ~52.

Totals, all 48 KIBA cells: fp32 ~250–360 GPU-h; mixed precision ~165–235. Random +
cold-drug only (24 cells): fp32 ~175–190; mixed precision ~90–130.
