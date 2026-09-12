# DAVIS cold-pair volume-matched control — ColdSite-DTI, binary

Cold-pair trains on 15,190 DAVIS rows against 21,039 for random, because it discards
every pair with only one unseen entity. This control asks how much of a cold-pair
accuracy drop is *less data* rather than *a harder task*: ColdSite-DTI is retrained on the
**random** split with its training set cut to **15,190 rows**, three seeds, each seed
drawing its own subsample. Validation (3,006 rows) and test (6,011 rows) are the full
random sets, so these cells are scored on exactly the random test set.

Design: `paper/methods_data_and_evaluation.md` §2.3. Notebook:
`notebooks/colab_volume_control.ipynb` (Colab T4, 2026-09-12). Binary task, DAVIS
pKd ≥ 7.0, the same trainer (`src/model/train.py --train-subsample 15190`) and
checkpoint rule (minimum validation loss among epochs ≥ 10) as the audit grid.

## Test metrics

| seed | AUROC | AUPRC | accuracy | best epoch | training rows |
|---|---|---|---|---|---|
| 1 | 0.8841 | 0.4723 | 0.9273 | 14 | 15,190 |
| 2 | 0.9065 | 0.5776 | 0.9355 | 38 | 15,190 |
| 3 | 0.8708 | 0.4493 | 0.9250 | 14 | 15,190 |
| **mean ± sd** | **0.8871 ± 0.0180** | **0.4998 ± 0.0684** | 0.9292 ± 0.0055 | | |

Sample standard deviation (ddof = 1) over three training seeds. Accuracy is shown for
completeness only: 7.7% of the random test pairs bind, so a model that always says "does
not bind" scores 0.92. Seed 2 kept improving on validation until epoch 38, the others
stopped improving at 14, and seed 2 also scores highest; the spread is real, and a gap
smaller than it is not a finding.

*Best epoch* is 1-indexed (`selection.best_epoch` in the results file). The same file's
top-level `best_epoch` is `train.py`'s 0-indexed epoch counter and reads one lower
(13, 37, 13).

## How to read it

Compared against ColdSite-DTI's full **random** and **cold-pair** binary cells from the
DAVIS audit grid (Kaggle, account 1), once they exist:

| comparison | measures |
|---|---|
| full random − 0.887 | the AUROC cost of training on cold-pair's volume alone |
| 0.887 − cold-pair | the part of the cold-pair drop that is genuine difficulty |

## Where the files are

Google Drive, folder `coldsite-volume-control` (owner mahimagarwal5@gmail.com), nine
files, three per seed:

```
davis_random_binary_seed{1,2,3}_trainsub15190_results.json
coldsite_dti_davis_random_binary_seed{1,2,3}_trainsub15190.pt
coldsite_dti_davis_random_binary_seed{1,2,3}_trainsub15190_history.json
```

The numbers above were read from the three `_results.json` files there on 2026-09-12;
each also records `n_train_rows: 15190` and `volume_matched_control: true`. The
`_trainsub15190` suffix exists because `train.py` names a cut-down run exactly like a
full one; **never copy these files into a grid results folder**, where they would replace
the real random cells.
