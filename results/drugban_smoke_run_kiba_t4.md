# DrugBAN smoke run, KIBA random, seed 1, Kaggle T4 (ACC3 notebook)

Run by the user from the ACC3 notebook's cells 1-4 plus the checklist cell in `docs/drugban_smoke_run.md`; the numbers below were pasted into the session
and are recorded here because the budget (`src/cloud/budget.py`) derives DrugBAN's hours from them. Only what was observed is written; nothing is inferred.
The environment was the notebook's own (its cell 3b installs the torch / DGL versions pinned for ACC3). The raw notebook output was not saved to the repository.

| arm | exit code | epochs run | mean_epoch_s | note |
|---|---|---|---|---|
| fp32 (full precision) | 0 | 2 | 320.4 | validation AUROC after epoch 2: 0.8183 |
| amp (`--amp`, float16 autocast) | 1 | 0 | n/a | first training batch gives a NaN loss: "loss is nan at epoch 1 train batch 1/1293"; the status file was then not written (FileNotFoundError in the checklist cell) |

Batches per epoch: 1293 = ceil(82,778 / 64) (KIBA random, batch size 64), so full precision costs about 0.25 s per batch including validation.

Consequences recorded in the code: `src/cloud/recipes.py::uses_amp` keeps DrugBAN in full precision on KIBA; `src/cloud/budget.py` uses 320.4 s per epoch at KIBA random.
Not measured: early-stopping epochs of DrugBAN on KIBA (the budget uses the DAVIS best epochs plus patience, a declared extrapolation), and cold_drug epoch time (scaled by batches per epoch).
