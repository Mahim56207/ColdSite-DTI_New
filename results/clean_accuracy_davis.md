# Test accuracy on targets unseen by sequence — DAVIS

Option A (2026-09-13): each cell re-scored by its own trainer's test pass; AUROC on every test row and on the rows whose target is unseen by sequence (`src/evaluation/clean_accuracy.py`, `results/sequence_audit_davis.md`).

| model | level | seed | recorded | re-scored, all rows | reproduces? | unseen by sequence | change | rows kept |
|---|---|---|---|---|---|---|---|---|
| deepdta | cold_target | 1 | 0.9039 | 0.9039 | yes | 0.8801 | -0.0238 | 5168 of 5984 |
| deepdta | cold_target | 2 | 0.9104 | 0.9104 | yes | 0.8861 | -0.0243 | 5168 of 5984 |
| deepdta | cold_target | 3 | 0.9081 | 0.9081 | yes | 0.8861 | -0.0220 | 5168 of 5984 |
| deepdta | cold_pair | 1 | 0.7680 | 0.7680 | yes | 0.7989 | +0.0309 | 1001 of 1144 |
| deepdta | cold_pair | 2 | 0.7125 | 0.7125 | yes | 0.7293 | +0.0168 | 1001 of 1144 |
| deepdta | cold_pair | 3 | 0.7027 | 0.7027 | yes | 0.7183 | +0.0156 | 1001 of 1144 |
| coldsite_dti | cold_target | 1 | 0.8501 | 0.8501 | yes | 0.8195 | -0.0307 | 5168 of 5984 |
| coldsite_dti | cold_target | 2 | 0.8513 | 0.8513 | yes | 0.8496 | -0.0017 | 5168 of 5984 |
| coldsite_dti | cold_target | 3 | 0.8696 | 0.8696 | yes | 0.8366 | -0.0330 | 5168 of 5984 |
| coldsite_dti | cold_pair | 1 | 0.7375 | 0.7375 | yes | 0.7516 | +0.0141 | 1001 of 1144 |
| coldsite_dti | cold_pair | 2 | 0.5568 | 0.5568 | yes | 0.5102 | -0.0466 | 1001 of 1144 |
| coldsite_dti | cold_pair | 3 | 0.5765 | 0.5765 | yes | 0.5590 | -0.0175 | 1001 of 1144 |
| hyperattentiondti | cold_target | 1 | 0.9152 | 0.9152 | yes | 0.8938 | -0.0214 | 5168 of 5984 |
| hyperattentiondti | cold_target | 2 | 0.9161 | 0.9161 | yes | 0.8947 | -0.0214 | 5168 of 5984 |
