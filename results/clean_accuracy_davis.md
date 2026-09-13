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
| hyperattentiondti | cold_target | 3 | 0.9132 | 0.9132 | yes | 0.8919 | -0.0213 | 5168 of 5984 |
| hyperattentiondti | cold_pair | 1 | 0.6964 | 0.6964 | yes | 0.7159 | +0.0195 | 1001 of 1144 |
| hyperattentiondti | cold_pair | 2 | 0.6552 | 0.6552 | yes | 0.6610 | +0.0058 | 1001 of 1144 |
| hyperattentiondti | cold_pair | 3 | 0.7301 | 0.7301 | yes | 0.7614 | +0.0312 | 1001 of 1144 |
| moltrans | cold_target | 1 | 0.8680 | 0.8672 | yes | 0.8239 | -0.0432 | 5168 of 5984 |
| moltrans | cold_target | 2 | 0.8785 | 0.8783 | yes | 0.8389 | -0.0395 | 5168 of 5984 |
| moltrans | cold_target | 3 | 0.8756 | 0.8756 | yes | 0.8346 | -0.0410 | 5168 of 5984 |
| moltrans | cold_pair | 1 | 0.5899 | 0.5916 | yes | 0.5480 | -0.0436 | 1001 of 1144 |
| moltrans | cold_pair | 2 | 0.5674 | 0.5552 | **NO** | 0.4940 | -0.0612 | 1001 of 1144 |
| moltrans | cold_pair | 3 | 0.5483 | 0.5490 | yes | 0.5427 | -0.0063 | 1001 of 1144 |
