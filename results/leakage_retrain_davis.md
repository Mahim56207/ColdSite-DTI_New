# What DAVIS's sequence leakage is worth — deepdta, retrained without it

Three training sets, one test set each level (`src/data/seqclean_splits.py`, `src/evaluation/leakage_retrain.py`). `seqmatched` keeps the leak at `seqclean`'s row count and positive count, so **seqmatched − seqclean is the leak** and **original − seqmatched is the smaller training set**. Test AUROC, mean ± sd over seeds, scored by the trainer's own test pass on CPU.

## cold-target

Test rows: 816 on targets seen by sequence in the original training set, 5168 on the rest.

| arm | train rows | all rows | leaked targets | unleaked targets |
|---|---|---|---|---|
| original | 21080 | 0.907 ± 0.003 | 0.950 ± 0.005 | 0.884 ± 0.003 |
| seqmatched | 17748 | 0.888 ± 0.011 | 0.945 ± 0.003 | 0.857 ± 0.015 |
| seqclean | 17748 | 0.869 ± 0.013 | 0.829 ± 0.005 | 0.876 ± 0.017 |

**seqmatched − seqclean = the leak**, per seed (same seed both sides):

| seed | all | leaked | unleaked |
|---|---|---|---|
| 1 | +0.005 | +0.122 | -0.037 |
| 2 | +0.015 | +0.108 | -0.029 |
| 3 | +0.037 | +0.118 | +0.009 |
| **mean** | **+0.019** | **+0.116** | **-0.019** |

**original − seqmatched = fewer training rows**, per seed (same seed both sides):

| seed | all | leaked | unleaked |
|---|---|---|---|
| 1 | +0.015 | +0.002 | +0.021 |
| 2 | +0.034 | +0.014 | +0.045 |
| 3 | +0.010 | -0.000 | +0.015 |
| **mean** | **+0.020** | **+0.005** | **+0.027** |

## cold-pair

Test rows: 143 on targets seen by sequence in the original training set, 1001 on the rest.

| arm | train rows | all rows | leaked targets | unleaked targets |
|---|---|---|---|---|
| original | 15190 | 0.728 ± 0.035 | 0.689 ± 0.006 | 0.749 ± 0.044 |
| seqmatched | 12936 | 0.666 ± 0.027 | 0.648 ± 0.031 | 0.686 ± 0.030 |
| seqclean | 12936 | 0.686 ± 0.018 | 0.669 ± 0.034 | 0.688 ± 0.033 |

**seqmatched − seqclean = the leak**, per seed (same seed both sides):

| seed | all | leaked | unleaked |
|---|---|---|---|
| 1 | -0.024 | -0.043 | -0.012 |
| 2 | -0.007 | +0.048 | +0.001 |
| 3 | -0.030 | -0.068 | +0.003 |
| **mean** | **-0.021** | **-0.021** | **-0.003** |

**original − seqmatched = fewer training rows**, per seed (same seed both sides):

| seed | all | leaked | unleaked |
|---|---|---|---|
| 1 | +0.092 | +0.035 | +0.112 |
| 2 | +0.026 | +0.008 | +0.014 |
| 3 | +0.067 | +0.079 | +0.064 |
| **mean** | **+0.062** | **+0.040** | **+0.063** |

## Checks

- PASS: all 18 cells' re-scored AUROC reproduce their recorded value within 0.005
- PASS: every seed of an arm trained on the same number of rows
- PASS: seqclean and seqmatched trained on the same number of rows (the control is volume-matched)
