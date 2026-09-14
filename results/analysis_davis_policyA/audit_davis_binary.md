# Audit grid

| Model | Warm | Cold-Drug | Cold-Target | Cold-Pair | Drop |
|---|---|---|---|---|---|
| coldsite_dti | 0.015 ± 0.007 | 0.022 ± 0.009 | 0.017 ± 0.002 | 0.013 ± 0.005 | +0.002 |
| hyperattentiondti | 0.034 ± 0.006 | 0.040 ± 0.031 | 0.024 ± 0.008 | 0.022 ± 0.010 | +0.012 |
| uniform_control | 0.020 ± 0.001 | 0.020 ± 0.001 | 0.018 ± 0.005 | 0.017 ± 0.001 | +0.003 |

`±` is the standard deviation over seeds. `!` marks a cell with fewer than 3 seeds — not a usable estimate.

## Significance (Holm-Bonferroni over the whole grid)

1 of 12 cells survive correction at alpha = 0.05.

- `hyperattentiondti|davis|random` p=0.001996 (threshold 0.004167) -> yes
- `coldsite_dti|davis|cold_drug` p=0.005988 (threshold 0.004545) -> no
- `hyperattentiondti|davis|cold_drug` p=0.0499 (threshold 0.005) -> no
- `hyperattentiondti|davis|cold_target` p=0.1078 (threshold 0.005556) -> no
- `hyperattentiondti|davis|cold_pair` p=0.3014 (threshold 0.00625) -> no
- `uniform_control|davis|cold_drug` p=0.4311 (threshold 0.007143) -> no
- `uniform_control|davis|random` p=0.6128 (threshold 0.008333) -> no
- `coldsite_dti|davis|cold_target` p=0.6966 (threshold 0.01) -> no
- `uniform_control|davis|cold_pair` p=0.6966 (threshold 0.0125) -> no
- `uniform_control|davis|cold_target` p=0.7784 (threshold 0.01667) -> no
- `coldsite_dti|davis|cold_pair` p=0.9381 (threshold 0.025) -> no
- `coldsite_dti|davis|random` p=1 (threshold 0.05) -> no

## Confound control (kinase vs non-kinase)

**No stratified comparison was possible.** Fewer than 20 non-kinase targets were available in every cell. The unstratified ladder must NOT be presented as if the kinase confound were absent -- state it as an explicit limitation in the Discussion, or enlarge the antiviral subset (Track A, Priority 1).