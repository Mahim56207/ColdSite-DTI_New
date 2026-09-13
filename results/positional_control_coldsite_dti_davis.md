# Positional control — coldsite_dti, davis

Does precision@k survive when each protein is given **another protein's** attention map? A borrowed map keeps the model's positional habit and loses everything specific to the protein. See `src/evaluation/positional_control.py`. k = 10; 1000 reassignments per cell; non-kinase sites exclude cotransport ions (the primary setting).

| seed | level | arm | p@10 | borrowed, absolute (p) | borrowed, relative (p) | same-residue shuffle (p) | first 10 residues | last 10 | top-10 near an end | sites near an end |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | random | kinase | 0.022 | 0.021 (0.406) | 0.020 (0.234) | 0.018 (0.049*) | 0.003 | 0.001 | 0.18 | 0.24 |
| 1 | random | non_kinase | 0.008 | 0.014 (0.912) | 0.012 (0.832) | 0.016 (0.974) | 0.000 | 0.005 | 0.20 | 0.09 |
| 1 | cold_drug | kinase | 0.025 | 0.020 (0.026*) | 0.020 (0.049*) | 0.023 (0.210) | 0.003 | 0.001 | 0.15 | 0.24 |
| 1 | cold_drug | non_kinase | 0.013 | 0.014 (0.576) | 0.012 (0.430) | 0.015 (0.668) | 0.000 | 0.005 | 0.21 | 0.09 |
| 1 | cold_target | kinase | 0.014 | 0.021 (0.953) | 0.018 (0.812) | 0.025 (0.987) | 0.000 | 0.000 | 0.17 | 0.25 |
| 1 | cold_target | non_kinase | 0.005 | 0.013 (0.984) | 0.012 (0.966) | 0.009 (0.909) | 0.000 | 0.005 | 0.24 | 0.09 |
| 1 | cold_pair | kinase | 0.018 | 0.018 (0.516) | 0.019 (0.509) | 0.014 (0.194) | 0.000 | 0.000 | 0.13 | 0.26 |
| 1 | cold_pair | non_kinase | 0.060 | 0.014 (0.001*) | 0.012 (0.001*) | 0.044 (0.008*) | 0.000 | 0.005 | 0.17 | 0.09 |
| 2 | random | kinase | 0.009 | 0.021 (1.000) | 0.020 (1.000) | 0.014 (0.996) | 0.003 | 0.001 | 0.17 | 0.24 |
| 2 | random | non_kinase | 0.050 | 0.012 (0.001*) | 0.013 (0.001*) | 0.045 (0.284) | 0.000 | 0.005 | 0.24 | 0.09 |
| 2 | cold_drug | kinase | 0.025 | 0.020 (0.021*) | 0.020 (0.057) | 0.024 (0.348) | 0.003 | 0.001 | 0.17 | 0.24 |
| 2 | cold_drug | non_kinase | 0.035 | 0.013 (0.001*) | 0.012 (0.001*) | 0.026 (0.076) | 0.000 | 0.005 | 0.19 | 0.09 |
| 2 | cold_target | kinase | 0.019 | 0.020 (0.579) | 0.019 (0.546) | 0.015 (0.203) | 0.000 | 0.000 | 0.18 | 0.25 |
| 2 | cold_target | non_kinase | 0.020 | 0.013 (0.103) | 0.012 (0.077) | 0.028 (0.945) | 0.000 | 0.005 | 0.20 | 0.09 |
| 2 | cold_pair | kinase | 0.011 | 0.019 (0.960) | 0.018 (0.927) | 0.009 (0.279) | 0.000 | 0.000 | 0.15 | 0.26 |
| 2 | cold_pair | non_kinase | 0.048 | 0.013 (0.001*) | 0.012 (0.001*) | 0.038 (0.048*) | 0.000 | 0.005 | 0.19 | 0.09 |
| 3 | random | kinase | 0.012 | 0.020 (1.000) | 0.020 (0.999) | 0.019 (1.000) | 0.003 | 0.001 | 0.18 | 0.24 |
| 3 | random | non_kinase | 0.022 | 0.014 (0.055) | 0.012 (0.021*) | 0.017 (0.173) | 0.000 | 0.005 | 0.19 | 0.09 |
| 3 | cold_drug | kinase | 0.013 | 0.020 (0.998) | 0.020 (0.996) | 0.020 (1.000) | 0.003 | 0.001 | 0.16 | 0.24 |
| 3 | cold_drug | non_kinase | 0.038 | 0.013 (0.001*) | 0.012 (0.001*) | 0.021 (0.004*) | 0.000 | 0.005 | 0.18 | 0.09 |
| 3 | cold_target | kinase | 0.019 | 0.019 (0.490) | 0.018 (0.414) | 0.019 (0.530) | 0.000 | 0.000 | 0.19 | 0.25 |
| 3 | cold_target | non_kinase | 0.018 | 0.013 (0.178) | 0.011 (0.097) | 0.022 (0.818) | 0.000 | 0.005 | 0.22 | 0.09 |
| 3 | cold_pair | kinase | 0.008 | 0.019 (0.995) | 0.019 (0.987) | 0.024 (1.000) | 0.000 | 0.000 | 0.16 | 0.26 |
| 3 | cold_pair | non_kinase | 0.023 | 0.013 (0.035*) | 0.013 (0.037*) | 0.018 (0.197) | 0.000 | 0.005 | 0.18 | 0.09 |

`*` = the real maps beat the null (p < 0.05, before correction). *Same-residue shuffle* = attention permuted only among residues of the same amino acid (keeps residue-type preference, removes location). *Near an end* = within the first or last 10% of the chain.
