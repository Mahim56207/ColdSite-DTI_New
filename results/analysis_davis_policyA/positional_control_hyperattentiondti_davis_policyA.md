# Positional control — hyperattentiondti, davis

Does precision@k survive when each protein is given **another protein's** attention map? A borrowed map keeps the model's positional habit and loses everything specific to the protein. See `src/evaluation/positional_control.py`. k = 10; 1000 reassignments per cell; non-kinase sites exclude cotransport ions (the primary setting).

| seed | level | arm | p@10 | borrowed, absolute (p) | borrowed, relative (p) | same-residue shuffle (p) | in-span shuffle (p) | top-10 in site span | span / chain | first 10 residues | last 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | random | kinase | 0.035 | 0.021 (0.001*) | 0.020 (0.001*) | 0.021 (0.001*) | 0.026 (0.003*) | 0.29 | 0.23 | 0.004 | 0.001 |
| 1 | random | non_kinase | 0.015 | 0.013 (0.280) | 0.013 (0.312) | 0.012 (0.185) | 0.014 (0.365) | 0.42 | 0.38 | 0.000 | 0.005 |
| 1 | cold_drug | kinase | 0.025 | 0.022 (0.185) | 0.021 (0.124) | 0.021 (0.037*) | 0.021 (0.066) | 0.24 | 0.23 | 0.004 | 0.001 |
| 1 | cold_drug | non_kinase | 0.017 | 0.013 (0.284) | 0.013 (0.248) | 0.013 (0.240) | 0.014 (0.351) | 0.41 | 0.38 | 0.000 | 0.005 |
| 1 | cold_target | kinase | 0.015 | 0.020 (0.864) | 0.020 (0.813) | 0.018 (0.760) | 0.019 (0.839) | 0.21 | 0.22 | 0.000 | 0.000 |
| 1 | cold_target | non_kinase | 0.017 | 0.013 (0.272) | 0.013 (0.233) | 0.012 (0.200) | 0.013 (0.245) | 0.39 | 0.38 | 0.000 | 0.005 |
| 1 | cold_pair | kinase | 0.013 | 0.019 (0.935) | 0.019 (0.930) | 0.016 (0.810) | 0.023 (0.983) | 0.25 | 0.21 | 0.000 | 0.000 |
| 1 | cold_pair | non_kinase | 0.007 | 0.013 (0.964) | 0.013 (0.948) | 0.010 (0.855) | 0.013 (0.955) | 0.40 | 0.38 | 0.000 | 0.005 |
| 2 | random | kinase | 0.040 | 0.022 (0.001*) | 0.021 (0.001*) | 0.021 (0.001*) | 0.024 (0.001*) | 0.26 | 0.23 | 0.004 | 0.001 |
| 2 | random | non_kinase | 0.013 | 0.013 (0.543) | 0.014 (0.559) | 0.011 (0.307) | 0.014 (0.591) | 0.44 | 0.38 | 0.000 | 0.005 |
| 2 | cold_drug | kinase | 0.076 | 0.023 (0.001*) | 0.022 (0.001*) | 0.024 (0.001*) | 0.029 (0.001*) | 0.32 | 0.23 | 0.004 | 0.001 |
| 2 | cold_drug | non_kinase | 0.008 | 0.013 (0.892) | 0.013 (0.902) | 0.015 (0.948) | 0.013 (0.858) | 0.39 | 0.38 | 0.000 | 0.005 |
| 2 | cold_target | kinase | 0.031 | 0.020 (0.035*) | 0.020 (0.049*) | 0.018 (0.013*) | 0.025 (0.202) | 0.28 | 0.22 | 0.000 | 0.000 |
| 2 | cold_target | non_kinase | 0.013 | 0.014 (0.561) | 0.013 (0.511) | 0.013 (0.511) | 0.014 (0.592) | 0.40 | 0.38 | 0.000 | 0.005 |
| 2 | cold_pair | kinase | 0.022 | 0.018 (0.233) | 0.018 (0.267) | 0.016 (0.134) | 0.019 (0.288) | 0.22 | 0.21 | 0.000 | 0.000 |
| 2 | cold_pair | non_kinase | 0.010 | 0.014 (0.815) | 0.013 (0.768) | 0.009 (0.397) | 0.013 (0.752) | 0.41 | 0.38 | 0.000 | 0.005 |
| 3 | random | kinase | 0.028 | 0.022 (0.011*) | 0.021 (0.013*) | 0.023 (0.041*) | 0.026 (0.208) | 0.28 | 0.23 | 0.004 | 0.001 |
| 3 | random | non_kinase | 0.017 | 0.012 (0.202) | 0.013 (0.297) | 0.019 (0.717) | 0.012 (0.207) | 0.39 | 0.38 | 0.000 | 0.005 |
| 3 | cold_drug | kinase | 0.019 | 0.023 (0.914) | 0.021 (0.687) | 0.020 (0.659) | 0.020 (0.678) | 0.24 | 0.23 | 0.004 | 0.001 |
| 3 | cold_drug | non_kinase | 0.015 | 0.012 (0.312) | 0.013 (0.388) | 0.010 (0.144) | 0.013 (0.363) | 0.40 | 0.38 | 0.000 | 0.005 |
| 3 | cold_target | kinase | 0.025 | 0.020 (0.123) | 0.017 (0.067) | 0.017 (0.053) | 0.022 (0.223) | 0.25 | 0.22 | 0.000 | 0.000 |
| 3 | cold_target | non_kinase | 0.022 | 0.014 (0.058) | 0.012 (0.040*) | 0.012 (0.009*) | 0.012 (0.022*) | 0.39 | 0.38 | 0.000 | 0.005 |
| 3 | cold_pair | kinase | 0.032 | 0.020 (0.021*) | 0.019 (0.030*) | 0.021 (0.039*) | 0.024 (0.103) | 0.27 | 0.21 | 0.000 | 0.000 |
| 3 | cold_pair | non_kinase | 0.005 | 0.014 (0.990) | 0.013 (0.988) | 0.014 (0.990) | 0.013 (0.977) | 0.41 | 0.38 | 0.000 | 0.005 |

`*` = the real maps beat the null (p < 0.05, before correction). *Same-residue shuffle* = attention permuted only among residues of the same amino acid (keeps residue-type preference, removes location). *In-span shuffle* = attention permuted within the stretch from the first to the last site, and separately outside it (keeps how much attention reaches that stretch -- for the KLIFS pocket, the kinase domain -- removes where in it). *Near an end* = within the first or last 10% of the chain.
