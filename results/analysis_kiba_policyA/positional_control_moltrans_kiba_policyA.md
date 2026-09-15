# Positional control — moltrans, kiba

Does precision@k survive when each protein is given **another protein's** attention map? A borrowed map keeps the model's positional habit and loses everything specific to the protein. See `src/evaluation/positional_control.py`. k = 10; 1000 reassignments per cell; non-kinase sites exclude cotransport ions (the primary setting).

| seed | level | arm | p@10 | borrowed, absolute (p) | borrowed, relative (p) | same-residue shuffle (p) | in-span shuffle (p) | top-10 in site span | span / chain | first 10 residues | last 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | random | kinase | 0.021 | 0.023 (0.648) | 0.022 (0.648) | 0.020 (0.389) | 0.022 (0.683) | 0.24 | 0.25 | 0.001 | 0.000 |
| 1 | random | non_kinase | 0.005 | 0.012 (0.966) | 0.011 (0.953) | 0.008 (0.896) | 0.009 (0.925) | 0.34 | 0.38 | 0.000 | 0.005 |
| 1 | cold_drug | kinase | 0.028 | 0.023 (0.124) | 0.023 (0.150) | 0.022 (0.031*) | 0.024 (0.108) | 0.26 | 0.25 | 0.001 | 0.000 |
| 1 | cold_drug | non_kinase | 0.015 | 0.012 (0.286) | 0.012 (0.308) | 0.017 (0.706) | 0.011 (0.199) | 0.35 | 0.38 | 0.000 | 0.005 |
| 2 | random | kinase | 0.053 | 0.035 (0.001*) | 0.030 (0.001*) | 0.023 (0.001*) | 0.026 (0.001*) | 0.28 | 0.25 | 0.001 | 0.000 |
| 2 | random | non_kinase | 0.013 | 0.014 (0.581) | 0.010 (0.301) | 0.011 (0.388) | 0.007 (0.078) | 0.25 | 0.38 | 0.000 | 0.005 |
| 2 | cold_drug | kinase | 0.063 | 0.031 (0.001*) | 0.028 (0.001*) | 0.025 (0.001*) | 0.030 (0.001*) | 0.33 | 0.25 | 0.001 | 0.000 |
| 2 | cold_drug | non_kinase | 0.015 | 0.009 (0.107) | 0.012 (0.289) | 0.015 (0.569) | 0.009 (0.086) | 0.31 | 0.38 | 0.000 | 0.005 |
| 3 | random | kinase | 0.022 | 0.023 (0.592) | 0.025 (0.754) | 0.022 (0.496) | 0.025 (0.773) | 0.27 | 0.25 | 0.001 | 0.000 |
| 3 | random | non_kinase | 0.020 | 0.012 (0.060) | 0.012 (0.065) | 0.013 (0.088) | 0.012 (0.062) | 0.39 | 0.38 | 0.000 | 0.005 |
| 3 | cold_drug | kinase | 0.016 | 0.021 (0.879) | 0.024 (0.975) | 0.022 (0.988) | 0.024 (0.996) | 0.26 | 0.25 | 0.001 | 0.000 |
| 3 | cold_drug | non_kinase | 0.017 | 0.012 (0.210) | 0.012 (0.184) | 0.011 (0.119) | 0.012 (0.162) | 0.33 | 0.38 | 0.000 | 0.005 |

`*` = the real maps beat the null (p < 0.05, before correction). *Same-residue shuffle* = attention permuted only among residues of the same amino acid (keeps residue-type preference, removes location). *In-span shuffle* = attention permuted within the stretch from the first to the last site, and separately outside it (keeps how much attention reaches that stretch -- for the KLIFS pocket, the kinase domain -- removes where in it). *Near an end* = within the first or last 10% of the chain.
