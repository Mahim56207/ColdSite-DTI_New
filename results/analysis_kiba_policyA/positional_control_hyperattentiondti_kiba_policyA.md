# Positional control — hyperattentiondti, kiba

Does precision@k survive when each protein is given **another protein's** attention map? A borrowed map keeps the model's positional habit and loses everything specific to the protein. See `src/evaluation/positional_control.py`. k = 10; 1000 reassignments per cell; non-kinase sites exclude cotransport ions (the primary setting).

| seed | level | arm | p@10 | borrowed, absolute (p) | borrowed, relative (p) | same-residue shuffle (p) | in-span shuffle (p) | top-10 in site span | span / chain | first 10 residues | last 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | random | kinase | 0.018 | 0.024 (0.954) | 0.023 (0.900) | 0.025 (0.987) | 0.024 (0.973) | 0.25 | 0.25 | 0.001 | 0.000 |
| 1 | random | non_kinase | 0.013 | 0.013 (0.457) | 0.013 (0.527) | 0.014 (0.594) | 0.013 (0.493) | 0.37 | 0.38 | 0.000 | 0.005 |
| 1 | cold_drug | kinase | 0.025 | 0.024 (0.398) | 0.024 (0.363) | 0.022 (0.202) | 0.026 (0.610) | 0.28 | 0.25 | 0.001 | 0.000 |
| 1 | cold_drug | non_kinase | 0.018 | 0.013 (0.131) | 0.013 (0.155) | 0.012 (0.081) | 0.013 (0.178) | 0.41 | 0.38 | 0.000 | 0.005 |
| 2 | random | kinase | 0.023 | 0.025 (0.754) | 0.023 (0.555) | 0.022 (0.386) | 0.023 (0.533) | 0.25 | 0.25 | 0.001 | 0.000 |
| 2 | random | non_kinase | 0.013 | 0.013 (0.458) | 0.011 (0.364) | 0.011 (0.326) | 0.013 (0.526) | 0.40 | 0.38 | 0.000 | 0.005 |
| 2 | cold_drug | kinase | 0.023 | 0.024 (0.628) | 0.023 (0.482) | 0.021 (0.360) | 0.025 (0.810) | 0.28 | 0.25 | 0.001 | 0.000 |
| 2 | cold_drug | non_kinase | 0.018 | 0.015 (0.267) | 0.013 (0.146) | 0.011 (0.085) | 0.014 (0.235) | 0.39 | 0.38 | 0.000 | 0.005 |
| 3 | random | kinase | 0.055 | 0.024 (0.001*) | 0.024 (0.001*) | 0.025 (0.001*) | 0.032 (0.001*) | 0.33 | 0.25 | 0.001 | 0.000 |
| 3 | random | non_kinase | 0.015 | 0.014 (0.327) | 0.013 (0.318) | 0.011 (0.156) | 0.013 (0.290) | 0.41 | 0.38 | 0.000 | 0.005 |
| 3 | cold_drug | kinase | 0.026 | 0.024 (0.296) | 0.025 (0.368) | 0.023 (0.172) | 0.028 (0.744) | 0.31 | 0.25 | 0.001 | 0.000 |
| 3 | cold_drug | non_kinase | 0.012 | 0.013 (0.577) | 0.012 (0.563) | 0.013 (0.603) | 0.012 (0.583) | 0.41 | 0.38 | 0.000 | 0.005 |

`*` = the real maps beat the null (p < 0.05, before correction). *Same-residue shuffle* = attention permuted only among residues of the same amino acid (keeps residue-type preference, removes location). *In-span shuffle* = attention permuted within the stretch from the first to the last site, and separately outside it (keeps how much attention reaches that stretch -- for the KLIFS pocket, the kinase domain -- removes where in it). *Near an end* = within the first or last 10% of the chain.
