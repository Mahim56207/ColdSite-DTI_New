# Positional control — moltrans, kiba

Does precision@k survive when each protein is given **another protein's** attention map? A borrowed map keeps the model's positional habit and loses everything specific to the protein. See `src/evaluation/positional_control.py`. k = 10; 1000 reassignments per cell; non-kinase sites exclude cotransport ions (the primary setting).

| seed | level | arm | p@10 | borrowed, absolute (p) | borrowed, relative (p) | same-residue shuffle (p) | in-span shuffle (p) | top-10 in site span | span / chain | first 10 residues | last 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | random | kinase | 0.122 | 0.143 (0.978) | 0.150 (0.994) | 0.142 (0.994) | 0.135 (0.997) | 0.24 | 0.27 | 0.005 | 0.008 |
| 1 | random | non_kinase | 0.005 | 0.012 (0.966) | 0.011 (0.953) | 0.008 (0.896) | 0.009 (0.925) | 0.34 | 0.38 | 0.000 | 0.005 |
| 1 | cold_drug | kinase | 0.172 | 0.145 (0.006*) | 0.157 (0.086) | 0.157 (0.032*) | 0.155 (0.002*) | 0.28 | 0.27 | 0.005 | 0.008 |
| 1 | cold_drug | non_kinase | 0.015 | 0.012 (0.286) | 0.012 (0.308) | 0.017 (0.706) | 0.011 (0.199) | 0.35 | 0.38 | 0.000 | 0.005 |
| 2 | random | kinase | 0.216 | 0.159 (0.001*) | 0.153 (0.001*) | 0.159 (0.001*) | 0.186 (0.001*) | 0.33 | 0.27 | 0.005 | 0.008 |
| 2 | random | non_kinase | 0.013 | 0.014 (0.581) | 0.010 (0.301) | 0.011 (0.388) | 0.007 (0.078) | 0.25 | 0.38 | 0.000 | 0.005 |
| 2 | cold_drug | kinase | 0.232 | 0.153 (0.001*) | 0.164 (0.001*) | 0.152 (0.001*) | 0.193 (0.001*) | 0.34 | 0.27 | 0.005 | 0.008 |
| 2 | cold_drug | non_kinase | 0.015 | 0.009 (0.107) | 0.012 (0.289) | 0.015 (0.569) | 0.009 (0.086) | 0.31 | 0.38 | 0.000 | 0.005 |
| 3 | random | kinase | 0.146 | 0.162 (0.952) | 0.174 (0.998) | 0.152 (0.787) | 0.158 (0.981) | 0.29 | 0.27 | 0.005 | 0.008 |
| 3 | random | non_kinase | 0.020 | 0.012 (0.060) | 0.012 (0.065) | 0.013 (0.088) | 0.012 (0.062) | 0.39 | 0.38 | 0.000 | 0.005 |
| 3 | cold_drug | kinase | 0.158 | 0.152 (0.288) | 0.163 (0.671) | 0.149 (0.105) | 0.158 (0.453) | 0.28 | 0.27 | 0.005 | 0.008 |
| 3 | cold_drug | non_kinase | 0.017 | 0.012 (0.210) | 0.012 (0.184) | 0.011 (0.119) | 0.012 (0.162) | 0.33 | 0.38 | 0.000 | 0.005 |

`*` = the real maps beat the null (p < 0.05, before correction). *Same-residue shuffle* = attention permuted only among residues of the same amino acid (keeps residue-type preference, removes location). *In-span shuffle* = attention permuted within the stretch from the first to the last site, and separately outside it (keeps how much attention reaches that stretch -- for the KLIFS pocket, the kinase domain -- removes where in it). *Near an end* = within the first or last 10% of the chain.
