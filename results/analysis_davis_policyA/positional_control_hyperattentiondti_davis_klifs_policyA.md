# Positional control — hyperattentiondti, davis

Does precision@k survive when each protein is given **another protein's** attention map? A borrowed map keeps the model's positional habit and loses everything specific to the protein. See `src/evaluation/positional_control.py`. k = 10; 1000 reassignments per cell; non-kinase sites exclude cotransport ions (the primary setting).

| seed | level | arm | p@10 | borrowed, absolute (p) | borrowed, relative (p) | same-residue shuffle (p) | in-span shuffle (p) | top-10 in site span | span / chain | first 10 residues | last 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | random | kinase | 0.286 | 0.157 (0.001*) | 0.151 (0.001*) | 0.150 (0.001*) | 0.221 (0.001*) | 0.39 | 0.25 | 0.011 | 0.010 |
| 1 | random | non_kinase | 0.015 | 0.013 (0.280) | 0.013 (0.312) | 0.012 (0.185) | 0.014 (0.365) | 0.42 | 0.38 | 0.000 | 0.005 |
| 1 | cold_drug | kinase | 0.199 | 0.151 (0.001*) | 0.147 (0.001*) | 0.152 (0.001*) | 0.167 (0.001*) | 0.29 | 0.25 | 0.011 | 0.011 |
| 1 | cold_drug | non_kinase | 0.017 | 0.013 (0.284) | 0.013 (0.248) | 0.013 (0.240) | 0.014 (0.351) | 0.41 | 0.38 | 0.000 | 0.005 |
| 1 | cold_target | kinase | 0.157 | 0.141 (0.160) | 0.143 (0.217) | 0.132 (0.040*) | 0.144 (0.093) | 0.25 | 0.25 | 0.000 | 0.000 |
| 1 | cold_target | non_kinase | 0.017 | 0.013 (0.272) | 0.013 (0.233) | 0.012 (0.200) | 0.013 (0.245) | 0.39 | 0.38 | 0.000 | 0.005 |
| 1 | cold_pair | kinase | 0.238 | 0.139 (0.001*) | 0.150 (0.001*) | 0.153 (0.001*) | 0.181 (0.001*) | 0.32 | 0.24 | 0.000 | 0.021 |
| 1 | cold_pair | non_kinase | 0.007 | 0.013 (0.964) | 0.013 (0.948) | 0.010 (0.855) | 0.013 (0.955) | 0.40 | 0.38 | 0.000 | 0.005 |
| 2 | random | kinase | 0.215 | 0.154 (0.001*) | 0.150 (0.001*) | 0.155 (0.001*) | 0.197 (0.001*) | 0.34 | 0.25 | 0.011 | 0.010 |
| 2 | random | non_kinase | 0.013 | 0.013 (0.543) | 0.014 (0.559) | 0.011 (0.307) | 0.014 (0.591) | 0.44 | 0.38 | 0.000 | 0.005 |
| 2 | cold_drug | kinase | 0.248 | 0.156 (0.001*) | 0.152 (0.001*) | 0.146 (0.001*) | 0.206 (0.001*) | 0.36 | 0.25 | 0.011 | 0.011 |
| 2 | cold_drug | non_kinase | 0.008 | 0.013 (0.892) | 0.013 (0.902) | 0.015 (0.948) | 0.013 (0.858) | 0.39 | 0.38 | 0.000 | 0.005 |
| 2 | cold_target | kinase | 0.219 | 0.144 (0.001*) | 0.150 (0.001*) | 0.144 (0.001*) | 0.184 (0.002*) | 0.33 | 0.25 | 0.000 | 0.000 |
| 2 | cold_target | non_kinase | 0.013 | 0.014 (0.561) | 0.013 (0.511) | 0.013 (0.511) | 0.014 (0.592) | 0.40 | 0.38 | 0.000 | 0.005 |
| 2 | cold_pair | kinase | 0.144 | 0.136 (0.281) | 0.140 (0.415) | 0.130 (0.143) | 0.142 (0.435) | 0.25 | 0.24 | 0.000 | 0.021 |
| 2 | cold_pair | non_kinase | 0.010 | 0.014 (0.815) | 0.013 (0.768) | 0.009 (0.397) | 0.013 (0.752) | 0.41 | 0.38 | 0.000 | 0.005 |
| 3 | random | kinase | 0.229 | 0.158 (0.001*) | 0.152 (0.001*) | 0.151 (0.001*) | 0.188 (0.001*) | 0.33 | 0.25 | 0.011 | 0.010 |
| 3 | random | non_kinase | 0.017 | 0.012 (0.202) | 0.013 (0.297) | 0.019 (0.717) | 0.012 (0.207) | 0.39 | 0.38 | 0.000 | 0.005 |
| 3 | cold_drug | kinase | 0.127 | 0.151 (1.000) | 0.144 (0.996) | 0.139 (0.981) | 0.145 (1.000) | 0.26 | 0.25 | 0.011 | 0.011 |
| 3 | cold_drug | non_kinase | 0.015 | 0.012 (0.312) | 0.013 (0.388) | 0.010 (0.144) | 0.013 (0.363) | 0.40 | 0.38 | 0.000 | 0.005 |
| 3 | cold_target | kinase | 0.179 | 0.147 (0.020*) | 0.140 (0.014*) | 0.131 (0.001*) | 0.159 (0.018*) | 0.28 | 0.25 | 0.000 | 0.000 |
| 3 | cold_target | non_kinase | 0.022 | 0.014 (0.058) | 0.012 (0.040*) | 0.012 (0.009*) | 0.012 (0.022*) | 0.39 | 0.38 | 0.000 | 0.005 |
| 3 | cold_pair | kinase | 0.170 | 0.143 (0.022*) | 0.143 (0.045*) | 0.152 (0.086) | 0.179 (0.799) | 0.32 | 0.24 | 0.000 | 0.021 |
| 3 | cold_pair | non_kinase | 0.005 | 0.014 (0.990) | 0.013 (0.988) | 0.014 (0.990) | 0.013 (0.977) | 0.41 | 0.38 | 0.000 | 0.005 |

`*` = the real maps beat the null (p < 0.05, before correction). *Same-residue shuffle* = attention permuted only among residues of the same amino acid (keeps residue-type preference, removes location). *In-span shuffle* = attention permuted within the stretch from the first to the last site, and separately outside it (keeps how much attention reaches that stretch -- for the KLIFS pocket, the kinase domain -- removes where in it). *Near an end* = within the first or last 10% of the chain.
