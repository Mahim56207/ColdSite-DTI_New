# Positional control — coldsite_dti, davis

Does precision@k survive when each protein is given **another protein's** attention map? A borrowed map keeps the model's positional habit and loses everything specific to the protein. See `src/evaluation/positional_control.py`. k = 10; 1000 reassignments per cell; non-kinase sites exclude cotransport ions (the primary setting).

| seed | level | arm | p@10 | borrowed, absolute (p) | borrowed, relative (p) | same-residue shuffle (p) | in-span shuffle (p) | top-10 in site span | span / chain | first 10 residues | last 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | random | kinase | 0.175 | 0.140 (0.001*) | 0.138 (0.001*) | 0.145 (0.001*) | 0.175 (0.516) | 0.31 | 0.24 | 0.010 | 0.009 |
| 1 | cold_drug | kinase | 0.276 | 0.143 (0.001*) | 0.139 (0.001*) | 0.167 (0.001*) | 0.236 (0.001*) | 0.42 | 0.24 | 0.010 | 0.009 |
| 1 | cold_target | kinase | 0.219 | 0.140 (0.001*) | 0.135 (0.001*) | 0.153 (0.001*) | 0.197 (0.013*) | 0.35 | 0.24 | 0.000 | 0.003 |
| 1 | cold_pair | kinase | 0.300 | 0.143 (0.001*) | 0.140 (0.001*) | 0.161 (0.001*) | 0.271 (0.014*) | 0.47 | 0.23 | 0.000 | 0.019 |
| 2 | random | kinase | 0.208 | 0.142 (0.001*) | 0.139 (0.001*) | 0.149 (0.001*) | 0.184 (0.001*) | 0.32 | 0.24 | 0.010 | 0.009 |
| 2 | cold_drug | kinase | 0.243 | 0.141 (0.001*) | 0.140 (0.001*) | 0.139 (0.001*) | 0.209 (0.001*) | 0.37 | 0.24 | 0.010 | 0.009 |
| 2 | cold_target | kinase | 0.191 | 0.131 (0.001*) | 0.136 (0.001*) | 0.141 (0.001*) | 0.181 (0.131) | 0.32 | 0.24 | 0.000 | 0.003 |
| 2 | cold_pair | kinase | 0.259 | 0.140 (0.001*) | 0.138 (0.001*) | 0.157 (0.001*) | 0.225 (0.002*) | 0.39 | 0.23 | 0.000 | 0.019 |
| 3 | random | kinase | 0.236 | 0.140 (0.001*) | 0.140 (0.001*) | 0.161 (0.001*) | 0.199 (0.001*) | 0.35 | 0.24 | 0.010 | 0.009 |
| 3 | cold_drug | kinase | 0.318 | 0.145 (0.001*) | 0.141 (0.001*) | 0.155 (0.001*) | 0.249 (0.001*) | 0.44 | 0.24 | 0.010 | 0.009 |
| 3 | cold_target | kinase | 0.278 | 0.138 (0.001*) | 0.129 (0.001*) | 0.155 (0.001*) | 0.231 (0.001*) | 0.41 | 0.24 | 0.000 | 0.003 |
| 3 | cold_pair | kinase | 0.240 | 0.134 (0.001*) | 0.137 (0.001*) | 0.144 (0.001*) | 0.231 (0.199) | 0.40 | 0.23 | 0.000 | 0.019 |

`*` = the real maps beat the null (p < 0.05, before correction). *Same-residue shuffle* = attention permuted only among residues of the same amino acid (keeps residue-type preference, removes location). *In-span shuffle* = attention permuted within the stretch from the first to the last site, and separately outside it (keeps how much attention reaches that stretch -- for the KLIFS pocket, the kinase domain -- removes where in it). *Near an end* = within the first or last 10% of the chain.
