# Positional control — hyperattentiondti, kiba

Does precision@k survive when each protein is given **another protein's** attention map? A borrowed map keeps the model's positional habit and loses everything specific to the protein. See `src/evaluation/positional_control.py`. k = 10; 1000 reassignments per cell; non-kinase sites exclude cotransport ions (the primary setting).

| seed | level | arm | p@10 | borrowed, absolute (p) | borrowed, relative (p) | same-residue shuffle (p) | in-span shuffle (p) | top-10 in site span | span / chain | first 10 residues | last 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | random | kinase | 0.202 | 0.157 (0.001*) | 0.161 (0.001*) | 0.154 (0.001*) | 0.187 (0.012*) | 0.33 | 0.27 | 0.005 | 0.008 |
| 1 | random | non_kinase | 0.013 | 0.013 (0.457) | 0.013 (0.527) | 0.014 (0.594) | 0.013 (0.493) | 0.37 | 0.38 | 0.000 | 0.005 |
| 1 | cold_drug | kinase | 0.171 | 0.153 (0.009*) | 0.158 (0.124) | 0.149 (0.004*) | 0.182 (0.968) | 0.32 | 0.27 | 0.005 | 0.008 |
| 1 | cold_drug | non_kinase | 0.018 | 0.013 (0.131) | 0.013 (0.155) | 0.012 (0.081) | 0.013 (0.178) | 0.41 | 0.38 | 0.000 | 0.005 |
| 2 | random | kinase | 0.204 | 0.154 (0.001*) | 0.155 (0.001*) | 0.160 (0.001*) | 0.173 (0.001*) | 0.31 | 0.27 | 0.005 | 0.008 |
| 2 | random | non_kinase | 0.013 | 0.013 (0.458) | 0.011 (0.364) | 0.011 (0.326) | 0.013 (0.526) | 0.40 | 0.38 | 0.000 | 0.005 |
| 2 | cold_drug | kinase | 0.198 | 0.156 (0.001*) | 0.159 (0.001*) | 0.153 (0.001*) | 0.178 (0.002*) | 0.32 | 0.27 | 0.005 | 0.008 |
| 2 | cold_drug | non_kinase | 0.018 | 0.015 (0.267) | 0.013 (0.146) | 0.011 (0.085) | 0.014 (0.235) | 0.39 | 0.38 | 0.000 | 0.005 |
| 3 | random | kinase | 0.216 | 0.157 (0.001*) | 0.161 (0.001*) | 0.154 (0.001*) | 0.196 (0.006*) | 0.35 | 0.27 | 0.005 | 0.008 |
| 3 | random | non_kinase | 0.015 | 0.014 (0.327) | 0.013 (0.318) | 0.011 (0.156) | 0.013 (0.290) | 0.41 | 0.38 | 0.000 | 0.005 |
| 3 | cold_drug | kinase | 0.219 | 0.158 (0.001*) | 0.163 (0.001*) | 0.157 (0.001*) | 0.186 (0.001*) | 0.33 | 0.27 | 0.005 | 0.008 |
| 3 | cold_drug | non_kinase | 0.012 | 0.013 (0.577) | 0.012 (0.563) | 0.013 (0.603) | 0.012 (0.583) | 0.41 | 0.38 | 0.000 | 0.005 |

`*` = the real maps beat the null (p < 0.05, before correction). *Same-residue shuffle* = attention permuted only among residues of the same amino acid (keeps residue-type preference, removes location). *In-span shuffle* = attention permuted within the stretch from the first to the last site, and separately outside it (keeps how much attention reaches that stretch -- for the KLIFS pocket, the kinase domain -- removes where in it). *Near an end* = within the first or last 10% of the chain.
