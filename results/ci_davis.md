# Confidence intervals — precision@10, resampling proteins

95% percentile intervals from 10,000 resamples of the proteins a cell scored, each protein carried in with all of its seeds (`src/evaluation/bootstrap_ci.py`). The mean is the same number the ladder reports; the interval says how precisely a cell of this size measures it.

| ground truth | model | level | n | precision@10 | 95% CI | width |
|---|---|---|---|---|---|---|
| uniprot | coldsite_dti | random | 349 | 0.015 | 0.013–0.018 | 0.005 |
| uniprot | coldsite_dti | cold-drug | 349 | 0.022 | 0.019–0.024 | 0.006 |
| uniprot | coldsite_dti | cold-target | 68 | 0.017 | 0.012–0.023 | 0.010 |
| uniprot | coldsite_dti | cold-pair | 72 | 0.013 | 0.007–0.020 | 0.013 |
| uniprot | hyperattentiondti | random | 349 | 0.034 | 0.030–0.038 | 0.008 |
| uniprot | hyperattentiondti | cold-drug | 349 | 0.040 | 0.037–0.044 | 0.007 |
| uniprot | hyperattentiondti | cold-target | 68 | 0.025 | 0.018–0.031 | 0.013 |
| uniprot | hyperattentiondti | cold-pair | 72 | 0.022 | 0.016–0.029 | 0.013 |
| uniprot | moltrans | random | 349 | 0.021 | 0.017–0.026 | 0.008 |
| uniprot | moltrans | cold-drug | 349 | 0.026 | 0.021–0.031 | 0.010 |
| uniprot | moltrans | cold-target | 68 | 0.026 | 0.018–0.035 | 0.017 |
| uniprot | moltrans | cold-pair | 72 | 0.020 | 0.013–0.027 | 0.014 |
| klifs | coldsite_dti | random | 350 | 0.219 | 0.208–0.229 | 0.021 |
| klifs | coldsite_dti | cold-drug | 350 | 0.299 | 0.286–0.312 | 0.025 |
| klifs | coldsite_dti | cold-target | 67 | 0.243 | 0.218–0.268 | 0.050 |
| klifs | coldsite_dti | cold-pair | 71 | 0.271 | 0.241–0.303 | 0.062 |
| klifs | hyperattentiondti | random | 350 | 0.242 | 0.232–0.253 | 0.021 |
| klifs | hyperattentiondti | cold-drug | 350 | 0.193 | 0.183–0.203 | 0.019 |
| klifs | hyperattentiondti | cold-target | 67 | 0.186 | 0.163–0.209 | 0.046 |
| klifs | hyperattentiondti | cold-pair | 71 | 0.185 | 0.165–0.206 | 0.040 |
| klifs | moltrans | random | 350 | 0.157 | 0.143–0.170 | 0.027 |
| klifs | moltrans | cold-drug | 350 | 0.154 | 0.142–0.166 | 0.024 |
| klifs | moltrans | cold-target | 67 | 0.176 | 0.150–0.201 | 0.051 |
| klifs | moltrans | cold-pair | 71 | 0.136 | 0.114–0.158 | 0.044 |

The widest interval is coldsite_dti at cold-pair against klifs (71 proteins, 0.062 wide): a cell that small cannot separate anything from chance, and the audit should say so rather than quote its point estimate.
