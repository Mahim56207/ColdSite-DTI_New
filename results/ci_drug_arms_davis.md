# Confidence intervals — precision@10, resampling proteins

95% percentile intervals from 10,000 resamples of the proteins a cell scored, each protein carried in with all of its seeds (`src/evaluation/bootstrap_ci.py`). The mean is the same number the ladder reports; the interval says how precisely a cell of this size measures it.

| ground truth | model | level | n | precision@10 | 95% CI | width |
|---|---|---|---|---|---|---|
| drug | coldsite_dti | random | 27 | 0.034 | 0.015–0.057 | 0.041 |
| drug | coldsite_dti | cold-drug | 55 | 0.038 | 0.029–0.047 | 0.018 |
| drug | coldsite_dti | cold-target | 10 | 0.035 | 0.010–0.067 | 0.057 |
| drug | coldsite_dti | cold-pair | 6 | 0.017 | 0.006–0.028 | 0.022 |
| drug | hyperattentiondti | random | 36 | 0.035 | 0.027–0.044 | 0.018 |
| drug | hyperattentiondti | cold-drug | 58 | 0.020 | 0.013–0.029 | 0.017 |
| drug | hyperattentiondti | cold-target | 20 | 0.058 | 0.043–0.073 | 0.030 |
| drug | hyperattentiondti | cold-pair | 6 | 0.067 | 0.044–0.089 | 0.044 |
| drug | moltrans | random | 36 | 0.011 | 0.005–0.019 | 0.015 |
| drug | moltrans | cold-drug | 58 | 0.028 | 0.020–0.037 | 0.018 |
| drug | moltrans | cold-target | 20 | 0.037 | 0.025–0.048 | 0.023 |
| drug | moltrans | cold-pair | 6 | 0.006 | 0.000–0.017 | 0.017 |
| swapped | coldsite_dti | random | 20 | 0.025 | 0.005–0.053 | 0.048 |
| swapped | coldsite_dti | cold-drug | 36 | 0.032 | 0.020–0.046 | 0.025 |
| swapped | coldsite_dti | cold-target | 4 | 0.038 | 0.007–0.082 | 0.075 |
| swapped | coldsite_dti | cold-pair | 3 | 0.011 | 0.000–0.033 | 0.033 |
| swapped | hyperattentiondti | random | 29 | 0.033 | 0.025–0.041 | 0.016 |
| swapped | hyperattentiondti | cold-drug | 39 | 0.015 | 0.005–0.027 | 0.022 |
| swapped | hyperattentiondti | cold-target | 14 | 0.060 | 0.043–0.076 | 0.033 |
| swapped | hyperattentiondti | cold-pair | 3 | 0.067 | 0.033–0.100 | 0.067 |
| swapped | moltrans | random | 29 | 0.009 | 0.002–0.021 | 0.018 |
| swapped | moltrans | cold-drug | 39 | 0.056 | 0.042–0.072 | 0.030 |
| swapped | moltrans | cold-target | 14 | 0.045 | 0.033–0.057 | 0.024 |
| swapped | moltrans | cold-pair | 3 | 0.000 | 0.000–0.000 | 0.000 |
| paired | coldsite_dti | random | 20 | 0.032 | 0.010–0.062 | 0.052 |
| paired | coldsite_dti | cold-drug | 36 | 0.030 | 0.019–0.043 | 0.023 |
| paired | coldsite_dti | cold-target | 4 | 0.046 | 0.008–0.096 | 0.087 |
| paired | coldsite_dti | cold-pair | 3 | 0.011 | 0.000–0.033 | 0.033 |
| paired | hyperattentiondti | random | 29 | 0.036 | 0.028–0.045 | 0.017 |
| paired | hyperattentiondti | cold-drug | 39 | 0.025 | 0.015–0.036 | 0.021 |
| paired | hyperattentiondti | cold-target | 14 | 0.060 | 0.040–0.079 | 0.038 |
| paired | hyperattentiondti | cold-pair | 3 | 0.078 | 0.067–0.100 | 0.033 |
| paired | moltrans | random | 29 | 0.011 | 0.005–0.021 | 0.016 |
| paired | moltrans | cold-drug | 39 | 0.036 | 0.025–0.048 | 0.023 |
| paired | moltrans | cold-target | 14 | 0.048 | 0.036–0.060 | 0.024 |
| paired | moltrans | cold-pair | 3 | 0.000 | 0.000–0.000 | 0.000 |

The widest interval is coldsite_dti at cold-target against paired (4 proteins, 0.087 wide): a cell that small cannot separate anything from chance, and the audit should say so rather than quote its point estimate.
