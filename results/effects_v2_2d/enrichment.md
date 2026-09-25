# Enrichment over chance, with confidence intervals

Mean precision@10 divided by mean chance over the same resampled targets. TWO-WAY bootstrap (`--resample seeds_and_targets`), 10,000 resamples, 95% percentile intervals, `default_rng(0)`: each resample draws the targets with replacement and, independently, the seeds with replacement from the cell's seeds, and averages over the drawn (target, seed) grid, so an interval carries seed-to-seed variance as well as target sampling. Point values are identical to the target-only tables of `results/effects_v2/` (amendment section 5); only the intervals differ. Effect sizes only: no threshold is applied. `excess` is precision minus chance. Ceiling is the recorded mean ceiling.

| family | ground truth | dataset | model | level | n | precision@10 | 95% CI | chance | ceiling | enrichment | 95% CI | excess | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | UniProt | DAVIS | coldsite_dti | random | 349 | 0.015 | 0.009–0.023 | 0.0204 | 0.989 | 0.74 | 0.43–1.11 | -0.0053 | -0.0116–0.0022 |
| P1 | UniProt | DAVIS | coldsite_dti | cold-drug | 349 | 0.022 | 0.013–0.030 | 0.0204 | 0.989 | 1.06 | 0.62–1.45 | 0.0013 | -0.0077–0.0092 |
| P1 | UniProt | DAVIS | coldsite_dti | cold-target | 68 | 0.017 | 0.010–0.024 | 0.0194 | 0.991 | 0.89 | 0.54–1.25 | -0.0022 | -0.0088–0.0048 |
| P1 | UniProt | DAVIS | coldsite_dti | cold-pair | 72 | 0.013 | 0.005–0.022 | 0.0190 | 0.990 | 0.68 | 0.27–1.15 | -0.0060 | -0.0139–0.0028 |
| P1 | UniProt | DAVIS | drugban | random | 349 | 0.018 | 0.014–0.022 | 0.0204 | 0.989 | 0.87 | 0.67–1.07 | -0.0027 | -0.0068–0.0015 |
| P1 | UniProt | DAVIS | drugban | cold-drug | 349 | 0.021 | 0.017–0.025 | 0.0204 | 0.989 | 1.03 | 0.86–1.20 | 0.0006 | -0.0028–0.0042 |
| P1 | UniProt | DAVIS | drugban | cold-target | 68 | 0.023 | 0.013–0.033 | 0.0194 | 0.991 | 1.16 | 0.68–1.69 | 0.0032 | -0.0061–0.0136 |
| P1 | UniProt | DAVIS | drugban | cold-pair | 72 | 0.027 | 0.017–0.039 | 0.0190 | 0.990 | 1.44 | 0.90–2.08 | 0.0083 | -0.0018–0.0203 |
| P1 | UniProt | DAVIS | hyperattentiondti | random | 349 | 0.034 | 0.027–0.041 | 0.0204 | 0.989 | 1.66 | 1.33–2.00 | 0.0135 | 0.0068–0.0205 |
| P1 | UniProt | DAVIS | hyperattentiondti | cold-drug | 349 | 0.040 | 0.019–0.074 | 0.0204 | 0.989 | 1.98 | 0.94–3.65 | 0.0200 | -0.0013–0.0540 |
| P1 | UniProt | DAVIS | hyperattentiondti | cold-target | 68 | 0.025 | 0.014–0.036 | 0.0194 | 0.991 | 1.27 | 0.73–1.88 | 0.0051 | -0.0053–0.0167 |
| P1 | UniProt | DAVIS | hyperattentiondti | cold-pair | 72 | 0.022 | 0.011–0.035 | 0.0190 | 0.990 | 1.17 | 0.58–1.82 | 0.0032 | -0.0079–0.0156 |
| P1 | UniProt | DAVIS | moltrans | random | 349 | 0.021 | 0.016–0.027 | 0.0204 | 0.989 | 1.05 | 0.79–1.33 | 0.0010 | -0.0044–0.0068 |
| P1 | UniProt | DAVIS | moltrans | cold-drug | 349 | 0.026 | 0.019–0.034 | 0.0204 | 0.989 | 1.27 | 0.93–1.66 | 0.0056 | -0.0015–0.0135 |
| P1 | UniProt | DAVIS | moltrans | cold-target | 68 | 0.026 | 0.010–0.044 | 0.0194 | 0.991 | 1.37 | 0.53–2.32 | 0.0071 | -0.0090–0.0249 |
| P1 | UniProt | DAVIS | moltrans | cold-pair | 72 | 0.020 | 0.004–0.038 | 0.0190 | 0.990 | 1.05 | 0.22–1.95 | 0.0009 | -0.0149–0.0182 |
| P2 | UniProt | KIBA | coldsite_dti | random | 211 | 0.019 | 0.011–0.027 | 0.0228 | 0.995 | 0.82 | 0.50–1.18 | -0.0042 | -0.0115–0.0040 |
| P2 | UniProt | KIBA | coldsite_dti | cold-drug | 212 | 0.020 | 0.013–0.027 | 0.0228 | 0.995 | 0.88 | 0.58–1.17 | -0.0028 | -0.0097–0.0039 |
| P2 | UniProt | KIBA | hyperattentiondti | random | 211 | 0.030 | 0.016–0.051 | 0.0228 | 0.995 | 1.34 | 0.68–2.25 | 0.0077 | -0.0072–0.0284 |
| P2 | UniProt | KIBA | hyperattentiondti | cold-drug | 212 | 0.021 | 0.016–0.027 | 0.0228 | 0.995 | 0.94 | 0.71–1.19 | -0.0014 | -0.0067–0.0043 |
| P2 | UniProt | KIBA | moltrans | random | 211 | 0.032 | 0.016–0.052 | 0.0228 | 0.995 | 1.39 | 0.72–2.28 | 0.0088 | -0.0063–0.0290 |
| P2 | UniProt | KIBA | moltrans | cold-drug | 212 | 0.036 | 0.016–0.061 | 0.0228 | 0.995 | 1.57 | 0.70–2.70 | 0.0129 | -0.0067–0.0387 |
| S3-D | KLIFS pocket | DAVIS | coldsite_dti | random | 350 | 0.219 | 0.192–0.242 | 0.1425 | 0.998 | 1.54 | 1.36–1.69 | 0.0763 | 0.0507–0.0983 |
| S3-D | KLIFS pocket | DAVIS | coldsite_dti | cold-drug | 350 | 0.299 | 0.261–0.337 | 0.1424 | 0.998 | 2.10 | 1.83–2.36 | 0.1567 | 0.1184–0.1932 |
| S3-D | KLIFS pocket | DAVIS | coldsite_dti | cold-target | 67 | 0.243 | 0.195–0.299 | 0.1398 | 1.000 | 1.74 | 1.40–2.13 | 0.1030 | 0.0564–0.1575 |
| S3-D | KLIFS pocket | DAVIS | coldsite_dti | cold-pair | 71 | 0.271 | 0.229–0.322 | 0.1366 | 1.000 | 1.99 | 1.69–2.35 | 0.1348 | 0.0942–0.1837 |
| S3-D | KLIFS pocket | DAVIS | drugban | random | 350 | 0.139 | 0.128–0.150 | 0.1425 | 0.998 | 0.98 | 0.91–1.04 | -0.0036 | -0.0130–0.0062 |
| S3-D | KLIFS pocket | DAVIS | drugban | cold-drug | 350 | 0.146 | 0.131–0.160 | 0.1424 | 0.998 | 1.02 | 0.93–1.11 | 0.0032 | -0.0103–0.0160 |
| S3-D | KLIFS pocket | DAVIS | drugban | cold-target | 67 | 0.145 | 0.121–0.170 | 0.1398 | 1.000 | 1.04 | 0.90–1.17 | 0.0050 | -0.0138–0.0243 |
| S3-D | KLIFS pocket | DAVIS | drugban | cold-pair | 71 | 0.138 | 0.117–0.162 | 0.1366 | 1.000 | 1.01 | 0.87–1.17 | 0.0019 | -0.0183–0.0228 |
| S3-D | KLIFS pocket | DAVIS | hyperattentiondti | random | 350 | 0.242 | 0.210–0.281 | 0.1425 | 0.998 | 1.70 | 1.48–1.98 | 0.0998 | 0.0686–0.1393 |
| S3-D | KLIFS pocket | DAVIS | hyperattentiondti | cold-drug | 350 | 0.193 | 0.135–0.247 | 0.1424 | 0.998 | 1.35 | 0.94–1.73 | 0.0503 | -0.0081–0.1040 |
| S3-D | KLIFS pocket | DAVIS | hyperattentiondti | cold-target | 67 | 0.186 | 0.146–0.226 | 0.1398 | 1.000 | 1.33 | 1.08–1.60 | 0.0463 | 0.0107–0.0832 |
| S3-D | KLIFS pocket | DAVIS | hyperattentiondti | cold-pair | 71 | 0.185 | 0.138–0.238 | 0.1366 | 1.000 | 1.36 | 1.03–1.73 | 0.0489 | 0.0036–0.1002 |
| S3-D | KLIFS pocket | DAVIS | moltrans | random | 350 | 0.157 | 0.122–0.189 | 0.1425 | 0.998 | 1.10 | 0.86–1.32 | 0.0142 | -0.0197–0.0458 |
| S3-D | KLIFS pocket | DAVIS | moltrans | cold-drug | 350 | 0.154 | 0.125–0.182 | 0.1424 | 0.998 | 1.08 | 0.89–1.27 | 0.0119 | -0.0162–0.0388 |
| S3-D | KLIFS pocket | DAVIS | moltrans | cold-target | 67 | 0.176 | 0.108–0.238 | 0.1398 | 1.000 | 1.26 | 0.77–1.70 | 0.0359 | -0.0315–0.0972 |
| S3-D | KLIFS pocket | DAVIS | moltrans | cold-pair | 71 | 0.136 | 0.090–0.183 | 0.1366 | 1.000 | 0.99 | 0.67–1.32 | -0.0009 | -0.0447–0.0444 |
| S3-K | KLIFS pocket | KIBA | coldsite_dti | random | 210 | 0.166 | 0.145–0.186 | 0.1516 | 0.996 | 1.09 | 0.97–1.21 | 0.0142 | -0.0049–0.0326 |
| S3-K | KLIFS pocket | KIBA | coldsite_dti | cold-drug | 211 | 0.152 | 0.119–0.193 | 0.1514 | 0.996 | 1.00 | 0.79–1.27 | 0.0005 | -0.0318–0.0411 |
| S3-K | KLIFS pocket | KIBA | hyperattentiondti | random | 210 | 0.207 | 0.190–0.226 | 0.1516 | 0.996 | 1.37 | 1.26–1.49 | 0.0559 | 0.0400–0.0735 |
| S3-K | KLIFS pocket | KIBA | hyperattentiondti | cold-drug | 211 | 0.189 | 0.162–0.215 | 0.1514 | 0.996 | 1.25 | 1.07–1.41 | 0.0372 | 0.0111–0.0620 |
| S3-K | KLIFS pocket | KIBA | moltrans | random | 210 | 0.161 | 0.118–0.213 | 0.1516 | 0.996 | 1.07 | 0.78–1.40 | 0.0099 | -0.0328–0.0601 |
| S3-K | KLIFS pocket | KIBA | moltrans | cold-drug | 211 | 0.188 | 0.150–0.232 | 0.1514 | 0.996 | 1.24 | 1.00–1.53 | 0.0361 | -0.0002–0.0795 |
| S1 | UniProt | DAVIS | coldsite_dti_ig | random | 349 | 0.021 | 0.012–0.031 | 0.0204 | 0.989 | 1.03 | 0.61–1.51 | 0.0006 | -0.0080–0.0104 |
| S1 | UniProt | DAVIS | coldsite_dti_ig | cold-drug | 349 | 0.055 | 0.032–0.072 | 0.0204 | 0.989 | 2.70 | 1.59–3.56 | 0.0346 | 0.0121–0.0520 |
| S1 | UniProt | DAVIS | coldsite_dti_ig | cold-target | 68 | 0.044 | 0.019–0.074 | 0.0194 | 0.991 | 2.25 | 1.01–3.80 | 0.0243 | 0.0001–0.0541 |
| S1 | UniProt | DAVIS | coldsite_dti_ig | cold-pair | 72 | 0.014 | 0.003–0.029 | 0.0190 | 0.990 | 0.76 | 0.18–1.50 | -0.0046 | -0.0154–0.0097 |
| S1 | UniProt | DAVIS | moltrans_ig | random | 349 | 0.018 | 0.013–0.024 | 0.0204 | 0.989 | 0.90 | 0.66–1.16 | -0.0021 | -0.0069–0.0032 |
| S1 | UniProt | DAVIS | moltrans_ig | cold-drug | 349 | 0.027 | 0.021–0.033 | 0.0204 | 0.989 | 1.33 | 1.03–1.63 | 0.0067 | 0.0005–0.0129 |
| S1 | UniProt | DAVIS | moltrans_ig | cold-target | 68 | 0.029 | 0.017–0.043 | 0.0194 | 0.991 | 1.52 | 0.88–2.24 | 0.0100 | -0.0024–0.0235 |
| S1 | UniProt | DAVIS | moltrans_ig | cold-pair | 72 | 0.021 | 0.010–0.034 | 0.0190 | 0.990 | 1.10 | 0.53–1.84 | 0.0019 | -0.0090–0.0157 |
| S1-klifs | KLIFS pocket | DAVIS | coldsite_dti_ig | random | 350 | 0.280 | 0.255–0.304 | 0.1425 | 0.998 | 1.96 | 1.80–2.13 | 0.1372 | 0.1139–0.1606 |
| S1-klifs | KLIFS pocket | DAVIS | coldsite_dti_ig | cold-drug | 350 | 0.451 | 0.342–0.627 | 0.1424 | 0.998 | 3.16 | 2.40–4.40 | 0.3083 | 0.2006–0.4854 |
| S1-klifs | KLIFS pocket | DAVIS | coldsite_dti_ig | cold-target | 67 | 0.325 | 0.267–0.381 | 0.1398 | 1.000 | 2.32 | 1.91–2.72 | 0.1851 | 0.1287–0.2388 |
| S1-klifs | KLIFS pocket | DAVIS | coldsite_dti_ig | cold-pair | 71 | 0.314 | 0.247–0.379 | 0.1366 | 1.000 | 2.30 | 1.81–2.77 | 0.1770 | 0.1112–0.2412 |
| S1-klifs | KLIFS pocket | DAVIS | moltrans_ig | random | 350 | 0.168 | 0.120–0.207 | 0.1425 | 0.998 | 1.18 | 0.84–1.44 | 0.0252 | -0.0222–0.0634 |
| S1-klifs | KLIFS pocket | DAVIS | moltrans_ig | cold-drug | 350 | 0.165 | 0.140–0.194 | 0.1424 | 0.998 | 1.16 | 0.99–1.35 | 0.0228 | -0.0019–0.0504 |
| S1-klifs | KLIFS pocket | DAVIS | moltrans_ig | cold-target | 67 | 0.162 | 0.128–0.197 | 0.1398 | 1.000 | 1.16 | 0.92–1.41 | 0.0224 | -0.0115–0.0564 |
| S1-klifs | KLIFS pocket | DAVIS | moltrans_ig | cold-pair | 71 | 0.125 | 0.090–0.164 | 0.1366 | 1.000 | 0.92 | 0.67–1.19 | -0.0112 | -0.0457–0.0256 |
| S2 | UniProt | KIBA | hyperattentiondti_ig | random | 211 | 0.042 | 0.024–0.065 | 0.0228 | 0.995 | 1.82 | 1.04–2.86 | 0.0187 | 0.0009–0.0422 |
| S2 | UniProt | KIBA | hyperattentiondti_ig | cold-drug | 212 | 0.043 | 0.015–0.077 | 0.0228 | 0.995 | 1.90 | 0.63–3.39 | 0.0204 | -0.0084–0.0544 |
| S2 | UniProt | KIBA | moltrans_ig | random | 211 | 0.031 | 0.017–0.049 | 0.0228 | 0.995 | 1.34 | 0.73–2.16 | 0.0078 | -0.0061–0.0264 |
| S2 | UniProt | KIBA | moltrans_ig | cold-drug | 212 | 0.031 | 0.022–0.042 | 0.0228 | 0.995 | 1.37 | 0.97–1.85 | 0.0085 | -0.0006–0.0190 |
| S2-klifs | KLIFS pocket | KIBA | hyperattentiondti_ig | random | 210 | 0.272 | 0.239–0.308 | 0.1516 | 0.996 | 1.79 | 1.58–2.03 | 0.1204 | 0.0879–0.1559 |
| S2-klifs | KLIFS pocket | KIBA | hyperattentiondti_ig | cold-drug | 211 | 0.257 | 0.201–0.329 | 0.1514 | 0.996 | 1.70 | 1.33–2.18 | 0.1059 | 0.0502–0.1785 |
| S2-klifs | KLIFS pocket | KIBA | moltrans_ig | random | 210 | 0.190 | 0.144–0.243 | 0.1516 | 0.996 | 1.25 | 0.96–1.60 | 0.0386 | -0.0059–0.0911 |
| S2-klifs | KLIFS pocket | KIBA | moltrans_ig | cold-drug | 211 | 0.174 | 0.152–0.196 | 0.1514 | 0.996 | 1.15 | 1.01–1.29 | 0.0225 | 0.0013–0.0433 |
