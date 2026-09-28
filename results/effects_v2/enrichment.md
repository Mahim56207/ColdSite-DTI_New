# Enrichment over chance, with confidence intervals

Mean precision@10 divided by mean chance over the same resampled proteins; 95% percentile intervals, 10,000 resamples, unit = target, each protein carried in with all its seeds averaged, `default_rng(0)` (`src/evaluation/effects_v2.py`, amendment section 5). Effect sizes only: no threshold is applied. `excess` is precision minus chance. Ceiling is the recorded mean ceiling.

| family | ground truth | dataset | model | level | n | precision@10 | 95% CI | chance | ceiling | enrichment | 95% CI | excess | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | UniProt | DAVIS | coldsite_dti | random | 349 | 0.015 | 0.013–0.018 | 0.0204 | 0.989 | 0.74 | 0.62–0.86 | -0.0053 | -0.0078–-0.0028 |
| P1 | UniProt | DAVIS | coldsite_dti | cold-drug | 349 | 0.022 | 0.019–0.024 | 0.0204 | 0.989 | 1.06 | 0.93–1.19 | 0.0013 | -0.0014–0.0039 |
| P1 | UniProt | DAVIS | coldsite_dti | cold-target | 68 | 0.017 | 0.012–0.023 | 0.0194 | 0.991 | 0.89 | 0.63–1.15 | -0.0022 | -0.0071–0.0029 |
| P1 | UniProt | DAVIS | coldsite_dti | cold-pair | 72 | 0.013 | 0.007–0.020 | 0.0190 | 0.990 | 0.68 | 0.40–1.04 | -0.0060 | -0.0113–0.0008 |
| P1 | UniProt | DAVIS | drugban | random | 349 | 0.018 | 0.015–0.020 | 0.0204 | 0.989 | 0.87 | 0.75–0.99 | -0.0027 | -0.0052–-0.0001 |
| P1 | UniProt | DAVIS | drugban | cold-drug | 349 | 0.021 | 0.018–0.024 | 0.0204 | 0.989 | 1.03 | 0.90–1.17 | 0.0006 | -0.0020–0.0034 |
| P1 | UniProt | DAVIS | drugban | cold-target | 68 | 0.023 | 0.016–0.030 | 0.0194 | 0.991 | 1.16 | 0.84–1.50 | 0.0032 | -0.0029–0.0098 |
| P1 | UniProt | DAVIS | drugban | cold-pair | 72 | 0.027 | 0.021–0.034 | 0.0190 | 0.990 | 1.44 | 1.12–1.77 | 0.0083 | 0.0023–0.0146 |
| P1 | UniProt | DAVIS | hyperattentiondti | random | 349 | 0.034 | 0.030–0.038 | 0.0204 | 0.989 | 1.66 | 1.49–1.84 | 0.0135 | 0.0100–0.0171 |
| P1 | UniProt | DAVIS | hyperattentiondti | cold-drug | 349 | 0.040 | 0.037–0.044 | 0.0204 | 0.989 | 1.98 | 1.81–2.15 | 0.0200 | 0.0167–0.0234 |
| P1 | UniProt | DAVIS | hyperattentiondti | cold-target | 68 | 0.025 | 0.018–0.031 | 0.0194 | 0.991 | 1.27 | 0.95–1.62 | 0.0051 | -0.0010–0.0118 |
| P1 | UniProt | DAVIS | hyperattentiondti | cold-pair | 72 | 0.022 | 0.016–0.029 | 0.0190 | 0.990 | 1.17 | 0.86–1.50 | 0.0032 | -0.0026–0.0096 |
| P1 | UniProt | DAVIS | moltrans | random | 349 | 0.021 | 0.017–0.026 | 0.0204 | 0.989 | 1.05 | 0.86–1.25 | 0.0010 | -0.0028–0.0050 |
| P1 | UniProt | DAVIS | moltrans | cold-drug | 349 | 0.026 | 0.021–0.031 | 0.0204 | 0.989 | 1.27 | 1.05–1.51 | 0.0056 | 0.0010–0.0104 |
| P1 | UniProt | DAVIS | moltrans | cold-target | 68 | 0.026 | 0.018–0.035 | 0.0194 | 0.991 | 1.37 | 0.93–1.87 | 0.0071 | -0.0014–0.0163 |
| P1 | UniProt | DAVIS | moltrans | cold-pair | 72 | 0.020 | 0.013–0.027 | 0.0190 | 0.990 | 1.05 | 0.70–1.42 | 0.0009 | -0.0056–0.0081 |
| P2 | UniProt | KIBA | coldsite_dti | random | 211 | 0.019 | 0.015–0.022 | 0.0228 | 0.995 | 0.82 | 0.68–0.95 | -0.0042 | -0.0072–-0.0011 |
| P2 | UniProt | KIBA | coldsite_dti | cold-drug | 212 | 0.020 | 0.016–0.024 | 0.0228 | 0.995 | 0.88 | 0.71–1.05 | -0.0028 | -0.0066–0.0011 |
| P2 | UniProt | KIBA | hyperattentiondti | random | 211 | 0.030 | 0.026–0.035 | 0.0228 | 0.995 | 1.34 | 1.12–1.56 | 0.0077 | 0.0029–0.0125 |
| P2 | UniProt | KIBA | hyperattentiondti | cold-drug | 212 | 0.021 | 0.018–0.025 | 0.0228 | 0.995 | 0.94 | 0.79–1.09 | -0.0014 | -0.0049–0.0021 |
| P2 | UniProt | KIBA | moltrans | random | 211 | 0.032 | 0.026–0.038 | 0.0228 | 0.995 | 1.39 | 1.14–1.64 | 0.0088 | 0.0032–0.0145 |
| P2 | UniProt | KIBA | moltrans | cold-drug | 212 | 0.036 | 0.030–0.042 | 0.0228 | 0.995 | 1.57 | 1.32–1.82 | 0.0129 | 0.0074–0.0185 |
| S3-D | KLIFS pocket | DAVIS | coldsite_dti | random | 350 | 0.219 | 0.208–0.229 | 0.1425 | 0.998 | 1.54 | 1.48–1.59 | 0.0763 | 0.0684–0.0841 |
| S3-D | KLIFS pocket | DAVIS | coldsite_dti | cold-drug | 350 | 0.299 | 0.286–0.312 | 0.1424 | 0.998 | 2.10 | 2.02–2.18 | 0.1567 | 0.1462–0.1671 |
| S3-D | KLIFS pocket | DAVIS | coldsite_dti | cold-target | 67 | 0.243 | 0.218–0.268 | 0.1398 | 1.000 | 1.74 | 1.58–1.90 | 0.1030 | 0.0822–0.1238 |
| S3-D | KLIFS pocket | DAVIS | coldsite_dti | cold-pair | 71 | 0.271 | 0.241–0.303 | 0.1366 | 1.000 | 1.99 | 1.78–2.21 | 0.1348 | 0.1074–0.1636 |
| S3-D | KLIFS pocket | DAVIS | drugban | random | 350 | 0.139 | 0.130–0.148 | 0.1425 | 0.998 | 0.98 | 0.93–1.02 | -0.0036 | -0.0099–0.0028 |
| S3-D | KLIFS pocket | DAVIS | drugban | cold-drug | 350 | 0.146 | 0.137–0.154 | 0.1424 | 0.998 | 1.02 | 0.98–1.07 | 0.0032 | -0.0031–0.0095 |
| S3-D | KLIFS pocket | DAVIS | drugban | cold-target | 67 | 0.145 | 0.124–0.167 | 0.1398 | 1.000 | 1.04 | 0.93–1.14 | 0.0050 | -0.0097–0.0201 |
| S3-D | KLIFS pocket | DAVIS | drugban | cold-pair | 71 | 0.138 | 0.121–0.157 | 0.1366 | 1.000 | 1.01 | 0.90–1.13 | 0.0019 | -0.0134–0.0179 |
| S3-D | KLIFS pocket | DAVIS | hyperattentiondti | random | 350 | 0.242 | 0.232–0.253 | 0.1425 | 0.998 | 1.70 | 1.64–1.76 | 0.0998 | 0.0917–0.1079 |
| S3-D | KLIFS pocket | DAVIS | hyperattentiondti | cold-drug | 350 | 0.193 | 0.183–0.203 | 0.1424 | 0.998 | 1.35 | 1.30–1.41 | 0.0503 | 0.0432–0.0578 |
| S3-D | KLIFS pocket | DAVIS | hyperattentiondti | cold-target | 67 | 0.186 | 0.163–0.209 | 0.1398 | 1.000 | 1.33 | 1.21–1.45 | 0.0463 | 0.0294–0.0630 |
| S3-D | KLIFS pocket | DAVIS | hyperattentiondti | cold-pair | 71 | 0.185 | 0.165–0.206 | 0.1366 | 1.000 | 1.36 | 1.23–1.48 | 0.0489 | 0.0321–0.0656 |
| S3-D | KLIFS pocket | DAVIS | moltrans | random | 350 | 0.157 | 0.143–0.170 | 0.1425 | 0.998 | 1.10 | 1.02–1.18 | 0.0142 | 0.0033–0.0252 |
| S3-D | KLIFS pocket | DAVIS | moltrans | cold-drug | 350 | 0.154 | 0.142–0.166 | 0.1424 | 0.998 | 1.08 | 1.01–1.15 | 0.0119 | 0.0014–0.0220 |
| S3-D | KLIFS pocket | DAVIS | moltrans | cold-target | 67 | 0.176 | 0.150–0.201 | 0.1398 | 1.000 | 1.26 | 1.09–1.43 | 0.0359 | 0.0122–0.0592 |
| S3-D | KLIFS pocket | DAVIS | moltrans | cold-pair | 71 | 0.136 | 0.114–0.158 | 0.1366 | 1.000 | 0.99 | 0.86–1.13 | -0.0009 | -0.0192–0.0180 |
| S3-K | KLIFS pocket | KIBA | coldsite_dti | random | 210 | 0.166 | 0.155–0.177 | 0.1516 | 0.996 | 1.09 | 1.04–1.15 | 0.0142 | 0.0063–0.0221 |
| S3-K | KLIFS pocket | KIBA | coldsite_dti | cold-drug | 211 | 0.152 | 0.140–0.164 | 0.1514 | 0.996 | 1.00 | 0.94–1.06 | 0.0005 | -0.0085–0.0097 |
| S3-K | KLIFS pocket | KIBA | hyperattentiondti | random | 210 | 0.207 | 0.195–0.220 | 0.1516 | 0.996 | 1.37 | 1.29–1.44 | 0.0559 | 0.0450–0.0667 |
| S3-K | KLIFS pocket | KIBA | hyperattentiondti | cold-drug | 211 | 0.189 | 0.176–0.202 | 0.1514 | 0.996 | 1.25 | 1.17–1.32 | 0.0372 | 0.0258–0.0489 |
| S3-K | KLIFS pocket | KIBA | moltrans | random | 210 | 0.161 | 0.146–0.176 | 0.1516 | 0.996 | 1.07 | 0.98–1.15 | 0.0099 | -0.0036–0.0230 |
| S3-K | KLIFS pocket | KIBA | moltrans | cold-drug | 211 | 0.188 | 0.171–0.204 | 0.1514 | 0.996 | 1.24 | 1.14–1.34 | 0.0361 | 0.0213–0.0508 |
| S1 | UniProt | DAVIS | coldsite_dti_ig | random | 349 | 0.021 | 0.018–0.024 | 0.0204 | 0.989 | 1.03 | 0.89–1.18 | 0.0006 | -0.0023–0.0037 |
| S1 | UniProt | DAVIS | coldsite_dti_ig | cold-drug | 349 | 0.055 | 0.051–0.060 | 0.0204 | 0.989 | 2.70 | 2.49–2.91 | 0.0346 | 0.0304–0.0389 |
| S1 | UniProt | DAVIS | coldsite_dti_ig | cold-target | 68 | 0.044 | 0.035–0.053 | 0.0194 | 0.991 | 2.25 | 1.80–2.73 | 0.0243 | 0.0156–0.0336 |
| S1 | UniProt | DAVIS | coldsite_dti_ig | cold-pair | 72 | 0.014 | 0.006–0.027 | 0.0190 | 0.990 | 0.76 | 0.31–1.41 | -0.0046 | -0.0131–0.0079 |
| S1 | UniProt | DAVIS | moltrans_ig | random | 349 | 0.018 | 0.015–0.022 | 0.0204 | 0.989 | 0.90 | 0.72–1.09 | -0.0021 | -0.0057–0.0018 |
| S1 | UniProt | DAVIS | moltrans_ig | cold-drug | 349 | 0.027 | 0.023–0.032 | 0.0204 | 0.989 | 1.33 | 1.11–1.55 | 0.0067 | 0.0023–0.0111 |
| S1 | UniProt | DAVIS | moltrans_ig | cold-target | 68 | 0.029 | 0.021–0.040 | 0.0194 | 0.991 | 1.52 | 1.04–2.08 | 0.0100 | 0.0009–0.0203 |
| S1 | UniProt | DAVIS | moltrans_ig | cold-pair | 72 | 0.021 | 0.012–0.032 | 0.0190 | 0.990 | 1.10 | 0.63–1.69 | 0.0019 | -0.0070–0.0130 |
| S1-klifs | KLIFS pocket | DAVIS | coldsite_dti_ig | random | 350 | 0.280 | 0.262–0.297 | 0.1425 | 0.998 | 1.96 | 1.85–2.08 | 0.1372 | 0.1213–0.1531 |
| S1-klifs | KLIFS pocket | DAVIS | coldsite_dti_ig | cold-drug | 350 | 0.451 | 0.430–0.471 | 0.1424 | 0.998 | 3.16 | 3.00–3.33 | 0.3083 | 0.2885–0.3281 |
| S1-klifs | KLIFS pocket | DAVIS | coldsite_dti_ig | cold-target | 67 | 0.325 | 0.291–0.361 | 0.1398 | 1.000 | 2.32 | 2.09–2.58 | 0.1851 | 0.1540–0.2172 |
| S1-klifs | KLIFS pocket | DAVIS | coldsite_dti_ig | cold-pair | 71 | 0.314 | 0.270–0.357 | 0.1366 | 1.000 | 2.30 | 2.00–2.62 | 0.1770 | 0.1359–0.2187 |
| S1-klifs | KLIFS pocket | DAVIS | moltrans_ig | random | 350 | 0.168 | 0.155–0.181 | 0.1425 | 0.998 | 1.18 | 1.10–1.25 | 0.0252 | 0.0147–0.0359 |
| S1-klifs | KLIFS pocket | DAVIS | moltrans_ig | cold-drug | 350 | 0.165 | 0.153–0.177 | 0.1424 | 0.998 | 1.16 | 1.08–1.24 | 0.0228 | 0.0119–0.0335 |
| S1-klifs | KLIFS pocket | DAVIS | moltrans_ig | cold-target | 67 | 0.162 | 0.140–0.185 | 0.1398 | 1.000 | 1.16 | 1.01–1.31 | 0.0224 | 0.0012–0.0429 |
| S1-klifs | KLIFS pocket | DAVIS | moltrans_ig | cold-pair | 71 | 0.125 | 0.103–0.149 | 0.1366 | 1.000 | 0.92 | 0.76–1.08 | -0.0112 | -0.0326–0.0109 |
| S2 | UniProt | KIBA | hyperattentiondti_ig | random | 211 | 0.042 | 0.034–0.050 | 0.0228 | 0.995 | 1.82 | 1.49–2.18 | 0.0187 | 0.0112–0.0267 |
| S2 | UniProt | KIBA | hyperattentiondti_ig | cold-drug | 212 | 0.043 | 0.036–0.051 | 0.0228 | 0.995 | 1.90 | 1.58–2.22 | 0.0204 | 0.0134–0.0277 |
| S2 | UniProt | KIBA | moltrans_ig | random | 211 | 0.031 | 0.025–0.036 | 0.0228 | 0.995 | 1.34 | 1.11–1.59 | 0.0078 | 0.0024–0.0135 |
| S2 | UniProt | KIBA | moltrans_ig | cold-drug | 212 | 0.031 | 0.026–0.037 | 0.0228 | 0.995 | 1.37 | 1.13–1.62 | 0.0085 | 0.0031–0.0141 |
| S2-klifs | KLIFS pocket | KIBA | hyperattentiondti_ig | random | 210 | 0.272 | 0.252–0.292 | 0.1516 | 0.996 | 1.79 | 1.67–1.93 | 0.1204 | 0.1015–0.1391 |
| S2-klifs | KLIFS pocket | KIBA | hyperattentiondti_ig | cold-drug | 211 | 0.257 | 0.239–0.275 | 0.1514 | 0.996 | 1.70 | 1.57–1.83 | 0.1059 | 0.0880–0.1234 |
| S2-klifs | KLIFS pocket | KIBA | moltrans_ig | random | 210 | 0.190 | 0.173–0.208 | 0.1516 | 0.996 | 1.25 | 1.16–1.35 | 0.0386 | 0.0240–0.0536 |
| S2-klifs | KLIFS pocket | KIBA | moltrans_ig | cold-drug | 211 | 0.174 | 0.158–0.190 | 0.1514 | 0.996 | 1.15 | 1.06–1.24 | 0.0225 | 0.0085–0.0363 |
