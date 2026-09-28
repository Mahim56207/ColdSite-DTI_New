# Seed spread against distance from chance

Rule 1.13 of the amendment (`seed_agreement.py`): the spread exceeds the signal iff max − min of the seeds' precision@10 is larger than |mean − chance|; chance is the cell's recorded chance. `exact` repeats the test with the rebuilt per-protein chance.

| family | ground truth | dataset | model | level | per-seed precision@10 | spread | SD | |mean − chance| | spread > signal | same, exact chance | enrichment (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | UniProt | DAVIS | coldsite_dti | random | 0.022923 / 0.009456 / 0.012894 | 0.0135 | 0.0070 | 0.0053 | yes | yes | 0.74 (0.62–0.86) |
| P1 | UniProt | DAVIS | coldsite_dti | cold-drug | 0.026934 / 0.026648 / 0.011461 | 0.0155 | 0.0089 | 0.0012 | yes | yes | 1.06 (0.93–1.19) |
| P1 | UniProt | DAVIS | coldsite_dti | cold-target | 0.014706 / 0.019118 / 0.017647 | 0.0044 | 0.0022 | 0.0022 | yes | yes | 0.89 (0.63–1.15) |
| P1 | UniProt | DAVIS | coldsite_dti | cold-pair | 0.018056 / 0.012500 / 0.008333 | 0.0097 | 0.0049 | 0.0058 | yes | yes | 0.68 (0.40–1.04) |
| P1 | UniProt | DAVIS | drugban | random | 0.019198 / 0.014613 / 0.019198 | 0.0046 | 0.0026 | 0.0027 | yes | yes | 0.87 (0.75–0.99) |
| P1 | UniProt | DAVIS | drugban | cold-drug | 0.021777 / 0.020630 / 0.020630 | 0.0011 | 0.0007 | 0.0005 | yes | yes | 1.03 (0.90–1.17) |
| P1 | UniProt | DAVIS | drugban | cold-target | 0.017647 / 0.027941 / 0.022059 | 0.0103 | 0.0052 | 0.0031 | yes | yes | 1.16 (0.84–1.50) |
| P1 | UniProt | DAVIS | drugban | cold-pair | 0.036111 / 0.022222 / 0.023611 | 0.0139 | 0.0076 | 0.0085 | yes | yes | 1.44 (1.12–1.77) |
| P1 | UniProt | DAVIS | hyperattentiondti | random | 0.034670 / 0.039255 / 0.027794 | 0.0115 | 0.0058 | 0.0136 | no | no | 1.66 (1.49–1.84) |
| P1 | UniProt | DAVIS | hyperattentiondti | cold-drug | 0.024928 / 0.076791 / 0.019484 | 0.0573 | 0.0316 | 0.0199 | yes | yes | 1.98 (1.81–2.15) |
| P1 | UniProt | DAVIS | hyperattentiondti | cold-target | 0.017647 / 0.030882 / 0.025000 | 0.0132 | 0.0066 | 0.0051 | yes | yes | 1.27 (0.95–1.62) |
| P1 | UniProt | DAVIS | hyperattentiondti | cold-pair | 0.012500 / 0.022222 / 0.031944 | 0.0194 | 0.0097 | 0.0034 | yes | yes | 1.17 (0.86–1.50) |
| P1 | UniProt | DAVIS | moltrans | random | 0.024355 / 0.020630 / 0.019198 | 0.0052 | 0.0027 | 0.0011 | yes | yes | 1.05 (0.86–1.25) |
| P1 | UniProt | DAVIS | moltrans | cold-drug | 0.021777 / 0.031805 / 0.024355 | 0.0100 | 0.0052 | 0.0055 | yes | yes | 1.27 (1.05–1.51) |
| P1 | UniProt | DAVIS | moltrans | cold-target | 0.011765 / 0.038235 / 0.029412 | 0.0265 | 0.0135 | 0.0071 | yes | yes | 1.37 (0.93–1.87) |
| P1 | UniProt | DAVIS | moltrans | cold-pair | 0.004167 / 0.031944 / 0.023611 | 0.0278 | 0.0143 | 0.0011 | yes | yes | 1.05 (0.70–1.42) |
| P2 | UniProt | KIBA | coldsite_dti | random | 0.018483 / 0.011374 / 0.026066 | 0.0147 | 0.0073 | 0.0042 | yes | yes | 0.82 (0.68–0.95) |
| P2 | UniProt | KIBA | coldsite_dti | cold-drug | 0.024528 / 0.020283 / 0.015094 | 0.0094 | 0.0047 | 0.0028 | yes | yes | 0.88 (0.71–1.05) |
| P2 | UniProt | KIBA | hyperattentiondti | random | 0.017062 / 0.021801 / 0.052607 | 0.0355 | 0.0193 | 0.0077 | yes | yes | 1.34 (1.12–1.56) |
| P2 | UniProt | KIBA | hyperattentiondti | cold-drug | 0.018868 / 0.020283 / 0.025000 | 0.0061 | 0.0032 | 0.0013 | yes | yes | 0.94 (0.79–1.09) |
| P2 | UniProt | KIBA | moltrans | random | 0.020379 / 0.052607 / 0.021801 | 0.0322 | 0.0182 | 0.0088 | yes | yes | 1.39 (1.14–1.64) |
| P2 | UniProt | KIBA | moltrans | cold-drug | 0.027830 / 0.062736 / 0.016509 | 0.0462 | 0.0241 | 0.0130 | yes | yes | 1.57 (1.32–1.82) |
| S3-D | KLIFS pocket | DAVIS | coldsite_dti | random | 0.191143 / 0.226286 / 0.238857 | 0.0477 | 0.0247 | 0.0762 | no | no | 1.54 (1.48–1.59) |
| S3-D | KLIFS pocket | DAVIS | coldsite_dti | cold-drug | 0.300000 / 0.259429 / 0.338000 | 0.0786 | 0.0393 | 0.1565 | no | no | 2.10 (2.02–2.18) |
| S3-D | KLIFS pocket | DAVIS | coldsite_dti | cold-target | 0.228358 / 0.204478 / 0.295522 | 0.0910 | 0.0472 | 0.1032 | no | no | 1.74 (1.58–1.90) |
| S3-D | KLIFS pocket | DAVIS | coldsite_dti | cold-pair | 0.309859 / 0.259155 / 0.245070 | 0.0648 | 0.0341 | 0.1355 | no | no | 1.99 (1.78–2.21) |
| S3-D | KLIFS pocket | DAVIS | drugban | random | 0.144571 / 0.134286 / 0.138000 | 0.0103 | 0.0052 | 0.0036 | yes | yes | 0.98 (0.93–1.02) |
| S3-D | KLIFS pocket | DAVIS | drugban | cold-drug | 0.151714 / 0.133143 / 0.152000 | 0.0189 | 0.0108 | 0.0029 | yes | yes | 1.02 (0.98–1.07) |
| S3-D | KLIFS pocket | DAVIS | drugban | cold-target | 0.144776 / 0.144776 / 0.144776 | 0.0000 | 0.0000 | 0.0052 | no | no | 1.04 (0.93–1.14) |
| S3-D | KLIFS pocket | DAVIS | drugban | cold-pair | 0.138028 / 0.133803 / 0.143662 | 0.0099 | 0.0049 | 0.0027 | yes | yes | 1.01 (0.90–1.13) |
| S3-D | KLIFS pocket | DAVIS | hyperattentiondti | random | 0.285143 / 0.213143 / 0.228571 | 0.0720 | 0.0379 | 0.0997 | no | no | 1.70 (1.64–1.76) |
| S3-D | KLIFS pocket | DAVIS | hyperattentiondti | cold-drug | 0.199143 / 0.248857 / 0.130286 | 0.1186 | 0.0595 | 0.0501 | yes | yes | 1.35 (1.30–1.41) |
| S3-D | KLIFS pocket | DAVIS | hyperattentiondti | cold-target | 0.156716 / 0.219403 / 0.182090 | 0.0627 | 0.0315 | 0.0465 | yes | yes | 1.33 (1.21–1.45) |
| S3-D | KLIFS pocket | DAVIS | hyperattentiondti | cold-pair | 0.238028 / 0.143662 / 0.174648 | 0.0944 | 0.0481 | 0.0496 | yes | yes | 1.36 (1.23–1.48) |
| S3-D | KLIFS pocket | DAVIS | moltrans | random | 0.168571 / 0.182571 / 0.118857 | 0.0637 | 0.0335 | 0.0141 | yes | yes | 1.10 (1.02–1.18) |
| S3-D | KLIFS pocket | DAVIS | moltrans | cold-drug | 0.169429 / 0.169714 / 0.123714 | 0.0460 | 0.0265 | 0.0116 | yes | yes | 1.08 (1.01–1.15) |
| S3-D | KLIFS pocket | DAVIS | moltrans | cold-target | 0.104478 / 0.200000 / 0.222388 | 0.1179 | 0.0626 | 0.0360 | yes | yes | 1.26 (1.09–1.43) |
| S3-D | KLIFS pocket | DAVIS | moltrans | cold-pair | 0.091549 / 0.171831 / 0.143662 | 0.0803 | 0.0407 | 0.0001 | yes | yes | 0.99 (0.86–1.13) |
| S3-K | KLIFS pocket | KIBA | coldsite_dti | random | 0.173333 / 0.176667 / 0.147143 | 0.0295 | 0.0162 | 0.0147 | yes | yes | 1.09 (1.04–1.15) |
| S3-K | KLIFS pocket | KIBA | coldsite_dti | cold-drug | 0.194787 / 0.140284 / 0.120853 | 0.0739 | 0.0383 | 0.0005 | yes | yes | 1.00 (0.94–1.06) |
| S3-K | KLIFS pocket | KIBA | hyperattentiondti | random | 0.198571 / 0.203810 / 0.220000 | 0.0214 | 0.0112 | 0.0565 | no | no | 1.37 (1.29–1.44) |
| S3-K | KLIFS pocket | KIBA | hyperattentiondti | cold-drug | 0.163981 / 0.190995 / 0.210900 | 0.0469 | 0.0235 | 0.0372 | yes | yes | 1.25 (1.17–1.32) |
| S3-K | KLIFS pocket | KIBA | moltrans | random | 0.121905 / 0.216190 / 0.146190 | 0.0943 | 0.0490 | 0.0105 | yes | yes | 1.07 (0.98–1.15) |
| S3-K | KLIFS pocket | KIBA | moltrans | cold-drug | 0.171090 / 0.233175 / 0.158294 | 0.0749 | 0.0401 | 0.0361 | yes | yes | 1.24 (1.14–1.34) |
| S1 | UniProt | DAVIS | coldsite_dti_ig | random | 0.012894 / 0.018911 / 0.031232 | 0.0183 | 0.0093 | 0.0007 | yes | yes | 1.03 (0.89–1.18) |
| S1 | UniProt | DAVIS | coldsite_dti_ig | cold-drug | 0.071920 / 0.062751 / 0.030372 | 0.0415 | 0.0218 | 0.0345 | yes | yes | 2.70 (2.49–2.91) |
| S1 | UniProt | DAVIS | coldsite_dti_ig | cold-target | 0.073529 / 0.020588 / 0.036765 | 0.0529 | 0.0271 | 0.0242 | yes | yes | 2.25 (1.80–2.73) |
| S1 | UniProt | DAVIS | coldsite_dti_ig | cold-pair | 0.013889 / 0.013889 / 0.015278 | 0.0014 | 0.0008 | 0.0044 | no | no | 0.76 (0.31–1.41) |
| S1 | UniProt | DAVIS | moltrans_ig | random | 0.019484 / 0.018052 / 0.017479 | 0.0020 | 0.0010 | 0.0020 | yes | no | 0.90 (0.72–1.09) |
| S1 | UniProt | DAVIS | moltrans_ig | cold-drug | 0.027794 / 0.024069 / 0.029513 | 0.0054 | 0.0028 | 0.0067 | no | no | 1.33 (1.11–1.55) |
| S1 | UniProt | DAVIS | moltrans_ig | cold-target | 0.027941 / 0.030882 / 0.029412 | 0.0029 | 0.0015 | 0.0100 | no | no | 1.52 (1.04–2.08) |
| S1 | UniProt | DAVIS | moltrans_ig | cold-pair | 0.018056 / 0.023611 / 0.020833 | 0.0056 | 0.0028 | 0.0020 | yes | yes | 1.10 (0.63–1.69) |
| S1-klifs | KLIFS pocket | DAVIS | coldsite_dti_ig | random | 0.267714 / 0.276571 / 0.294857 | 0.0271 | 0.0138 | 0.1371 | no | no | 1.96 (1.85–2.08) |
| S1-klifs | KLIFS pocket | DAVIS | coldsite_dti_ig | cold-drug | 0.352286 / 0.633714 / 0.366286 | 0.2814 | 0.1586 | 0.3081 | no | no | 3.16 (3.00–3.33) |
| S1-klifs | KLIFS pocket | DAVIS | coldsite_dti_ig | cold-target | 0.341791 / 0.353731 / 0.279104 | 0.0746 | 0.0401 | 0.1853 | no | no | 2.32 (2.09–2.58) |
| S1-klifs | KLIFS pocket | DAVIS | coldsite_dti_ig | cold-pair | 0.326761 / 0.264789 / 0.349296 | 0.0845 | 0.0438 | 0.1778 | no | no | 2.30 (2.00–2.62) |
| S1-klifs | KLIFS pocket | DAVIS | moltrans_ig | random | 0.115714 / 0.199714 / 0.187714 | 0.0840 | 0.0454 | 0.0251 | yes | yes | 1.18 (1.10–1.25) |
| S1-klifs | KLIFS pocket | DAVIS | moltrans_ig | cold-drug | 0.192857 / 0.156857 / 0.146000 | 0.0469 | 0.0245 | 0.0226 | yes | yes | 1.16 (1.08–1.24) |
| S1-klifs | KLIFS pocket | DAVIS | moltrans_ig | cold-target | 0.143284 / 0.170149 / 0.173134 | 0.0299 | 0.0164 | 0.0226 | yes | yes | 1.16 (1.01–1.31) |
| S1-klifs | KLIFS pocket | DAVIS | moltrans_ig | cold-pair | 0.104225 / 0.150704 / 0.121127 | 0.0465 | 0.0235 | 0.0105 | yes | yes | 0.92 (0.76–1.08) |
| S2 | UniProt | KIBA | hyperattentiondti_ig | random | 0.024645 / 0.063033 / 0.036967 | 0.0384 | 0.0196 | 0.0187 | yes | yes | 1.82 (1.49–2.18) |
| S2 | UniProt | KIBA | hyperattentiondti_ig | cold-drug | 0.034906 / 0.013679 / 0.081132 | 0.0675 | 0.0345 | 0.0205 | yes | yes | 1.90 (1.58–2.22) |
| S2 | UniProt | KIBA | moltrans_ig | random | 0.019431 / 0.049763 / 0.022749 | 0.0303 | 0.0166 | 0.0078 | yes | yes | 1.34 (1.11–1.59) |
| S2 | UniProt | KIBA | moltrans_ig | cold-drug | 0.037736 / 0.032075 / 0.024057 | 0.0137 | 0.0069 | 0.0086 | yes | yes | 1.37 (1.13–1.62) |
| S2-klifs | KLIFS pocket | KIBA | hyperattentiondti_ig | random | 0.301905 / 0.257619 / 0.256190 | 0.0457 | 0.0260 | 0.1209 | no | no | 1.79 (1.67–1.93) |
| S2-klifs | KLIFS pocket | KIBA | hyperattentiondti_ig | cold-drug | 0.215640 / 0.218957 / 0.337441 | 0.1218 | 0.0694 | 0.1059 | yes | yes | 1.70 (1.57–1.83) |
| S2-klifs | KLIFS pocket | KIBA | moltrans_ig | random | 0.148571 / 0.245714 / 0.176190 | 0.0971 | 0.0501 | 0.0392 | yes | yes | 1.25 (1.16–1.35) |
| S2-klifs | KLIFS pocket | KIBA | moltrans_ig | cold-drug | 0.182464 / 0.177251 / 0.162085 | 0.0204 | 0.0106 | 0.0225 | no | no | 1.15 (1.06–1.24) |
