# Faithfulness — davis_seed1

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 1.1603 | 0.3589 | **0.8014** | 2.9209 | 2.8755 | 1.1918 | 200 | yes |
| Cold-Drug | 2.0266 | 0.2257 | **1.8009** | 2.1277 | 2.4878 | 1.8538 | 200 | yes |
| Cold-Target | 0.2293 | 0.0664 | **0.1630** | 0.5867 | 0.6458 | 0.2331 | 200 | yes |
| Cold-Pair | 0.7793 | 0.1600 | **0.6193** | 1.2069 | 2.0880 | 0.7414 | 200 | yes |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
