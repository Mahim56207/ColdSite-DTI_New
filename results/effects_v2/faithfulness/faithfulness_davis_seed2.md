# Faithfulness — davis_seed2

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 2.3502 | 0.3573 | **1.9929** | 4.3991 | 4.4983 | 2.1340 | 200 | yes |
| Cold-Drug | 1.0545 | 0.0889 | **0.9656** | 2.0444 | 1.9399 | 1.0793 | 200 | yes |
| Cold-Target | 1.0528 | 0.1620 | **0.8908** | 3.6744 | 4.2203 | 1.0049 | 200 | yes |
| Cold-Pair | 1.1022 | 0.1577 | **0.9444** | 3.0077 | 3.5687 | 1.0530 | 200 | yes |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
