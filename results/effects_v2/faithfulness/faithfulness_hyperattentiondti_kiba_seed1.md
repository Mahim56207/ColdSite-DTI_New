# Faithfulness — hyperattentiondti_kiba_seed1

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 0.2628 | 0.1782 | **0.0846** | 2.8707 | 2.8685 | 0.3286 | 200 | yes |
| Cold-Drug | 0.1421 | 0.1232 | **0.0189** | 1.4495 | 1.4510 | 0.1897 | 200 | yes |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
