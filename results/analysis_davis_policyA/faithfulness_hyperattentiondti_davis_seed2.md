# Faithfulness — hyperattentiondti_davis_seed2

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 0.4137 | 0.2503 | **0.1634** | 2.9132 | 2.9096 | 0.4674 | 200 | yes |
| Cold-Drug | 0.3585 | 0.2115 | **0.1471** | 3.8367 | 3.8334 | 0.3790 | 200 | yes |
| Cold-Target | 0.2842 | 0.1678 | **0.1163** | 1.2770 | 1.2782 | 0.2885 | 200 | yes |
| Cold-Pair | 0.1762 | 0.1147 | **0.0615** | 0.8661 | 0.8657 | 0.1726 | 200 | yes |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
