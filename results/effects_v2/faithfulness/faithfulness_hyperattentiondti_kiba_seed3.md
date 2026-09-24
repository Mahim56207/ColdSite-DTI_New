# Faithfulness — hyperattentiondti_kiba_seed3

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 0.4004 | 0.1788 | **0.2215** | 3.1377 | 3.1360 | 0.4527 | 200 | yes |
| Cold-Drug | 0.2766 | 0.1492 | **0.1274** | 2.0057 | 1.9993 | 0.3059 | 200 | yes |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
