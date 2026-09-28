# Faithfulness — hyperattentiondti_davis_seed3

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 0.3158 | 0.1743 | **0.1415** | 2.0056 | 2.0194 | 0.3270 | 200 | yes |
| Cold-Drug | 0.2339 | 0.1790 | **0.0549** | 2.7260 | 2.7242 | 0.2453 | 200 | yes |
| Cold-Target | 0.1570 | 0.1150 | **0.0420** | 1.0135 | 1.0114 | 0.1746 | 200 | yes |
| Cold-Pair | 0.1496 | 0.1025 | **0.0471** | 0.8503 | 0.8483 | 0.1628 | 200 | yes |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
