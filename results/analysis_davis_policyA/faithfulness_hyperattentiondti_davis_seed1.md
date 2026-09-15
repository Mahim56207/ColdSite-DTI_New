# Faithfulness — hyperattentiondti_davis_seed1

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 0.6051 | 0.3574 | **0.2478** | 3.5651 | 3.5597 | 0.7088 | 200 | yes |
| Cold-Drug | 0.3562 | 0.2179 | **0.1384** | 3.8031 | 3.8007 | 0.3962 | 200 | yes |
| Cold-Target | 0.2007 | 0.1690 | **0.0316** | 1.2476 | 1.2543 | 0.1964 | 200 | yes |
| Cold-Pair | 0.1859 | 0.1273 | **0.0586** | 1.1877 | 1.1859 | 0.1822 | 200 | yes |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
