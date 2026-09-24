# Faithfulness — hyperattentiondti_kiba_seed2

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 0.2702 | 0.1809 | **0.0893** | 2.9986 | 2.9973 | 0.3071 | 200 | yes |
| Cold-Drug | 0.2766 | 0.1372 | **0.1394** | 1.6072 | 1.6099 | 0.2902 | 200 | yes |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
