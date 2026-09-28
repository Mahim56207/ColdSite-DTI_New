# Faithfulness — davis_seed3

| Level | comp. | random control | **delta** | suff. | suff. random | AOPC | n | load-bearing? |
|---|---|---|---|---|---|---|---|---|
| Warm | 1.0992 | 0.3292 | **0.7700** | 4.1308 | 4.1291 | 1.1326 | 200 | yes |
| Cold-Drug | 1.0497 | 0.2790 | **0.7707** | 3.5080 | 3.5317 | 1.0100 | 200 | yes |
| Cold-Target | 0.6334 | 0.1373 | **0.4961** | 1.7809 | 1.9970 | 0.6646 | 200 | yes |
| Cold-Pair | 0.6225 | 0.1908 | **0.4317** | 1.7508 | 1.8499 | 0.5915 | 200 | yes |

`delta` = comprehensiveness minus its random-masking control, and it is the only column that is a result. Masking anything moves the prediction, so the raw comprehensiveness means nothing on its own.

`load-bearing? no` means masking the attended residues perturbed the prediction no more than masking arbitrary ones — the explanation is decoration at that level. That is a finding, not a failed run.
