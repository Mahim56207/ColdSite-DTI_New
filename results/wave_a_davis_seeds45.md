# Wave A, DAVIS seeds 4 and 5: the 24 trained cells (2026-10-18)

Generated from the complete-markers and results files in `~/ColdSite-results/wave_a_davis_seeds45/merged` (outside the repository; 178 files, SHA-256 in `MANIFEST.sha256` there). All 24 cells were verified by the harness's marker check (checkpoint and results hashes) and every predictions file matches its marker's hash. Trained on Kaggle T4 GPUs, ACC1 (seed 4) on account `mahim5` and ACC2 (seed 5) on `mahim234`, harness hash `52bd3335…`, notebooks pinned to commit `5d55c63`. **torch 2.11.0+cu128 for every cell**, whereas seeds 1 to 3 were trained under torch 2.10 (`docs/PROTOCOL_AMENDMENT_v2.md`, Addendum A7 item 6).

`resumed after` is the epoch after which the cell was continued in a second session from its resume file; the harness's own tests show a resumed cell ends where an uninterrupted one does (CPU, tiny trainer), not verified on these GPUs.

| seed | model | level | test AUROC | test AUPRC | best epoch | resumed after | torch | GPU | checkpoint sha256 | predictions sha256 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | HyperAttentionDTI | random | 0.9383 | 0.6761 | 16 |  | 2.11.0+cu128 | Tesla T4 | 0639dd5b640e… | ed33b09d4298… |
| 4 | HyperAttentionDTI | cold-drug | 0.6811 | 0.1806 | 11 |  | 2.11.0+cu128 | Tesla T4 | f096af0a0855… | d57ba53fd285… |
| 4 | HyperAttentionDTI | cold-target | 0.9224 | 0.6540 | 12 |  | 2.11.0+cu128 | Tesla T4 | 66b801dcffd5… | b809a79d29b5… |
| 4 | HyperAttentionDTI | cold-pair | 0.7230 | 0.2008 | 13 |  | 2.11.0+cu128 | Tesla T4 | 109bff24efb2… | 73022be73628… |
| 4 | MolTrans | random | 0.9182 | 0.6206 | 13 |  | 2.11.0+cu128 | Tesla T4 | bae9d8e10a02… | 639b01b9d0e3… |
| 4 | MolTrans | cold-drug | 0.7249 | 0.1357 | 10 |  | 2.11.0+cu128 | Tesla T4 | d2c4d3d7cb51… | 24353d0a39df… |
| 4 | MolTrans | cold-target | 0.8428 | 0.5165 | 10 |  | 2.11.0+cu128 | Tesla T4 | 332823c690e3… | 34d87eb49184… |
| 4 | MolTrans | cold-pair | 0.4910 | 0.0831 | 11 | 16 | 2.11.0+cu128 | Tesla T4 | 0a6e27be1cb7… | 41b0ed728047… |
| 4 | XAttn-Ref | random | 0.9172 | 0.5924 | 39 |  | 2.11.0+cu128 | Tesla T4 | ca30148670c6… | ddecdf974654… |
| 4 | XAttn-Ref | cold-drug | 0.7194 | 0.1681 | 9 |  | 2.11.0+cu128 | Tesla T4 | 971a699bae44… | 624f22b8ee89… |
| 4 | XAttn-Ref | cold-target | 0.8682 | 0.5143 | 10 | 3 | 2.11.0+cu128 | Tesla T4 | 6a3965a33175… | c196d6acf66b… |
| 4 | XAttn-Ref | cold-pair | 0.6288 | 0.1366 | 9 | 1 | 2.11.0+cu128 | Tesla T4 | e0c942265689… | 19f853e8c8b0… |
| 5 | HyperAttentionDTI | random | 0.9356 | 0.6683 | 14 |  | 2.11.0+cu128 | Tesla T4 | 184295bd0db2… | 7f5571970462… |
| 5 | HyperAttentionDTI | cold-drug | 0.7240 | 0.2514 | 10 |  | 2.11.0+cu128 | Tesla T4 | 9108d5d7d4bf… | 3f8a2dfe6ecb… |
| 5 | HyperAttentionDTI | cold-target | 0.9209 | 0.6477 | 14 |  | 2.11.0+cu128 | Tesla T4 | d750a19544f4… | 2f4da486eb39… |
| 5 | HyperAttentionDTI | cold-pair | 0.6075 | 0.1297 | 11 |  | 2.11.0+cu128 | Tesla T4 | 63a9286437e0… | 72026f6ee550… |
| 5 | MolTrans | random | 0.9215 | 0.6310 | 12 |  | 2.11.0+cu128 | Tesla T4 | e0fddb97596d… | 4cf251f0c2a8… |
| 5 | MolTrans | cold-drug | 0.6386 | 0.1227 | 10 |  | 2.11.0+cu128 | Tesla T4 | aeb42f3c949d… | 406e9416cd38… |
| 5 | MolTrans | cold-target | 0.8819 | 0.5381 | 10 |  | 2.11.0+cu128 | Tesla T4 | 0d9a72e780cb… | 4c9a46ffe798… |
| 5 | MolTrans | cold-pair | 0.5950 | 0.1368 | 10 | 23 | 2.11.0+cu128 | Tesla T4 | 693807aed820… | a74be673a512… |
| 5 | XAttn-Ref | random | 0.9212 | 0.6088 | 32 |  | 2.11.0+cu128 | Tesla T4 | c52c21911cb9… | 1b2d63f6de5b… |
| 5 | XAttn-Ref | cold-drug | 0.7002 | 0.1957 | 10 | 20 | 2.11.0+cu128 | Tesla T4 | 61cc272727cd… | 04a214570131… |
| 5 | XAttn-Ref | cold-target | 0.8710 | 0.4615 | 10 | 1 | 2.11.0+cu128 | Tesla T4 | a161421aa5c6… | 970b5174f498… |
| 5 | XAttn-Ref | cold-pair | 0.6894 | 0.1623 | 9 | 1 | 2.11.0+cu128 | Tesla T4 | 8ee25ae0fbb3… | 9747e8b560a0… |

The two XAttn-Ref and one MolTrans cells per account that were paused at the session's self-stop were reported FAILED / partial by the runner in the first session: the XAttn-Ref launcher turns a deadline pause into a failed validation (exit code 1) although the resume file was saved correctly; this is cosmetic (`docs/wave_a_launch.md`).
Quota (harness ledger, per account, both sessions): ACC1 22.4 and ACC2 22.3 of 25.5 usable GPU-hours.
