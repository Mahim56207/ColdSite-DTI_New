# Sequence audit — davis

What the model reads for each target. Produced by `src/data/sequence_audit.py`; changes nothing, reports only.

**379 distinct sequences for 442 targets.**

## 1. Variants identical to their wild type

74 variant targets: **54 carry exactly the wild-type sequence**, 0 differ from it, 20 have no wild-type entry to compare.

Identical: ABL1(E255K), ABL1(F317I), ABL1(F317I)p, ABL1(F317L), ABL1(F317L)p, ABL1(H396P), ABL1(H396P)p, ABL1(M351T), ABL1(Q252H), ABL1(Q252H)p, ABL1(T315I), ABL1(T315I)p, ABL1(Y253F), BRAF(V600E), EGFR(E746A750del), EGFR(G719C), EGFR(G719S), EGFR(L747E749del), EGFR(L747S752del), EGFR(L747T751del), EGFR(L858R), EGFR(L858RT790M), EGFR(L861Q), EGFR(S752I759del), EGFR(T790M), FGFR3(G697C), FLT3(D835H), FLT3(D835Y), FLT3(ITD), FLT3(K663Q), FLT3(N841I), FLT3(R834Q), KIT(A829P), KIT(D816H), KIT(D816V), KIT(L576P), KIT(V559D), KIT(V559D-T670I), KIT(V559D-V654A), LRRK2(G2019S), MET(M1250T), MET(Y1235D), PIK3CA(C420R), PIK3CA(E542K), PIK3CA(E545A), PIK3CA(E545K), PIK3CA(H1047L), PIK3CA(H1047Y), PIK3CA(I800L), PIK3CA(M1043I), PIK3CA(Q546K), RET(M918T), RET(V804L), RET(V804M)

## 2. Test targets unseen by name but seen by sequence

| level | part | targets | unseen by name | identical sequence in training | rows affected |
|---|---|---|---|---|---|
| random | test | 442 | 0 | 0 | 0 of 6011 (0.0%) |
| random | valid | 442 | 0 | 0 | 0 of 3006 (0.0%) |
| cold_drug | test | 442 | 0 | 0 | 0 of 5746 (0.0%) |
| cold_drug | valid | 442 | 0 | 0 | 0 of 2652 (0.0%) |
| cold_target | test | 88 | 88 | 12 | 816 of 5984 (13.6%) |
| cold_target | valid | 44 | 44 | 2 | 136 of 2992 (4.5%) |
| cold_pair | test | 88 | 88 | 11 | 143 of 1144 (12.5%) |
| cold_pair | valid | 44 | 44 | 6 | 36 of 264 (13.6%) |

cold_target test targets seen by sequence: ABL1(F317I), EGFR(G719S), EGFR(L858RT790M), FLT3(D835H), FLT3(K663Q), FLT3(R834Q), KIT(L576P), MET, PIK3CA(E542K), RET(V804M), RSK3(KinDom.2-C-terminal), TYK2(JH1domain-catalytic)

cold_pair test targets seen by sequence: ABL1p, EGFR(G719S), EGFR(L858RT790M), FGFR3(G697C), FLT3(N841I), MET, PIK3CA(H1047L), PIK3CA(H1047Y), PIK3CA(Q546K), RET(V804M), RPS6KA4(KinDom.2-C-terminal)

## 3. Sequences without the kinase pocket

Targets whose sequence holds fewer than 40 of the 85 KLIFS pocket residues (`data/davis_klifs_pocket_report.json`): **10**

- BTK: 13 of 85
- EPHA6: 0 of 85
- EPHB2: 0 of 85
- MLK1: 0 of 85
- RET: 0 of 85
- RET(M918T): 0 of 85
- RET(V804L): 0 of 85
- RET(V804M): 0 of 85
- ROCK2: 0 of 85
- TRKB: 0 of 85
