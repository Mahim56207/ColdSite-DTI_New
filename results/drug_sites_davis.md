# Drug-specific ground truth — KLIFS ligand contacts

217 drug-target pairs with a co-crystal structure, covering 101 proteins and 28 drugs (`src/data/klifs_ligand_contacts.py`).

- contacted residues per pair: median 19, range 3-27
- structures per pair: median 3, range 1-18
- a residue counts when it is contacted in at least 50% of the pair's structures

- of the 175 pairs with more than one structure, the agreed set is 93% the size of the union — crystals of one pair do not contact identical residues

## What each split can support

| level | test rows | scorable pairs | proteins | drugs | swappable |
|---|---|---|---|---|---|
| random | 6011 | **38** | 29 | 17 | 30 |
| cold-drug | 5746 | **60** | 57 | 6 | 39 |
| cold-target | 5984 | **39** | 20 | 19 | 27 |
| cold-pair | 1144 | **12** | 12 | 5 | 6 |

A pair is scorable only where that exact drug has been crystallised with that exact kinase. At cold-pair there are too few pairs for a precision@k test to be worth much, which is a fact about what DAVIS supports, not a result.
