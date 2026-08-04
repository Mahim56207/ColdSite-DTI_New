# Binding-site ground truth — format and how to use it

Track A deliverable for 124AD0067 (docs/01_GUIDE_124AD0008.md Step 4).

## Do not read the JSON directly

```python
# WRONG -- costs about a third of the score, silently
sites = json.load(open("data/davis_ground_truth_sites.json"))["AAK1"]
precision_at_k(attention, {s["start"] for s in sites}, k=10)

# RIGHT
from src.data.ground_truth import load_site_sets
sites = load_site_sets("data/davis_ground_truth_sites.json", max_len=1000)
precision_at_k(attention, sites["AAK1"].positions, k=10)
```

UniProt reports **1-indexed, inclusive residue ranges**. `precision_at_k`
expects a **0-indexed set of array positions**. Feeding one to the other does
not raise — it returns a plausible-looking number that is wrong. On a model
with perfect attention it returns 0.67 instead of 1.00
(`tests/test_ground_truth.py::test_off_by_one_regression`).

## File format

`data/{dataset}_ground_truth_sites.json`

```json
{
  "AAK1": [
    {"start": 176, "end": 176, "type": "Active site", "description": "Proton acceptor"},
    {"start": 52,  "end": 60,  "type": "Nucleotide binding", "description": "ATP"}
  ]
}
```

| field | meaning |
|---|---|
| `start`, `end` | 1-indexed, **inclusive** UniProt residue numbers. `52–60` is nine residues. |
| `type` | UniProt feature type. Only `Binding site`, `Active site`, `Nucleotide binding` are collected. |
| `description` | free text, informational only |

A sibling `*_provenance.json` records, per target, which UniProt accession was
used and how it was resolved.

## What the adapter gives you

`load_site_sets()` returns `{target_id: SiteSet}`:

| attribute | meaning |
|---|---|
| `.positions` | 0-indexed set — this is what the metric consumes |
| `.usable` | `False` if nothing survived filtering; such proteins must be **skipped**, not scored 0.0 |
| `.is_variant` / `.resolved_from` | `ABL1(T315I)p` → `True` / `ABL1` |
| `.n_dropped_truncation` etc. | what was discarded and why, for the Methods section |

## Three decisions that must be recorded in the paper

**1. Truncation.** The model reads at most `max_protein_len` residues (default
1000). Annotated sites past that point can never be retrieved.

- `truncation="exclude"` (default) — drop them, and drop the protein if nothing
  is left. Measures explanation quality on residues the model can see.
- `truncation="keep"` — retain them; the protein stays in the average and its
  ceiling reflects the unreachable sites.

Current impact at `max_len=1000`: **283 positions / 15 targets** dropped in
DAVIS, **165 positions / 10 targets** in KIBA.

Note the asymmetry: this does not change raw precision@k, because there is no
attention out past the cut. It changes which proteins are averaged over, and
the achievable ceiling.

**2. Mutant targets.** DAVIS names point mutants separately (`ABL1(T315I)p`).
UniProt has no entry for them, so all 72 variant IDs resolve to wild-type
accessions — every ABL1 variant gets an identical site list. Defensible;
must be stated.

**3. The ceiling.** With fewer than *k* annotated sites, perfect attention
still cannot reach 1.0 — six sites at k=20 caps at 0.30. Report
`mean_normalised` alongside `mean_precision_at_k`, or the numbers are not
comparable across proteins.

## Current coverage

Run `python -m src.data.ground_truth` to regenerate these numbers.

| | DAVIS | KIBA |
|---|---|---|
| targets in file | 409 | 224 |
| usable at `max_len=1000` | 394 | 214 |
| distinct wild-type accessions | 336 | 214 |
| total 0-indexed positions | 4,902 | 2,732 |

> ⚠️ **These files need regenerating.** They were produced by a fetcher that
> also collected UniProt's `Site` catch-all type, which carries protease
> cleavage points and chromosomal breakpoints — positions no drug binds to,
> counted as correct answers. The adapter currently filters them by description
> heuristic (125 dropped in DAVIS, 67 in KIBA), which has known false negatives.
> The fix is a re-fetch:
>
> ```bash
> python -m src.data.fetch_binding_sites --dataset davis
> python -m src.data.fetch_binding_sites --dataset kiba
> ```
>
> That writes `type` on every record, after which the adapter filters exactly
> and `dropped_feature_type` becomes non-zero instead of `dropped_description`.
> DAVIS coverage is also 409 of 442 targets — the 33 unresolved ones need a
> manual gene→UniProt lookup.
