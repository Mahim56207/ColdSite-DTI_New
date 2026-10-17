#!/usr/bin/env python3
"""Assemble paper/v3_ismb/MANUSCRIPT.md (with source tags) and MANUSCRIPT_clean.md (tags/comments stripped) from the section files.

Order: title + front matter, abstract, 1 Introduction, 2 Methods, 3 Results (figures placed at the end of 3.2, 3.4, 3.6), 4 Discussion,
declarations (placeholders only), references. The Supplement stays its own file (journal rule: one separate supplementary file).
Nothing is rewritten here; sections are concatenated as written.
"""
import os, re
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "v3_ismb")
rd = lambda n: open(os.path.join(D, n), encoding="utf-8").read().strip() + "\n"

caps = {}
for m in re.finditer(r"\*\*(Figure \d|Supplementary Figure S\d)\. .*?(?=\n\n\*\*|\Z)", rd("figure_captions.md"), flags=re.S):
    caps[m.group(1)] = m.group(0)
img = {"Figure 1": "fig1_seed_instability", "Figure 2": "fig2_enrichment_forest", "Figure 5": "fig5_faithfulness", "Figure 6": "fig6_leakage"}
def fig(name):
    return f"\n![{name}](../v2/figures/{img[name]}.png)\n\n{caps[name]}\n"

abstract = rd("abstract.md")
abstract = abstract.split("\n---\n")[0]                      # drop the optional Key points block
results = rd("results.md")
results = results.replace("\n## 3.3 ", fig("Figure 1") + "\n## 3.3 ", 1)
results = results.replace("\n## 3.5 ", fig("Figure 2") + fig("Figure 5") + "\n## 3.5 ", 1)
tail = "\nEvery result in this section concern"
tail = "\nEvery result in this section concerns kinase"
assert tail in results
results = results.replace(tail, fig("Figure 6") + tail, 1)

front = ("<!-- v3 (ISMB / Bioinformatics format), assembled by scripts/build_v3_manuscript.py. Venue: ISMB 2027 Proceedings -> Bioinformatics (unverified call; see VENUE_SPEC.md). -->\n\n"
         "**Mahim Agarwal** [affiliation and e-mail: to be added by the author]; supervised by **Dr. Chandra Mohan Dasari**.\n")
decl = ("\n## Declarations [placeholders: wording and required items to be matched to the journal template, which is unverified]\n\n"
        "* **Funding:** [to be completed by the author]\n"
        "* **Conflict of interest:** XAttn-Ref is the authors' own model (Section 2.2); [formal statement to be completed by the author]\n"
        "* **Data and code availability:** see the abstract; [archived release DOI and checkpoint location to be added]\n"
        "* **Acknowledgements:** [to be completed by the author]\n")
doc = "\n".join([abstract, front, rd("introduction.md"), rd("methods.md"), results, rd("discussion.md"), decl, rd("references.md")])
doc = re.sub(r"\n{3,}", "\n\n", doc)
open(os.path.join(D, "MANUSCRIPT.md"), "w", encoding="utf-8").write(doc)
clean = re.sub(r"<!--.*?-->", "", doc, flags=re.S)
clean = re.sub(r"[ \t]+\n", "\n", clean); clean = re.sub(r"\n{3,}", "\n\n", clean)
open(os.path.join(D, "MANUSCRIPT_clean.md"), "w", encoding="utf-8").write(clean)

def words(t):
    return len(re.sub(r"<!--.*?-->", "", t, flags=re.S).split())
print("section words (headings and captions included, comments excluded):")
for n in ("abstract", "introduction", "methods", "results", "discussion"):
    print(f"  {n:13s}{words(rd(n + '.md')):6d}")
body = sum(words(rd(n + '.md')) for n in ("introduction", "methods", "results", "discussion"))
capw = sum(words(c) for k, c in caps.items() if k in img)
print(f"  main text (Introduction..Discussion): {body}; main-figure captions: {capw}; supplement: {words(rd('supplement.md'))}")
