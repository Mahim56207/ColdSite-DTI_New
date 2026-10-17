# Overleaf project (generated; do not edit by hand)

Rebuild with `python scripts/build_v3_manuscript.py && python scripts/build_v3_latex.py`. Source of truth is the markdown in `paper/v3_ismb/`; edit that and regenerate.

**Upload:** Overleaf -> New Project -> Upload Project -> `paper/v3_ismb/overleaf.zip`. Set the main document to `main.tex`, compiler pdfLaTeX (default), TeX Live current.
The Supplement is a separate document: set the main document to `supplement.tex` (or make a second project) and compile it the same way.

**Not verified:**
* The project has **not been compiled** (the machine that built it has no TeX engine). I checked brace and environment balance, that every `\cite` key exists in `references.bib`,
  and that every figure file exists. Expect to look at the first Overleaf compile for warnings; the likely ones are table widths in the Supplement (Tables S3 and S5 are wide) and BibTeX style details.
* **Layout is a neutral article + natbib (plainnat) layout, not the journal's template.** The template for the *Bioinformatics* journal on Overleaf is dated 2017, and which template ISMB 2027
  requires is unknown until the call is out. `manuscript_body.tex` holds the body text alone, for pasting into the official class later.
* `references.bib` comes from saved Crossref/DataCite records; Holm 1979 and Sundararajan 2017 come from `paper/references.md` (initials only, no registry record).
* Figure references in the text ("Figure 1") are plain text, not `\ref`; LaTeX numbers captions in order of placement (main text 1 to 4, which matches the text; the Supplement's figures print as S1 and S2).
* Placeholders (affiliation, e-mail, DOI, funding, conflict of interest, acknowledgements) are still marked in the text.
