# Notebooks — which file to import

**Import the `.ipynb` files into Kaggle. Never the `.py` files.**

| import this into Kaggle | what it trains |
|---|---|
| `kaggle_drugban_davis.ipynb` | DrugBAN, 12 DAVIS cells (4 levels × 3 seeds) |
| `kaggle_coldsite_kiba.ipynb` | ColdSite-DTI, 6 KIBA cells (random + cold-drug × 3 seeds) |

One-click import (opens the notebook as a new Kaggle notebook):

    https://www.kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/Mahim56207/ColdSite-DTI_New/main/notebooks/<name>.ipynb

Use the **raw** URL, as above. A `github.com/.../blob/...` URL is a web page, and Kaggle
imports the page's HTML instead of the notebook.

**You have the right file** if the first cell is a formatted markdown heading and the
table of contents on the right lists numbered sections. If you see one red code cell
beginning `"""Build notebooks/...`, you imported a builder script — delete it and import
the `.ipynb`.

## The `.py` files here are generators, not notebooks

| file | generates |
|---|---|
| `build_drugban_nb.py` + `runner_patch.py` | `kaggle_drugban_davis.ipynb` |
| `build_coldsite_kiba_nb.py` + `runner_patch_coldsite.py` | `kaggle_coldsite_kiba.ipynb` |

They exist so a notebook's contents are reviewable as code and rebuild byte-identically:

    python notebooks/build_coldsite_kiba_nb.py

Edit the builder, rebuild, commit both. Never hand-edit a generated `.ipynb`: the next
rebuild would silently undo it.
