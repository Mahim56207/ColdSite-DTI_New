"""PHASE 1a: hash every file under results/, record git state and mtime, flag files
newer than the draft. Independent of the repo's own provenance tooling."""
import hashlib, os, subprocess, sys, datetime as dt, csv, collections
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
os.chdir(ROOT)
DRAFT = "paper/v2/INSTRUCTOR_DRAFT.md"
OUT = os.path.join(ROOT, "results", "certification", "results_manifest.tsv")

def sh(*a):
    return subprocess.run(a, capture_output=True, text=True, cwd=ROOT).stdout

draft_mtime = os.stat(DRAFT).st_mtime
draft_iso = dt.datetime.fromtimestamp(draft_mtime).astimezone().isoformat(timespec="seconds")
print(f"DRAFT {DRAFT}\n  sha256={hashlib.sha256(open(DRAFT,'rb').read()).hexdigest()}\n  mtime={draft_iso}")
print("  git log:", sh("git","log","-3","--format=%h %cI %s","--",DRAFT).strip().replace("\n","\n           "))
print("  git status of draft:", repr(sh("git","status","--porcelain","--",DRAFT)))

tracked = set(sh("git","ls-files","-z","results").split("\0")) - {""}
status = {}
for line in sh("git","status","--porcelain","--ignored","-uall","--","results").splitlines():
    status[line[3:].strip('"')] = line[:2]
last_commit = {}
cur = None
for line in sh("git","log","--name-only","--format=@@%cI","--","results").splitlines():
    if line.startswith("@@"): cur = line[2:]
    elif line and line not in last_commit: last_commit[line] = cur

rows = []
for dp, dn, fn in os.walk("results"):
    for f in sorted(fn):
        if f in ('results_manifest.tsv', 'traceability_matrix.tsv', 'draft_numbers.tsv', 'untagged_numbers.tsv'): continue   # this script's own outputs
        p = os.path.join(dp, f)
        try:
            b = open(p, "rb").read()
        except OSError as e:
            rows.append((p, "UNREADABLE", str(e), 0, "", "", "", "", ""))
            continue
        st = os.stat(p)
        if p in tracked:
            g = "tracked-dirty" if status.get(p, "  ").strip() else "tracked-clean"
        else:
            g = {"!!":"ignored","??":"untracked"}.get(status.get(p,"  "), "unknown")
        rows.append((p, hashlib.sha256(b).hexdigest(), hashlib.md5(b).hexdigest(), st.st_size,
                     dt.datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds"),
                     g, last_commit.get(p, ""), "NEWER_THAN_DRAFT" if st.st_mtime > draft_mtime else "", ""))
rows.sort()
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(["path","sha256","md5","bytes","mtime","git_state","last_commit","flag","-"])
    w.writerows(rows)
print(f"\nFILES HASHED: {len(rows)}   manifest -> {OUT}")
print("GIT STATE:", dict(collections.Counter(r[5] for r in rows)))
print("UNREADABLE:", [r[0] for r in rows if r[1]=="UNREADABLE"])
newer = [r for r in rows if r[7]]
print(f"\nFILES WITH mtime AFTER DRAFT ({draft_iso}): {len(newer)}")
for r in newer[:60]: print("  ", r[4], r[5], r[0])
dirty = [r for r in rows if r[5]=="tracked-dirty"]
print(f"\nTRACKED FILES WITH UNCOMMITTED EDITS: {len(dirty)}")
for r in dirty: print("  ", r[0])
d0 = draft_iso
late = [r for r in rows if r[6] and r[6] > "2026-09-25T13:56:55+05:30"]
print(f"\nTRACKED FILES COMMITTED AFTER d27b7d9 (draft's last commit): {len(late)}")
for r in late: print("  ", r[6], r[0])
print("\nPRIOR HASH RECORDS IN REPO (anything to compare against):")
print(sh("find",".","-not","-path","./.git/*","(","-iname","*manifest*","-o","-iname","*.sha256","-o","-iname","*checksum*",")").strip() or "  none")
print("\nCITED-BY-DRAFT ANCHOR CHECK is done in p1_trace.py")
print("\nsha256 of every file whose name the draft cites is printed by p1_trace.py; full list in the TSV.")
