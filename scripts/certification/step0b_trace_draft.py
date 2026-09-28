"""PHASE 1b/1c: (2) extract every number in the draft prose, (3) trace every source tag to the
exact raw file+line/row and re-check the value, (4) flag mismatches / uncovered numbers.
Independent of scripts/check_number_provenance.py."""
import re, os, csv, json, hashlib, io, sys, collections, datetime as dt
from decimal import Decimal, InvalidOperation
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2]); os.chdir(ROOT)
HERE = os.path.join(ROOT, "results", "certification")
DRAFT = os.environ.get("DRAFT","paper/v2/INSTRUCTOR_DRAFT.md")
OUTP = os.environ.get("OUTP","")
draft_mtime = os.stat(DRAFT).st_mtime
manifest = {r["path"]: r for r in csv.DictReader(open(os.path.join(HERE, "results_manifest.tsv")), delimiter="\t")}
text = open(DRAFT, encoding="utf-8").read()
lines = text.split("\n")

NUM = re.compile(r"(?<![\w.])[-−+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
def norm(s): return s.replace("−", "-").replace(",", "").lstrip("+")
def dec(s):
    try: return Decimal(norm(s))
    except InvalidOperation: return None
def decimals(s):
    s = norm(s); return len(s.split(".")[1]) if "." in s else 0
def close(val, tok):
    v, t = dec(val), dec(tok)
    if v is None or t is None: return False
    tol = Decimal("0.5") * (Decimal(10) ** -decimals(val)) + Decimal("1e-12")
    return abs(v - t) <= tol
_hcache = {}
def file_info(path):
    if path in manifest:
        m = manifest[path]; return m["sha256"][:16], m["git_state"], ("YES" if m["flag"] else "no")
    if not os.path.exists(path): return "MISSING", "", ""
    if path not in _hcache:
        _hcache[path] = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    import subprocess
    tr = subprocess.run(["git","ls-files","--error-unmatch",path],capture_output=True,cwd=ROOT).returncode==0
    dirty = subprocess.run(["git","status","--porcelain","--",path],capture_output=True,text=True,cwd=ROOT).stdout.strip()
    st = ("tracked-dirty" if dirty else "tracked-clean") if tr else "untracked/ignored"
    return _hcache[path], st, ("YES" if os.stat(path).st_mtime > draft_mtime else "no")

# ---------- 1. tags -------------------------------------------------------------------------
tag_re = re.compile(r"<!--\s*(.*?)\s*-->", re.S)
tags = []   # dict(line, raw, kind, ...)
for m in tag_re.finditer(text):
    ln = text.count("\n", 0, m.start()) + 1
    raw = m.group(1)
    if raw.startswith("src:"):
        body = raw[4:].strip()
        if body.startswith("derived:"):
            tags.append(dict(line=ln, raw=raw, kind="derived", body=body[8:].strip()))
            continue
        if " = " not in body: tags.append(dict(line=ln, raw=raw, kind="malformed")); continue
        left, vals = body.rsplit(" = ", 1)
        d = dict(line=ln, raw=raw, vals=[v for v in re.findall(NUM.pattern, vals)] or [vals.strip()])
        if "#$" in left:
            d.update(kind="json", path=left.split("#$")[0], sel=left.split("#$")[1])
        elif "#rows" in left:
            d.update(kind="csvrows", path=left.split("#")[0])
        elif "#" in left and "->" in left:
            p, rest = left.split("#", 1); flt, cols = rest.split("->", 1)
            d.update(kind="csv", path=p, filt=dict(kv.split("=", 1) for kv in flt.split("&")), cols=[c.strip() for c in cols.split(",")])
        elif re.search(r":\d+$", left):
            p, l = left.rsplit(":", 1); d.update(kind="line", path=p, lineno=int(l))
        else: d.update(kind="malformed")
        tags.append(d)
    elif raw.startswith("claim"): tags.append(dict(line=ln, raw=raw, kind="claim"))
    else: tags.append(dict(line=ln, raw=raw, kind="comment"))

def check(t):
    """return (status, expected, observed)"""
    k = t["kind"]
    if k in ("claim", "comment"): return "MANUAL (prose claim; Phase 2)", "", ""
    if k == "malformed": return "FAIL: unparseable tag", "", ""
    if k == "derived":
        m = re.match(r"(.+?)=\s*([-\d.,]+)\s*$", t["body"])
        expr, res = m.group(1), m.group(2)
        if not re.fullmatch(r"[\d.\s+\-*/()]+", expr): return "FAIL: derived expr not arithmetic", expr, ""
        val = eval(expr); ok = abs(val - float(res)) <= 0.5 * 10 ** -decimals(res) + 1e-12
        return ("MATCH" if ok else "FAIL: arithmetic"), res, f"{expr}= {val:.6g}"
    p = t["path"]
    if not os.path.exists(p): return "FAIL: FILE MISSING", ",".join(t["vals"]), ""
    if k == "line":
        fl = open(p, encoding="utf-8", errors="replace").read().split("\n")
        if t["lineno"] > len(fl): return "FAIL: line beyond EOF", ",".join(t["vals"]), f"file has {len(fl)} lines"
        ltxt = fl[t["lineno"] - 1]; toks = NUM.findall(ltxt)
        miss = [v for v in t["vals"] if not any(close(v, x) for x in toks)]
        # tolerate percent-form (0.72 shown as 72)
        miss2 = [v for v in miss if not any(dec(v) is not None and dec(x) is not None and abs(dec(x) * 100 - dec(v)) <= Decimal("0.5") * Decimal(10) ** -decimals(v) for x in toks)]
        note = "" if not miss or len(miss2) < len(miss) and not miss2 else ""
        if miss2: return "FAIL: value not on that line", ",".join(t["vals"]), ltxt.strip()[:140]
        return ("MATCH*pct-form" if miss else "MATCH"), ",".join(t["vals"]), ltxt.strip()[:140]
    if k == "csvrows":
        rows = list(csv.reader(open(p, encoding="utf-8")))
        n = len(rows) - 1
        return ("MATCH" if close(t["vals"][0], str(n)) else "FAIL: row count"), t["vals"][0], f"{n} data rows (+1 header)"
    if k == "csv":
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        hit = [r for r in rows if all((r.get(a) or "").strip() == b.strip() for a, b in t["filt"].items())]
        if len(hit) != 1: return f"FAIL: selector matched {len(hit)} rows", ",".join(t["vals"]), str(t["filt"])
        r = hit[0]; obs = []
        if any(c not in r for c in t["cols"]): return "FAIL: column missing", ",".join(t["cols"]), ",".join(r.keys())[:120]
        obs = [r[c] for c in t["cols"]]
        if len(obs) != len(t["vals"]): return "FAIL: value/column count differs", ",".join(t["vals"]), ",".join(obs)
        bad = [(v, o) for v, o in zip(t["vals"], obs) if not close(v, (NUM.search(o).group(0) if NUM.search(o) else o))]
        return ("MATCH" if not bad else "FAIL: value differs"), ",".join(t["vals"]), ",".join(o[:10] for o in obs)
    if k == "json":
        obj = json.load(open(p)); cur = obj
        for part in re.findall(r"[^.\[\]]+|\[\d+\]", t["sel"]):
            cur = cur[int(part[1:-1])] if part.startswith("[") else cur[part]
        return ("MATCH" if close(t["vals"][0], str(cur)) else "FAIL: value differs"), t["vals"][0], str(cur)

results = []
for t in tags:
    st, exp, obs = check(t)
    sha, gs, newer = ("", "", "") if t["kind"] in ("claim", "comment", "derived", "malformed") else file_info(t["path"])
    where = t.get("path", "") + (f":{t['lineno']}" if t["kind"] == "line" else "") + ("" if t["kind"] != "csv" else "#" + "&".join(f"{a}={b}" for a, b in t["filt"].items()) + "->" + ",".join(t["cols"]))
    results.append(dict(draft_line=t["line"], kind=t["kind"], where=where, expected=exp, observed=obs, status=st, sha=sha, git=gs, newer=newer, vals=t.get("vals", [])))

with open(os.path.join(HERE, OUTP+"traceability_matrix.tsv"), "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(["draft_line","kind","raw_file(:line|#selector)","draft_value","raw_value","status","sha256[:16]","git_state","modified_after_draft"])
    for r in results: w.writerow([r["draft_line"], r["kind"], r["where"], r["expected"], r["observed"], r["status"], r["sha"], r["git"], r["newer"]])

# ---------- 2. every number in the prose ---------------------------------------------------
def strip_comments(s): return tag_re.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), s)   # keep line numbers
prose = strip_comments(text).split("\n")
occ = []
for i, ln in enumerate(prose, 1):
    if ln.startswith("```"): continue
    code_spans = [(m.start(), m.end()) for m in re.finditer(r"`[^`]*`|\]\([^)]*\)", ln)]
    for m in NUM.finditer(ln):
        s, e = m.span(); tok = m.group(0); pre = ln[max(0, s - 14):s]
        cat = "VALUE"
        if any(a <= s < b for a, b in code_spans): cat = "CODE/LINK"
        elif re.match(r"^#+\s", ln) and s < 12: cat = "HEADING-NUM"
        elif re.search(r"(§|Section[s]?\s?S?|Figure[s]?\s|Fig\.\s|Table[s]?\s(S)?|Box\s|Supplementary\s(Table\s)?S?|ref(erence)?s?\s|\bT|\bk\s\|)\s*$", pre): cat = "LABEL/REF"
        elif re.fullmatch(r"(19|20)\d\d", tok) and (pre.rstrip().endswith(("(", ",", ";", "IG", "et al.,", "&")) or re.search(r"\(\s*$|,\s*$", pre) or ln[e:e+1] in (")", ",", ";")): cat = "YEAR"
        elif re.match(r"^\s*[|]?\s*\d+\s*[|.]\s", ln[s:s+6]) and s < 6: cat = "LIST-NUM"
        occ.append(dict(line=i, tok=tok, cat=cat, ctx=ln[max(0, s-35):e+25].strip()))
# blocks (paragraph = run of non-blank lines) and tag values per block
blk = {}; b = 0
for i, ln in enumerate(lines, 1):
    if ln.strip() == "": b += 1
    blk[i] = b
blk_vals = collections.defaultdict(list)
for r in results:
    for v in r["vals"]: blk_vals[blk[r["draft_line"]]].append((v, r))
# derived results
for t, r in zip(tags, results):
    if t["kind"] == "derived":
        for x in NUM.findall(t["body"]): blk_vals[blk[t["line"]]].append((x, r))
for r in results:
    r.setdefault("_", 0)
# text corpus for uncovered numbers
corpus = []
for base in ("results", "docs"):
    for dp, dn, fn in os.walk(base):
        for f in fn:
            if f.endswith((".md", ".csv", ".txt", ".json", ".log")): corpus.append(os.path.join(dp, f))
def corpus_hits(tok, cap=3):
    pat = re.compile(r"(?<![\d.])" + re.escape(norm(tok)) + r"(?![\d])"); hits = []; n = 0
    for p in corpus:
        try:
            for j, l in enumerate(open(p, encoding="utf-8", errors="ignore"), 1):
                if pat.search(l.replace(",", "")):
                    n += 1
                    if len(hits) < cap: hits.append(f"{p}:{j}")
        except OSError: pass
    return n, hits
cov = collections.Counter(); unc = []
for o in occ:
    if o["cat"] != "VALUE": cov[o["cat"]] += 1; continue
    tagged = [(v, r) for v, r in blk_vals[blk[o["line"]]] if dec(v) is not None and dec(o["tok"]) is not None and abs(dec(v) - dec(o["tok"])) <= Decimal("1e-12")]
    if tagged:
        o["covered"] = tagged[0][1]["status"]; cov["VALUE tagged & " + ("MATCH" if all(r["status"].startswith("MATCH") for _, r in tagged) else "CHECK")] += 1
    else:
        big = "." in norm(o["tok"]) or len(norm(o["tok"]).lstrip("-")) >= 3
        o["covered"] = None; o["big"] = big; unc.append(o); cov["VALUE untagged (" + ("decimal/3+digit" if big else "small integer") + ")"] += 1
with open(os.path.join(HERE, OUTP+"draft_numbers.tsv"), "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t"); w.writerow(["draft_line","token","category","tag_status","context"])
    for o in occ: w.writerow([o["line"], o["tok"], o["cat"], o.get("covered") or ("" if o["cat"] != "VALUE" else "UNTAGGED"), o["ctx"]])

# ---------- report ---------------------------------------------------------------------------
kinds = collections.Counter(r["kind"] for r in results)
stat = collections.Counter(("MATCH" if r["status"].startswith("MATCH") else r["status"].split(":")[0] if r["status"].startswith("FAIL") else "MANUAL") for r in results)
print(f"TAGS FOUND: {len(tags)}   by kind: {dict(kinds)}")
print(f"TAG CHECK RESULTS: {dict(stat)}")
print(f"NUMERIC TOKENS IN PROSE (comments stripped): {len(occ)}   by class: {dict(collections.Counter(o['cat'] for o in occ))}")
print("COVERAGE OF VALUE TOKENS:", dict((k, v) for k, v in cov.items() if k.startswith("VALUE")))
print("\n=== FATAL INTEGRITY CANDIDATES: tags that FAIL against the raw file ===")
fails = [r for r in results if r["status"].startswith("FAIL")]
for r in fails: print(f"L{r['draft_line']:>4} {r['status']} | draft={r['expected']} | raw={r['observed'][:100]} | {r['where']}")
print(f"({len(fails)} failing)")
print("\n=== pct-form matches (draft shows 72, raw shows 0.72) ===")
for r in results:
    if r["status"] == "MATCH*pct-form": print(f"L{r['draft_line']:>4} draft={r['expected']} raw={r['observed'][:80]} {r['where']}")
print("\n=== CITED FILES: git anchor and modification-after-draft ===")
cited = collections.OrderedDict()
for r in results:
    if r["where"] and r["kind"] not in ("derived",): cited.setdefault(r["where"].split(":")[0].split("#")[0], (r["sha"], r["git"], r["newer"]))
by_git = collections.Counter(v[1] for v in cited.values())
print(f"distinct cited files: {len(cited)}   git state: {dict(by_git)}   modified after draft: {sum(1 for v in cited.values() if v[2]=='YES')}   missing: {sum(1 for v in cited.values() if v[0]=='MISSING')}")
for p, (sha, gs, nw) in cited.items(): print(f"  {sha:16} {gs:18} newer={nw:3} {p}")
print("\n=== SOURCES OUTSIDE results/ (weak provenance: author's own narrative docs, code constants) ===")
for p in cited:
    if not p.startswith("results/"): print("  ", p)
print(f"\n=== UNTAGGED VALUE TOKENS (decimal or 3+ digits): candidate provenance by string search (results/ + docs/) ===")
big = [o for o in unc if o.get("big")]
rows = []
for o in big:
    n, h = corpus_hits(o["tok"]); rows.append((o, n, h))
with open(os.path.join(HERE, OUTP+"untagged_numbers.tsv"), "w") as fh:
    fh.write("line\ttoken\thits\tfirst_hits\tcontext\n")
    for o, n, h in rows: fh.write(f"{o['line']}\t{o['tok']}\t{n}\t{' | '.join(h)}\t{o['ctx']}\n")
zero = [(o, n, h) for o, n, h in rows if n == 0]
print(f"{len(big)} untagged decimal/3+digit tokens; {len(zero)} have NO string match anywhere in results/ or docs/")
for o, n, h in zero: print(f"  L{o['line']:>4} '{o['tok']}' :: {o['ctx']}")
print(f"\n{len([o for o in unc if not o.get('big')])} untagged small integers (cannot be string-traced; need semantic check in Phase 2/3)")
print("\nFiles written:", *[os.path.join(HERE, f) for f in (OUTP+"traceability_matrix.tsv",OUTP+"draft_numbers.tsv",OUTP+"untagged_numbers.tsv","results_manifest.tsv")], sep="\n  ")
