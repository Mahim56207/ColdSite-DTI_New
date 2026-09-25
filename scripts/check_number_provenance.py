#!/usr/bin/env python3
"""Number-provenance checker for the manuscript (plan T20/T21, problem P29).

Every number written in digits in the manuscript prose must be backed by a hidden source tag in
the same block (paragraph, list item or table row), and every tag must resolve to an existing file
and a value that matches what the prose states.

Tag grammar (an HTML comment, invisible when rendered):

    <!-- src: PATH[LOCATOR] = V1[, V2 ...] -->

PATH is relative to the repository root (a leading ``~`` is expanded; such tags are reported as
EXTERNAL, i.e. outside the repository, but still checked). LOCATOR selects the value(s):

* CSV   ``#col=val&col=val->colA[,colB]``  exactly one row must match every ``col=val``; the stated
                                          values are compared, in order, with ``colA``, ``colB``...
* rows  ``#rows``                         the number of data rows (header excluded) of a CSV file.
* JSON  ``#$.key.key[0].key``             one value (dotted path, integer indices in brackets).
* text  ``:N``                            line N (1-based) of any other file; each stated value must
                                          appear on that line as a number token.

A value computed from sourced values (a ratio, a difference) is tagged as

    <!-- src: derived: EXPR = V -->

where EXPR uses only numbers and ``+ - * / ( )``. Every number in EXPR must itself be a value stated
by a sourced tag in the same block, and EXPR must round to V at V's stated decimals.

A CSV/JSON value matches when the source value rounds to the stated value at the stated number of
decimals (|source - stated| <= half a unit in the last stated decimal). A text value matches when
the same number token appears on the line (thousands separators and the sign character normalised),
or a token in scientific or underscore notation (``5e-5``, ``10_000``) on the line equals it by value.

What is exempt from the coverage rule (and why): numbers inside inline code or HTML comments
(identifiers, paths, the tags themselves); numbers glued to a letter, ``@``, ``_``, ``/`` or ``-``
before them (``T4``, ``precision@10``, ``P1``, ``S3-D``); numbers after ``Figure``, ``Fig.``,
``Table``, ``Box``, ``Section`` or ``§`` (cross-references); bare four-digit integers 1900-2099
(years in citations); heading lines; list-marker numerals at the start of a line (also inside a
blockquote). Numbers spelled
as words ("three seeds") are NOT checked -- a declared limit of this tool.

Usage:
    python scripts/check_number_provenance.py [FILES...]        # default: paper/v2/*.md
    python scripts/check_number_provenance.py --words [FILES...]  # also print word counts
Exit status 0 iff there are no failures.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TAG_RE = re.compile(r"<!--\s*src:\s*(?P<body>.*?)\s*-->", re.S)
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
CODE_RE = re.compile(r"`[^`]*`")
LINK_URL_RE = re.compile(r"\]\([^)]*\)")
# a number token: optional sign, digits with optional thousands separators, optional decimals
NUM_RE = re.compile(r"(?P<sign>[−-])?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")
XREF_RE = re.compile(r"(?:Figure|Fig\.|Table|Box|Section|§|Supplementary Table|Supplementary Figure)\s*S?$")


def normalise(token: str) -> str:
    """Canonical text of a number: ASCII minus, no thousands separators."""
    t = token.replace("−", "-").replace(",", "")
    if t.startswith("+"):
        t = t[1:]
    return t


def decimals(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def prose_numbers(block: str) -> list:
    """Number tokens in a block's visible prose that the coverage rule applies to."""
    text = COMMENT_RE.sub(" ", block)
    text = CODE_RE.sub(" ", text)
    text = LINK_URL_RE.sub("]", text)
    out = []
    for m in NUM_RE.finditer(text):
        before = text[:m.start("num")]
        prev = before[-1:]
        if prev and (prev.isalpha() or prev in "@_/."):
            continue                      # "T4", "precision@10", "P1", "v2.1"
        sign = m.group("sign") or ""
        if sign:
            p2 = text[:m.start()][-1:]
            if p2.isalpha():
                continue                  # "S3-D", "E4-K": part of a name
            if p2.isdigit():
                sign = ""                 # "0.030-0.038": a range, not a negative number
        after = text[m.end():m.end() + 1]
        if after.isalpha() and after != "x":
            continue                      # "3D", "2nd": a name or an ordinal
        if XREF_RE.search(before.rstrip()):
            continue
        num = m.group("num")
        if re.fullmatch(r"(19|20)\d\d", num):
            continue
        out.append(normalise(sign + num))
    return out


def blocks(md: str) -> list:
    """(line_number, block_text) for paragraphs, list items and table rows; headings dropped."""
    out, cur, cur_start = [], [], None

    def flush():
        nonlocal cur, cur_start
        if cur:
            out.append((cur_start, "\n".join(cur)))
        cur, cur_start = [], None

    in_comment = False
    for i, line in enumerate(md.split("\n"), 1):
        line = re.sub(r"^\s*>\s?", "", line)          # a blockquote (a box) is prose like any other
        stripped = line.strip()
        if in_comment:
            cur.append(line)
            if "-->" in line:
                in_comment = False
            continue
        if stripped.startswith("#"):
            flush()
            continue
        if not stripped:
            flush()
            continue
        if stripped.startswith("|"):
            flush()
            if re.fullmatch(r"\|[\s:|-]+\|?", stripped):
                continue
            out.append((i, line))
            continue
        if re.match(r"^\s*([*+-]|\d+\.)\s", line):
            flush()
            line = re.sub(r"^\s*\d+\.\s", " ", line)
        if cur_start is None:
            cur_start = i
        cur.append(line)
        if "<!--" in line and "-->" not in line[line.index("<!--"):]:
            in_comment = True
    flush()
    return out


def parse_tag(body: str):
    if " = " not in body:
        raise ValueError(f"tag has no ' = ': {body!r}")
    ref, values = body.rsplit(" = ", 1)
    ref = ref.strip()
    vals = [v.strip() for v in values.split(",") if v.strip()]
    if "#" in ref:
        path, loc = ref.split("#", 1)
        kind = "json" if loc.startswith("$") else "rows" if loc == "rows" else "csv"
    elif re.search(r":\d+$", ref):
        path, loc = ref.rsplit(":", 1)
        kind = "text"
    else:
        raise ValueError(f"tag has no locator (#... or :LINE): {body!r}")
    return path.strip(), kind, loc, vals


def resolve_path(path: str):
    external = path.startswith("~")
    full = os.path.expanduser(path) if external else os.path.join(ROOT, path)
    return full, external


def csv_values(full: str, loc: str) -> list:
    if "->" not in loc:
        raise ValueError("CSV locator needs ->column")
    cond, cols = loc.rsplit("->", 1)
    conds = [c.split("=", 1) for c in cond.split("&") if c]
    with open(full, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if all(r.get(k) == v for k, v in conds)]
    if len(rows) != 1:
        raise ValueError(f"{len(rows)} rows match {cond!r} (need exactly 1)")
    out = []
    for c in cols.split(","):
        if c not in rows[0]:
            raise ValueError(f"no column {c!r}")
        out.append(rows[0][c])
    return out


def json_value(full: str, loc: str):
    with open(full) as fh:
        obj = json.load(fh)
    for part in re.findall(r"\.([^.\[\]]+)|\[(\d+)\]", loc[1:]):
        key, idx = part
        obj = obj[int(idx)] if idx else obj[key]
    return obj


def numeric_match(source, stated: str) -> bool:
    try:
        x = float(str(source))
    except ValueError:
        # a formatted cell such as "0.367 ± 0.027": compare with its leading number only
        m = re.match(r"\s*([−-]?\d+(?:\.\d+)?)\s*±", str(source))
        if not m:
            return False
        x = float(normalise(m.group(1)))
    v = float(normalise(stated))
    return abs(x - v) <= 0.5 * 10 ** (-decimals(normalise(stated))) + 1e-12


def check_tag(body: str) -> tuple:
    """(values stated, problems, external?)"""
    try:
        path, kind, loc, vals = parse_tag(body)
    except ValueError as exc:
        return [], [str(exc)], False
    full, external = resolve_path(path)
    if not os.path.isfile(full):
        return vals, [f"file not found: {path}"], external
    problems = []
    try:
        if kind == "csv":
            src = csv_values(full, loc)
            if len(src) != len(vals):
                problems.append(f"{len(vals)} values stated for {len(src)} column(s)")
            for s, v in zip(src, vals):
                if not numeric_match(s, v):
                    problems.append(f"stated {v} but {path} has {s}")
        elif kind == "rows":
            with open(full, newline="") as fh:
                n = sum(1 for _ in csv.reader(fh)) - 1
            if len(vals) != 1 or not numeric_match(n, vals[0]):
                problems.append(f"stated {vals} but {path} has {n} data rows")
        elif kind == "json":
            if len(vals) != 1:
                problems.append("a JSON tag states exactly one value")
            elif not numeric_match(json_value(full, loc), vals[0]):
                problems.append(f"stated {vals[0]} but {path}{loc} = {json_value(full, loc)}")
        else:
            n = int(loc)
            with open(full) as fh:
                lines = fh.read().split("\n")
            if not 1 <= n <= len(lines):
                problems.append(f"{path} has no line {n}")
            else:
                on_line = {normalise(m.group(0)) for m in NUM_RE.finditer(lines[n - 1])}
                on_line |= {x.lstrip("-") for x in on_line}
                # scientific notation in code ("lr=5e-5", "10_000") is compared by value
                sci = [float(t.replace("_", "")) for t in
                       re.findall(r"\d[\d_]*(?:\.\d+)?e[-+]?\d+|\d{1,3}(?:_\d{3})+", lines[n - 1])]
                for v in vals:
                    if normalise(v) not in on_line and normalise(v).lstrip("-") not in on_line \
                            and not any(numeric_match(x, v) for x in sci):
                        problems.append(f"{v} not on {path}:{n}")
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        problems.append(f"{path}: {exc}")
    return vals, problems, external


def check_derived(expr: str, value: str, stated: set) -> list:
    if not re.fullmatch(r"[\d.\s+*/()−-]+", expr):
        return [f"derived expression has characters other than numbers and + - * / ( ): {expr!r}"]
    problems = [f"derived operand {n} is not a sourced value in this block"
                for n in re.findall(r"\d+(?:\.\d+)?", expr) if n not in stated]
    try:
        result = eval(expr.replace("−", "-"), {"__builtins__": {}}, {})
    except (SyntaxError, ZeroDivisionError) as exc:
        return problems + [f"derived expression does not evaluate: {exc}"]
    if not numeric_match(result, value):
        problems.append(f"derived {expr} = {result:.6g}, stated {value}")
    return problems


def check_file(path: str) -> dict:
    with open(path) as fh:
        md = fh.read()
    failures, external, n_tags, n_numbers = [], [], 0, 0
    for line_no, block in blocks(md):
        stated, derived = set(), []
        for m in TAG_RE.finditer(block):
            n_tags += 1
            body = m.group("body")
            if body.startswith("derived:"):
                expr, _, value = body[len("derived:"):].rpartition(" = ")
                derived.append((expr.strip(), value.strip()))
                stated.add(normalise(value.strip()))
                continue
            vals, problems, ext = check_tag(body)
            stated |= {normalise(v) for v in vals}
            stated |= {normalise(v).lstrip("-") for v in vals}
            if ext:
                external.append(f"{path}:{line_no}: {m.group('body')[:90]}")
            failures += [f"{path}:{line_no}: bad tag: {p}" for p in problems]
        for expr, value in derived:
            failures += [f"{path}:{line_no}: bad derived tag: {p}" for p in check_derived(expr, value, stated)]
        for num in prose_numbers(block):
            n_numbers += 1
            if num not in stated and num.lstrip("-") not in stated:
                failures.append(f"{path}:{line_no}: untagged number {num!r}")
    return {"failures": failures, "external": external, "tags": n_tags, "numbers": n_numbers}


def word_count(md: str) -> int:
    text = COMMENT_RE.sub(" ", md)
    text = "\n".join(l for l in text.split("\n") if not l.strip().startswith(("#", "|")))
    return len(re.findall(r"[A-Za-z0-9][\w'’.,%×−-]*", text))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--words", action="store_true", help="print word counts (prose; headings, tables, comments excluded)")
    args = ap.parse_args(argv)
    files = args.files or sorted(glob.glob(os.path.join(ROOT, "paper", "v2", "*.md")))
    total_fail, total_tags, total_nums = 0, 0, 0
    for f in files:
        r = check_file(f)
        total_fail += len(r["failures"])
        total_tags += r["tags"]
        total_nums += r["numbers"]
        rel = os.path.relpath(f, ROOT)
        print(f"{rel}: {r['numbers']} numbers, {r['tags']} tags, {len(r['failures'])} failures, "
              f"{len(r['external'])} external tags"
              + (f", {word_count(open(f).read())} words" if args.words else ""))
        for line in r["failures"]:
            print("  FAIL " + os.path.relpath(line, ROOT) if line.startswith(ROOT) else "  FAIL " + line)
        for line in r["external"]:
            print("  EXTERNAL " + line)
    print(f"TOTAL: {total_nums} numbers, {total_tags} tags, {total_fail} failures")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
