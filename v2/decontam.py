"""Drop training records that overlap the Decision Index suite (rows we train on score as wrong).

Two checks against every suite row (state + instructions + option descriptions):
  1. exact match of any normalized text segment of >= 5 words (segments split on . ? ! : ; newlines),
     which catches short items such as intent queries reused across splits;
  2. >= 20% of a record's 13-word shingles appearing in the suite (GPT-3-style n-gram check).

Usage: python decontam.py --suite DIR --in records.jsonl --out clean.jsonl [--report r.json]
       python decontam.py --suite DIR --build-index idx.npz   (cache the suite index)
"""
import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

SPLIT = re.compile(r"[.?!:;\n]+")
WORD = re.compile(r"[a-z0-9]+")
N = 13


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little", signed=True)


def leaves(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from leaves(v)
    elif isinstance(x, list):
        for v in x:
            yield from leaves(v)


def record_text(r):
    parts = list(leaves(r.get("state")))
    for q in r["questions"].values():
        parts += list(leaves(q.get("instructions"))) + list(leaves(q.get("criteria", {})))
    return parts


def segments(parts):
    for p in parts:
        for seg in SPLIT.split(p.lower()):
            w = WORD.findall(seg)
            if len(w) >= 5:
                yield h64(" ".join(w))


def shingles(parts):
    w = WORD.findall(" ".join(parts).lower())
    return [h64(" ".join(w[i:i + N])) for i in range(len(w) - N + 1)]


BOILERPLATE_ROWS = 20  # a segment/shingle in more suite rows than this is template text, not test content


def build_index(suite_dir):
    seg_df, sh_df = Counter(), Counter()
    for name in ("selected-rows.jsonl.gz", "added-rows.jsonl.gz"):
        for line in gzip.open(Path(suite_dir, name), "rt"):
            parts = record_text(json.loads(line))
            seg_df.update(set(segments(parts)))
            sh_df.update(set(shingles(parts)))
    keep = lambda df: np.array(sorted(k for k, n in df.items() if n <= BOILERPLATE_ROWS), dtype=np.int64)
    print(f"boilerplate excluded: {sum(n > BOILERPLATE_ROWS for n in seg_df.values())} segments, "
          f"{sum(n > BOILERPLATE_ROWS for n in sh_df.values())} shingles")
    return keep(seg_df), keep(sh_df)


def member(x, sorted_arr):
    """Vectorized membership test against an already-sorted index (binary search)."""
    i = np.searchsorted(sorted_arr, x)
    i[i == sorted_arr.size] = 0
    return sorted_arr[i] == x


def contaminated(r, segs, shs):
    parts = record_text(r)
    s = np.fromiter(segments(parts), dtype=np.int64)
    if s.size and member(s, segs).any():
        return "segment"
    sh = np.array(shingles(parts), dtype=np.int64)
    if sh.size and member(sh, shs).mean() >= 0.2:
        return "ngram"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True)
    p.add_argument("--index", help="cached .npz index")
    p.add_argument("--build-index")
    p.add_argument("--in", dest="inp")
    p.add_argument("--out")
    p.add_argument("--report")
    p.add_argument("--corpus-df", type=int, default=0,
                   help="two passes: a suite segment that also matches in at least N records of THIS input file is domain "
                        "boilerplate (license headers, traceback lines), not test content, and is ignored (0 = off)")
    p.add_argument("--allow", action="append", default=[],
                   help="generic phrase (>= 5 words) to ignore as a segment match, e.g. an OS error string; repeatable")
    a = p.parse_args()
    if a.build_index:
        segs, shs = build_index(a.suite)
        np.savez(a.build_index, segs=segs, shs=shs)
        print(f"index: {segs.size} segments, {shs.size} shingles")
        return
    if a.index and Path(a.index).exists():
        z = np.load(a.index)
        segs, shs = z["segs"], z["shs"]
    else:
        segs, shs = build_index(a.suite)
    if a.allow:
        drop = np.array([h64(" ".join(WORD.findall(x.lower()))) for x in a.allow], dtype=np.int64)
        segs = segs[~np.isin(segs, drop)]
        print(f"allow-listed {len(a.allow)} generic phrases", flush=True)
    if a.corpus_df:
        df = Counter()
        with open(a.inp) as f:
            for line in f:
                s_ = np.unique(np.fromiter(segments(record_text(json.loads(line))), dtype=np.int64))
                if s_.size:
                    df.update(s_[member(s_, segs)].tolist())
        common = np.array([h for h, n in df.items() if n >= a.corpus_df], dtype=np.int64)
        segs = segs[~np.isin(segs, common)]
        print(f"corpus boilerplate: {common.size} suite segments match in >= {a.corpus_df} input records; ignored "
              f"({len(df) - common.size} rarer matching segments still block)", flush=True)
    kept, dropped = 0, Counter()
    with open(a.inp) as f, open(a.out, "w") as g:
        for line in f:
            r = json.loads(line)
            why = contaminated(r, segs, shs)
            if why:
                dropped[(r["source"], why)] += 1
            else:
                g.write(line)
                kept += 1
    rep = {"kept": kept, "dropped": sum(dropped.values()),
           "by_source": {f"{s}:{w}": n for (s, w), n in sorted(dropped.items())}}
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
