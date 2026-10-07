"""Paired per-benchmark regression check between two runs on the same leaderboard-sample rows.

For each benchmark: questions the old model got right and the new one wrong (losses) vs the reverse (wins), with an
exact two-sided McNemar p-value. A benchmark FAILS the gate if the new model has more losses than wins with p < 0.2
on benchmarks where the old model was leading the board ("protected").

  regression_check.py runs/eval-v3/results/lb-sample-v4 runs/eval-v3/results/lb-sample-v5
"""
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = {"HoVer", "ChessBench", "BFCL", "RAGTruth", "ToolRet", "NLI4CT", "New Yorker", "When2Call", "MuSR", "Humicroedit",
             "GSM8K", "BANKING77", "FinEntity", "SATA-Bench", "HellaSwag", "ANLI", "ContractNLI", "BPoMP", "BRIGHT", "CLINC150"}


def load(run):
    out = {}
    for l in open(Path(run) / "results.jsonl"):
        r = json.loads(l)
        out[r["run_id"]] = r
    return out


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    old, new = load(sys.argv[1]), load(sys.argv[2])
    rows = {}
    for l in gzip.open(ROOT / "runs/suite-sample3000/selected-rows.jsonl.gz", "rt"):
        r = json.loads(l)
        rows[r["_evaluation"]["run_id"]] = r
    board = json.load(open(ROOT / "runs/board/index.json"))
    short = {v["dataset"] if "dataset" in v else k: v["short"] for k, v in board["benchmarks"].items()}
    stats = defaultdict(lambda: [0, 0, 0])  # losses, wins, both
    for sha, o in old.items():
        n = new.get(sha)
        if not n or o.get("status") != "ok" or n.get("status") != "ok":
            continue
        exp = (rows.get(sha) or {}).get("expected") or {}
        oa = (o.get("response") or {}).get("answers", {})
        na = (n.get("response") or {}).get("answers", {})
        name = o["dataset"]
        for q, gold in exp.items():
            if q not in oa or q not in na:
                continue
            def right(a):
                if a.get("type") == "noul":
                    return (a["noul"] >= 0.5) == (gold in (True, "true", "yes"))
                return a.get("choice") == gold
            ro, rn = right(oa[q]), right(na[q])
            if ro and not rn:
                stats[name][0] += 1
            elif rn and not ro:
                stats[name][1] += 1
            elif ro and rn:
                stats[name][2] += 1
    fails = []
    print(f"{'benchmark':40s} {'loss':>5} {'win':>5} {'net':>5} {'p':>6}")
    for name, (b, c, both) in sorted(stats.items(), key=lambda kv: kv[1][1] - kv[1][0]):
        p = mcnemar(b, c)
        prot = any(name.startswith(x) or x.startswith(name.split()[0]) for x in PROTECTED)
        bad = prot and b > c and p < 0.2
        warn = prot and not bad and b - c >= 5
        if bad:
            fails.append(name)
        print(f"{name[:40]:40s} {b:5d} {c:5d} {c - b:+5d} {p:6.2f} {'PROTECTED' if prot else ''} {'<-- FAIL' if bad else '<-- warn' if warn else ''}")
    print("\nGATE:", "PASS" if not fails else f"FAIL on {fails}")


if __name__ == "__main__":
    main()
