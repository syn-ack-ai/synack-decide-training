"""Compare two Decision Index result dirs from different engines on the same rows: answer agreement, probability
differences, per-request latency and total wall time.
  compare_engines.py runs/eval-v3/results/lb-sample-v5 runs/eval-v3/results/lb-sample-v5-vllm"""
import json
import sys
from datetime import datetime


def load(d):
    return {r["run_id"]: r for r in map(json.loads, open(f"{d}/results.jsonl")) if r.get("status") == "ok"}


def wall(rows):
    t = [datetime.fromisoformat(r[k]) for r in rows.values() for k in ("started_utc", "completed_utc")]
    return (max(t) - min(t)).total_seconds()


def pct(xs, q):
    xs = sorted(xs)
    return xs[int(q * (len(xs) - 1))]


A, B = load(sys.argv[1]), load(sys.argv[2])
ids = [i for i in A if i in B]
n_q = same = 0
dps = []
for i in ids:
    for k, a in A[i]["response"]["answers"].items():
        b = B[i]["response"]["answers"].get(k)
        if b is None:
            continue
        n_q += 1
        if a["type"] == "noul":
            same += (a["noul"] >= .5) == (b["noul"] >= .5)
            dps.append(abs(a["noul"] - b["noul"]))
        else:
            same += a["choice"] == b["choice"]
            dps.append(max(abs(a["probabilities"][c] - b["probabilities"].get(c, 0)) for c in a["probabilities"]))
la = [A[i]["model_request_wall_ms"] for i in ids]
lb = [B[i]["model_request_wall_ms"] for i in ids]
print(f"{len(ids)} shared rows ({len(A)} / {len(B)} ok), {n_q} questions: same answer {same}/{n_q} ({100 * same / n_q:.1f}%); "
      f"max|dp| median {pct(dps, .5):.4f}, p90 {pct(dps, .9):.4f}, max {max(dps):.3f}")
print(f"per-request ms  median {pct(la, .5):.0f} -> {pct(lb, .5):.0f};  p90 {pct(la, .9):.0f} -> {pct(lb, .9):.0f};  "
      f"sum {sum(la) / 1000 / 60:.1f} -> {sum(lb) / 1000 / 60:.1f} min ({sum(la) / sum(lb):.1f}x)")
print(f"run wall time {wall(A) / 60:.1f} -> {wall(B) / 60:.1f} min")
