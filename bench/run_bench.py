"""Benchmark a local model on tev1's 1,300 v1 development records (test + policy_transfer)."""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from decide_mlx import Decider

RECORDS = Path(__file__).resolve().parents[1] / "tev1/data/v1/records"


def summary(rows):
    lat = sorted(r["latency_ms"] for r in rows)
    return {"n": len(rows), "correct": sum(r["correct"] for r in rows),
            "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
            "unconstrained_valid": sum(r["unconstrained_valid"] for r in rows),
            "median_ms": round(statistics.median(lat), 1),
            "p95_ms": round(lat[min(len(lat) - 1, int(.95 * len(lat)))], 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--adapter")
    p.add_argument("--name", required=True)
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    jobs = [(split, json.loads(line)) for split in ("test", "policy_transfer")
            for line in (RECORDS / f"{split}.jsonl").read_text().splitlines()]
    random.Random(42).shuffle(jobs)
    jobs = jobs[:args.limit] if args.limit else jobs

    d = Decider(args.model, args.adapter)
    d.decide(jobs[0][1])  # warm-up, excluded
    out_dir = Path(__file__).resolve().parent / "results" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with (out_dir / "results.jsonl").open("w") as f:
        for i, (split, rec) in enumerate(jobs, 1):
            r = d.decide(rec)
            row = {"split": split, "id": rec["id"], "source": rec["source"], "gold": rec["answer"],
                   "gold_key": rec["answer_key"], "correct": r["label"] == rec["answer"], **r}
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} acc={sum(x['correct'] for x in rows) / i:.3f}", flush=True)

    report = {"model": args.model, "adapter": args.adapter, "all": summary(rows)}
    for split in ("test", "policy_transfer"):
        rr = [r for r in rows if r["split"] == split]
        if rr:
            by = defaultdict(list)
            for r in rr:
                by[r["source"]].append(r)
            report[split] = {**summary(rr), "by_source": {s: summary(v) for s, v in sorted(by.items())}}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["all"]))


if __name__ == "__main__":
    main()
