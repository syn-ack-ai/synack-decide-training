"""MLX twin of bench_tev1_v2.py: a v2 model on tev1's 1,300 records (test + policy_transfer)."""
import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--name", required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    from engine_mlx import label_probs, load_mlx
    tok, model, label_ids = load_mlx(a.model, a.adapter)
    rec = ROOT / "tev1/data/v1/records"
    jobs = [(s, json.loads(l)) for s in ("test", "policy_transfer") for l in (rec / f"{s}.jsonl").read_text().splitlines()]
    random.Random(42).shuffle(jobs)
    jobs = jobs[:a.limit] if a.limit else jobs
    out = ROOT / "bench/results" / a.name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (split, r) in enumerate(jobs):
        crit = {o["key"]: o["description"] for o in r["options"]}
        q = {"type": "choice", "instructions": r["question"], "criteria": crit}
        t = time.perf_counter()
        probs, _ = label_probs(tok, model, label_ids, r["state"], q)
        ms = (time.perf_counter() - t) * 1000
        keys = [o["key"] for o in r["options"]]
        best = max(probs, key=probs.get)
        pred = LABELS[keys.index(best)]
        rows.append({"split": split, "id": r["id"], "source": r["source"], "gold": r["answer"], "pred": pred,
                     "correct": pred == r["answer"], "confidence": probs[best], "latency_ms": ms, "warmup": i == 0})
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(jobs)} acc={sum(x['correct'] for x in rows) / len(rows):.3f}", flush=True)
    (out / "results.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    lat = sorted(x["latency_ms"] for x in rows if not x["warmup"])
    by = defaultdict(list)
    for x in rows:
        by[f"{x['split']}/{x['source']}"].append(x["correct"])
    rep = {"model": a.model, "adapter": a.adapter, "correct": sum(x["correct"] for x in rows), "n": len(rows),
           "accuracy": round(sum(x["correct"] for x in rows) / len(rows), 4),
           "median_ms": round(statistics.median(lat), 1), "p95_ms": round(lat[int(.95 * len(lat))], 1),
           "by_source": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by.items())}}
    (out / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
