"""Our v2 model on tev1's 1,300 records (test + policy_transfer), comparable to bench/run_bench_cuda.py.

Each tev1 record is converted to a v2 question (same state, question, options in their original
order and labels), so the only difference from Tev1's run is the model and its prompt wording.
"""
import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

from engine_v2 import load_v2
from common import LABELS, SYSTEM, decision_user

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models/gemma4-12b-it"))
    p.add_argument("--adapter")
    p.add_argument("--four-bit", action="store_true")
    p.add_argument("--name", required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    rec_dir = ROOT / "tev1/data/v1/records"
    jobs = [(s, json.loads(l)) for s in ("test", "policy_transfer") for l in (rec_dir / f"{s}.jsonl").read_text().splitlines()]
    random.Random(42).shuffle(jobs)
    jobs = jobs[:a.limit] if a.limit else jobs
    tok, model, label_ids = load_v2(a.model, a.adapter, a.four_bit)
    rows = []
    out = ROOT / "bench/results" / a.name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w") as f:
        for i, (split, r) in enumerate(jobs):
            keys = [o["key"] for o in r["options"]]
            crit = {o["key"]: o["description"] for o in r["options"]}
            q = {"type": "choice", "instructions": r["question"], "criteria": crit}
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": decision_user(r["state"], q, keys, crit)}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=False)
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.inference_mode():
                logits = model(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1).logits[0, -1]
                pr = torch.softmax(logits[label_ids[:len(keys)]].float(), -1)
                free = int(logits.argmax())
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 1000
            best = int(pr.argmax())
            row = {"split": split, "id": r["id"], "source": r["source"], "gold": r["answer"],
                   "pred": LABELS[best], "correct": LABELS[best] == r["answer"], "confidence": float(pr[best]),
                   "unconstrained_valid": tok.decode([free]).strip() in LABELS[:len(keys)], "latency_ms": ms}
            if i > 0:  # first call is warm-up
                rows.append(row)
            f.write(json.dumps(row) + "\n")
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(jobs)} acc={sum(x['correct'] for x in rows) / max(len(rows), 1):.3f}", flush=True)
    # score all 1,300 (warm-up row scored too, its latency excluded)
    allrows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    lat = sorted(r["latency_ms"] for r in rows)
    by = defaultdict(list)
    for r in allrows:
        by[(r["split"], r["source"])].append(r["correct"])
    rep = {"model": a.model, "adapter": a.adapter, "four_bit": a.four_bit, "device": torch.cuda.get_device_name(0),
           "correct": sum(r["correct"] for r in allrows), "n": len(allrows),
           "accuracy": round(sum(r["correct"] for r in allrows) / len(allrows), 4),
           "unconstrained_valid": sum(r["unconstrained_valid"] for r in allrows),
           "median_ms": round(statistics.median(lat), 1), "p95_ms": round(lat[int(.95 * len(lat))], 1),
           "by_source": {f"{s}/{src}": f"{sum(v)}/{len(v)}" for (s, src), v in sorted(by.items())}}
    (out / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
