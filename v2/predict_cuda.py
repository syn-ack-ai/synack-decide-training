"""P(option) for single-question decision records with a CUDA model (+ PEFT adapter). Writes {"id", "p_<key>"}.

Usage: python predict_cuda.py --model google/gemma-4-26B-A4B-it --adapter adapters/best \
         --records test_records.jsonl --qkey needs_review --key yes --out preds.jsonl
"""
import argparse
import json
import time

import torch

from engine_v2 import V2Engine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--records", required=True)
    p.add_argument("--qkey", default="needs_review")
    p.add_argument("--key", default="yes")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    eng = V2Engine(model=a.model, adapter=a.adapter)
    rows = [json.loads(l) for l in open(a.records)]
    t0 = time.time()
    with open(a.out, "w") as f:
        for i, r in enumerate(rows, 1):
            probs, _ = eng.probs(r["state"], r["questions"][a.qkey])
            f.write(json.dumps({"id": r["id"], f"p_{a.key}": probs[a.key]}) + "\n")
            if i % 250 == 0:
                torch.cuda.synchronize()
                print(f"{i}/{len(rows)} {i / (time.time() - t0):.1f}/s", flush=True)
    print(f"done {len(rows)} in {(time.time() - t0) / 60:.1f} min ({(time.time() - t0) / len(rows) * 1000:.0f} ms/record)", flush=True)


if __name__ == "__main__":
    main()
