"""Full option probabilities for every question of decision records (CUDA or MLX). Writes one line per record:
{"id", "source", "expected", "probs": {qkey: {option: p}}}.

  predict_records.py --backend cuda --model SkyPanther/synack-decide-26b-a4b [--adapter A] --records r.jsonl --out p.jsonl
  predict_records.py --backend mlx --model models/systemone-v3-mlx8 --records r.jsonl --out p.jsonl
"""
import argparse
import json
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["cuda", "mlx"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--records", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.backend == "cuda":
        from engine_v2 import V2Engine
        eng = V2Engine(model=a.model, adapter=a.adapter)
        probs = eng.probs
    else:
        from engine_mlx import label_probs, load_mlx
        tok, model, ids = load_mlx(a.model, a.adapter)
        probs = lambda state, q: label_probs(tok, model, ids, state, q)  # noqa: E731
    rows = [json.loads(l) for l in open(a.records)]
    t0 = time.time()
    with open(a.out, "w") as f:
        for i, r in enumerate(rows, 1):
            out = {qk: probs(r["state"], q)[0] for qk, q in r["questions"].items()}
            f.write(json.dumps({"id": r["id"], "source": r.get("source"), "expected": r.get("expected"), "probs": out}) + "\n")
            if i % 200 == 0:
                print(f"{i}/{len(rows)} {i / (time.time() - t0):.1f}/s", flush=True)
    print(f"done {len(rows)} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
