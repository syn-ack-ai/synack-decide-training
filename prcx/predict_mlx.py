"""P(needs_review = yes) for PR decision records with an MLX model (+ optional adapter).

Same prompt and readout as training (v2/common.py). Writes {"id", "p_yes"} per record.
Usage: python predict_mlx.py --model ../models/gemma4-26b-a4b-qat-mlx4 --adapter ../v2/adapters/run1-moe/best \
         --records data/dataset/test_records.jsonl --out preds/moe_zeroshot.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v2"))
from engine_mlx import label_probs, load_mlx  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--records", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    tok, model, ids = load_mlx(a.model, a.adapter)
    rows = [json.loads(l) for l in open(a.records)][:a.limit]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(a.out, "w") as f:
        for i, r in enumerate(rows, 1):
            q = r["questions"]["needs_review"]
            probs, _ = label_probs(tok, model, ids, r["state"], q)
            f.write(json.dumps({"id": r["id"], "p_yes": probs["yes"]}) + "\n")
            if i % 200 == 0:
                print(f"{i}/{len(rows)} {i / (time.time() - t0):.2f}/s", flush=True)
    print(f"done {len(rows)} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
