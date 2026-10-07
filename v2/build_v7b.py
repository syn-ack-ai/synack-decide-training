"""v7b: the v7 mix (fresh LoRA on the Gemma base) plus the fixes for v7's losses, without benchmaxing (docs/V6_PLAN.md).

  start       v7's train_sd.jsonl from SkyPanther/systemone-train-v7 (teachers already filled: Kimi/Gemma or v5)
  injection   v4's balanced 50/50 PR author-claim rows (the v4 fix; v7 only had the natural-rate set)
  reasoning   our own generators, unused so far: web-of-lies, navigate, ordering, tracking, temporal, arithmetic,
              boolean, CRUXEval-style code execution (plus v6's 1,288 synthetic CRUXEval-style rows)
  2nd pass    every hf/* and synthetic/* row appears twice (v7 saw the reasoning core once; v5 several times)
New rows are marked "replay": the job fills their teacher with v5's probabilities (applied only where it agrees with gold).

  python v2/build_v7b.py --v7-sd PATH/train_sd.jsonl   -> v2/data/v7b/{train.jsonl, valid.jsonl}
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v2"))
from build_v6 import examples, jsonl, record_id, write  # noqa: E402

TL = ROOT / "datasets/systemone-teacher-labels-v1"
GENERATORS = ("synthetic/web_of_lies", "synthetic/navigate", "synthetic/ordering", "synthetic/tracking",
              "synthetic/temporal", "synthetic/arithmetic", "synthetic/boolean", "synthetic/cruxeval")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v7-sd", required=True, help="v7's train_sd.jsonl (teachers filled)")
    p.add_argument("--out", default=str(ROOT / "v2/data/v7b"))
    p.add_argument("--seed", type=int, default=20261008)
    a = p.parse_args()
    rng = random.Random(a.seed)

    base = jsonl(a.v7_sd)
    for e in base:
        e.pop("replay", None)
    assert all(e.get("teacher") for e in base), "v7 train_sd rows should all carry a teacher"

    injection = [e for e in jsonl(ROOT / "v2/data/v4/train.jsonl") if e["source"] == "prcx/injected"]
    used = {record_id(e["id"]) for v in ("v3", "v4", "v5", "v6", "v7") for e in jsonl(ROOT / f"v2/data/{v}/train.jsonl")}
    used |= {r["id"] for f in ("records/train.jsonl", "records/valid.jsonl") for r in jsonl(TL / f)}
    gen = examples([r for r in jsonl(TL / "pools/synthetic.jsonl") if r["source"] in GENERATORS and r["id"] not in used])
    crux_v6 = [e for e in jsonl(ROOT / "v2/data/v6/train.jsonl") if e["source"] == "synthetic/cruxeval" and not e.get("replay")]
    new = injection + gen + crux_v6
    for e in new:
        e.pop("teacher", None)
        e["replay"] = True

    first = base + new
    second = []
    for e in first:
        if e["source"].startswith(("hf/", "synthetic/")):
            c = dict(e)
            c["id"] = e["id"] + "~2"
            second.append(c)
    train = first + second
    rng.shuffle(train)
    out = Path(a.out)
    write(out / "train.jsonl", train)
    write(out / "valid.jsonl", jsonl(ROOT / "v2/data/v5/valid.jsonl"))
    print(f"train {len(train)}: v7 base {len(base)}, new {len(new)} (injection {len(injection)}, generators {len(gen)}, "
          f"v6 cruxeval {len(crux_v6)}), second pass {len(second)}; replay rows for v5 targets "
          f"{sum(1 for e in train if e.get('replay'))}")
    for k, v in collections.Counter(e["source"] for e in gen + crux_v6).most_common():
        print(f"  {v:6d} {k}")


if __name__ == "__main__":
    main()
