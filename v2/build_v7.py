"""v7 training mix: ONE fresh LoRA on google/gemma-4-26B-A4B-it with everything that proved out (docs/V6_PLAN.md).

  base mix    all v3 examples: the v3 records (HF train splits, Tev1 data, code-verified synthetic tasks) re-expanded exactly as in v3,
              with teacher targets 0.8 Kimi-K3 + 0.2 Gemma-4-31B where both exist (else the one that exists)
  PRs         all 13,064 labelled public training PRs (one copy each, as in v3)
  additions   v4 ordinal questions, v5 PR author claims at the natural 28% rate, v5's added train-split sets
              (board/*, financial sentiment, SmartBugs); optionally v6's pool additions (--include-v6)
Rows without teacher labels are marked "replay": the HF job fills their teacher with the best stacked model's own
probabilities (v5 unless v6 proved better), so v7 inherits what v3-v6 learned without their accumulated drift.
Training applies a teacher only where it agrees with gold.

  python v2/build_v7.py [--include-v6]   -> v2/data/v7/{train.jsonl, valid.jsonl}
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v2"))
from build_v6 import examples, jsonl, mix, record_id, write  # noqa: E402

TL = ROOT / "datasets/systemone-teacher-labels-v1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "v2/data/v7"))
    p.add_argument("--seed", type=int, default=20261007)
    p.add_argument("--include-v6", action="store_true", help="add v6's pool rows (CLINC, ESCI, CRUXEval, SuperGPQA, MMLU-aux)")
    a = p.parse_args()
    rng = random.Random(a.seed)

    kimi = {x["qid"]: x["probs"] for x in jsonl(TL / "labels/kimi-k3/train.jsonl")}
    gemma = {x["qid"]: x["probs"] for x in jsonl(TL / "labels/gemma-4-31b-it/train.jsonl")}
    tq = {q: mix(kimi.get(q), gemma.get(q)) for q in set(kimi) | set(gemma)}

    base = []
    for r in jsonl(TL / "records/train.jsonl"):
        for ex in examples([r], copies=None, teacher=tq):
            base.append(ex)
    prs = examples(jsonl(ROOT / "prcx/data/dataset/train_records.jsonl"), copies=1)
    have = {e["id"] for e in base} | {e["id"] for e in prs}
    v3_extra = [e for e in jsonl(ROOT / "v2/data/v3/train.jsonl") if e["id"] not in have]  # v3's gold-only MNLI/BoolQ rows

    v4 = [e for e in jsonl(ROOT / "v2/data/v4/train.jsonl") if e["source"] == "synthetic/ordinal"]
    v5 = [e for e in jsonl(ROOT / "v2/data/v5/train.jsonl") if not e.get("replay")]  # v5's new rows only
    adds = v4 + v5
    if a.include_v6:
        v6 = [e for e in jsonl(ROOT / "v2/data/v6/train.jsonl")
              if not e.get("replay") and not e.get("teacher") and not e["source"].startswith("prcx/")]
        adds += v6

    train = base + prs + v3_extra + adds
    for e in train:
        e.pop("replay", None)
        if not e.get("teacher"):
            e["replay"] = True  # the job fills these with the stacked model's probabilities
            e.pop("teacher", None)
    seen, dedup = set(), []
    for e in train:
        if e["id"] not in seen:
            seen.add(e["id"])
            dedup.append(e)
    rng.shuffle(dedup)
    out = Path(a.out)
    write(out / "train.jsonl", dedup)
    valid = jsonl(ROOT / "v2/data/v5/valid.jsonl")
    write(out / "valid.jsonl", valid)
    n_t = sum(1 for e in dedup if e.get("teacher"))
    print(f"train {len(dedup)} ({len(train) - len(dedup)} duplicate ids dropped): teacher 0.8/0.2 {n_t}, "
          f"model-teacher (replay) {len(dedup) - n_t}; valid {len(valid)}")
    for k, v in collections.Counter(e["source"].split("/")[0] for e in dedup).most_common():
        print(f"  {v:6d} {k}")


if __name__ == "__main__":
    main()
