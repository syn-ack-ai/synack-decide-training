"""Select the v2 training mix (per-source quotas, class-balanced) and a disjoint validation set.

Writes mix_train_records.jsonl / mix_valid_records.jsonl (decision records). Quotas follow the
Decision Index area weights (Knowledge .26, Language .26, Retrieval .20, Tools .18, Arts .10)
plus a tev1 share for the tev1 1,300-record goal.
"""
import argparse
import json
import random
from collections import defaultdict

QUOTAS = {  # source prefix -> questions
    "tev1/": 9000,
    "hf/gsm8k": 2000, "hf/supergpqa": 1500, "hf/mmlu_aux": 1000, "hf/chess": 1500, "hf/corr2cause": 800,
    "synthetic/": 3700,
    "hf/anli": 2000, "hf/winogrande": 1500, "hf/hellaswag": 1500, "hf/nli4ct": 600, "hf/ragtruth": 1400,
    "hf/banking77": 1200, "hf/clinc": 2000, "hf/esci": 1800,
    "hf/when2call": 924,
    "hf/newyorker": 2000,
}
VALID_PER_SOURCE = 60


def bucket(src):
    return next((p for p in QUOTAS if src.startswith(p)), None)


def balanced(rows, n, rng):
    """Round-robin over (sub-source, gold) so no class or sub-family dominates."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["source"], json.dumps(list(r["expected"].values()), default=str))].append(r)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups)
    rng.shuffle(keys)
    out = []
    while len(out) < n and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out-prefix", default="data/mix")
    a = p.parse_args()
    rng = random.Random(20261002)
    by = defaultdict(list)
    for path in a.inputs:
        for line in open(path):
            r = json.loads(line)
            b = bucket(r["source"])
            if b:
                by[b].append(r)
    train, valid = [], []
    for b, n in QUOTAS.items():
        rows = by.get(b, [])
        rng.shuffle(rows)
        v, rest = rows[:VALID_PER_SOURCE], rows[VALID_PER_SOURCE:]
        t = balanced(rest, n, rng)
        valid += v
        train += t
        print(f"{b:18s} available {len(rows):6d}  train {len(t):5d}  valid {len(v)}")
    rng.shuffle(train)
    for name, rows in (("train", train), ("valid", valid)):
        with open(f"{a.out_prefix}_{name}_records.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows)
    print(f"train {len(train)} records, valid {len(valid)}")


if __name__ == "__main__":
    main()
