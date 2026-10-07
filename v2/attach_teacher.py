"""Combine teacher labels per question and attach them to expanded training examples.

Teacher files hold {"qid", "probs": {option_key: p}}. Per question, the listed teachers' key
distributions are averaged (source-specific overrides allowed); questions with no API label fall
back to --fallback files. Each shuffled copy gets the averaged distribution mapped to its labels.
Usage: python attach_teacher.py --records mix_train_records.jsonl --out mix_train.jsonl \
         --teacher data/teacher_api_gemma31.jsonl [--teacher data/teacher_api_kimi.jsonl] \
         --override hf/corr2cause=data/teacher_api_dsv4.jsonl --override hf/supergpqa=data/teacher_api_dsv4.jsonl \
         --fallback data/teacher_mix.jsonl
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import expand  # noqa: E402


def load(path):
    out = {}
    for line in open(path):
        x = json.loads(line)
        out[x["qid"]] = x["probs"]
    return out


def average(dists):
    keys = set().union(*dists)
    avg = {k: sum(d.get(k, 0.0) for d in dists) / len(dists) for k in keys}
    z = sum(avg.values()) or 1.0
    return {k: v / z for k, v in avg.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--teacher", action="append", default=[])
    p.add_argument("--override", action="append", default=[], help="source_prefix=teacher.jsonl (added to the average)")
    p.add_argument("--fallback", action="append", default=[])
    a = p.parse_args()
    teachers = [load(t) for t in a.teacher]
    overrides = [(s, load(f)) for s, f in (o.split("=", 1) for o in a.override)]
    fallbacks = [load(f) for f in a.fallback]
    stats, agree = Counter(), Counter()
    with open(a.records) as f, open(a.out, "w") as g:
        for line in f:
            r = json.loads(line)
            per_q = {}
            for qkey in r["questions"]:
                qid = f"{r['id']}#{qkey}"
                ds = [t[qid] for t in teachers if qid in t]
                ds += [t[qid] for s, t in overrides if r["source"].startswith(s) and qid in t]
                if not ds:
                    ds = [t[qid] for t in fallbacks if qid in t][:1]
                    stats["fallback" if ds else "none"] += 1
                else:
                    stats[f"{len(ds)}_teachers"] += 1
                if ds:
                    per_q[qkey] = average(ds)
            for ex in expand(r):
                qkey = ex["id"].split("#")[1]
                dist = per_q.get(qkey)
                if dist:
                    order = ex["key_order"]  # option keys in this copy's label order
                    ex["teacher"] = {lab: dist.get(k, 0.0) for lab, k in zip(ex["labels"], order)}
                    t_best = max(ex["teacher"], key=ex["teacher"].get)
                    agree["agree" if t_best == ex["gold"] else "disagree"] += 1
                ex.pop("key_order", None)
                g.write(json.dumps(ex, ensure_ascii=False) + "\n")
                stats["examples"] += 1
    print(json.dumps({**stats, **agree}))


if __name__ == "__main__":
    main()
