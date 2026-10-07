"""v6 training mix, stacked on merged v5 (docs/V6_PLAN.md).

  replay      v5's training mix, marked "replay": the HF job overwrites their teachers with v5's own probabilities
              (self-distillation), which anchors what v5 already does well.
  teacher     teacher-labelled records not in the replay, with targets 0.8 Kimi-K3 + 0.2 Gemma-4-31B
              (the best mix on the 1,018-question teacher validation set).
  pr          labelled public PRs not in the replay, gold only (PR AUC drifted 0.780 -> 0.776 over v3-v5).
  new         unused rows of the decontaminated pools for v5's dips and the knowledge gap, gold only:
              CLINC150, Amazon ESCI, synthetic CRUXEval, SuperGPQA, MMLU auxiliary train.

  python v2/build_v6.py   -> v2/data/v6/{train.jsonl, valid.jsonl}
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v2"))
from common import expand  # noqa: E402

TL = ROOT / "datasets/systemone-teacher-labels-v1"
NEW = {"hf/clinc": 2500, "hf/esci": 1500, "synthetic/cruxeval": 1300, "hf/supergpqa": 1500, "hf/mmlu_aux": 1000}


def jsonl(path):
    return [json.loads(l) for l in open(path)]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def record_id(example_id):
    return example_id.rsplit("#", 2)[0]


def mix(kimi, gemma, wk=0.8):
    if not kimi or not gemma:
        return kimi or gemma
    keys = set(kimi) | set(gemma)
    p = {k: wk * kimi.get(k, 0.0) + (1 - wk) * gemma.get(k, 0.0) for k in keys}
    z = sum(p.values()) or 1.0
    return {k: v / z for k, v in p.items()}


def examples(records, copies=1, teacher=None):
    out = []
    for r in records:
        for ex in expand(r, copies=copies):
            order = ex.pop("key_order")
            dist = teacher.get(f"{r['id']}#{ex['id'].split('#')[-2]}") if teacher else None
            if dist:
                ex["teacher"] = {lab: dist.get(k, 0.0) for lab, k in zip(ex["labels"], order)}
            out.append(ex)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "v2/data/v6"))
    p.add_argument("--seed", type=int, default=20261006)
    p.add_argument("--replay", type=int, default=15000)
    p.add_argument("--teacher-records", type=int, default=5000)
    p.add_argument("--pr", type=int, default=3000)
    a = p.parse_args()
    rng = random.Random(a.seed)

    v5 = jsonl(ROOT / "v2/data/v5/train.jsonl")
    rng.shuffle(v5)
    replay = v5[:a.replay]
    for e in replay:
        e["replay"] = True
        e.pop("teacher", None)  # replaced by v5's own probabilities in the job
    in_replay = {record_id(e["id"]) for e in replay}

    kimi = {x["qid"]: x["probs"] for x in jsonl(TL / "labels/kimi-k3/train.jsonl")}
    gemma = {x["qid"]: x["probs"] for x in jsonl(TL / "labels/gemma-4-31b-it/train.jsonl")}
    tq = {q: mix(kimi.get(q), gemma.get(q)) for q in set(kimi) & set(gemma)}
    trec = [r for r in jsonl(TL / "records/train.jsonl") if r["id"] not in in_replay
            and all(f"{r['id']}#{k}" in tq for k in r["questions"])]
    rng.shuffle(trec)
    teacher_ex = examples(trec[:a.teacher_records], teacher=tq)

    prs = [r for r in jsonl(ROOT / "prcx/data/dataset/train_records.jsonl") if r["id"] not in in_replay]
    rng.shuffle(prs)
    pr_ex = examples(prs[:a.pr])

    used = set(in_replay)
    for v in ("v3", "v4", "v5"):
        used |= {record_id(json.loads(l)["id"]) for l in open(ROOT / f"v2/data/{v}/train.jsonl")}
    used |= {json.loads(l)["id"] for f in ("records/train.jsonl", "records/valid.jsonl") for l in open(TL / f)}
    pool = collections.defaultdict(list)
    for f in ("pools/hf.jsonl", "pools/synthetic.jsonl"):
        for r in jsonl(TL / f):
            if r["source"] in NEW and r["id"] not in used:
                pool[r["source"]].append(r)
    new_train, new_valid = [], []
    for src, n in NEW.items():
        rs = pool[src]
        rng.shuffle(rs)
        k = min(80, len(rs) // 10)
        new_valid += rs[:k]
        new_train += rs[k:k + n]
    new_ex = examples(new_train)

    train = replay + teacher_ex + pr_ex + new_ex
    rng.shuffle(train)
    out = Path(a.out)
    write(out / "train.jsonl", train)
    valid = jsonl(ROOT / "v2/data/v5/valid.jsonl") + examples(new_valid)
    write(out / "valid.jsonl", valid)
    print(f"train {len(train)}: replay {len(replay)}, teacher 0.8/0.2 {len(teacher_ex)}, PR {len(pr_ex)}, "
          f"new {len(new_ex)}; valid {len(valid)}")
    for k, v in collections.Counter(e["source"] for e in new_ex).most_common():
        print(f"  {v:6d} {k}")


if __name__ == "__main__":
    main()
