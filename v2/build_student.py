"""Student mix (Gemma 4 E2B): the v7 training rows with v7 as the teacher.

Teacher per example: 0.5 v7 + 0.5 x (0.8 Kimi-K3 + 0.2 Gemma-4-31B) = 0.5 v7 + 0.4 Kimi + 0.1 Gemma where the v7 mix
already had Kimi/Gemma targets; v7 alone everywhere else (PRs, generated tasks, the newer sets). On the 1,018-question
teacher validation set this blend scored 83.9% / NLL 0.494 (v7 alone 82.7% / 0.513, Kimi alone 80.5% / 0.672).
Keeps the v7 recipe's second pass over hf/* and synthetic/* rows (same targets as the first copy).

  python v2/build_student.py   -> v2/data/student/{train.jsonl, valid.jsonl}
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "v2/data"


def main():
    v7 = {x["id"]: x["v7"] for x in map(json.loads, open(D / "student/teacher_v7.jsonl"))}
    out, n_blend, n_v7, n_missing = [], 0, 0, 0
    for line in open(D / "v7b/train.jsonl"):
        ex = json.loads(line)
        base_id = ex["id"][:-2] if ex["id"].endswith("~2") else ex["id"]
        t7 = v7.get(base_id)
        if t7 is None:
            n_missing += 1
            continue
        ext = None if ex.get("replay") else ex.get("teacher")
        if ext:
            keys = set(t7) | set(ext)
            mix = {k: 0.5 * t7.get(k, 0.0) + 0.5 * ext.get(k, 0.0) for k in keys}
            z = sum(mix.values()) or 1.0
            ex["teacher"] = {k: v / z for k, v in mix.items()}
            n_blend += 1
        else:
            ex["teacher"] = t7
            n_v7 += 1
        ex.pop("replay", None)
        out.append(ex)
    (D / "student").mkdir(exist_ok=True)
    with open(D / "student/train.jsonl", "w") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(D / "student/train_1pass.jsonl", "w") as f:  # one pass over the unique examples (no "~2" copies)
        for ex in out:
            if not ex["id"].endswith("~2"):
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    (D / "student/valid.jsonl").write_text((D / "v7b/valid.jsonl").read_text())
    print(f"student train {len(out)} (blend v7+Kimi+Gemma {n_blend}, v7 only {n_v7}, skipped without v7 label {n_missing})")


if __name__ == "__main__":
    main()
