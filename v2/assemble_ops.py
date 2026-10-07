"""Assemble the v8 ops slice from decontaminated records and teacher labels (docs/V6_PLAN.md, "v8 (future planning)").

Gold tasks keep their labels; v7's probabilities (teachers/ops_v7.jsonl) are applied at training time only where
v7 agrees with gold, as for every other v7 row.
Triage has no gold. A triage question is kept only when Kimi-K3 and Gemma-4-31B pick the same answer, and the answer
does not contradict an operator alert label (BGL/Thunderbird): an alert line may not be "noise" or "no action".
The kept answer becomes the record's expected value, and the 0.8 Kimi + 0.2 Gemma blend is its teacher.

  python assemble_ops.py --clean DIR --kimi K.jsonl --gemma G.jsonl --v7 V.jsonl --out OUT
"""
import argparse
import collections
import json
from pathlib import Path

GOLD_TASKS = ("template", "same_event", "bgl_window", "hdfs_session", "hadoop_fault", "rca")


def load(path):
    out = {}
    for line in open(path):
        x = json.loads(line)
        out[x["qid"]] = x["probs"]
    return out


def argmax(p, noul):
    return p.get("true", 0) >= 0.5 if noul else max(p, key=p.get)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean", required=True)
    p.add_argument("--kimi", required=True)
    p.add_argument("--gemma", required=True)
    p.add_argument("--v7", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    clean, out = Path(a.clean), Path(a.out)
    (out / "records").mkdir(parents=True, exist_ok=True)
    (out / "teachers").mkdir(exist_ok=True)
    stats = collections.Counter()

    for t in GOLD_TASKS:
        rows = [json.loads(l) for l in open(clean / f"{t}.jsonl")]
        with open(out / "records" / f"{t}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[f"gold {t}"] = len(rows)

    K, G = load(a.kimi), load(a.gemma)
    kept, blend = [], []
    for line in open(clean / "triage.jsonl"):
        r = json.loads(line)
        alert = r.get("alert_label") not in (None, "-")
        qs, exp = {}, {}
        for q, spec in r["questions"].items():
            qid = f"{r['id']}#{q}"
            k, g = K.get(qid), G.get(qid)
            noul = spec["type"] == "noul"
            if not k or not g:
                stats["triage q missing a teacher"] += 1
                continue
            if argmax(k, noul) != argmax(g, noul):
                stats[f"triage {q} teachers disagree"] += 1
                continue
            ans = argmax(k, noul)
            if alert and ((q == "severity" and ans == "noise") or (q == "act" and ans is False)):
                stats[f"triage {q} contradicts alert label"] += 1
                continue
            qs[q], exp[q] = spec, ans
            mix = {o: 0.8 * k.get(o, 0) + 0.2 * g.get(o, 0) for o in set(k) | set(g)}
            z = sum(mix.values()) or 1.0
            blend.append({"qid": qid, "teacher": "0.8 kimi-k3 + 0.2 gemma-4-31b", "probs": {o: v / z for o, v in mix.items()}})
            stats[f"triage {q} kept"] += 1
        if qs:
            kept.append({**{k: v for k, v in r.items() if k not in ("questions", "expected")},
                         "questions": qs, "expected": exp, "teacher_only": True})
    with open(out / "records" / "triage.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["triage records kept"] = len(kept)
    for name, src in (("ops_triage_blend.jsonl", blend), ):
        with open(out / "teachers" / name, "w") as f:
            for x in src:
                f.write(json.dumps(x) + "\n")
    for name, path in (("ops_kimi_k3.jsonl", a.kimi), ("ops_gemma31.jsonl", a.gemma), ("ops_v7.jsonl", a.v7)):
        (out / "teachers" / name).write_text(Path(path).read_text())
    json.dump(dict(stats), open(out / "assemble_stats.json", "w"), indent=1)
    for k, v in sorted(stats.items()):
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
