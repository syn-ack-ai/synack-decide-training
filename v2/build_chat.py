"""Chat-distillation prompts for the E4B conversational experiment.

Training prompts: English conversation openers from OpenAssistant oasst2 (Apache-2.0), reviewed and not deleted,
20-2,000 characters, de-duplicated, with anything sharing a 13-word window with an evaluation set removed.
Evaluation sets kept out of training: MT-Bench (80 two-turn questions), IFEval (541 instruction-following prompts),
GSM8K test, plus 300 held-out oasst2 prompts for judged comparisons.

  python build_chat.py --data DIR --out DIR [--n-train 9000 --n-heldout 300]
"""
import argparse
import gzip
import json
import random
import re
from pathlib import Path

WORD = re.compile(r"[a-z0-9]+")


def grams(text, n=13):
    w = WORD.findall(text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-train", type=int, default=9000)
    p.add_argument("--n-heldout", type=int, default=300)
    p.add_argument("--seed", type=int, default=20261006)
    a = p.parse_args()
    d, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq
    evals = {"mt_bench": [json.loads(l) for l in open(d / "raw/question.jsonl")],
             "ifeval": [json.loads(l) for l in open(d / "ifeval_input_data.jsonl")],
             "gsm8k": pq.read_table(d / "main/test-00000-of-00001.parquet").to_pylist()}
    eval_text = ([t for q in evals["mt_bench"] for t in q["prompt"]] + [q["prompt"] for q in evals["ifeval"]]
                 + [q["question"] for q in evals["gsm8k"]])
    eval_grams = set().union(*(grams(t) for t in eval_text))

    seen, keep, dropped = set(), [], 0
    for line in gzip.open(d / "2023-11-05_oasst2_prompts.messages.jsonl.gz", "rt"):
        r = json.loads(line)
        if (r.get("lang") != "en" or r.get("parent_id") or r.get("deleted") or r.get("review_result") is False
                or r.get("synthetic") or not 20 <= len(r["text"]) <= 2000):
            continue
        key = " ".join(WORD.findall(r["text"].lower()))[:300]
        if key in seen:
            continue
        seen.add(key)
        if grams(r["text"]) & eval_grams:
            dropped += 1
            continue
        keep.append({"id": f"oasst2:{r['message_id']}", "messages": [{"role": "user", "content": r["text"].strip()}]})
    random.Random(a.seed).shuffle(keep)
    held, train = keep[:a.n_heldout], keep[a.n_heldout:a.n_heldout + a.n_train]
    for name, rows in (("train_prompts", train), ("heldout_prompts", held)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name, rows in evals.items():
        with open(out / f"eval_{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"{len(keep)} clean openers ({dropped} dropped for eval overlap); train {len(train)}, held-out {len(held)}; "
          f"evals: MT-Bench {len(evals['mt_bench'])}, IFEval {len(evals['ifeval'])}, GSM8K {len(evals['gsm8k'])}")


if __name__ == "__main__":
    main()
