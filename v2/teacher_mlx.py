"""Label decision records with a teacher's option probabilities (MLX, Apple Silicon).

For every question: render the v2 prompt (options in their stored order), one forward pass,
softmax over the option labels -> {option_key: prob}. Resumable: skips ids already in --out.
Usage: python teacher_mlx.py --model ../models/gemma4-31b-it-mlx-8bit --in clean.jsonl --out teacher.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, SYSTEM, decision_user, question_criteria  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=6000)
    p.add_argument("--shard", default="0/1", help="k/n: process every n-th record starting at k")
    p.add_argument("--shortest-first", action="store_true", help="label short records first")
    a = p.parse_args()
    k, n = map(int, a.shard.split("/"))
    model, tok = load(a.model)
    label_ids = []
    for l in LABELS:
        ids = tok.encode(l, add_special_tokens=False)
        assert len(ids) == 1, (l, ids)
        label_ids.append(ids[0])
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["qid"] for l in open(a.out)}
    t0, cnt = time.time(), 0
    lines = open(a.inp).readlines()
    order = sorted(range(len(lines)), key=lambda i: len(lines[i])) if a.shortest_first else range(len(lines))
    with open(a.out, "a") as g:
        for i in order:
            line = lines[i]
            if i % n != k:
                continue
            r = json.loads(line)
            for qkey, q in r["questions"].items():
                qid = f"{r['id']}#{qkey}"
                if qid in done or r["expected"].get(qkey) is None:
                    continue
                crit = question_criteria(q)
                keys = list(crit)
                if not 2 <= len(keys) <= len(LABELS):
                    continue
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": decision_user(r["state"], q, keys, crit)}]
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)
                if len(ids) > a.max_tokens:
                    continue
                logits = model(mx.array(ids)[None])[0, -1]
                probs = mx.softmax(logits[mx.array(label_ids[:len(keys)])].astype(mx.float32)).tolist()
                g.write(json.dumps({"qid": qid, "probs": dict(zip(keys, probs))}) + "\n")
                cnt += 1
                if cnt % 200 == 0:
                    g.flush()
                    print(f"{cnt} labelled, {cnt / (time.time() - t0):.2f}/s", flush=True)
    print(f"done: {cnt} labelled in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
