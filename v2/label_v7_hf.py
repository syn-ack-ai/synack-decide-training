"""v7 (bf16) as a teacher on a GPU: option probabilities for every unique training example, same prompt and readout
as training (batched like self_teacher.py). Second-pass copies (ids ending "~2") reuse the first copy's targets.
Writes {"id", "v7": {label: p}}.

  label_v7_hf.py --model SkyPanther/synack-decide-26b-a4b --inp train.jsonl --out teacher_v7_bf16.jsonl
"""
import argparse
import json
import time

from engine_v2 import V2Engine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--token-budget", type=int, default=32768)
    a = p.parse_args()
    eng = V2Engine(model=a.model, token_budget=a.token_budget)
    rows, seen = [], set()
    for line in open(a.inp):
        ex = json.loads(line)
        if ex["id"].endswith("~2") or ex["id"] in seen:
            continue
        seen.add(ex["id"])
        rows.append(ex)
    prompts = []
    for ex in rows:
        ids = eng.tok.apply_chat_template(ex["messages"][:-1], add_generation_prompt=True, enable_thinking=False,
                                          tokenize=True, return_dict=False)
        prompts.append((ex["labels"], ids))
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i][1]))
    print(f"{len(rows)} unique examples to label", flush=True)
    t0, done, agree = time.time(), 0, 0
    with open(a.out, "w") as f:
        chunk, cmax = [], 0

        def flush(chunk):
            nonlocal done, agree
            for i, probs in zip(chunk, eng._batch_probs([prompts[i] for i in chunk])):
                f.write(json.dumps({"id": rows[i]["id"], "v7": {k: round(v, 6) for k, v in probs.items()}}) + "\n")
                agree += max(probs, key=probs.get) == rows[i]["gold"]
            done += len(chunk)
            if done // 5000 != (done - len(chunk)) // 5000:
                el = time.time() - t0
                print(f"{done}/{len(rows)} {done / el:.1f}/s eta {(len(rows) - done) / (done / el) / 60:.0f} min", flush=True)

        for i in order:
            L = len(prompts[i][1])
            if chunk and max(cmax, L) * (len(chunk) + 1) > a.token_budget:
                flush(chunk)
                chunk, cmax = [], 0
            chunk.append(i)
            cmax = max(cmax, L)
        if chunk:
            flush(chunk)
    print(f"done {done} in {(time.time() - t0) / 60:.1f} min; v7 agrees with gold on {agree / done:.3f}", flush=True)


if __name__ == "__main__":
    main()
