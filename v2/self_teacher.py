"""Self-distillation targets: the current model's own option probabilities on replay examples.

Rows marked "replay": true get teacher = {label: p} from --model (one batched forward per chunk, same prompt and
readout as inference). Training then pulls the next model toward these distributions wherever they agree with gold,
which keeps what the current model already does well ("learning without forgetting").

  self_teacher.py --model SkyPanther/synack-decide-26b-a4b --inp train.jsonl --out train_sd.jsonl
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
    rows = [json.loads(l) for l in open(a.inp)]
    todo = [i for i, r in enumerate(rows) if r.get("replay")]
    prompts = {}
    for i in todo:
        m = rows[i]["messages"]
        ids = eng.tok.apply_chat_template(m[:-1], add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=False)
        prompts[i] = (rows[i]["labels"], ids)
    order = sorted(todo, key=lambda i: len(prompts[i][1]))
    t0, done, chunk, cmax = time.time(), 0, [], 0
    agree = 0

    def flush(chunk):
        nonlocal done, agree
        for i, probs in zip(chunk, eng._batch_probs([prompts[i] for i in chunk])):
            rows[i]["teacher"] = {k: round(v, 5) for k, v in probs.items()}
            agree += max(probs, key=probs.get) == rows[i]["gold"]
        done += len(chunk)
        if done % 2000 < len(chunk):
            print(f"{done}/{len(todo)} {done / (time.time() - t0):.1f}/s", flush=True)

    for i in order:
        L = len(prompts[i][1])
        if chunk and max(cmax, L) * (len(chunk) + 1) > a.token_budget:
            flush(chunk)
            chunk, cmax = [], 0
        chunk.append(i)
        cmax = max(cmax, L)
    if chunk:
        flush(chunk)
    with open(a.out, "w") as f:
        for r in rows:
            r.pop("replay", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"self-teacher done: {len(todo)} rows in {(time.time() - t0) / 60:.1f} min; current model agrees with gold on {agree / max(len(todo), 1):.3f}", flush=True)


if __name__ == "__main__":
    main()
