"""v7 (bf16, exact) option probabilities for decision records, in the teacher-file format of v2/teacher_api.py:
{"qid": "<record id>#<question>", "teacher": "v7-bf16", "probs": {option_key: p}} with keys as in the record
("true"/"false" for yes/no). Same prompt and readout as engine_v2 (softmax over every option label, no top-k cut).
Questions are sorted by length and batched under a token budget. Resumable.

  python label_records_v7.py --model SkyPanther/synack-decide-26b-a4b --inp records.jsonl --out ops_v7_bf16.jsonl
"""
import argparse
import json
import time
from pathlib import Path

from engine_v2 import V2Engine
from decision_index.engines.base import Unsupported


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--inp", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--token-budget", type=int, default=32768)
    a = p.parse_args()
    done = {json.loads(l)["qid"] for l in open(a.out)} if Path(a.out).exists() else set()
    eng = V2Engine(model=a.model, token_budget=a.token_budget, prefix_reuse=False)
    items, skipped = [], 0
    for path in a.inp:
        for line in open(path):
            r = json.loads(line)
            for q, spec in r["questions"].items():
                qid = f"{r['id']}#{q}"
                if qid in done:
                    continue
                try:
                    keys, ids = eng._prompt(r["state"], spec)
                except Unsupported:
                    skipped += 1
                    continue
                items.append((qid, keys, ids))
    items.sort(key=lambda x: len(x[2]))
    print(f"{len(items)} questions to label ({len(done)} done, {skipped} unsupported)", flush=True)
    t0, n = time.time(), 0
    with open(a.out, "a") as f:
        chunk, cmax = [], 0

        def flush(chunk):
            nonlocal n
            for (qid, _, _), probs in zip(chunk, eng._batch_probs([(k, ids) for _, k, ids in chunk])):
                f.write(json.dumps({"qid": qid, "teacher": "v7-bf16", "probs": {k: round(v, 6) for k, v in probs.items()}}) + "\n")
            n += len(chunk)
            f.flush()
            if n // 2000 != (n - len(chunk)) // 2000:
                el = time.time() - t0
                print(f"{n}/{len(items)} {n / el:.1f}/s eta {(len(items) - n) / (n / el) / 60:.0f} min", flush=True)

        for it in items:
            L = len(it[2])
            if chunk and max(cmax, L) * (len(chunk) + 1) > a.token_budget:
                flush(chunk)
                chunk, cmax = [], 0
            chunk.append(it)
            cmax = max(cmax, L)
        if chunk:
            flush(chunk)
    print(f"LABEL_DONE {n} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
