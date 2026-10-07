"""v7 as a teacher: option probabilities for training examples, computed by a llama.cpp server running v7 (GGUF).

Each example is tokenized exactly as in training (chat template, enable_thinking=False) and sent as token ids to
/completion with n_probs, so the teacher sees the identical prompt. Probabilities of the example's label tokens are
renormalized; labels outside the returned top-n share the leftover mass. Resumable; writes {"id", "v7": {label: p}}.

  python v2/label_v7_teacher.py --url http://GPU_HOST:8080/v7b/completion --inp v2/data/v7b/train.jsonl --out teacher_v7.jsonl
"""
import argparse
import json
import math
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default="SkyPanther/synack-decide-26b-a4b")
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    label_id = {l: tok.encode(l, add_special_tokens=False)[0] for l in LABELS}
    done = {json.loads(l)["id"] for l in open(a.out)} if Path(a.out).exists() else set()
    rows, seen = [], set()
    for line in open(a.inp):
        ex = json.loads(line)
        if ex["id"].endswith("~2") or ex["id"] in seen or ex["id"] in done:
            continue  # second-pass copies reuse the first copy's targets
        seen.add(ex["id"])
        rows.append(ex)
    print(f"{len(rows)} examples to label ({len(done)} already done)", flush=True)
    lock, t0, n = threading.Lock(), time.time(), [0]
    out = open(a.out, "a")

    def one(ex):
        ids = tok.apply_chat_template(ex["messages"][:-1], add_generation_prompt=True, enable_thinking=False,
                                      tokenize=True, return_dict=False)
        labels = ex["labels"]
        body = json.dumps({"prompt": ids, "n_predict": 1, "n_probs": min(300, max(20, 2 * len(labels))),
                           "temperature": 0, "cache_prompt": False}).encode()
        for attempt in range(5):
            try:
                r = json.load(urllib.request.urlopen(urllib.request.Request(a.url, data=body, headers={"Content-Type": "application/json"}), timeout=600))
                break
            except Exception:
                if attempt == 4:
                    return None
                time.sleep(10 * (attempt + 1))
        top = {t["id"]: math.exp(t["logprob"]) for t in r["completion_probabilities"][0]["top_logprobs"]}
        got = {l: top[label_id[l]] for l in labels if label_id[l] in top}
        if not got:
            return None
        rest = [l for l in labels if l not in got]
        left = max(0.0, 1.0 - sum(got.values()))
        probs = {l: got.get(l, left / len(rest) if rest else 0.0) for l in labels}
        z = sum(probs.values()) or 1.0
        return {"id": ex["id"], "v7": {l: round(v / z, 6) for l, v in probs.items()}}

    with ThreadPoolExecutor(a.threads) as pool:
        for res in pool.map(one, rows):
            with lock:
                if res:
                    out.write(json.dumps(res) + "\n")
                n[0] += 1
                if n[0] % 1000 == 0:
                    out.flush()
                    el = time.time() - t0
                    print(f"{n[0]}/{len(rows)} {n[0] / el:.1f}/s eta {(len(rows) - n[0]) / (n[0] / el) / 3600:.1f} h", flush=True)
    out.close()
    print("LABEL_DONE", flush=True)


if __name__ == "__main__":
    main()
