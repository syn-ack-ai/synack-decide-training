"""On-policy distillation, step 2 (teacher side, Mac): Gemma 4 31B in MLX scores the student's own answers. For every
response position it saves the teacher's top-k token ids and probabilities (renormalised over the top k), which become
the training target at that position. The prompt and response ids are used exactly as the student produced them; the
same Gemma 4 tokenizer and chat template make that valid. Resumable.

  python opd_score.py --samples samples.jsonl --out scored.jsonl [--teacher models/gemma4-31b-it-mlx-8bit --top-k 8]
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="models/gemma4-31b-it-mlx-8bit")
    p.add_argument("--samples", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    model, ttok = load(a.teacher)
    # Gemma 4 31B's template (thinking off) ends the generation prompt with an empty thought block that E4B's template
    # omits; give the teacher its own expected prefix so it scores the student's answer in-distribution.
    probe = ttok.apply_chat_template([{"role": "user", "content": "x"}], add_generation_prompt=True, enable_thinking=False)
    probe = probe if isinstance(probe, list) else ttok.encode(probe, add_special_tokens=False)
    tail = ttok.encode("<|channel>thought\n<channel|>", add_special_tokens=False)
    extra = tail if probe[-len(tail):] == tail else []
    print(f"teacher prefix added after the student prompt: {ttok.decode(extra)!r}", flush=True)
    done = {json.loads(l)["id"] for l in open(a.out)} if Path(a.out).exists() else set()
    rows = [r for r in map(json.loads, open(a.samples)) if r["id"] not in done and r["response_ids"]][:a.limit or None]
    t0 = time.time()
    with open(a.out, "a") as f:
        for i, r in enumerate(rows, 1):
            P0, R = r["prompt_ids"], r["response_ids"]
            P = P0 + (extra if P0[-len(extra):] != extra else []) if extra else P0
            x = mx.array([P + R])
            logits = model(x)[0, len(P) - 1:len(P) + len(R) - 1].astype(mx.float32)  # position j predicts response token j
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            idx = mx.argpartition(-logp, a.top_k, axis=-1)[:, :a.top_k]
            val = mx.take_along_axis(logp, idx, axis=-1)
            prob = mx.exp(val)
            prob = prob / prob.sum(axis=-1, keepdims=True)
            tok_lp = mx.take_along_axis(logp, mx.array(R)[:, None], axis=-1)[:, 0]  # teacher log-prob of what the student wrote
            mx.eval(idx, prob, tok_lp)
            f.write(json.dumps({"id": r["id"], "prompt_ids": P0, "response_ids": R,
                                "kd_ids": idx.tolist(), "kd_probs": [[round(v, 5) for v in row] for row in prob.tolist()],
                                "teacher_token_logprob": [round(v, 4) for v in tok_lp.tolist()]}) + "\n")
            if i % 50 == 0:
                f.flush()
                el = time.time() - t0
                print(f"{i}/{len(rows)} scored, {el / 60:.1f} min, eta {(len(rows) - i) * el / i / 60:.0f} min", flush=True)
    print("SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
