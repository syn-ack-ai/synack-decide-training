"""On-policy distillation, step 1 (student side): the current student (Gemma 4 E4B + adapters) answers prompts by
sampling, and the exact token ids are saved so the teacher scores precisely what the student produced.

Output: {"id", "prompt_ids", "response_ids", "text"} per prompt. Works on CUDA (4090) or MPS (Mac).

  python opd_sample.py --prompts P.jsonl --out samples.jsonl [--adapter A ...] [--limit N --max-new 640 --temperature 1.0]
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--adapter", action="append", default=[])
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-new", type=int, default=640)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=24)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev)
    if a.adapter:
        from peft import PeftModel
        for ad in a.adapter:
            model = PeftModel.from_pretrained(model, ad).merge_and_unload()
    model.eval()
    done = {json.loads(l)["id"] for l in open(a.out)} if Path(a.out).exists() else set()
    rows = [r for r in map(json.loads, open(a.prompts)) if r["id"] not in done][:a.limit or None]
    rows.sort(key=lambda r: len(r["messages"][-1]["content"]))
    eot = [tok.convert_tokens_to_ids("<turn|>"), tok.eos_token_id]
    t0, n = time.time(), 0
    with open(a.out, "a") as f, torch.no_grad():
        for s in range(0, len(rows), a.batch):
            chunk = rows[s:s + a.batch]
            prompts = [tok.apply_chat_template(r["messages"], add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=False) for r in chunk]
            L = max(map(len, prompts))
            pad = tok.pad_token_id
            ids = torch.tensor([[pad] * (L - len(x)) + x for x in prompts], device=dev)
            mask = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in prompts], device=dev)
            gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=a.max_new, do_sample=True,
                                 temperature=a.temperature, top_p=1.0, eos_token_id=eot, pad_token_id=pad)
            for r, pr, g in zip(chunk, prompts, gen[:, L:].tolist()):
                resp = []
                for t in g:
                    resp.append(t)
                    if t in eot:
                        break
                if resp and resp[-1] not in eot:  # hit the length limit: keep it, the teacher still scores every token
                    pass
                f.write(json.dumps({"id": r["id"], "prompt_ids": pr, "response_ids": resp,
                                    "text": tok.decode(resp, skip_special_tokens=True)}, ensure_ascii=False) + "\n")
                n += 1
            f.flush()
            el = time.time() - t0
            print(f"{n}/{len(rows)} sampled, {el / 60:.1f} min, eta {(len(rows) - n) * el / max(n, 1) / 60:.0f} min", flush=True)
    print("SAMPLE_DONE", flush=True)


if __name__ == "__main__":
    main()
