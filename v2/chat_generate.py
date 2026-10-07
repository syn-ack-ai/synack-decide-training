"""Generate chat answers for the before/after evaluation of a Gemma 4 chat model (optionally with a LoRA adapter).

Sets (from build_chat.py): MT-Bench (80 questions, two turns: the second turn sees the model's own first answer),
IFEval (541), GSM8K test (first --gsm8k-n), held-out oasst2 prompts (300). Greedy decoding, thinking off,
batched with left padding. Writes one JSONL per set: {"id", "set", "turns": [...prompts], "answers": [...]}.

  python chat_generate.py --model google/gemma-4-E4B-it [--adapter DIR] --data DIR --out DIR
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GSM = "{q}\n\nSolve this step by step. End with a final line of the form '#### <number>'."
LETTERS = "ABCDEFGHIJ"
MMLU = "{q}\n\n{opts}\n\nThink it through briefly, then end with a final line of the form 'Answer: <letter>'."
TQA = "{q}\n\nReply with just the answer: a few words, no explanation."


@torch.no_grad()
def generate(model, tok, convs, max_new, bs):
    """convs: list of message lists -> list of answer strings (same order)."""
    order = sorted(range(len(convs)), key=lambda i: -len(json.dumps(convs[i])))
    out = [None] * len(convs)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        texts = [tok.apply_chat_template(convs[i], add_generation_prompt=True, enable_thinking=False, tokenize=False) for i in idx]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for k, i in enumerate(idx):
            out[i] = tok.decode(gen[k, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--adapter", action="append", default=[], help="LoRA adapters merged in order (stage 1, on-policy round, ...)")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sets", default="mt_bench,ifeval,gsm8k,mmlu_pro,triviaqa,heldout")
    p.add_argument("--gsm8k-n", type=int, default=250)
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args()
    d, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    if a.adapter:
        from peft import PeftModel
        for ad in a.adapter:  # "path" or "path@0.5" to blend the adapter in at partial strength
            path, _, scale = ad.partition("@")
            model = PeftModel.from_pretrained(model, path)
            if scale:
                with torch.no_grad():
                    for n_, p_ in model.named_parameters():
                        if "lora_B" in n_:
                            p_.mul_(float(scale))
            model = model.merge_and_unload()
    model.eval()
    for name in a.sets.split(","):
        t0 = time.time()
        if name == "mt_bench":
            qs = [json.loads(l) for l in open(d / "eval_mt_bench.jsonl")]
            first = generate(model, tok, [[{"role": "user", "content": q["prompt"][0]}] for q in qs], 1024, a.batch)
            second = generate(model, tok, [[{"role": "user", "content": q["prompt"][0]}, {"role": "assistant", "content": f},
                                            {"role": "user", "content": q["prompt"][1]}] for q, f in zip(qs, first)], 1024, a.batch)
            rows = [{"id": q["question_id"], "set": name, "category": q["category"], "turns": q["prompt"], "answers": [f, s]}
                    for q, f, s in zip(qs, first, second)]
        elif name == "ifeval":
            qs = [json.loads(l) for l in open(d / "eval_ifeval.jsonl")]
            ans = generate(model, tok, [[{"role": "user", "content": q["prompt"]}] for q in qs], 1280, a.batch)
            rows = [{"id": q["key"], "set": name, "turns": [q["prompt"]], "answers": [x]} for q, x in zip(qs, ans)]
        elif name == "gsm8k":
            qs = [json.loads(l) for l in open(d / "eval_gsm8k.jsonl")][:a.gsm8k_n]
            ans = generate(model, tok, [[{"role": "user", "content": GSM.format(q=q["question"])}] for q in qs], 768, a.batch)
            rows = [{"id": i, "set": name, "turns": [q["question"]], "answers": [x], "gold": q["answer"]} for i, (q, x) in enumerate(zip(qs, ans))]
        elif name == "mmlu_pro":
            qs = [json.loads(l) for l in open(d / "eval_mmlu_pro.jsonl")]
            conv = [[{"role": "user", "content": MMLU.format(q=q["question"], opts="\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"])))}] for q in qs]
            ans = generate(model, tok, conv, 2048, a.batch)  # long worked answers ran out of room at 1,024
            rows = [{"id": q["id"], "set": name, "category": q["category"], "turns": [q["question"]], "answers": [x], "gold": q["answer"]} for q, x in zip(qs, ans)]
        elif name == "triviaqa":
            qs = [json.loads(l) for l in open(d / "eval_triviaqa.jsonl")]
            ans = generate(model, tok, [[{"role": "user", "content": TQA.format(q=q["question"])}] for q in qs], 64, a.batch)
            rows = [{"id": q["id"], "set": name, "turns": [q["question"]], "answers": [x], "aliases": q["aliases"]} for q, x in zip(qs, ans)]
        else:
            qs = [json.loads(l) for l in open(d / "heldout_prompts.jsonl")]
            ans = generate(model, tok, [q["messages"] for q in qs], 1024, a.batch)
            rows = [{"id": q["id"], "set": name, "turns": [q["messages"][-1]["content"]], "answers": [x]} for q, x in zip(qs, ans)]
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} in {(time.time() - t0) / 60:.1f} min", flush=True)
    print("GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
