"""Agent evaluation for a Gemma 4 chat model: answer factual questions with or without tools (web_search via SearXNG,
fetch_url), batched across questions. Generation stops at each tool call; the call is parsed, run, and its result
appended in Gemma's native format before generation continues (at most --max-steps calls per question).
Scores: the final answer contains a gold alias (normalised). Also reports tool use per question.

  SEARX_URL=http://SEARXNG_HOST:8888/search python agent_eval.py --data eval_agent.jsonl --out DIR --mode tools [--adapter A ...]
"""
import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gen_agent import SYSTEM, TOOLS, fetch_url, norm, web_search

CALL = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", re.S)
ARG = re.compile(r'(\w+):<\|"\|>(.*?)<\|"\|>', re.S)
NO_TOOLS = "You are a helpful assistant. Answer the question directly and concisely."


def parse_calls(text):
    return [(name, dict(ARG.findall(body))) for name, body in CALL.findall(text)]


@torch.no_grad()
def step(model, tok, prompts, max_new, stop_ids):
    try:
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=stop_ids, pad_token_id=tok.pad_token_id)
        return [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=False) for g in gen]
    except torch.OutOfMemoryError:  # later rounds carry several search results/pages
        pass
    # retry outside the except block: the live exception's traceback holds the failed attempt's GPU tensors
    enc = gen = None
    torch.cuda.empty_cache()
    if len(prompts) == 1:  # one conversation too long for the GPU: give up on it (scored as wrong), keep the run going
        print(f"OOM on a single {len(tok(prompts[0], add_special_tokens=False)['input_ids'])}-token prompt; dropped", flush=True)
        return [""]
    h = len(prompts) // 2
    return step(model, tok, prompts[:h], max_new, stop_ids) + step(model, tok, prompts[h:], max_new, stop_ids)


LOOP = asyncio.new_event_loop()  # one loop for the whole run (gen_agent's search locks bind to their first loop)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--adapter", action="append", default=[], help="LoRA adapters merged in order (stage 1, stage 2, ...)")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["tools", "no_tools"], required=True)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev)
    if a.adapter:
        from peft import PeftModel
        for ad in a.adapter:
            model = PeftModel.from_pretrained(model, ad).merge_and_unload()
    model.eval()
    stop_ids = [tok.convert_tokens_to_ids(t) for t in ("<turn|>", "<tool_call|>")] + [tok.eos_token_id]
    qs = [json.loads(l) for l in open(a.data)][:a.limit or None]
    tools = TOOLS if a.mode == "tools" else None
    sysmsg = SYSTEM if a.mode == "tools" else NO_TOOLS
    convs = [[{"role": "system", "content": sysmsg}, {"role": "user", "content": q["question"]}] for q in qs]
    calls = [0] * len(qs)
    final = [None] * len(qs)
    t0 = time.time()

    async def run_tools(batch_calls):
        sem = asyncio.Semaphore(32)
        async with httpx.AsyncClient() as web:
            async def one(name, args):
                async with sem:
                    if name == "web_search":
                        return await web_search(web, args.get("query", ""))
                    if name == "fetch_url":
                        return await fetch_url(web, args.get("url", ""))
                    return "Unknown tool."
            return await asyncio.gather(*(one(n, ar) for n, ar in batch_calls))

    for rnd in range(a.max_steps + 1):
        active = [i for i in range(len(qs)) if final[i] is None]
        if not active:
            break
        pending = []  # the round's tool calls, run together after all its generations
        for s in range(0, len(active), a.batch):
            idx = active[s:s + a.batch]
            prompts = [tok.apply_chat_template(convs[i], tools=tools, add_generation_prompt=True, enable_thinking=False, tokenize=False) for i in idx]
            outs = step(model, tok, prompts, 768, stop_ids)
            todo = []
            for i, o in zip(idx, outs):
                pc = parse_calls(o + ("<tool_call|>" if "<|tool_call>" in o and "<tool_call|>" not in o else ""))
                if a.mode == "tools" and pc and rnd < a.max_steps:
                    convs[i].append({"role": "assistant", "content": "", "tool_calls": [
                        {"id": f"c{calls[i]}_{k}", "type": "function", "function": {"name": n, "arguments": ar}} for k, (n, ar) in enumerate(pc)]})
                    for k, (n, ar) in enumerate(pc):
                        todo.append((i, f"c{calls[i]}_{k}", n, ar))
                    calls[i] += len(pc)
                else:
                    final[i] = re.sub(r"<[^>]*\|>|<\|[^>]*>", "", o).strip()
            pending += todo
        if pending:
            res = LOOP.run_until_complete(run_tools([(n, ar) for _, _, n, ar in pending]))
            for (i, cid, n, _), r in zip(pending, res):
                convs[i].append({"role": "tool", "tool_call_id": cid, "name": n, "content": r})
        print(f"round {rnd}: {sum(f is None for f in final)} still working, {(time.time() - t0) / 60:.1f} min", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    score = {}
    with open(out / f"agent_{a.mode}.jsonl", "w") as f:
        for q, fin, c, cv in zip(qs, final, calls, convs):
            ok = bool(fin) and any(al and f" {norm(al)} " in f" {norm(fin)} " for al in q["aliases"])
            score.setdefault(q["kind"], []).append((ok, c))
            f.write(json.dumps({"id": q["id"], "kind": q["kind"], "correct": ok, "tool_calls": c, "final": fin, "messages": cv}, ensure_ascii=False) + "\n")
    for k, v in score.items():
        print(f"{a.mode} {k}: accuracy {sum(o for o, _ in v) / len(v):.3f}  mean tool calls {sum(c for _, c in v) / len(v):.2f}  "
              f"used tools {sum(c > 0 for _, c in v) / len(v):.2f}  (n={len(v)})", flush=True)
    print("AGENT_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
