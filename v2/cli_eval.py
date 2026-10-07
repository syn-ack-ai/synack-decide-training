"""CLI eval for a Gemma 4 chat model (stage 3): verifiable shell tasks (cli_tasks.py) solved with or without the
run_shell tool, using the same system prompt and tool as gen_cli.py. In tools mode each run_shell call runs in a fresh
sandbox copy of the task directory; the final ```bash block is then run on a clean copy and checked. Scores pass rate
per family. Needs the sandbox runner (sandbox_runner.py) on 127.0.0.1:18899.

  python cli_eval.py --data cli_test_tasks.jsonl --out DIR --mode tools [--adapter A ...]
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_eval import parse_calls, step
from gen_cli import BASH, SYSTEM, TOOLS, sandbox, show

NO_TOOLS = ("You are a careful command-line assistant working in the user's project directory. Give the final answer: a "
            "single shell command (pipes and && are fine) in one ```bash block at the end. Be exact with flags.")


async def run_all(jobs):
    sem = asyncio.Semaphore(4)
    async with httpx.AsyncClient() as web:
        return await asyncio.gather(*(sandbox(web, sem, setup, cmd, check) for setup, cmd, check in jobs))


LOOP = asyncio.new_event_loop()  # one loop for the whole run (gen_agent's search locks bind to their first loop)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--adapter", action="append", default=[], help="LoRA adapters merged in order")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["tools", "no_tools"], required=True)
    p.add_argument("--max-steps", type=int, default=5)
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
    ts = [json.loads(l) for l in open(a.data)][:a.limit or None]
    tools = TOOLS if a.mode == "tools" else None
    sysmsg = SYSTEM if a.mode == "tools" else NO_TOOLS
    convs = [[{"role": "system", "content": sysmsg}, {"role": "user", "content": t["question"]}] for t in ts]
    calls, final = [0] * len(ts), [None] * len(ts)
    t0 = time.time()
    for rnd in range(a.max_steps + 1):
        active = [i for i in range(len(ts)) if final[i] is None]
        if not active:
            break
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
                    todo += [(i, f"c{calls[i]}_{k}", n, ar) for k, (n, ar) in enumerate(pc)]
                    calls[i] += len(pc)
                else:
                    final[i] = o.replace("<turn|>", "").strip()
            if todo:
                res = LOOP.run_until_complete(run_all([(ts[i]["setup"], ar.get("command", "") if n == "run_shell" else "echo 'Unknown tool.' >&2; exit 1", "true")
                                           for i, _, n, ar in todo]))
                for (i, cid, n, _), r in zip(todo, res):
                    convs[i].append({"role": "tool", "tool_call_id": cid, "name": n, "content": show(r)})
        print(f"round {rnd}: {sum(f is None for f in final)} still working, {(time.time() - t0) / 60:.1f} min", flush=True)
    cmds = [(BASH.findall(f or "") or [None])[-1] for f in final]
    checks = LOOP.run_until_complete(run_all([(t["setup"], c.strip(), t["check"]) for t, c in zip(ts, cmds) if c]))
    it = iter(checks)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    score = {}
    with open(out / f"cli_{a.mode}.jsonl", "w") as f:
        for t, c, n, cv, fin in zip(ts, cmds, calls, convs, final):
            ok = bool(c) and bool(next(it).get("check_passed"))
            score.setdefault(t["family"], []).append((ok, n))
            f.write(json.dumps({"id": t["id"], "family": t["family"], "correct": ok, "tool_calls": n, "final_command": c,
                                "final": fin, "messages": cv}, ensure_ascii=False) + "\n")
    allv = [x for v in score.values() for x in v]
    for k, v in sorted(score.items()) + [("ALL", allv)]:
        print(f"{a.mode} {k}: pass {sum(o for o, _ in v) / len(v):.3f}  mean tool calls {sum(c for _, c in v) / len(v):.2f}  (n={len(v)})", flush=True)
    print("CLI_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
