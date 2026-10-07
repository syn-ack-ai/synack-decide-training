"""Stage-3 CLI trajectories: a teacher (default Gemma-4-31B via OpenRouter) solves verifiable shell tasks
(v2/cli_tasks.py) with a run_shell tool backed by the sandbox (v2/sandbox_runner.py, through the SSH tunnel). Each
run_shell call runs in a fresh copy of the task directory (changes don't persist between calls). The final answer must
end with one ```bash block; that command is run on a clean copy and checked. Only verified conversations are kept,
in OpenAI message format (TOOLS below), ready for train_sft.py --messages.

  python gen_cli.py --inp cli_train.jsonl --out cli_traj.jsonl [--limit 10 --budget 1]
"""
import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"
RUNNER = "http://127.0.0.1:18899/run"
SYSTEM = ("You are a careful command-line assistant working in the user's project directory. You can run shell commands "
          "with run_shell; each call runs in a fresh copy of the directory, so changes do not persist between calls. Look "
          "around and test your command when it helps, then give the final answer: a single shell command (pipes and && are "
          "fine) in one ```bash block at the end. Be exact with flags.")
TOOLS = [{"type": "function", "function": {"name": "run_shell", "description": "Run a shell command in a fresh copy of the user's directory and return stdout, stderr and the exit code.",
                                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
BASH = re.compile(r"```(?:bash|sh|shell)?\n(.*?)```", re.S)


async def sandbox(client, sem, setup, command, check="true"):
    async with sem:
        try:
            r = await client.post(RUNNER, json={"setup": setup, "command": command, "check": check, "timeout": 15}, timeout=120)
            return r.json()
        except Exception as e:
            return {"stdout": "", "stderr": f"sandbox error: {type(e).__name__}", "exit_code": -1, "check_passed": False}


def show(res):
    out = (res.get("stdout") or "")[:1500]
    err = (res.get("stderr") or "")[:800]
    return f"exit code {res.get('exit_code')}\nstdout:\n{out}" + (f"\nstderr:\n{err}" if err else "")


async def call(api, a, msgs, stats):
    body = {"model": a.model, "messages": msgs, "tools": TOOLS, "max_tokens": 800, "temperature": 0.3,
            "reasoning": {"enabled": False}, "provider": {"require_parameters": True, "ignore": ["Venice", "Novita"]}}
    for attempt in range(4):
        try:
            r = await api.post(URL, json=body)
            if r.status_code == 200:
                d = r.json()
                stats["cost"] += float((d.get("usage") or {}).get("cost") or 0)
                return d["choices"][0]
            await asyncio.sleep(2 ** attempt)
        except (httpx.HTTPError, KeyError, ValueError, IndexError):
            await asyncio.sleep(2 ** attempt)
    return None


async def solve(t, a, api, web, sem, sbx, stats):
    async with sem:
        try:
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": t["question"]}]
            calls = 0
            for _ in range(a.max_steps + 1):
                c = await call(api, a, msgs, stats)
                if not c:
                    return None
                m = c.get("message") or {}
                tcs = m.get("tool_calls") or []
                if not tcs:
                    msgs.append({"role": "assistant", "content": (m.get("content") or "").strip()})
                    break
                if calls >= a.max_steps:
                    return None
                parsed = []
                for tc in tcs:
                    try:
                        parsed.append(json.loads(tc["function"].get("arguments") or "{}"))
                    except json.JSONDecodeError:
                        return None
                msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["function"]["name"], "arguments": p}} for tc, p in zip(tcs, parsed)]})
                for tc, p in zip(tcs, parsed):
                    calls += 1
                    res = await sandbox(web, sbx, t["setup"], p.get("command", ""))
                    msgs.append({"role": "tool", "tool_call_id": tc["id"], "name": "run_shell", "content": show(res)})
            final = msgs[-1]["content"] if msgs[-1]["role"] == "assistant" else ""
            blocks = BASH.findall(final or "")
            if not blocks:
                stats["no_command"] += 1
                return None
            v = await sandbox(web, sbx, t["setup"], blocks[-1].strip(), t["check"])
            ok = bool(v.get("check_passed"))
            stats["kept" if ok else "rejected"] += 1
            return {"id": t["id"], "family": t["family"], "correct": ok, "tool_calls": calls, "final_command": blocks[-1].strip(), "messages": msgs}
        except Exception:
            stats["errors"] += 1
            return None


async def main(a):
    done = {json.loads(l)["id"] for l in open(a.out)} if Path(a.out).exists() else set()
    tasks = [t for t in map(json.loads, open(a.inp)) if t["id"] not in done][:a.limit or None]
    key = (Path.home() / ".config/openrouter/key").read_text().strip()
    stats = {"cost": 0.0, "kept": 0, "rejected": 0, "no_command": 0, "errors": 0}
    sem, sbx, t0, n = asyncio.Semaphore(a.concurrency), asyncio.Semaphore(4), time.time(), 0
    print(f"{len(tasks)} tasks ({len(done)} done)", flush=True)
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"}, timeout=180) as api, httpx.AsyncClient() as web:
        with open(a.out, "a") as f:
            for fut in asyncio.as_completed([asyncio.create_task(solve(t, a, api, web, sem, sbx, stats)) for t in tasks]):
                x = await fut
                if x:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
                    f.flush()
                    n += 1
                    if n % 50 == 0:
                        print(f"{n}/{len(tasks)} {stats} {(time.time() - t0) / 60:.0f} min", flush=True)
                if stats["cost"] > a.budget:
                    print("budget reached", flush=True)
                    break
    print(f"done: {n} written, {stats}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="google/gemma-4-31b-it")
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int)
    p.add_argument("--budget", type=float, default=1.0)
    asyncio.run(main(p.parse_args()))
