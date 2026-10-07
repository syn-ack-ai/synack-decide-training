"""Teacher responses for chat distillation through OpenRouter (or any OpenAI-compatible endpoint).

For each prompt record {"id", "messages"} it writes {"id", "teacher", "response", "finish", "tokens": [[token,
logprob, [[alt, logprob], ...]], ...]} with the top-k alternatives of every generated token (when the provider returns
logprobs), so a same-tokenizer student (Gemma 4 E4B for Gemma 4 31B) can be trained on the teacher's token
distribution, not only its text. Resumable; stops at --budget dollars. Key from ~/.config/openrouter/key.

  python gen_teacher.py --model google/gemma-4-31b-it --inp train_prompts.jsonl --out teacher.jsonl [--limit 10]
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"


async def one(client, sem, a, r, stats):
    body = {"model": a.model, "messages": r["messages"], "max_tokens": a.max_tokens, "temperature": a.temperature,
            "reasoning": {"enabled": False}}
    if a.top_logprobs:
        body.update(logprobs=True, top_logprobs=a.top_logprobs,
                    provider={"require_parameters": True, "ignore": ["Venice", "Novita"]})
    async with sem:
        for attempt in range(4):
            try:
                res = await client.post(a.url, json=body)
                if res.status_code == 200:
                    d = res.json()
                    stats["cost"] += float((d.get("usage") or {}).get("cost") or 0)
                    c = d["choices"][0]
                    text = (c.get("message") or {}).get("content") or ""
                    if not text.strip():
                        stats["empty"] += 1
                        return None
                    lp = ((c.get("logprobs") or {}).get("content")) or []
                    toks = [[t["token"], round(t["logprob"], 4), [[x["token"], round(x["logprob"], 4)] for x in t.get("top_logprobs", [])]] for t in lp]
                    stats["tokens"] += len(toks)
                    return {"id": r["id"], "teacher": a.model, "provider": d.get("provider"), "response": text,
                            "finish": c.get("finish_reason"), "tokens": toks}
                if res.status_code in (429, 500, 502, 503):
                    await asyncio.sleep(2 ** attempt)
                    continue
                stats[f"http_{res.status_code}"] = stats.get(f"http_{res.status_code}", 0) + 1
                if stats[f"http_{res.status_code}"] <= 3:
                    print("HTTP", res.status_code, res.text[:200], flush=True)
                return None
            except (httpx.HTTPError, KeyError, ValueError, IndexError):
                await asyncio.sleep(2 ** attempt)
    return None


async def main(a):
    done = {json.loads(l)["id"] for l in open(a.out)} if Path(a.out).exists() else set()
    rows = [r for r in map(json.loads, open(a.inp)) if r["id"] not in done]
    rows = rows[:a.limit] if a.limit else rows
    print(f"{a.model}: {len(rows)} prompts ({len(done)} done)", flush=True)
    key = (Path.home() / ".config/openrouter/key").read_text().strip()
    headers = {"X-Title": "systemone-chat-teacher"}
    if a.url.startswith("https://openrouter.ai/"):
        headers["Authorization"] = f"Bearer {key}"
    stats = {"cost": 0.0, "empty": 0, "tokens": 0}
    sem, t0, n = asyncio.Semaphore(a.concurrency), time.time(), 0
    queue = asyncio.Queue()
    for r in rows:
        queue.put_nowait(r)
    f = open(a.out, "a")

    async def worker(client):
        nonlocal n
        while not queue.empty() and stats["cost"] <= a.budget:
            r = queue.get_nowait()
            x = await one(client, sem, a, r, stats)
            if x:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                if n % 200 == 0:
                    print(f"{n}/{len(rows)} cost=${stats['cost']:.3f} tokens={stats['tokens']} {(time.time() - t0) / 60:.1f} min "
                          f"{ {k: v for k, v in stats.items() if k not in ('cost', 'tokens') and v} }", flush=True)

    async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(120, connect=20)) as client:
        await asyncio.gather(*(worker(client) for _ in range(a.concurrency)))
    f.close()
    if stats["cost"] > a.budget:
        print(f"budget ${a.budget} reached", flush=True)
    print(f"done: {n} responses, ${stats['cost']:.3f}, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--url", default=URL)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-logprobs", type=int, default=8)
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--budget", type=float, default=3.0)
    asyncio.run(main(p.parse_args()))
