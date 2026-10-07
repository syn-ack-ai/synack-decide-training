"""Label decision records with option probabilities from an OpenRouter teacher (top_logprobs).

Same prompt as training (v2/common.py). The first generated token is the option label; its
top-20 logprobs give the distribution. Options missing from the top 20 share the leftover mass.
Resumable (skips qids already in --out). Key read from ~/.config/openrouter/key.
Usage: python teacher_api.py --model google/gemma-4-31b-it --in mix_train_records.jsonl --out t.jsonl
"""
import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, SYSTEM, decision_user, question_criteria  # noqa: E402

KEY = (Path.home() / ".config/openrouter/key").read_text().strip()
URL = "https://openrouter.ai/api/v1/chat/completions"


PROVIDER = {"require_parameters": True, "ignore": ["Venice", "Novita"]}
MIN_LABEL_MASS = 0.5


def to_probs(top, labels):
    seen = {}
    for t in top:
        tok = t["token"].strip()
        if tok in labels and tok not in seen:
            seen[tok] = math.exp(t["logprob"])
    if not seen:
        return None
    mass = sum(seen.values())
    if mass < MIN_LABEL_MASS:  # first token mostly not a label (e.g. a thinking channel): reject, don't invent a distribution
        return None
    rest = [l for l in labels if l not in seen]
    left = max(0.0, 1.0 - mass)
    p = {l: seen.get(l, left / len(rest) if rest else 0.0) for l in labels}
    z = sum(p.values())
    return {l: v / z for l, v in p.items()}


async def label_one(client, sem, model, r, qkey, q, stats):
    crit = question_criteria(q)
    keys = list(crit)
    if not 2 <= len(keys) <= len(LABELS):
        return None
    labels = LABELS[:len(keys)]
    body = {"model": model, "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 20,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": decision_user(r["state"], q, keys, crit)}],
            "provider": PROVIDER, "reasoning": {"enabled": False}}
    if not URL.startswith("https://openrouter.ai/"):  # local llama.cpp: its Gemma template opens a thinking channel unless told not to
        body = {k: v for k, v in body.items() if k not in ("provider", "reasoning")}
        body["chat_template_kwargs"] = {"enable_thinking": False}
    async with sem:
        for attempt in range(4):
            try:
                res = await client.post(URL, json=body)
                if res.status_code == 200:
                    d = res.json()
                    stats["cost"] += float((d.get("usage") or {}).get("cost") or 0)
                    lp = d["choices"][0].get("logprobs") or {}
                    content = lp.get("content") or []
                    if not content:
                        stats["no_logprobs"] += 1
                        return None
                    p = to_probs(content[0].get("top_logprobs") or [], labels)
                    if p is None:
                        stats["no_label"] += 1
                        return None
                    return {"qid": f"{r['id']}#{qkey}", "teacher": model,
                            "probs": {keys[labels.index(l)]: v for l, v in p.items()}}
                if res.status_code in (429, 500, 502, 503):
                    await asyncio.sleep(2 ** attempt)
                    continue
                stats["http_" + str(res.status_code)] = stats.get("http_" + str(res.status_code), 0) + 1
                if stats["http_" + str(res.status_code)] <= 3:
                    print("HTTP", res.status_code, res.text[:200], flush=True)
                return None
            except (httpx.HTTPError, KeyError, ValueError):
                await asyncio.sleep(2 ** attempt)
    return None


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--concurrency", type=int, default=48)
    p.add_argument("--budget", type=float, default=10.0, help="stop when reported spend exceeds this (USD)")
    p.add_argument("--order", help="comma-separated providers to use, in order, with no fallback (e.g. Morph,StreamLake)")
    p.add_argument("--no-gold", action="store_true", help="also label questions without an expected answer (teacher-only rows)")
    p.add_argument("--url", help="OpenAI-compatible chat endpoint instead of OpenRouter (e.g. a local llama.cpp server, for a free test)")
    a = p.parse_args()
    done = {json.loads(l)["qid"] for l in open(a.out)} if Path(a.out).exists() else set()
    jobs = []
    for line in open(a.inp):
        r = json.loads(line)
        for qkey, q in r["questions"].items():
            if f"{r['id']}#{qkey}" not in done and (a.no_gold or r["expected"].get(qkey) is not None):
                jobs.append((r, qkey, q))
    jobs = jobs[:a.limit] if a.limit else jobs
    print(f"{a.model}: {len(jobs)} questions to label ({len(done)} already done)", flush=True)
    global PROVIDER, URL
    if a.url:
        URL = a.url
    if a.order:
        PROVIDER = {"require_parameters": True, "order": a.order.split(","), "allow_fallbacks": False}
    stats = {"cost": 0.0, "no_logprobs": 0, "no_label": 0}
    sem = asyncio.Semaphore(a.concurrency)
    t0, n_ok = time.time(), 0
    headers = {"X-Title": "systemone-teacher"}
    if URL.startswith("https://openrouter.ai/"):  # never send the key to a local test server
        headers["Authorization"] = f"Bearer {KEY}"
    async with httpx.AsyncClient(headers=headers, timeout=120) as client:
        with open(a.out, "a") as g:
            for start in range(0, len(jobs), 500):
                batch = jobs[start:start + 500]
                res = await asyncio.gather(*(label_one(client, sem, a.model, r, k, q, stats) for r, k, q in batch))
                for x in res:
                    if x:
                        g.write(json.dumps(x) + "\n")
                        n_ok += 1
                g.flush()
                print(f"{start + len(batch)}/{len(jobs)} ok={n_ok} cost=${stats['cost']:.3f} "
                      f"{(start + len(batch)) / (time.time() - t0):.1f}/s {dict((k, v) for k, v in stats.items() if k != 'cost' and v)}",
                      flush=True)
                if stats["cost"] > a.budget:
                    print(f"budget ${a.budget} reached; stopping", flush=True)
                    break
    print(f"done: {n_ok} labelled, cost ${stats['cost']:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
