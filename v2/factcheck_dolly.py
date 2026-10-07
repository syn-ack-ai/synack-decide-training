"""Fact-check teacher answers to Dolly prompts against Dolly's human reference answers, and correct the ones that
contradict it (stage 1 of the E4B experiment).

1. Check: the checker model (default Gemma-4-31B) reads the question, the human reference and the teacher answer, and
   replies CONSISTENT or CONTRADICTS. Only fact-bearing categories are checked (not brainstorming or creative writing).
2. Adjudicate: Dolly's references date from 2023 and can be stale, so for CONTRADICTS the question is searched
   (SearXNG on box) and the checker decides from the results: reference right -> correct the answer; teacher right ->
   keep it; unclear -> drop the example.
3. Correct: the teacher rewrites its answer with the reference in hand, keeping a complete and helpful style, agreeing on
   the facts and not adding details it is unsure of.

Outputs:
- --out: corrected teacher file in the same format. Rewritten answers carry no token distributions, so they train with
  cross-entropy only.
- --pairs: {"id", "prompt", "rejected", "chosen"} for an optional DPO stage.

  python factcheck_dolly.py --teacher teacher_gemma31_dolly.jsonl --prompts dolly_prompts.jsonl --dolly cleaned.json --out OUT --pairs PAIRS
"""
import argparse
import os
import asyncio
import json
import re
from pathlib import Path

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"
CHECK = ("Question:\n{q}\n\nReference answer (written by a person, treat as correct):\n{ref}\n\nCandidate answer:\n{cand}\n\n"
         "Does the candidate answer contradict the reference on any fact that matters for the question (a different name, "
         "date, number, place, choice or classification)? Extra correct detail is fine. Reply with exactly one word: "
         "CONSISTENT or CONTRADICTS.")
FIX = ("{q}\n\n(Private note for you, not to be mentioned in the answer: according to current web sources the correct answer "
       "is: \"{ref}\". Write a complete, helpful answer that agrees with it on every fact. Do not add names, dates, titles or "
       "numbers that you are not sure of.)")
ADJ = ("Question:\n{q}\n\nAnswer A:\n{ref}\n\nAnswer B:\n{cand}\n\nWeb search results (current):\n{res}\n\nThe two answers "
       "disagree. Based on the search results, which answer is correct now? Reply with exactly one word: A, B or UNCLEAR.")
SEARX = os.environ.get("SEARX_URL", "http://127.0.0.1:18888/search")  # SSH tunnel to SearXNG on box (macOS blocks LAN access for unapproved apps)


async def search(client, q, n=5):
    """Throttled SearXNG with the Wikipedia API as fallback (shared with gen_agent.py)."""
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parent))
    from gen_agent import fetch_url, web_search
    out = await web_search(client, q, n)
    if out.startswith(("No results", "Search failed")):
        return ""
    m = re.search(r"https?://\S+", out)  # also read the top result's page: snippets alone rarely settle a dispute
    page = await fetch_url(client, m.group(0), 2500) if m else ""
    return out + ("\n\nText of result [1]:\n" + page if page and not page.startswith("Fetch failed") else "")
CHECKED = {"open_qa", "general_qa", "closed_qa", "information_extraction", "classification"}


async def chat(client, sem, model, content, max_tokens, stats):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "reasoning": {"enabled": False}}
    async with sem:
        for attempt in range(4):
            try:
                r = await client.post(URL, json=body)
                if r.status_code == 200:
                    d = r.json()
                    stats["cost"] += float((d.get("usage") or {}).get("cost") or 0)
                    return d["choices"][0]["message"].get("content") or ""
                await asyncio.sleep(2 ** attempt)
            except (httpx.HTTPError, KeyError, ValueError):
                await asyncio.sleep(2 ** attempt)
    return None


async def main(a):
    prompts = {r["id"]: r for r in map(json.loads, open(a.prompts))}
    ref = {}
    for x in json.load(open(a.dolly)):
        if "category" in x:
            text = x["instruction"].strip() + ("\n\n" + x["input"].strip() if x["input"].strip() else "")
            ref.setdefault(text, x["output"].strip())
    rows = [json.loads(l) for l in open(a.teacher)]
    seen, uniq = set(), []
    for r in rows:  # keep the first answer per id (the restart left duplicates)
        if r["id"] not in seen:
            seen.add(r["id"])
            uniq.append(r)
    rows = uniq[:a.limit] if a.limit else uniq
    key = (Path.home() / ".config/openrouter/key").read_text().strip()
    stats = {"cost": 0.0}
    sem = asyncio.Semaphore(a.concurrency)
    search_sem = asyncio.Semaphore(4)  # be gentle with SearXNG's upstream engines
    plain = httpx.AsyncClient()
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"}, timeout=180) as client:
        async def handle(r):
            p = prompts.get(r["id"])
            if not p or p["category"] not in CHECKED or r.get("finish") != "stop":
                return r, "skipped", None
            q = p["messages"][0]["content"]
            h = ref.get(q)
            if not h:
                return r, "no_reference", None
            v = await chat(client, sem, a.model, CHECK.format(q=q, ref=h, cand=r["response"]), 8, stats)
            if not v or "CONTRADICT" not in v.upper():
                return r, "consistent" if v else "check_failed", None
            async with search_sem:  # the reference may be stale (Dolly is from 2023): let current web results decide
                res = await search(plain, q.split("\n\n")[0])
            if not res:
                return r, "unclear_dropped", None
            j = await chat(client, sem, a.model, ADJ.format(q=q, ref=h, cand=r["response"][:3000], res=res), 8, stats)
            j = (j or "").strip().upper()
            if j.startswith("B"):
                return r, "teacher_right_kept", None
            if not j.startswith("A"):
                return r, "unclear_dropped", None
            fixed = await chat(client, sem, a.model, FIX.format(q=q, ref=h.replace('"', "'")), 1024, stats)
            if not fixed:
                return r, "fix_failed", None
            new = {**r, "response": fixed.strip(), "tokens": [], "corrected": True}
            return new, "corrected", {"id": r["id"], "prompt": p["messages"], "rejected": r["response"], "chosen": fixed.strip(), "reference": h}
        res = await asyncio.gather(*(handle(r) for r in rows))
    counts = {}
    with open(a.out, "w") as f, open(a.pairs, "w") as g:
        for row, status, pair in res:
            counts[status] = counts.get(status, 0) + 1
            if status not in ("fix_failed", "unclear_dropped"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if pair:
                g.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"{len(rows)} answers: {counts}; cost ${stats['cost']:.3f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--dolly", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--model", default="google/gemma-4-31b-it")
    p.add_argument("--limit", type=int)
    p.add_argument("--concurrency", type=int, default=24)
    asyncio.run(main(p.parse_args()))
