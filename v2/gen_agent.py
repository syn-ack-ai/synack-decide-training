"""Search-agent trajectories for stage 2 of the E4B experiment: a teacher (default Gemma-4-31B via OpenRouter) answers
questions with two real tools, web_search (self-hosted SearXNG) and fetch_url (fetches a page and returns its text).
Questions with a known answer are kept only when the final answer contains it (rejection sampling); no-search prompts
are kept only when the teacher answered without tools. Output messages are in OpenAI format (system, user, assistant
with tool_calls, tool results, final assistant), ready for Gemma 4's chat template with tools=TOOLS.

  python gen_agent.py --inp questions.jsonl --out trajectories.jsonl [--limit 5 --budget 3]
"""
import argparse
import asyncio
import html
import json
import os
import re
import time
from pathlib import Path

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"
SEARX = os.environ.get("SEARX_URL", "http://127.0.0.1:18888/search")  # SSH tunnel to SearXNG on box
SYSTEM = ("You are a helpful assistant with two tools: web_search(query) returns the top web results, and fetch_url(url) "
          "returns the text of a page. Use them when a question depends on facts you are not certain of, on specifics "
          "(names, dates, numbers) or on recent events; you may search more than once and read pages. Do not use tools "
          "for tasks that need no outside facts, such as writing, maths or opinions. Base factual answers on what the "
          "sources say and cite them as [1], [2] with their URLs at the end. For important or contested claims, cross-check "
          "several independent sources; if they disagree, say so and show what each says. On contested political or "
          "ideological topics, prefer primary sources (official statements, original documents, data), use outlets with "
          "different perspectives, and present each side's claims with citations rather than a single narrative. If the "
          "sources do not settle the question, say so.")
FORCE = (" For this question, verify the answer with web_search before answering, even if you think you know it, "
         "and cite what you found.")
TOOLS = [{"type": "function", "function": {"name": "web_search", "description": "Search the web and return the top results (title, URL, snippet).",
                                            "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "the search query"}}, "required": ["query"]}}},
         {"type": "function", "function": {"name": "fetch_url", "description": "Fetch a web page and return its main text (first ~3,000 characters).",
                                            "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "the page URL"}}, "required": ["url"]}}}]
TAG = re.compile(r"<[^>]+>")
DROP = re.compile(r"<(script|style|noscript|svg|head)[^>]*>.*?</\1>", re.S | re.I)


def norm(t):
    t = re.sub(r"[^a-z0-9 ]", " ", (t or "").lower())
    return " ".join(w for w in t.split() if w not in ("a", "an", "the"))


WIKI = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "SynACK-research-agent/0.1 (https://github.com/syn-ack-ai)"}
_searx_lock = None
_searx_last = [0.0]


SEARX_ENGINES = os.environ.get("SEARX_ENGINES", "bing,yahoo,mojeek,startpage").split(",")
EXCLUDE = tuple(d for d in os.environ.get("EXCLUDE_DOMAINS", "").split(",") if d)  # always excluded
CONTESTED_EXCLUDE = ("wikipedia.org", "wikimedia.org", "wikiwand.com")  # not used for contested topics (user policy)
CONTESTED = [re.compile(l.strip(), re.I) for l in (Path(__file__).resolve().parent / "contested_topics.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]


def contested(text):
    return any(rx.search(text or "") for rx in CONTESTED)
SEARX_BACKOFF = [int(x) for x in os.environ.get("SEARX_BACKOFF", "0,10,30,60").split(",")]  # evals use "0": one pass, then the next backend
SEARX_GAP = float(os.environ.get("SEARX_GAP", "1.5"))  # seconds between queries overall (each engine sees 1/len(engines))
CACHE = Path(os.environ.get("SEARCH_CACHE", str(Path.home() / ".cache/synack-search/cache.sqlite")))
_rr = [0]
_db = [None]


def excluded(url, topic=False):
    host = re.sub(r"^https?://", "", url or "").split("/")[0].lower()
    return any(host == d or host.endswith("." + d) for d in EXCLUDE + (CONTESTED_EXCLUDE if topic else ()))


def cache_get(key):
    import sqlite3
    if _db[0] is None:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        _db[0] = sqlite3.connect(CACHE)
        _db[0].execute("create table if not exists c (k text primary key, v text)")
    row = _db[0].execute("select v from c where k=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def cache_put(key, val):
    _db[0].execute("insert or replace into c values (?, ?)", (key, json.dumps(val)))
    _db[0].commit()


async def searx(client, query, n):
    """SearXNG on box: one engine per query in rotation (spreads load so no engine blocks us), failover to the next
    engine, backoff and retry when all are empty, excluded domains filtered, results cached."""
    global _searx_lock
    topic = contested(query)
    key = ("searx-contested|" if topic else "searx|") + " ".join(query.lower().split())
    hit = cache_get(key)
    if hit is not None:
        return [tuple(x) for x in hit][:n]
    if _searx_lock is None:
        _searx_lock = asyncio.Lock()
    for backoff in SEARX_BACKOFF:
        if backoff:
            await asyncio.sleep(backoff)
        for _ in range(len(SEARX_ENGINES)):
            async with _searx_lock:
                wait = SEARX_GAP - (time.time() - _searx_last[0])
                if wait > 0:
                    await asyncio.sleep(wait)
                _searx_last[0] = time.time()
                eng = SEARX_ENGINES[_rr[0] % len(SEARX_ENGINES)]
                _rr[0] += 1
            try:
                r = await client.get(SEARX, params={"q": query[:300], "format": "json", "engines": eng}, timeout=25)
                res = [(x.get("title", ""), x.get("url", ""), x.get("content", "")) for x in r.json().get("results", [])
                       if not excluded(x.get("url", ""), topic)]
            except Exception:
                res = []
            if res and relevant(query, res):
                cache_put(key, res[:10])
                return res[:n]
    return []


STOP = set("a an the of in on at to for and or is are was were be by with what which who whom whose when where why how did do does "
           "from as that this it its into about than then".split())


def relevant(query, res, k=3, need=0.5):
    """Accept an engine's results only if one of the top k mentions at least half of the query's content words
    (e.g. Bing read 'who won the 2001 Wimbledon...' as the Korean currency)."""
    words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in STOP and len(w) > 2}
    if not words:
        return True
    for t, u, c in res[:k]:
        text = f"{t} {c} {u}".lower()
        if sum(w in text for w in words) >= need * len(words):
            return True
    return False


async def wiki(client, query, n):
    try:
        r = await client.get(WIKI, params={"action": "query", "list": "search", "srsearch": query[:300], "format": "json",
                                           "srlimit": n, "srprop": "snippet"}, headers=UA, timeout=20)
        hits = r.json().get("query", {}).get("search", [])
        return [(h["title"], "https://en.wikipedia.org/wiki/" + h["title"].replace(" ", "_"),
                 html.unescape(TAG.sub("", h.get("snippet", "")))) for h in hits]
    except Exception:
        return []


BRAVE_KEY = Path.home() / ".config/brave/key"
BRAVE_USED = Path.home() / ".config/brave/used.json"
BRAVE_MONTHLY = int(os.environ.get("BRAVE_MONTHLY", "1995"))  # free plan: 2,000/month (user: use it all; it is only used here)
_brave_lock = None
_brave_last = [0.0]


async def brave(client, query, n):
    """Brave Search API (free plan: 1 query/s, 2,000/month); a local counter stops at BRAVE_MONTHLY."""
    global _brave_lock
    if not BRAVE_KEY.exists():
        return []
    month = time.strftime("%Y-%m")
    used = json.loads(BRAVE_USED.read_text()) if BRAVE_USED.exists() else {}
    if used.get(month, 0) >= BRAVE_MONTHLY:
        return []
    if _brave_lock is None:
        _brave_lock = asyncio.Lock()
    async with _brave_lock:
        wait = 1.05 - (time.time() - _brave_last[0])
        if wait > 0:
            await asyncio.sleep(wait)
        _brave_last[0] = time.time()
        try:
            r = await client.get("https://api.search.brave.com/res/v1/web/search", params={"q": query[:300], "count": min(20, n * 3)},
                                 headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_KEY.read_text().strip()}, timeout=20)
            used[month] = used.get(month, 0) + 1
            BRAVE_USED.write_text(json.dumps(used))
            if r.status_code != 200:
                return []
            return [(x.get("title", ""), x.get("url", ""), TAG.sub("", html.unescape(x.get("description", ""))))
                    for x in r.json().get("web", {}).get("results", []) if not excluded(x.get("url", ""), contested(query))][:n]
        except Exception:
            return []


_searx_empty = [0, 0.0]  # consecutive empty results, time SearXNG was last skipped
BACKENDS = os.environ.get("SEARCH_BACKENDS", "searx,wiki").split(",")  # e.g. "brave,searx,wiki" for evaluation


async def web_search(client, query, n=5):
    if not query.strip():
        return "Search failed: empty query."
    res = []
    for be in BACKENDS:  # first backend with results wins
        if be == "brave":
            res = await brave(client, query, n)
        elif be == "searx":
            res = await searx(client, query, n)
        elif be == "wiki" and not contested(query):  # Wikipedia is fine for neutral facts, never for contested topics
            res = await wiki(client, query, n)
        if res:
            break
    if not res:
        return "No results."
    return "\n".join(f"[{i + 1}] {t.strip()} - {u}\n    {c.strip()[:300]}" for i, (t, u, c) in enumerate(res))


async def fetch_url(client, url, n=3000):
    if excluded(url, contested(url.replace("_", " "))):
        return "This source is not used for this topic; use other independent sources."
    try:
        r = await client.get(url, timeout=20, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (research agent)"})
        text = html.unescape(TAG.sub(" ", DROP.sub(" ", r.text)))
        text = re.sub(r"\s+", " ", text).strip()
        return text[:n] if text else "The page had no readable text."
    except Exception as e:
        return f"Fetch failed: {type(e).__name__}"


async def call(client, a, messages, stats):
    body = {"model": a.model, "messages": messages, "tools": TOOLS, "max_tokens": 1024, "temperature": 0.3,
            "reasoning": {"enabled": False}, "provider": {"require_parameters": True, "ignore": ["Venice", "Novita"]}}
    for attempt in range(4):
        try:
            r = await client.post(URL, json=body)
            if r.status_code == 200:
                d = r.json()
                stats["cost"] += float((d.get("usage") or {}).get("cost") or 0)
                return d["choices"][0]
            await asyncio.sleep(2 ** attempt)
        except (httpx.HTTPError, KeyError, ValueError, IndexError):
            await asyncio.sleep(2 ** attempt)
    return None


def args_of(t):
    try:
        x = json.loads(t["function"].get("arguments") or "{}")
        return x if isinstance(x, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return None


async def run(q, a, api, web, sem, stats):
    try:
        return await _run(q, a, api, web, sem, stats)
    except Exception as e:  # one bad conversation must never stop the whole run
        stats["errors"] = stats.get("errors", 0) + 1
        return None


async def _run(q, a, api, web, sem, stats):
    async with sem:
        sysmsg = SYSTEM + (FORCE if q["kind"] in ("single_fact", "multi_hop") else "")
        msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": q["question"]}]
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
            parsed = [args_of(t) for t in tcs]
            if any(x is None for x in parsed):
                return None  # malformed tool-call arguments: drop the conversation
            msgs.append({"role": "assistant", "content": m.get("content") or "",
                         "tool_calls": [{"id": t["id"], "type": "function", "function": {"name": t["function"]["name"],
                                         "arguments": x}} for t, x in zip(tcs, parsed)]})
            for t, args in zip(tcs, parsed):
                calls += 1
                if t["function"]["name"] == "web_search":
                    out = await web_search(web, args.get("query", ""))
                elif t["function"]["name"] == "fetch_url":
                    out = await fetch_url(web, args.get("url", ""))
                else:
                    out = "Unknown tool."
                msgs.append({"role": "tool", "tool_call_id": t["id"], "name": t["function"]["name"], "content": out})
        final = msgs[-1]["content"] if msgs[-1]["role"] == "assistant" else ""
        if not final:
            return None
        if q["aliases"]:
            ok = any(al and f" {norm(al)} " in f" {norm(final)} " for al in q["aliases"])
        else:
            ok = calls == 0  # no-search prompts: keep only answers that did not use tools
        if q["aliases"] and ok and calls == 0:
            ok = False  # factual questions must be answered from a search, not from memory
        if any(m["role"] == "tool" and m["content"].startswith(("No results", "Search failed")) for m in msgs) and calls <= 1:
            ok = False  # a failed lone search teaches nothing
        msgs[0] = {"role": "system", "content": SYSTEM}  # train with the normal prompt: E4B must learn to search on its own
        stats["kept" if ok else "rejected"] += 1
        return {"id": q["id"], "kind": q["kind"], "correct": ok, "tool_calls": calls, "messages": msgs}


async def main(a):
    done = {json.loads(l)["id"] for l in open(a.out)} if Path(a.out).exists() else set()
    qs = [q for q in map(json.loads, open(a.inp)) if q["id"] not in done]
    qs = qs[:a.limit] if a.limit else qs
    key = (Path.home() / ".config/openrouter/key").read_text().strip()
    stats = {"cost": 0.0, "kept": 0, "rejected": 0}
    sem, t0, n = asyncio.Semaphore(a.concurrency), time.time(), 0
    print(f"{len(qs)} questions ({len(done)} done)", flush=True)
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"}, timeout=180) as api, httpx.AsyncClient() as web:
        with open(a.out, "a") as f:
            tasks = [asyncio.create_task(run(q, a, api, web, sem, stats)) for q in qs]
            for fut in asyncio.as_completed(tasks):
                x = await fut
                if x:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
                    f.flush()
                    n += 1
                    if n % 100 == 0:
                        print(f"{n}/{len(qs)} kept={stats['kept']} rejected={stats['rejected']} cost=${stats['cost']:.2f} {(time.time() - t0) / 60:.0f} min", flush=True)
                if stats["cost"] > a.budget:
                    print("budget reached", flush=True)
                    for t in tasks:
                        t.cancel()
                    break
    print(f"done: {n} written ({stats['kept']} kept / {stats['rejected']} rejected), ${stats['cost']:.3f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="google/gemma-4-31b-it")
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--limit", type=int)
    p.add_argument("--budget", type=float, default=3.0)
    asyncio.run(main(p.parse_args()))
