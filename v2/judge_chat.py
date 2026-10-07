"""Pairwise LLM judge (default Kimi-K3 via OpenRouter) for two chat_generate.py runs on MT-Bench and held-out prompts.

Every question is judged twice with the answers swapped; a model wins a question only if it wins both orders,
otherwise it is a tie. MT-Bench is judged on the full two-turn conversation. Prints win/tie/loss of B vs A.

  python judge_chat.py --a GEN_A --b GEN_B --out judgements.jsonl [--sets mt_bench,heldout --limit N --budget 2]
"""
import argparse
import asyncio
import json
import re
from pathlib import Path

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM = ("You are an impartial judge of AI assistant answers. Compare two assistants' replies to the same user. "
          "Judge helpfulness, correctness, instruction following, depth and clarity; ignore length unless it helps "
          "the user. Reply with exactly one of: [[A]], [[B]] or [[TIE]].")


def render(turns, answers):
    return "\n\n".join(f"### User (turn {i + 1})\n{t}\n\n### Assistant (turn {i + 1})\n{a}" for i, (t, a) in enumerate(zip(turns, answers)))


async def judge(client, sem, model, turns, x, y, stats):
    user = (f"[Conversation with assistant A]\n{render(turns, x)}\n\n[Conversation with assistant B]\n{render(turns, y)}\n\n"
            "Which assistant is better? Reply with ONLY the verdict tag [[A]], [[B]] or [[TIE]] and nothing else.")
    body = {"model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "max_tokens": 60, "temperature": 0, "reasoning": {"enabled": False}}
    async with sem:
        for attempt in range(4):
            try:
                r = await client.post(URL, json=body)
                if r.status_code == 200:
                    d = r.json()
                    stats["cost"] += float((d.get("usage") or {}).get("cost") or 0)
                    text = d["choices"][0]["message"]["content"] or ""
                    m = re.search(r"\[\[(A|B|TIE)\]\]", text)
                    if m:
                        return m.group(1)
                    first = text.strip().split("\n")[0].lower()  # fallback: "**Assistant A** is better ..."
                    m = re.search(r"assistant (a|b)\b[^.\n]*\b(better|wins|superior|preferred)", first)
                    if m:
                        return m.group(1).upper()
                    return "TIE" if re.search(r"\b(tie|equal|equally)\b", first) else None
                await asyncio.sleep(2 ** attempt)
            except (httpx.HTTPError, KeyError, ValueError):
                await asyncio.sleep(2 ** attempt)
    return None


async def main(a):
    key = (Path.home() / ".config/openrouter/key").read_text().strip()
    stats = {"cost": 0.0}
    sem = asyncio.Semaphore(16)
    out = open(a.out, "w")
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"}, timeout=180) as client:
        for s in a.sets.split(","):
            A = {r["id"]: r for r in map(json.loads, open(Path(a.a) / f"{s}.jsonl"))}
            B = {r["id"]: r for r in map(json.loads, open(Path(a.b) / f"{s}.jsonl"))}
            ids = [i for i in A if i in B][:a.limit or None]

            async def both(i):
                t = A[i]["turns"]
                v1 = await judge(client, sem, a.judge, t, A[i]["answers"], B[i]["answers"], stats)
                v2 = await judge(client, sem, a.judge, t, B[i]["answers"], A[i]["answers"], stats)
                b_wins = v1 == "B" and v2 == "A"
                a_wins = v1 == "A" and v2 == "B"
                res = "B" if b_wins else "A" if a_wins else "tie"
                out.write(json.dumps({"set": s, "id": i, "order1": v1, "order2": v2, "result": res}) + "\n")
                return res
            res = await asyncio.gather(*(both(i) for i in ids))
            w, l, t = res.count("B"), res.count("A"), res.count("tie")
            print(f"{s}: B wins {w}, ties {t}, A wins {l} (n={len(res)}); B win rate {(w + t / 2) / len(res):.3f}  cost so far ${stats['cost']:.2f}", flush=True)
            if stats["cost"] > a.budget:
                print("budget reached", flush=True)
                break
    out.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="baseline generations dir")
    p.add_argument("--b", required=True, help="candidate generations dir")
    p.add_argument("--out", required=True)
    p.add_argument("--sets", default="mt_bench,heldout")
    p.add_argument("--judge", default="moonshotai/kimi-k3")
    p.add_argument("--limit", type=int)
    p.add_argument("--budget", type=float, default=2.0)
    asyncio.run(main(p.parse_args()))
