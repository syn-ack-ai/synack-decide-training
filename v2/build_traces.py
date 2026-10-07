"""Agent-trace failure detection records (Flow-1-style detection, one forward pass): docs/V6_PLAN.md, v8/v9 ideas.

  trace/swe_outcome   nebius/SWE-agent-trajectories (CC-BY-4.0): did this SWE-agent run actually fix the issue?
                      (yes/no; gold = the hidden tests passed). Train and test never share a GitHub issue.
  trace/whowhen_*     Kevin355/Who_and_When (held-out test only): which agent made the decisive mistake (choice over
                      the agents in the log) and at which step (choice over the steps).

Traces are compacted to fit the context: the task, then every step's action and the start of its output; when it is
still too long, the middle steps are elided (the start and the end, where failures surface, are kept).

  python build_traces.py --swe PARQUET --whowhen DIR --out DIR [--n-train 1200 --n-test 200 --max-chars 36000]
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

CODE = re.compile(r"```(?:\w+)?\n(.*?)```", re.S)


def clip(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + f" ...(+{len(s) - n} chars)"


def fit(steps, max_chars):
    """Keep all steps if they fit; otherwise the first 6 and as many final steps as fit, with an elision marker."""
    if sum(len(s) for s in steps) <= max_chars:
        return steps
    head, budget = steps[:6], max_chars - sum(len(s) for s in steps[:6]) - 80
    tail = []
    for s in reversed(steps[6:]):
        if budget - len(s) < 0:
            break
        tail.insert(0, s)
        budget -= len(s)
    return head + [f"... ({len(steps) - len(head) - len(tail)} steps omitted) ..."] + tail


def swe_steps(traj, out_chars=300, tail_steps=0, tail_chars=1200):
    out, k = [], 0
    msgs = traj[2:]  # [0] system prompt, [1] the issue (given separately as the task)
    n_ai = sum(x["role"] == "ai" for x in msgs)
    for x in msgs:
        t = x.get("text") or ""
        if x["role"] == "ai":
            k += 1
            m = CODE.findall(t)
            thought = CODE.sub("", t).strip()
            out.append(f"{k} ACTION: {clip(thought, 220)} $ {clip(m[-1] if m else '', 400)}")
        elif x["role"] == "user":
            late = k > n_ai - tail_steps  # the final steps (tests, last edits) keep more of their output
            out.append(f"{k} OUTPUT: {clip(t, tail_chars if late else out_chars)}")
    return out


def issue_text(traj):
    t = traj[1].get("text") or ""
    m = re.search(r"ISSUE:\n(.*?)\n(?:INSTRUCTIONS|$)", t, re.S)
    return clip(m.group(1) if m else t, 2000)


def swe_records(paths, n_train, n_test, max_chars, rng, tail_steps=0, patch_chars=0):
    cols = ["instance_id", "model_name", "target", "trajectory", "exit_status", "generated_patch"]
    rows = [r for path in paths for r in pq.read_table(path, columns=cols).to_pylist()]
    split = lambda r: "test" if int(hashlib.md5(r["instance_id"].encode()).hexdigest(), 16) % 10 == 0 else "train"
    out = {"train": [], "test": []}
    for part, n in (("train", n_train), ("test", n_test)):
        pool = [r for r in rows if split(r) == part]
        pos = [r for r in pool if r["target"]]
        neg = [r for r in pool if not r["target"]]
        for r in rng.sample(pos, min(n // 2, len(pos))) + rng.sample(neg, min(n // 2, len(neg))):
            steps = fit(swe_steps(r["trajectory"], tail_steps=tail_steps), max_chars)
            extra = {"submitted_patch": clip(r["generated_patch"], patch_chars)} if patch_chars else {}
            out[part].append({"id": f"trace/swe_outcome:{r['instance_id']}:{r['model_name']}:{hashlib.md5(json.dumps(r['generated_patch']).encode()).hexdigest()[:8]}",
                              "source": "trace/swe_outcome",
                              "state": {"agent": "SWE-agent, an LLM coding agent fixing a GitHub issue in a sandboxed repo",
                                        "task": issue_text(r["trajectory"]), "trajectory": steps,
                                        "ended_with": r["exit_status"], **extra},
                              "questions": {"q": {"type": "noul", "instructions": "Did the agent actually fix the issue, "
                                                  "i.e. would the repository's hidden tests for this issue now pass?"}},
                              "expected": {"q": bool(r["target"])}})
    return out


def whowhen_records(d, max_chars):
    out = []
    for part in ("Hand-Crafted", "Algorithm-Generated"):
        for i, r in enumerate(pq.read_table(Path(d) / f"{part}.parquet").to_pylist()):
            h = r["history"]
            if not h or r.get("mistake_step") in (None, ""):
                continue
            step = int(r["mistake_step"])
            if not 0 <= step < len(h) or len(h) > 255:
                continue
            name = lambda x: (x.get("name") or x.get("role") or "?").split(" (")[0]
            per = max(150, max_chars // len(h))
            log = [f"step {j} [{x.get('name') or x.get('role')}]: {clip(x.get('content'), per)}" for j, x in enumerate(h)]
            agents = sorted({name(x) for x in h if name(x) not in ("human", "user")})
            if r["mistake_agent"] not in agents:
                continue
            out.append({"id": f"trace/whowhen:{part}:{i}", "source": f"trace/whowhen_{part.lower()}",
                        "state": {"system": "a multi-agent LLM system working on a user's question; the final answer was wrong",
                                  "question": clip(r["question"], 1500), "log": log},
                        "questions": {"agent": {"type": "choice", "instructions": "Which agent made the decisive mistake that caused the wrong final answer?",
                                                "criteria": {a: None for a in agents}},
                                      "step": {"type": "choice", "instructions": "At which step did the decisive mistake happen?",
                                               "criteria": {f"step_{j}": None for j in range(len(h))}}},
                        "expected": {"agent": r["mistake_agent"], "step": f"step_{step}"}})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--swe", required=True, nargs="+", help="one or more parquet shards")
    p.add_argument("--tail-steps", type=int, default=0, help="keep up to 1,200 chars of output for the last N steps")
    p.add_argument("--patch-chars", type=int, default=0, help="include the submitted patch, clipped to N chars (0 = leave out)")
    p.add_argument("--whowhen", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-train", type=int, default=1200)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--max-chars", type=int, default=36000)
    p.add_argument("--seed", type=int, default=20261006)
    a = p.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    swe = swe_records(a.swe, a.n_train, a.n_test, a.max_chars, rng, a.tail_steps, a.patch_chars)
    ww = whowhen_records(a.whowhen, a.max_chars)
    for name, rows in (("swe_outcome_train", swe["train"]), ("swe_outcome_test", swe["test"]), ("whowhen_test", ww)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} records, max state chars {max((len(json.dumps(r['state'])) for r in rows), default=0)}")
    tr = {r["id"].split(":")[1] for r in swe["train"]}
    te = {r["id"].split(":")[1] for r in swe["test"]}
    print(f"issue overlap train/test: {len(tr & te)}")


if __name__ == "__main__":
    main()
