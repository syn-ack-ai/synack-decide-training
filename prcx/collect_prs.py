"""Collect merged PRs with review outcomes from the selected public repos (GitHub GraphQL via `gh`).

Per repo: newest merged PRs (default 400, since --since), with reviews, review-thread and comment
counts, commit timestamps and labels. Bot-authored PRs are kept but flagged. Resumable: repos with
an existing output file are skipped. Writes data/prs/<owner>__<repo>.jsonl.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
Q = """query($owner: String!, $name: String!, $after: String) { rateLimit { remaining resetAt cost }
repository(owner: $owner, name: $name) { pullRequests(states: MERGED, first: 40, after: $after,
  orderBy: {field: CREATED_AT, direction: DESC}) { pageInfo { hasNextPage endCursor } nodes {
  number title body createdAt mergedAt additions deletions changedFiles
  author { login } baseRefName baseRefOid headRefOid mergeCommit { oid }
  labels(first: 10) { nodes { name } }
  reviews(first: 40) { nodes { author { login } state submittedAt bodyText } }
  reviewThreads(first: 1) { totalCount }
  comments(first: 1) { totalCount }
  commits(last: 50) { totalCount nodes { commit { committedDate } } } } } } }"""


def gql(**vars):
    args = ["gh", "api", "graphql", "-f", f"query={Q}"]
    for k, v in vars.items():
        if v is not None:
            args += ["-f", f"{k}={v}"]
    for attempt in range(5):
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode == 0:
            return json.loads(r.stdout)["data"]
        if "rate limit" in r.stderr.lower() or "secondary" in r.stderr.lower():
            time.sleep(60 * (attempt + 1))
        else:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(r.stderr[:300])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-repo", type=int, default=400)
    p.add_argument("--since", default="2024-06-01")
    p.add_argument("--shard", default="0/1", help="k/n: handle every n-th repo starting at k")
    a = p.parse_args()
    repos = json.loads((ROOT / "data/repos.json").read_text())
    outdir = ROOT / "data/prs"
    outdir.mkdir(parents=True, exist_ok=True)
    total = 0
    shard_k, shard_n = map(int, a.shard.split("/"))
    for i, r in enumerate(repos):
        if i % shard_n != shard_k:
            continue
        out = outdir / (r["repo"].replace("/", "__") + ".jsonl")
        if out.exists():
            total += sum(1 for _ in open(out))
            continue
        owner, name = r["repo"].split("/")
        rows, after = [], None
        while len(rows) < a.per_repo:
            try:
                d = gql(owner=owner, name=name, after=after)
            except RuntimeError as e:
                print(f"{r['repo']}: error {e}", file=sys.stderr, flush=True)
                break
            prs = d["repository"]["pullRequests"]
            stop = False
            for n in prs["nodes"]:
                if n["createdAt"] < a.since:
                    stop = True
                    break
                n["body"] = (n["body"] or "")[:2000]
                for rv in n["reviews"]["nodes"]:
                    rv["bodyText"] = (rv["bodyText"] or "")[:500]
                n["repo"] = r["repo"]
                n["repo_language"] = r["language"]
                rows.append(n)
            if stop or not prs["pageInfo"]["hasNextPage"]:
                break
            after = prs["pageInfo"]["endCursor"]
            if d["rateLimit"]["remaining"] < 200:
                print("rate limit low; sleeping until reset", flush=True)
                time.sleep(900)
        out.write_text("".join(json.dumps(x) + "\n" for x in rows[:a.per_repo]))
        total += len(rows[:a.per_repo])
        print(f"[{i + 1}/{len(repos)}] {r['repo']}: {len(rows[:a.per_repo])} PRs (total {total}), "
              f"graphql left {d['rateLimit']['remaining'] if rows else '?'}", flush=True)
    print(f"done: {total} PRs")


if __name__ == "__main__":
    main()
