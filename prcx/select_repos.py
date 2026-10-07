"""Pick public repos that resemble an enterprise stack and actually review their PRs.

Search GitHub per language (plus Jenkins orgs), then sample each candidate's recent merged PRs and keep
repos where non-bot PRs commonly get human reviews. Writes data/repos.json.
Uses the logged-in `gh` CLI (GraphQL); costs roughly one query per candidate repo.
"""
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "repos.json"
QUOTAS = {  # search query -> how many repos to keep
    "language:Groovy": 20, "org:jenkinsci language:Java": 15, "topic:jenkins-shared-library": 8,
    "language:Java": 20, "language:Kotlin": 6, "language:Python": 22, "language:HCL": 14,
    "language:Go": 12, "language:Shell": 6, "language:TypeScript": 12,
}
BOTS = ("dependabot", "renovate", "github-actions", "snyk", "greenkeeper", "pre-commit-ci", "mergify", "copilot")


def gql(query, **vars):
    args = ["gh", "api", "graphql", "-f", f"query={query}"]
    for k, v in vars.items():
        args += ["-F" if isinstance(v, int) else "-f", f"{k}={v}"]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[:300])
    return json.loads(r.stdout)["data"]


SEARCH = """query($q: String!, $n: Int!) { search(query: $q, type: REPOSITORY, first: $n) { nodes { ... on Repository {
  nameWithOwner stargazerCount isFork isArchived primaryLanguage { name } pushedAt
  pullRequests(states: MERGED) { totalCount } } } } }"""
SAMPLE = """query($owner: String!, $name: String!) { repository(owner: $owner, name: $name) {
  pullRequests(states: MERGED, last: 40) { nodes { author { login } reviews(first: 20) { nodes { author { login } state } } } } } }"""


def is_bot(login):
    return not login or login.endswith("[bot]") or any(b in login.lower() for b in BOTS)


def review_rate(nwo):
    owner, name = nwo.split("/")
    prs = gql(SAMPLE, owner=owner, name=name)["repository"]["pullRequests"]["nodes"]
    human = [p for p in prs if not is_bot((p["author"] or {}).get("login"))]
    if len(human) < 10:
        return 0.0, len(human)
    reviewed = sum(any(r["author"] and r["author"]["login"] != (p["author"] or {}).get("login") and not is_bot(r["author"]["login"])
                       for r in p["reviews"]["nodes"]) for p in human)
    return reviewed / len(human), len(human)


def main():
    picked, seen = [], set()
    for q, n in QUOTAS.items():
        query = f"{q} stars:>150 pushed:>2026-04-01 archived:false fork:false sort:stars"
        cands = [c for c in gql(SEARCH, q=query, n=60)["search"]["nodes"] if c and c["nameWithOwner"] not in seen
                 and c["pullRequests"]["totalCount"] >= 150]
        kept = 0
        for c in cands:
            if kept >= n:
                break
            try:
                rate, h = review_rate(c["nameWithOwner"])
            except RuntimeError as e:
                print("skip", c["nameWithOwner"], e, file=sys.stderr)
                continue
            if rate >= 0.4:
                seen.add(c["nameWithOwner"])
                picked.append({"repo": c["nameWithOwner"], "query": q, "language": (c["primaryLanguage"] or {}).get("name"),
                               "stars": c["stargazerCount"], "merged_prs": c["pullRequests"]["totalCount"],
                               "human_review_rate": round(rate, 2)})
                kept += 1
        print(f"{q:32s} kept {kept}/{n} from {len(cands)} candidates", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(picked, indent=1))
    print(f"total {len(picked)} repos -> {OUT}")


if __name__ == "__main__":
    main()
