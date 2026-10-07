#!/usr/bin/env python3
"""Compare the complexity script with the systemone model on your own PRs (standard library only).

For each PR it runs YOUR scorer locally (calculate_pr_complexity.py), collects title, description and
diff from the local clone, asks the model server, and writes one CSV row:
  repo, pr, base, head, script_score, script_tier, p_needs_review, model_tier, prompt_tokens, latency_ms, error
Join the CSV with your review outcomes to compare predictors.

PR list, either:
  --prs prs.csv        columns: repo_path,pr,base,head,title,description   (title/description optional)
  --gh owner/repo --repo-path ~/code/repo --limit 200   (merged PRs via the `gh` CLI; heads are fetched)

  python3 pr_compare_client.py --scorer ~/engine/codefusion/lib/calculate_pr_complexity.py \
     --server http://SERVER_HOST:8766 --token-file ~/systemone_token --gh org/repo --repo-path ~/code/repo --out cmp.csv
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


def git(repo, *args, check=True):
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"git {' '.join(args[:3])}: {r.stderr.strip()[:200]}")
    return r.stdout


def gh_prs(nwo, repo, limit):
    out = subprocess.run(["gh", "pr", "list", "-R", nwo, "--state", "merged", "--limit", str(limit), "--json",
                          "number,title,body,baseRefOid,headRefOid"], capture_output=True, text=True, check=True).stdout
    rows = []
    for p in json.loads(out):
        git(repo, "fetch", "-q", "origin", f"refs/pull/{p['number']}/head", check=False)
        rows.append({"repo_path": repo, "pr": p["number"], "base": p["baseRefOid"], "head": p["headRefOid"],
                     "title": p["title"], "description": p.get("body") or ""})
    return rows


def score(scorer, repo, mb, head):
    env = {**os.environ, "PYTHONPATH": str(Path(scorer).resolve().parents[1])}
    r = subprocess.run([sys.executable, scorer, "--base", mb, "--head", head, "--output", "json"],
                       cwd=repo, capture_output=True, text=True, env=env, timeout=600)
    if r.returncode or not r.stdout.strip():
        raise RuntimeError(f"scorer exit {r.returncode}: {r.stderr.strip()[:200]}")
    return json.loads(r.stdout)


def ask(server, token, body):
    req = urllib.request.Request(server.rstrip("/") + "/v1/pr/route", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scorer", required=True, help="path to calculate_pr_complexity.py")
    p.add_argument("--server", required=True)
    p.add_argument("--token-file", required=True)
    p.add_argument("--prs")
    p.add_argument("--gh")
    p.add_argument("--repo-path")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--out", default="comparison.csv")
    a = p.parse_args()
    token = Path(a.token_file).expanduser().read_text().strip()
    if a.prs:
        rows = list(csv.DictReader(open(a.prs)))
    elif a.gh and a.repo_path:
        rows = gh_prs(a.gh, str(Path(a.repo_path).expanduser()), a.limit)
    else:
        p.error("give --prs, or --gh with --repo-path")
    fields = ["repo", "pr", "base", "head", "script_score", "script_tier", "p_needs_review", "model_tier",
              "prompt_tokens", "latency_ms", "error"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            repo = str(Path(r["repo_path"]).expanduser())
            row = {"repo": Path(repo).name, "pr": r.get("pr"), "base": r["base"], "head": r["head"]}
            try:
                mb = git(repo, "merge-base", r["base"], r["head"]).strip()
                sj = score(a.scorer, repo, mb, r["head"])
                if not r.get("title"):
                    r["title"] = git(repo, "log", "-1", "--format=%s", r["head"]).strip()
                diff = git(repo, "diff", "--no-color", "-U2", mb, r["head"])
                res = ask(a.server, token, {"title": r["title"], "description": r.get("description") or "",
                                            "script_json": sj, "diff": diff[:20000]})
                row.update(script_score=res["script_score"], script_tier=res["script_tier"],
                           p_needs_review=res["p_needs_review"], model_tier=res["recommended_tier"],
                           prompt_tokens=res["prompt_tokens"], latency_ms=res["latency_ms"])
            except Exception as e:  # noqa: BLE001 - record and continue
                row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            w.writerow(row)
            f.flush()
            print(f"[{i}/{len(rows)}] {row['repo']}#{row['pr']}: script {row.get('script_score')} "
                  f"model {row.get('p_needs_review')} {row.get('error') or ''}", flush=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
