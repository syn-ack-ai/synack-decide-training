"""Score collected PRs with the deterministic complexity script and extract compact diff summaries.

For each repo: blobless, no-checkout clone; for each PR fetch its head (refs/pull/N/head) and base,
take merge-base(base, head), run calculate_pr_complexity.py --base MB --head HEAD --output json, and
store the score JSON plus a size-capped unified diff. Parallel over repos. Resumable per PR.
Usage (on the worker): python score_prs.py --prs data/prs --out data/scored --workers 8
"""
import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCORER = ROOT / "vendor/lib/calculate_pr_complexity.py"
PY = ROOT / ".venv/bin/python"
DIFF_CAP = 12_000  # characters of unified diff kept per PR (model input is truncated further later)


def run(args, cwd, timeout=300):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def score_repo(pr_file, out_dir, repo_dir):
    rows = [json.loads(l) for l in open(pr_file)]
    if not rows:
        return pr_file.name, 0, 0
    repo = rows[0]["repo"]
    dest = Path(repo_dir) / repo.replace("/", "__")
    out = Path(out_dir) / pr_file.name
    done = set()
    if out.exists():
        for line in open(out):
            try:
                done.add(json.loads(line)["number"])
            except (json.JSONDecodeError, KeyError):
                pass  # half-written last line from an interrupted run; that PR is redone
    if not dest.exists():
        # Full clone (all blobs local) plus every PR head in one fetch: diffs and scoring then never hit the network.
        r = run(["git", "clone", "-q", "--no-checkout", f"https://github.com/{repo}.git", str(dest)], cwd=repo_dir, timeout=3600)
        if r.returncode:
            return pr_file.name, 0, len(rows)
        run(["git", "fetch", "-q", "origin", "+refs/pull/*/head:refs/pull/*/head"], cwd=dest, timeout=3600)
    env = {**os.environ, "GITHUB_API_URL": "https://api.github.com", "PYTHONPATH": str(ROOT / "vendor")}
    ok = fail = 0
    with open(out, "a") as g:
        for pr in rows:
            if pr["number"] in done:
                continue
            rec = {"repo": repo, "number": pr["number"]}
            try:
                have = run(["git", "cat-file", "-e", pr["headRefOid"]], cwd=dest).returncode == 0 and \
                    run(["git", "cat-file", "-e", pr["baseRefOid"]], cwd=dest).returncode == 0
                if not have:
                    run(["git", "fetch", "-q", "origin", f"refs/pull/{pr['number']}/head", pr["baseRefOid"]], cwd=dest, timeout=600)
                mb = run(["git", "merge-base", pr["baseRefOid"], pr["headRefOid"]], cwd=dest).stdout.strip()
                if not mb:
                    raise RuntimeError("no merge-base")
                s = subprocess.run([str(PY), str(SCORER), "--base", mb, "--head", pr["headRefOid"], "--output", "json"],
                                   cwd=dest, capture_output=True, text=True, timeout=600, env=env)
                rec["score_json"] = json.loads(s.stdout)
                d = run(["git", "diff", "--no-color", "-U2", mb, pr["headRefOid"]], cwd=dest, timeout=600).stdout
                rec["diff"] = d[:DIFF_CAP]
                rec["diff_chars"] = len(d)
                rec["merge_base"] = mb
                ok += 1
            except Exception as e:  # noqa: BLE001 - record and continue
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                fail += 1
            g.write(json.dumps(rec) + "\n")
            g.flush()
    return pr_file.name, ok, fail


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prs", default=str(ROOT / "data/prs"))
    p.add_argument("--out", default=str(ROOT / "data/scored"))
    p.add_argument("--repos", default=str(ROOT / "repos"))
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.repos).mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.prs).glob("*.jsonl"))
    with ProcessPoolExecutor(a.workers) as ex:
        for name, ok, fail in ex.map(score_repo, files, [a.out] * len(files), [a.repos] * len(files)):
            print(f"{name}: scored {ok}, failed {fail}", flush=True)


if __name__ == "__main__":
    main()
