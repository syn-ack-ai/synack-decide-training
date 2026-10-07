"""Join collected PRs with their complexity scores, derive review-outcome labels, build v2 decision records.

Outcome label "needs_review" (did the PR need substantive changes during review?):
  positive: a human CHANGES_REQUESTED review, or (commits pushed after the first human review AND >= 2
            review threads), or >= 4 review threads, or the PR was later reverted.
  negative: a human approved it, at most 1 review thread, and no commits after the first human review.
  otherwise ambiguous -> dropped. Bot-authored PRs and PRs with no human review are dropped.
Model input contains only pre-review information (title, short description, the script's JSON, a diff
excerpt); nothing about the review itself. Splits are by repository so evaluation measures transfer
to unseen codebases. Also writes features.csv for script / logistic-regression baselines.
"""
import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOTS = ("dependabot", "renovate", "github-actions", "snyk", "greenkeeper", "pre-commit-ci", "mergify", "copilot",
        "codecov", "sonarcloud", "coderabbit")
QUESTION = ("Will this pull request need substantive changes during code review before it is ready to merge? "
            "Judge from the change itself: size, risk signals, files touched and the diff.")
CRITERIA = {"yes": "Yes: reviewers are likely to request changes or several rounds of fixes.",
            "no": "No: it is likely to be approved with little or no rework."}
DIFF_CHARS = 3500


def is_bot(login):
    return not login or login.endswith("[bot]") or any(b in login.lower() for b in BOTS)


def outcome(pr, reverted):
    author = (pr["author"] or {}).get("login")
    if is_bot(author):
        return None, "bot_author"
    human = [r for r in pr["reviews"]["nodes"] if r["author"] and r["author"]["login"] != author
             and not is_bot(r["author"]["login"])]
    if not human:
        return None, "no_human_review"
    first = min(r["submittedAt"] for r in human if r["submittedAt"])
    rework = sum(1 for c in pr["commits"]["nodes"] if c["commit"]["committedDate"] > first)
    threads = pr["reviewThreads"]["totalCount"]
    changes_req = any(r["state"] == "CHANGES_REQUESTED" for r in human)
    approved = any(r["state"] == "APPROVED" for r in human)
    feats = {"changes_requested": int(changes_req), "rework_commits": rework, "review_threads": threads,
             "reverted": int(reverted)}
    if changes_req or (rework >= 1 and threads >= 2) or threads >= 4 or reverted:
        return "yes", feats
    if approved and threads <= 1 and rework == 0:
        return "no", feats
    return None, "ambiguous"


def script_view(sj):
    """Compact, model-facing view of the scorer's JSON (no review information)."""
    m = sj.get("metrics", {})
    files = sorted(sj.get("files", []), key=lambda f: -f.get("weighted_impact", 0))[:12]
    return {"score": sj.get("score"), "size": m.get("size_label"), "lines_added": m.get("lines_added"),
            "lines_deleted": m.get("lines_deleted"), "files_changed": m.get("files_changed"),
            "weighted_impact": m.get("weighted_impact"), "signals": sj.get("signals"), "flags": sj.get("flags"),
            "routing_tier": (sj.get("routing") or {}).get("tier"), "reasoning": sj.get("reasoning"),
            "top_files": [{"path": f["path"], "status": f.get("status"), "category": f.get("category"),
                           "+": f.get("additions"), "-": f.get("deletions")} for f in files]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prs", default=str(ROOT / "data/prs"))
    p.add_argument("--scored", default=str(ROOT / "data/scored"))
    p.add_argument("--out", default=str(ROOT / "data/dataset"))
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--valid-frac", type=float, default=0.10)
    p.add_argument("--diff-chars", type=int, default=DIFF_CHARS)
    p.add_argument("--slim", action="store_true", help="smaller script view: top 6 files, no reasoning text")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scored = {}
    for f in Path(a.scored).glob("*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if "score_json" in r and "diff" in r and "error" not in r:
                scored[(r["repo"], r["number"])] = r
    prs = [json.loads(l) for f in Path(a.prs).glob("*.jsonl") for l in open(f)]
    by_repo = defaultdict(list)
    for pr in prs:
        by_repo[pr["repo"]].append(pr)
    reverted = set()
    for repo, items in by_repo.items():
        titles = {pr["title"]: pr["number"] for pr in items}
        for pr in items:
            t = pr["title"]
            if t.lower().startswith("revert"):
                m = re.search(r"#(\d+)", t + " " + (pr["body"] or "")[:300])
                if m:
                    reverted.add((repo, int(m.group(1))))
                q = re.match(r'revert\s+"(.+)"', t, re.I)
                if q and q.group(1) in titles:
                    reverted.add((repo, titles[q.group(1)]))
    repos = sorted(by_repo)
    random.Random(20261003).shuffle(repos)
    n_test, n_val = int(len(repos) * a.test_frac), int(len(repos) * a.valid_frac)
    split_of = {r: "test" for r in repos[:n_test]}
    split_of.update({r: "valid" for r in repos[n_test:n_test + n_val]})
    drops, labels = Counter(), Counter()
    files = {s: open(out / f"{s}_records.jsonl", "w") for s in ("train", "valid", "test")}
    fcsv = open(out / "features.csv", "w", newline="")
    w = csv.writer(fcsv)
    w.writerow(["repo", "number", "split", "label", "score", "lines_added", "lines_deleted", "files_changed",
                "weighted_impact", "n_signals", "routing_standard", "changes_requested", "rework_commits",
                "review_threads", "reverted"])
    for pr in prs:
        key = (pr["repo"], pr["number"])
        lab, info = outcome(pr, key in reverted)
        if lab is None:
            drops[info] += 1
            continue
        s = scored.get(key)
        if not s:
            drops["not_scored"] += 1
            continue
        split = split_of.get(pr["repo"], "train")
        labels[(split, lab)] += 1
        view = script_view(s["score_json"])
        if a.slim:
            view = {k: v for k, v in view.items() if k != "reasoning"}
            view["top_files"] = view["top_files"][:6]
        state = {"title": pr["title"][:200], "description": (pr["body"] or "")[:400],
                 "language": pr.get("repo_language"), "complexity_script": view, "diff_excerpt": s["diff"][:a.diff_chars],
                 "diff_truncated": s["diff_chars"] > a.diff_chars}
        rec = {"id": f"prcx:{pr['repo']}#{pr['number']}", "source": f"prcx/{pr.get('repo_language')}", "state": state,
               "questions": {"needs_review": {"type": "choice", "instructions": QUESTION, "criteria": CRITERIA}},
               "expected": {"needs_review": lab}}
        files[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
        sig = view["signals"]
        n_sig = sum(1 for v in (sig.values() if isinstance(sig, dict) else sig or []) if v)
        w.writerow([pr["repo"], pr["number"], split, lab, view["score"], view["lines_added"], view["lines_deleted"],
                    view["files_changed"], view["weighted_impact"], n_sig, int(view["routing_tier"] == "standard"),
                    info["changes_requested"], info["rework_commits"], info["review_threads"], info["reverted"]])
    for f in files.values():
        f.close()
    fcsv.close()
    print(json.dumps({"prs": len(prs), "scored": len(scored), "dropped": dict(drops),
                      "labels": {f"{s}/{l}": n for (s, l), n in sorted(labels.items())},
                      "repos": {"train": len(repos) - n_test - n_val, "valid": n_val, "test": n_test}}, indent=1))


if __name__ == "__main__":
    main()
