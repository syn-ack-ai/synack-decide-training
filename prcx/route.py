"""Route a pull request: complexity script + systemone decision model -> P(needs substantive review).

Runs fully locally. For a git repository and a base/head pair it
  1. runs the deterministic complexity scorer (vendor/lib/calculate_pr_complexity.py),
  2. builds exactly the model input used in training (title, description, the scorer's JSON view,
     a 3,500-character diff excerpt),
  3. reads the model's calibrated probability for "yes, it needs substantive changes",
and recommends a reviewer tier: p >= --threshold -> "standard" (capable model), else "value" (cheap model).
The default threshold 0.27 routes the same share of PRs cheap as the script's 0.30 rule on held-out
public repos (where the model missed 28.3% of needs-review PRs vs the script's 32.7%).

  route.py --repo ~/code/service --base main --head HEAD
  route.py --serve 8766            # POST /route {"repo", "base", "head", "title"?, "description"?}
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "v2"))
from build_dataset import CRITERIA, DIFF_CHARS, QUESTION, script_view  # noqa: E402

SCORER = ROOT / "vendor/lib/calculate_pr_complexity.py"
SCORER_PY = ROOT / ".venv/bin/python"  # has pydriller; the scorer runs fine without it, one signal short
DEFAULT_MODEL = ROOT.parent / "models/systemone-v3-mlx8"
EXT_LANG = {".java": "Java", ".kt": "Kotlin", ".groovy": "Groovy", ".gradle": "Groovy", ".py": "Python",
            ".go": "Go", ".ts": "TypeScript", ".tsx": "TypeScript", ".js": "TypeScript", ".tf": "HCL",
            ".hcl": "HCL", ".sh": "Shell", ".bash": "Shell"}


def git(repo, *args):
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[:200]}")
    return r.stdout


def default_base(repo):
    for ref in ("origin/main", "origin/master", "main", "master"):
        if subprocess.run(["git", "rev-parse", "--verify", "-q", ref], cwd=repo, capture_output=True).returncode == 0:
            return ref
    raise RuntimeError("no base given and no main/master branch found")


def build_record(repo, base, head, title=None, description=None):
    repo = str(Path(repo).expanduser().resolve())
    base = base or default_base(repo)
    mb = git(repo, "merge-base", base, head).strip()
    head_sha = git(repo, "rev-parse", head).strip()
    env = {**os.environ, "PYTHONPATH": str(ROOT / "vendor"), "GITHUB_API_URL": "https://api.github.com"}
    py = str(SCORER_PY) if SCORER_PY.exists() else sys.executable
    s = subprocess.run([py, str(SCORER), "--base", mb, "--head", head_sha, "--output", "json"],
                       cwd=repo, capture_output=True, text=True, env=env, timeout=600)
    if s.returncode or not s.stdout.strip():
        raise RuntimeError(f"complexity scorer failed (exit {s.returncode}): {s.stderr.strip()[:300]}")
    sj = json.loads(s.stdout)
    diff = git(repo, "diff", "--no-color", "-U2", mb, head_sha)
    if title is None:
        title = git(repo, "log", "-1", "--format=%s", head_sha).strip()
    if description is None:
        description = git(repo, "log", "-1", "--format=%b", head_sha).strip()
    exts = collections.Counter(Path(f["path"]).suffix.lower() for f in sj.get("files", []))
    lang = next((EXT_LANG[e] for e, _ in exts.most_common() if e in EXT_LANG), None)
    state = {"title": title[:200], "description": (description or "")[:400], "language": lang,
             "complexity_script": script_view(sj), "diff_excerpt": diff[:DIFF_CHARS], "diff_truncated": len(diff) > DIFF_CHARS}
    q = {"type": "choice", "instructions": QUESTION, "criteria": CRITERIA}
    return state, q, sj, {"base": base, "merge_base": mb, "head": head_sha}


class Router:
    def __init__(self, model):
        from engine_mlx import label_probs, load_mlx
        t = time.time()
        self.tok, self.model, self.ids = load_mlx(str(model))
        self.label_probs = label_probs
        print(f"model loaded in {time.time() - t:.1f}s: {model}", file=sys.stderr, flush=True)

    def route(self, repo, base=None, head="HEAD", title=None, description=None, threshold=0.27):
        t0 = time.time()
        state, q, sj, refs = build_record(repo, base, head, title, description)
        t1 = time.time()
        probs, n_tok = self.label_probs(self.tok, self.model, self.ids, state, q)
        p = probs["yes"]
        return {"p_needs_review": round(p, 4), "recommended_tier": "standard" if p >= threshold else "value",
                "threshold": threshold, "script_score": sj.get("score"),
                "script_tier": (sj.get("routing") or {}).get("tier"), "script_reasoning": sj.get("reasoning"),
                "prompt_tokens": n_tok, "timing_ms": {"script_and_diff": round((t1 - t0) * 1000),
                                                      "model": round((time.time() - t1) * 1000)}, **refs}


def serve(router, port, threshold):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/route":
                self.send_error(404)
                return
            try:
                b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                out = router.route(b["repo"], b.get("base"), b.get("head", "HEAD"), b.get("title"),
                                   b.get("description"), float(b.get("threshold", threshold)))
                code = 200
            except Exception as e:  # noqa: BLE001 - report to the caller
                out, code = {"error": f"{type(e).__name__}: {e}"}, 400
            data = json.dumps(out).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    print(f"routing on http://127.0.0.1:{port}/route", file=sys.stderr, flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", default=".")
    p.add_argument("--base")
    p.add_argument("--head", default="HEAD")
    p.add_argument("--title")
    p.add_argument("--description")
    p.add_argument("--threshold", type=float, default=0.27)
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--serve", type=int, help="run an HTTP server on this port instead of a one-shot route")
    a = p.parse_args()
    router = Router(a.model)
    if a.serve:
        serve(router, a.serve, a.threshold)
    else:
        print(json.dumps(router.route(a.repo, a.base, a.head, a.title, a.description, a.threshold), indent=2))


if __name__ == "__main__":
    main()
