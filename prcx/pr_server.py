"""LAN decision server for PR routing comparisons (MLX, model loaded once).

Endpoints (all POST bodies JSON; send header "Authorization: Bearer <token>"):
  GET  /health
  POST /v1/pr/route        {"title", "description"?, "script_json", "diff", "language"?, "threshold"?}
  POST /v1/pr/route_batch  {"prs": [ ...same objects..., optionally with "id" ], "threshold"?}
  POST /v1/systemone       {"state", "questions"}   (Jev / TypeSafe wire format)
The server rebuilds the exact training input from the raw pieces (script view, 400-char description,
3,500-char diff excerpt), so clients cannot drift from the training format.
Privacy: request bodies are never logged or stored; only counts and timings are printed.

  pr_server.py --host 0.0.0.0 --port 8766 --token-file ~/.config/systemone/server_token
"""
import argparse
import collections
import json
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "v2"))
from build_dataset import CRITERIA, DIFF_CHARS, QUESTION, script_view  # noqa: E402
from common import LABELS  # noqa: E402
from engine_mlx import label_probs, load_mlx  # noqa: E402

DEFAULT_MODEL = ROOT.parent / "models/systemone-v3-mlx8"
EXT_LANG = {".java": "Java", ".kt": "Kotlin", ".groovy": "Groovy", ".gradle": "Groovy", ".py": "Python", ".go": "Go",
            ".ts": "TypeScript", ".tsx": "TypeScript", ".js": "TypeScript", ".tf": "HCL", ".hcl": "HCL", ".sh": "Shell"}
MAX_BODY = 50 * 1024 * 1024


class Model:
    def __init__(self, path):
        t = time.time()
        self.name = Path(path).name
        self.tok, self.model, self.ids = load_mlx(str(path))
        self.lock = threading.Lock()
        self.stats = collections.Counter()
        print(f"loaded {self.name} in {time.time() - t:.1f}s", flush=True)

    def probs(self, state, q):
        with self.lock:
            return label_probs(self.tok, self.model, self.ids, state, q)

    def route(self, pr, threshold):
        sj = pr["script_json"]
        if not isinstance(sj, dict) or "score" not in sj:
            raise ValueError('"script_json" must be the scorer\'s JSON output (with "score")')
        diff = pr.get("diff") or ""
        lang = pr.get("language")
        if not lang:
            exts = collections.Counter(Path(f.get("path", "")).suffix.lower() for f in sj.get("files", []))
            lang = next((EXT_LANG[e] for e, _ in exts.most_common() if e in EXT_LANG), None)
        state = {"title": (pr.get("title") or "")[:200], "description": (pr.get("description") or "")[:400],
                 "language": lang, "complexity_script": script_view(sj), "diff_excerpt": diff[:DIFF_CHARS],
                 "diff_truncated": len(diff) > DIFF_CHARS}
        t = time.time()
        p, n = self.probs(state, {"type": "choice", "instructions": QUESTION, "criteria": CRITERIA})
        self.stats["pr"] += 1
        out = {"p_needs_review": round(p["yes"], 4), "recommended_tier": "standard" if p["yes"] >= threshold else "value",
               "threshold": threshold, "script_score": sj.get("score"), "script_tier": (sj.get("routing") or {}).get("tier"),
               "prompt_tokens": n, "latency_ms": round((time.time() - t) * 1000, 1)}
        if "id" in pr:
            out["id"] = pr["id"]
        return out

    def systemone(self, state, questions):
        if not isinstance(questions, dict) or not questions:
            raise ValueError('"questions" must be a non-empty object')
        answers, n_tok = {}, 0
        for qid, q in questions.items():
            if q.get("type") not in ("choice", "noul"):
                raise ValueError(f'question "{qid}": type must be "choice" or "noul"')
            if q["type"] == "choice" and not 2 <= len(q.get("criteria") or {}) <= len(LABELS):
                raise ValueError(f'question "{qid}": choice needs 2-{len(LABELS)} criteria')
            p, n = self.probs(state, q)
            n_tok += n
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": round(p["true"], 6)}
            else:
                best = max(p, key=p.get)
                answers[qid] = {"type": "choice", "choice": best, "confidence": round(p[best], 6),
                                "probabilities": {k: round(v, 6) for k, v in p.items()}}
        self.stats["systemone"] += 1
        return {"model": self.name, "answers": answers, "usage": {"input_tokens": n_tok}}


def make_handler(model, token, default_threshold):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self):
            got = self.headers.get("Authorization", "")
            return secrets.compare_digest(got.encode(), f"Bearer {token}".encode())

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True, "model": model.name, "served": dict(model.stats)})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._authorized():
                return self._send(401, {"error": "missing or wrong bearer token"})
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BODY:
                return self._send(413, {"error": "request too large"})
            t0 = time.time()
            try:
                b = json.loads(self.rfile.read(n))
                thr = float(b.get("threshold", default_threshold))
                if self.path == "/v1/pr/route":
                    out = model.route(b, thr)
                elif self.path == "/v1/pr/route_batch":
                    out = {"results": []}
                    for pr in b.get("prs", []):
                        try:
                            out["results"].append(model.route(pr, thr))
                        except (ValueError, KeyError, TypeError) as e:
                            out["results"].append({"id": pr.get("id"), "error": str(e)})
                elif self.path == "/v1/systemone":
                    out = model.systemone(b.get("state", ""), b.get("questions"))
                else:
                    return self._send(404, {"error": "unknown endpoint"})
                self._send(200, out)
                print(f"{time.strftime('%H:%M:%S')} {self.path} ok {(time.time() - t0) * 1000:.0f} ms", flush=True)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})
                print(f"{time.strftime('%H:%M:%S')} {self.path} 400", flush=True)
            except Exception as e:  # noqa: BLE001 - never drop the connection silently
                self._send(500, {"error": f"{type(e).__name__}: {e}"})
                print(f"{time.strftime('%H:%M:%S')} {self.path} 500 {type(e).__name__}", flush=True)

        def log_message(self, *args):
            pass  # never log request lines or bodies

    return H


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--token-file", default=str(Path.home() / ".config/systemone/server_token"))
    p.add_argument("--threshold", type=float, default=0.27)
    a = p.parse_args()
    tf = Path(a.token_file).expanduser()
    if not tf.exists():
        tf.parent.mkdir(parents=True, exist_ok=True)
        tf.write_text(secrets.token_urlsafe(32))
        tf.chmod(0o600)
    token = tf.read_text().strip()
    model = Model(a.model)
    print(f"serving on http://{a.host}:{a.port}  (token in {tf})", flush=True)
    # Single-threaded on purpose: MLX GPU streams belong to the thread that loaded the model, and the model
    # serves one request at a time anyway.
    HTTPServer((a.host, a.port), make_handler(model, token, a.threshold)).serve_forever()


if __name__ == "__main__":
    main()
