"""Local System One decision server and CLI for the v2/v3 models (MLX, Apple Silicon).

Server (Jev / TypeSafe wire format, same as the Decision Index harness and llama.cpp /v1/systemone):
  serve_local.py --serve 8765
  POST /v1/systemone {"state": ..., "questions": {"id": {"type": "choice", "instructions": "...",
                      "criteria": {"opt": "description" | null, ...}} | {"type": "noul", "instructions": "..."}}}
  -> {"model", "answers": {"id": {"type": "choice", "choice", "probabilities", "confidence"} | {"type": "noul", "noul"}},
      "usage": {"input_tokens"}, "latency_ms"}

One-off CLI:
  serve_local.py --state "Customer: I was charged twice." --question "Which team?" --options billing,shipping,technical
  serve_local.py --state "..." --yesno "Is the customer angry?"
"""
import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS  # noqa: E402
from engine_mlx import label_probs, load_mlx  # noqa: E402

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models/systemone-v3-mlx8"


class Decider:
    def __init__(self, model):
        t = time.time()
        self.name = Path(model).name
        self.tok, self.model, self.ids = load_mlx(str(model))
        self.lock = threading.Lock()
        print(f"loaded {self.name} in {time.time() - t:.1f}s", file=sys.stderr, flush=True)

    def answer(self, state, questions):
        if not isinstance(questions, dict) or not questions:
            raise ValueError('"questions" must be a non-empty object')
        out, n_tok, t0 = {}, 0, time.time()
        with self.lock:
            for qid, q in questions.items():
                if q.get("type") not in ("choice", "noul"):
                    raise ValueError(f'question "{qid}": type must be "choice" or "noul"')
                if q["type"] == "choice" and not (2 <= len(q.get("criteria") or {}) <= len(LABELS)):
                    raise ValueError(f'question "{qid}": choice needs 2-{len(LABELS)} criteria')
                probs, n = label_probs(self.tok, self.model, self.ids, state, q)
                n_tok += n
                if q["type"] == "noul":
                    out[qid] = {"type": "noul", "noul": round(probs["true"], 6)}
                else:
                    best = max(probs, key=probs.get)
                    out[qid] = {"type": "choice", "choice": best, "confidence": round(probs[best], 6),
                                "probabilities": {k: round(v, 6) for k, v in probs.items()}}
        return {"model": self.name, "answers": out, "usage": {"input_tokens": n_tok},
                "latency_ms": round((time.time() - t0) * 1000, 1)}


def serve(decider, port):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path not in ("/v1/systemone", "/systemone"):
                return self._send(404, {"error": "not found; POST /v1/systemone"})
            try:
                b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._send(200, decider.answer(b.get("state", ""), b.get("questions")))
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})

        def log_message(self, *args):
            pass

    print(f"System One endpoint: http://127.0.0.1:{port}/v1/systemone", file=sys.stderr, flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--serve", type=int, help="port for the /v1/systemone server")
    p.add_argument("--state", default="")
    p.add_argument("--question", help="choice question (use with --options)")
    p.add_argument("--options", help="comma-separated options, or a JSON object {option: description}")
    p.add_argument("--yesno", help="yes/no question")
    a = p.parse_args()
    d = Decider(a.model)
    if a.serve:
        return serve(d, a.serve)
    qs = {}
    if a.question:
        if not a.options:
            p.error("--question needs --options")
        crit = json.loads(a.options) if a.options.strip().startswith("{") else {o.strip(): None for o in a.options.split(",")}
        qs["q"] = {"type": "choice", "instructions": a.question, "criteria": crit}
    if a.yesno:
        qs["yesno"] = {"type": "noul", "instructions": a.yesno}
    if not qs:
        p.error("give --question/--options and/or --yesno, or --serve PORT")
    print(json.dumps(d.answer(a.state, qs), indent=2))


if __name__ == "__main__":
    main()
