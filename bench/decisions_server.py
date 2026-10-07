"""Decisions-style HTTP endpoint over a local MLX decision model.

PROVISIONAL CONTRACT. OpenAI has not published the Decisions API schema (as of
2026-10-02). This mirrors the illustrative shape in the community guide at
huggingface.co/blog/sora-2/what-is-openai-decisions-api-a-practical-guide:

    POST /v1/decisions
    {"model": "...", "context": {...} | "...",
     "question": {"name": "route", "prompt": "Which workflow owns this case?",
                  "answers": ["refund_review", "technical_support"]}}

    -> {"id": "dec_...", "object": "decision", "model": "...", "question": "route",
        "choice": "refund_review", "confidence": 0.97,
        "scores": {"refund_review": 0.97, "technical_support": 0.03}}

Answers may also be objects {"key": ..., "description": ...}. Only `to_record`
and `to_response` know this shape; swap them when the official contract lands.
"""
import argparse
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from decide_mlx import Decider

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWX"


def to_record(body):
    q = body.get("question")
    if not isinstance(q, dict) or not isinstance(q.get("prompt"), str) or not q["prompt"].strip():
        raise ValueError("question.prompt is required")
    answers = q.get("answers")
    if not isinstance(answers, list) or not 2 <= len(answers) <= len(LABELS):
        raise ValueError(f"question.answers must list 2-{len(LABELS)} answers")
    options = []
    for label, a in zip(LABELS, answers):
        key, desc = (a, a) if isinstance(a, str) else (a.get("key"), a.get("description") or a.get("key"))
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Each answer needs a nonempty key")
        options.append({"label": label, "key": key, "description": desc})
    if len({o["key"] for o in options}) != len(options):
        raise ValueError("Answer keys must be unique")
    if "context" not in body:
        raise ValueError("context is required")
    return {"state": body["context"], "question": q["prompt"], "options": options}


def to_response(body, model_name, result):
    return {"id": f"dec_{uuid.uuid4().hex[:24]}", "object": "decision", "model": model_name,
            "question": body["question"].get("name"), "choice": result["key"],
            "confidence": round(result["confidence"], 6),
            "scores": {o["key"]: round(result["probs"][o["label"]], 6) for o in result["options"]},
            "usage": {"prompt_tokens": result["prompt_tokens"], "latency_ms": round(result["latency_ms"], 1)}}


def make_handler(decider, model_name):
    lock = threading.Lock()  # one MLX forward pass at a time

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != "/v1/decisions":
                return self._send(404, {"error": {"message": "Not found"}})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                record = to_record(body)
                with lock:
                    result = decider.decide(record)
                result["options"] = record["options"]
                self._send(200, to_response(body, model_name, result))
            except (ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": {"message": str(e), "type": "invalid_request_error"}})

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model")
    p.add_argument("--adapter")
    p.add_argument("--name", default="local-decider")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    decider = Decider(args.model, args.adapter)
    print(f"Serving {args.name} on http://127.0.0.1:{args.port}/v1/decisions", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(decider, args.name)).serve_forever()


if __name__ == "__main__":
    main()
