"""Load-on-demand front for llama-server: holds no GPU memory until a request arrives.

Listens on HOST:PORT (default 0.0.0.0:8080). The first request starts llama-server on 127.0.0.1:BACKEND_PORT via
START_CMD and waits for /health; requests are then proxied unchanged (so /v1/systemone works as before). After
IDLE_SECONDS without traffic the backend is stopped and the GPU is free again. GET /ondemand/status answers locally
without loading anything.

Several models: MODELS='{"v7b": "/path/a.gguf", "v5": "/path/b.gguf"}' and DEFAULT_MODEL. A path prefix picks the
model (/v5/v1/systemone -> v5); no prefix uses the default. Asking for a different model swaps the backend (one GPU).
Requests are handled one at a time (llama-server runs --parallel 1 anyway), so a swap never cuts a request.

  python ondemand.py            env: HOST, PORT, BACKEND_PORT, IDLE_SECONDS, START_CMD, LOAD_TIMEOUT, MODELS, DEFAULT_MODEL
"""
import http.client
import json
import os
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", "8081"))
IDLE_SECONDS = int(os.environ.get("IDLE_SECONDS", "600"))
LOAD_TIMEOUT = int(os.environ.get("LOAD_TIMEOUT", "300"))
START_CMD = os.environ.get("START_CMD", os.path.expanduser("~/synack/start-server.sh"))
MODELS = json.loads(os.environ.get("MODELS", "{}"))
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL") or (next(iter(MODELS)) if MODELS else None)
HOP = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "upgrade", "te", "trailer", "host"}

lock = threading.Lock()
request_lock = threading.Lock()
proc = None
loaded_model = None
last_used = 0.0
active = 0


def backend_up():
    try:
        c = http.client.HTTPConnection("127.0.0.1", BACKEND_PORT, timeout=2)
        c.request("GET", "/health")
        return c.getresponse().status == 200
    except OSError:
        return False


def ensure_backend(model=None):
    global proc, loaded_model
    if model != loaded_model and proc is not None and proc.poll() is None:
        stop_backend(f"switching {loaded_model} -> {model}")
    with lock:
        if proc is not None and proc.poll() is None and backend_up():
            return True
        if proc is None or proc.poll() is not None:
            env = dict(os.environ, HOST="127.0.0.1", PORT=str(BACKEND_PORT))
            if model:
                env["MODEL"] = MODELS[model]
            loaded_model = model
            proc = subprocess.Popen([START_CMD], env=env, start_new_session=True)
            print(f"[ondemand] starting backend pid {proc.pid} ({model or 'default model'})", flush=True)
        t0 = time.time()
        while time.time() - t0 < LOAD_TIMEOUT:
            if proc.poll() is not None:
                print(f"[ondemand] backend exited with {proc.returncode}", flush=True)
                return False
            if backend_up():
                print(f"[ondemand] backend ready in {time.time() - t0:.1f} s", flush=True)
                return True
            time.sleep(0.5)
        return False


def stop_backend(reason):
    global proc
    with lock:
        if proc is not None and proc.poll() is None:
            print(f"[ondemand] stopping backend ({reason})", flush=True)
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        proc = None


def idle_reaper():
    while True:
        time.sleep(15)
        if proc is not None and active == 0 and time.time() - last_used > IDLE_SECONDS:
            stop_backend(f"idle {IDLE_SECONDS}s")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _reply(self, status, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _proxy(self):
        global last_used, active
        if self.path == "/ondemand/status":
            loaded = proc is not None and proc.poll() is None
            return self._reply(200, {"loaded": loaded, "model": loaded_model if loaded else None,
                                     "models": list(MODELS), "default_model": DEFAULT_MODEL, "idle_seconds": IDLE_SECONDS,
                                     "seconds_since_last_request": round(time.time() - last_used) if last_used else None})
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        model, path = DEFAULT_MODEL, self.path
        first = self.path.split("/")[1] if self.path.count("/") > 1 else ""
        if first in MODELS:
            model, path = first, self.path[len(first) + 1:]
        active += 1
        request_lock.acquire()
        try:
            if not ensure_backend(model):
                return self._reply(503, {"error": "model failed to load; see journalctl --user -u synack-decide-ondemand"})
            c = http.client.HTTPConnection("127.0.0.1", BACKEND_PORT, timeout=LOAD_TIMEOUT)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            c.request(self.command, path, body=body, headers=headers)
            r = c.getresponse()
            data = r.read()
            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() not in HOP and k.lower() != "content-length":
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError as e:
            self._reply(502, {"error": f"backend error: {e}"})
        finally:
            request_lock.release()
            active -= 1
            last_used = time.time()

    do_GET = do_POST = do_PUT = do_DELETE = _proxy


if __name__ == "__main__":
    threading.Thread(target=idle_reaper, daemon=True).start()
    print(f"[ondemand] listening on {HOST}:{PORT}; backend 127.0.0.1:{BACKEND_PORT}; idle unload after {IDLE_SECONDS}s", flush=True)
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        srv.serve_forever()
    finally:
        stop_backend("shutdown")
