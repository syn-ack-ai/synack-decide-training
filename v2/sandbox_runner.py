"""Sandbox runner for stage 3 of the E4B experiment. It runs on box at 127.0.0.1:8899; reach it through an SSH tunnel.

POST /run with JSON:
  {"setup": "shell that creates fixtures", "command": "shell to run" | "python": "code", "check": "shell that verifies",
   "timeout": 15}

Each request runs in a fresh synack-sandbox container: no network, 1 CPU, 512 MB, 128 pids, read-only root, a private
/work directory, non-root user. It returns stdout, stderr and the exit code of the command, plus the check's output and
whether the check passed. At most 4 containers run at once.
"""
import json
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SLOTS = threading.Semaphore(4)
LIMIT = 20000


def run(req):
    work = tempfile.mkdtemp(prefix="sbx-")
    os.chmod(work, 0o777)
    code_dir = tempfile.mkdtemp(prefix="sbxc-")  # the command/program, mounted read-only at /sbx (outside /work)
    os.chmod(code_dir, 0o755)
    name = "sbx-" + uuid.uuid4().hex[:12]
    t = min(int(req.get("timeout", 15)), 60)
    if req.get("python") is not None:
        with open(os.path.join(code_dir, "main.py"), "w") as f:
            f.write(req["python"])
        cmdline = "python3 /sbx/main.py"
    else:  # written to a file so the command reaches bash exactly as given (no extra quoting or expansion)
        with open(os.path.join(code_dir, "cmd.sh"), "w") as f:
            f.write((req.get("command") or "") + "\n")
        cmdline = "bash /sbx/cmd.sh"
    for fn in os.listdir(code_dir):
        os.chmod(os.path.join(code_dir, fn), 0o644)
    setup, check = req.get("setup", ""), req.get("check", "true")
    # setup, command and check each start in /work (a `cd` in one must not leak into the next)
    script = (f"cd /work; ( {setup}\n ) > /tmp/setup.log 2>&1\n"
              f"cd /work; echo __CMD_START__; ( timeout {t} {cmdline} ) > /tmp/out 2> /tmp/err; echo $? > /tmp/rc; echo __CMD_END__\n"
              f"cat /tmp/out; echo __OUT_END__; cat /tmp/err; echo __ERR_END__; cat /tmp/rc; echo __RC_END__\n"
              f"cd /work; ( {check}\n ) 2>&1; echo __CHECK_RC__$?")
    cmd = ["docker", "run", "--rm", "--name", name, "--network", "none", "--memory", "512m", "--memory-swap", "512m",
           "--cpus", "1", "--pids-limit", "128", "--read-only", "--tmpfs", "/tmp:rw,size=64m",
           "--tmpfs", "/home/runner:rw,size=16m,uid=1000,gid=1000,mode=0700", "-v", f"{work}:/work", "-v", f"{code_dir}:/sbx:ro", "-w", "/work", "--user", "1000:1000",
           "synack-sandbox", "bash", "-c", script]
    timed_out = False
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=t + 20)
        o = p.stdout
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        o, timed_out = "", True
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(code_dir, ignore_errors=True)

    def between(a, b):
        return o.split(a, 1)[1].split(b, 1)[0].strip("\n")[:LIMIT] if a in o and b in o else ""

    rc = between("__ERR_END__", "__RC_END__")
    tail = o.split("__RC_END__", 1)[1] if "__RC_END__" in o else ""
    check_out, _, check_rc = tail.rpartition("__CHECK_RC__")
    return {"stdout": between("__CMD_END__", "__OUT_END__"), "stderr": between("__OUT_END__", "__ERR_END__"),
            "exit_code": int(rc) if rc.strip().lstrip("-").isdigit() else -1,
            "timed_out": timed_out or rc.strip() == "124",
            "check_output": check_out.strip()[:LIMIT], "check_passed": check_rc.strip() == "0"}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/run":
            self.send_response(404)
            self.end_headers()
            return
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with SLOTS:
            res = run(req)
        b = json.dumps(res).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8899), Handler).serve_forever()
