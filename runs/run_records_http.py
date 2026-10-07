"""Run decision records (one question each) against a /v1/systemone server and write {id, source, expected, probs}.

  python3 runs/run_records_http.py http://GPU_HOST:8080/v7b/v1/systemone records.jsonl out.jsonl
"""
import json
import sys
import time
import urllib.request

url, inp, out = sys.argv[1:4]
rows = [json.loads(l) for l in open(inp)]
t0 = time.time()
with open(out, "w") as f:
    for i, r in enumerate(rows, 1):
        body = json.dumps({"state": r["state"], "questions": r["questions"]}).encode()
        for attempt in range(3):
            try:
                res = json.load(urllib.request.urlopen(urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}), timeout=600))
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5)
        probs = {}
        for k, a in res["answers"].items():
            probs[k] = {"true": a["noul"], "false": 1 - a["noul"]} if a["type"] == "noul" else a["probabilities"]
        f.write(json.dumps({"id": r["id"], "source": r["source"], "expected": r["expected"], "probs": probs}) + "\n")
        if i % 400 == 0:
            print(f"{i}/{len(rows)} {i / (time.time() - t0):.1f}/s", flush=True)
print(f"done {len(rows)} in {(time.time() - t0) / 60:.1f} min", flush=True)
