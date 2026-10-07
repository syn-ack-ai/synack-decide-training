"""Compare llama.cpp /v1/systemone (GGUF) with the MLX engine, one model in memory at a time.

  compare_llamacpp.py llama URL N     -> runs/gguf_cmp_llama.jsonl   (only llama-server loaded)
  compare_llamacpp.py mlx N           -> runs/gguf_cmp_mlx.jsonl     (stop llama-server first)
  compare_llamacpp.py report
Env: CMP_TAG=v4 writes/reads runs/gguf_cmp_*_v4.jsonl; MLX_MODEL overrides models/systemone-v3-mlx8.
"""
import gzip, json, os, random, sys, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v2"))
TAG = f"_{os.environ['CMP_TAG']}" if os.environ.get("CMP_TAG") else ""
LLAMA_OUT, MLX_OUT = ROOT / f"runs/gguf_cmp_llama{TAG}.jsonl", ROOT / f"runs/gguf_cmp_mlx{TAG}.jsonl"


def items(n):
    out = []
    rows = [json.loads(l) for l in gzip.open(ROOT / "runs/suite-sample3000/selected-rows.jsonl.gz", "rt")]
    random.Random(1).shuffle(rows)
    for r in rows:
        if len(json.dumps(r["state"])) > 20000:
            continue
        for qk, q in r["questions"].items():
            if q.get("type") in ("choice", "noul") and (q["type"] == "noul" or 2 <= len(q.get("criteria") or {}) <= 255):
                out.append(("suite", r["state"], qk, q)); break
        if len(out) >= n:
            break
    prs = [json.loads(l) for l in open(ROOT / "prcx/data/dataset/test_records.jsonl")]
    random.Random(2).shuffle(prs)
    for r in prs[:n // 2]:
        qk, q = next(iter(r["questions"].items()))
        out.append(("pr", r["state"], qk, q))
    return out


mode = sys.argv[1] if __name__ == "__main__" else None
if mode == "llama":
    url, n = sys.argv[2], int(sys.argv[3])
    with open(LLAMA_OUT, "w") as f:
        for kind, state, qk, q in items(n):
            body = json.dumps({"state": state, "questions": {qk: q}}).encode()
            t = time.time()
            r = json.load(urllib.request.urlopen(urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})))
            a = r["answers"][qk]
            p = {"true": a["noul"], "false": 1 - a["noul"]} if q["type"] == "noul" else a["probabilities"]
            f.write(json.dumps({"kind": kind, "p": p, "tok": r["usage"]["input_tokens"], "s": time.time() - t}) + "\n")
elif mode == "mlx":
    from engine_mlx import label_probs, load_mlx
    tok, model, ids = load_mlx(os.environ.get("MLX_MODEL") or str(ROOT / "models/systemone-v3-mlx8"))
    with open(MLX_OUT, "w") as f:
        for kind, state, qk, q in items(int(sys.argv[2])):
            t = time.time()
            p, n_tok = label_probs(tok, model, ids, state, q)
            f.write(json.dumps({"kind": kind, "p": p, "tok": n_tok, "s": time.time() - t}) + "\n")
else:
    L = [json.loads(l) for l in open(LLAMA_OUT)]
    M = [json.loads(l) for l in open(MLX_OUT)]
    N = len(L)
    tok_eq = sum(a["tok"] == b["tok"] for a, b in zip(L, M))
    agree = sum(max(a["p"], key=a["p"].get) == max(b["p"], key=b["p"].get) for a, b in zip(L, M))
    d = sorted(max(abs(a["p"][k] - b["p"][k]) for k in b["p"]) for a, b in zip(L, M))
    print(f"{N} questions: token counts equal {tok_eq}/{N}, same argmax {agree}/{N}; max|dp| median {d[N // 2]:.3f} "
          f"p90 {d[int(N * .9)]:.3f} max {d[-1]:.3f}; llama.cpp {sum(a['s'] for a in L) / N * 1000:.0f} ms/q, "
          f"MLX {sum(b['s'] for b in M) / N * 1000:.0f} ms/q")
    for a, b in zip(L, M):
        if a["tok"] != b["tok"]:
            print("  tokens differ:", a["kind"], a["tok"], b["tok"])
