"""Check gguf/systemone.jinja against v2/common.py on real rows (Python jinja2 with llama.cpp-style tojson)."""
import gzip, json, sys
from pathlib import Path
import jinja2
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v2"))
from common import LABELS, SYSTEM, decision_user, question_criteria

def tojson(v, ensure_ascii=False, separators=None, indent=None):
    return json.dumps(v, ensure_ascii=ensure_ascii, separators=tuple(separators) if separators else (", ", ": "))

env = jinja2.Environment()
env.filters["tojson"] = tojson
tmpl = env.from_string((ROOT / "gguf/systemone.jinja").read_text())
tok = AutoTokenizer.from_pretrained(str(ROOT / "models/systemone-v3-merged-hf"))
n = bad = 0
for line in gzip.open(ROOT / "runs/suite-sample3000/selected-rows.jsonl.gz", "rt"):
    row = json.loads(line)
    for qk, q in row["questions"].items():
        if q.get("type") not in ("choice", "noul"):
            continue
        crit = question_criteria(q)
        keys = list(crit) if q["type"] == "choice" else ["false", "true"]  # nimble order for noul
        if not 2 <= len(keys) <= 255:
            continue
        ref = tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": decision_user(row["state"], q, keys, crit)}],
                                      add_generation_prompt=True, enable_thinking=False, tokenize=False)
        opts = [{"key": k, "description": (q.get("criteria") or {}).get(k) if q["type"] == "choice" else None,
                 "label": LABELS[i]} for i, k in enumerate(keys)]
        got = tmpl.render(id=qk, type=q["type"], instructions=q["instructions"], state=row["state"], options=opts, images=[])
        n += 1
        if got != ref:
            bad += 1
            if bad <= 3:
                i = next(j for j in range(min(len(got), len(ref))) if got[j] != ref[j]) if got[:len(ref)] != ref[:len(got)] else min(len(got), len(ref))
                print("MISMATCH", row["id"], qk, repr(ref[i-60:i+60]), "\n   got", repr(got[i-60:i+60]))
print(f"{n} questions, {bad} mismatches")
