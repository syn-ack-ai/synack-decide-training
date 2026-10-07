"""SynACK Decide v4 vs Jev-style models on the Jev Decision Index (board data 2026-09-28; ours = 3,000-row sample)."""
import json
d = json.load(open("runs/board/index.json"))
ours = json.load(open("runs/eval-v3/lb-score-v4/scores.json"))
by = {m["name"]: m for m in d["models"]}
cols = [("SynACK Decide 26B-A4B v4*", None), ("Jev (TypeSafe)", d["jev"]), ("Surogate Rune 26B-A4B v3", by["Surogate Rune 26B-A4B v3"]),
        ("pplx-decider-v1-27b", by["pplx-decider-v1-27b"]), ("Winnow-12B", by["Winnow-12B"]), ("Tev1-4B (Together AI)", by["Tev1-4B-experimental"])]
names = {int(k): v["short"] for k, v in d["benchmarks"].items()}

def index(m):
    return ours["decision_index"] if m is None else m["scores"]["balanced_skill"]
def cat(m, cid):
    if m is None:
        return next(a["skill"] for a in ours["areas"] if a["id"] == cid) * 100
    c = next((c for c in m["categories"] if c["id"] == cid), None)
    return c["skill"] * 100 if c and c.get("skill") is not None else None
def bench(m, b):
    r = (ours["benchmarks"] if m is None else m["results"]).get(str(b))
    return None if not r or r.get("score") is None or (m is None and not r.get("answered")) else r["score"] * 100

rows = [("Overall", "Decision Index", [index(m) for _, m in cols])]
for c in d["categories"][:5]:
    rows.append((c["label"], "Category score (chance-corrected)", [cat(m, c["id"]) for _, m in cols]))
pick = {"knowledge": ["GPQA Diamond", "MMLU-Pro", "GSM8K", "CRUXEval"], "language": ["ANLI", "HellaSwag", "ContractNLI"],
        "retrieval": ["BANKING77", "CLINC150+OOS", "HoVer"], "tools": ["BFCL", "API-Bank", "When2Call"],
        "arts": ["New Yorker", "Humicroedit"]}
for c in d["categories"][:5]:
    for b in c["panel"]:
        n = names[b]
        if any(n.startswith(p) or p.startswith(n) for p in pick.get(c["id"], [])):
            rows.append((c["label"], n, [bench(m, b) for _, m in cols]))
rows.append(("Speed", "Median latency (ms)", [ours["latency_ms"]["median"]] + [m["latency"].get("median") for _, m in cols[1:]]))
hdr = ["Group", "Benchmark"] + [c for c, _ in cols]
print("| " + " | ".join(hdr) + " |")
print("|" + "---|" * len(hdr))
for g, b, vals in rows:
    lower = b.startswith("Median latency")
    nums = [v for v in vals if v is not None]
    best = (min if lower else max)(nums) if nums else None
    cells = []
    for v in vals:
        s = "—" if v is None else (f"{v:.0f}" if lower else f"{v:.1f}")
        cells.append(f"**{s}**" if v is not None and v == best else s)
    print(f"| {g} | {b} | " + " | ".join(cells) + " |")
print("\nparams:", [(c, round((m["meta"].get("served_params") or 0) / 1e9, 1) if m and "meta" in m else None) for c, m in cols])
