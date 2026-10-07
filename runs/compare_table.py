"""Comparison table across our measured models (same 3,000-row leaderboard sample, same scorer)."""
import json
M = [("SynACK Decide v4", "runs/eval-v3/lb-score-v4"), ("SynACK Decide v3", "runs/eval-v3/lb-score"),
     ("Gemma-4-12B (our recipe)", "runs/run1-12b-sample3000/score-sample"),
     ("Gemma-4-26B-A4B 4-bit (our recipe)", "runs/run1-moe-sample3000/score-sample"),
     ("Tev1-4B (Together AI)", "runs/tev1-sample3000/score-sample")]
S = {n: json.load(open(f"{d}/scores.json")) for n, d in M}
print("Decision Index (sample est.):", {n: S[n]["decision_index"] for n in S})
for a in S["SynACK Decide v4"]["areas"]:
    print(f"\n== {a['label']}")
    print("   area skill:", {n: next(x for x in S[n]["areas"] if x["id"] == a["id"])["skill"] for n in S})
    for b in a["benchmarks"]:
        row = {n: S[n]["benchmarks"].get(str(b), {}) for n in S}
        name = row["SynACK Decide v4"].get("dataset")
        print(f"   {name:28s}", " | ".join(f"{(r.get('score') or 0) * 100:5.1f}" if r.get("answered") else "  —  " for r in row.values()),
              f"   ({row['SynACK Decide v4'].get('metric')}, n={row['SynACK Decide v4'].get('requests')})")
print("\nmedian latency ms:", {n: S[n]["latency_ms"]["median"] for n in S})
