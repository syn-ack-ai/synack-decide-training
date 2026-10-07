"""Compare two PR prediction files ({id, p_yes}) on their shared ids: AUC, rank correlation, same decision at thresholds.
  compare_pr_parity.py ref.jsonl other.jsonl [thr ...]"""
import json, sys
ROOT = __file__.rsplit("/runs/", 1)[0]
gold = {json.loads(l)["id"]: json.loads(l)["expected"]["needs_review"] == "yes" for l in open(f"{ROOT}/prcx/data/dataset/test_records.jsonl")}
load = lambda p: {d["id"]: d["p_yes"] for d in map(json.loads, open(p))}
A, B = load(sys.argv[1]), load(sys.argv[2])
ids = [i for i in B if i in A]
def auc(p):
    pos = [p[i] for i in ids if gold[i]]; neg = [p[i] for i in ids if not gold[i]]
    return sum((a > b) + .5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))
def ranks(p):
    o = sorted(ids, key=p.get); return {i: k for k, i in enumerate(o)}
ra, rb = ranks(A), ranks(B); n = len(ids)
rho = 1 - 6 * sum((ra[i] - rb[i]) ** 2 for i in ids) / (n * (n * n - 1))
print(f"{n} PRs: AUC {auc(A):.3f} (ref) vs {auc(B):.3f}; Spearman {rho:.3f}; mean p {sum(A[i] for i in ids)/n:.3f} vs {sum(B[i] for i in ids)/n:.3f}")
for t in map(float, sys.argv[3:] or ["0.38", "0.27"]):
    same = sum((A[i] >= t) == (B[i] >= t) for i in ids)
    print(f"  same decision at {t}: {same}/{n} ({100 * same / n:.1f}%)")
