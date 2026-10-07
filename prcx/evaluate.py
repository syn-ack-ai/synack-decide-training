"""Compare routing predictors on held-out repositories.

Predictors: the complexity script's score; logistic regression over the script's numeric features
(trained on train repos); and optionally model probabilities (--model-preds: jsonl of {"id", "p_yes"}).
Metrics on the test split: AUC; and miss rate (share of needs-review PRs routed cheap) when each
predictor routes the SAME share of PRs cheap as the script's 0.30 threshold does.
"""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FEATS = ["score", "lines_added", "lines_deleted", "files_changed", "weighted_impact", "n_signals", "routing_standard"]


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank, i, total = 0.0, 0, 0.0
    while i < len(s):  # average ranks over ties
        j = i
        while j < len(s) and s[j][0] == s[i][0]:
            j += 1
        r = (i + j + 1) / 2
        total += r * sum(1 for k in range(i, j) if s[k][1] == 1)
        i = j
    return (total - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def miss_at_share(rows, key, share):
    """Route the lowest-scoring `share` of PRs cheap; return the share of positives that went cheap."""
    order = sorted(rows, key=lambda r: r[key])
    cheap = order[:round(share * len(order))]
    pos = sum(r["y"] for r in rows)
    return sum(r["y"] for r in cheap) / pos if pos else float("nan")


def logistic(train, test, iters=3000, lr=0.1):
    def x(r):
        return [1.0] + [math.log1p(max(0.0, float(r[f]))) if f not in ("score", "routing_standard") else float(r[f]) for f in FEATS]
    X = [x(r) for r in train]
    mu = [sum(c) / len(X) for c in zip(*X)]
    sd = [max(1e-6, (sum((v - m) ** 2 for v in c) / len(X)) ** 0.5) for c, m in zip(zip(*X), mu)]
    norm = lambda v: [1.0] + [(a - m) / s for a, m, s in list(zip(v, mu, sd))[1:]]
    Xn, y = [norm(v) for v in X], [r["y"] for r in train]
    w = [0.0] * len(Xn[0])
    for _ in range(iters):  # plain batch gradient descent with light L2
        g = [0.0] * len(w)
        for xi, yi in zip(Xn, y):
            p = 1 / (1 + math.exp(-sum(a * b for a, b in zip(w, xi))))
            for k in range(len(w)):
                g[k] += (p - yi) * xi[k]
        w = [wk - lr * (gk / len(Xn) + 1e-3 * wk) for wk, gk in zip(w, g)]
    for r in test:
        r["logreg"] = 1 / (1 + math.exp(-sum(a * b for a, b in zip(w, norm(x(r))))))
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(ROOT / "data/dataset"))
    p.add_argument("--model-preds", action="append", default=[], help="name=path.jsonl with {id, p_yes}")
    p.add_argument("--split", default="test")
    a = p.parse_args()
    rows = list(csv.DictReader(open(Path(a.dataset) / "features.csv")))
    for r in rows:
        r["y"] = int(r["label"] == "yes")
        r["score"] = float(r["score"])
        r["id"] = f"prcx:{r['repo']}#{r['number']}"
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == a.split]
    logistic(train, test)
    preds = {}
    for spec in a.model_preds:
        name, path = spec.split("=", 1)
        m = {json.loads(l)["id"]: json.loads(l)["p_yes"] for l in open(path)}
        preds[name] = m
        for r in test:
            r[name] = m.get(r["id"], float("nan"))
    share = sum(r["score"] < 0.30 for r in test) / len(test)
    pos = sum(r["y"] for r in test)
    print(f"{a.split}: {len(test)} PRs from {len({r['repo'] for r in test})} held-out repos, {pos} need review "
          f"({pos / len(test):.0%}); script@0.30 routes {share:.0%} cheap")
    print(f"{'predictor':22s} {'AUC':>6s}  {'miss rate @ same cheap share':>28s}")
    for key in ["score", "logreg", *preds]:
        rr = [r for r in test if not (isinstance(r.get(key), float) and math.isnan(r[key]))]
        a_ = auc([r[key] for r in rr if r["y"]], [r[key] for r in rr if not r["y"]])
        print(f"{key:22s} {a_:6.3f}  {miss_at_share(rr, key, share):28.1%}   (n={len(rr)})")


if __name__ == "__main__":
    main()
