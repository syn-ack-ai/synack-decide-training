#!/bin/sh
# After the 8-bit MLX conversion: PR test + tev1 records, compared with the bf16 results.
cd ~/systemone/prcx
until [ -f ../models/systemone-v3-mlx8/config.json ] && ls ../models/systemone-v3-mlx8/*.safetensors >/dev/null 2>&1 && ! pgrep -f "mlx_lm.convert" >/dev/null; do sleep 30; done
M=../models/systemone-v3-mlx8
../mlxenv/bin/python predict_mlx.py --model $M --records data/dataset/test_records.jsonl --out preds/v3_mlx8_test.jsonl > preds_mlx8.log 2>&1
cd ../v2 && ../mlxenv/bin/python bench_tev1_mlx.py --model ../models/systemone-v3-mlx8 --name v3-mlx8 > ../prcx/tev1_mlx8.log 2>&1; cd ../prcx
{ echo "=== 8-bit MLX vs bf16 (PR test, 2,196 PRs, 16 held-out repos) ==="; grep "done" preds_mlx8.log | tail -1
  python3 evaluate.py --dataset data/dataset --model-preds v3_bf16=../runs/eval-v3/results/pr_preds_v3_best.jsonl --model-preds v3_mlx8=preds/v3_mlx8_test.jsonl | tail -5
  python3 - <<'PY'
import json, statistics
a = {json.loads(l)["id"]: json.loads(l)["p_yes"] for l in open("../runs/eval-v3/results/pr_preds_v3_best.jsonl")}
b = {json.loads(l)["id"]: json.loads(l)["p_yes"] for l in open("preds/v3_mlx8_test.jsonl")}
k = [i for i in a if i in b]; d = [abs(a[i] - b[i]) for i in k]
same = sum((a[i] >= 0.27) == (b[i] >= 0.27) for i in k)
ra = {i: r for r, i in enumerate(sorted(k, key=a.get))}; rb = {i: r for r, i in enumerate(sorted(k, key=b.get))}
n = len(k); rho = 1 - 6 * sum((ra[i] - rb[i]) ** 2 for i in k) / (n * (n * n - 1))
print(f"per-PR |p_bf16 - p_mlx8|: mean {statistics.mean(d):.4f}, median {statistics.median(d):.4f}, max {max(d):.3f}; rank correlation {rho:.4f}")
print(f"same routing decision at threshold 0.27: {same}/{n} ({same / n:.1%})")
PY
  echo "=== tev1 records ==="; grep -E "^\{" tev1_mlx8.log | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('8-bit MLX:', d['correct'], '/', d['n'], '| median', d['median_ms'], 'ms   (bf16 on RTX PRO 6000: 1158 / 1300)')"
  echo "MLX8 EVAL DONE"; } > mlx8_results.txt 2>&1
