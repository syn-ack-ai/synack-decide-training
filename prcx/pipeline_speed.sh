#!/bin/sh
# After Option A: does trimming the input (37% fewer tokens) cost accuracy?
cd ~/systemone/prcx
# (pipeline A already finished)
M=../models/gemma4-26b-a4b-qat-mlx4
../mlxenv/bin/python predict_mlx.py --model $M --adapter ../v2/adapters/run1-moe/best --records data/dataset-slim/test_records.jsonl --out preds/moe_zeroshot_slim.jsonl > preds_slim_zs.log 2>&1
../mlxenv/bin/python predict_mlx.py --model $M --adapter ../v2/adapters/moe-pr-small/best --records data/dataset-slim/test_records.jsonl --out preds/moe_pr_small_slim.jsonl > preds_slim_ft.log 2>&1
{ echo "=== SPEED TEST: full vs trimmed inputs ==="; grep "done" preds_zeroshot.log | tail -1 | sed 's/^/zero-shot full:  /'; grep "done" preds_slim_zs.log | tail -1 | sed 's/^/zero-shot slim:  /'; grep "done" preds_ft_small.log | tail -1 | sed 's/^/fine-tuned full: /'; grep "done" preds_slim_ft.log | tail -1 | sed 's/^/fine-tuned slim: /';
  python3 evaluate.py --dataset data/dataset --model-preds zs_full=preds/moe_zeroshot_test.jsonl --model-preds zs_slim=preds/moe_zeroshot_slim.jsonl --model-preds ft_full=preds/moe_pr_small_test.jsonl --model-preds ft_slim=preds/moe_pr_small_slim.jsonl; echo "SPEED TEST DONE"; } > speed_results.txt 2>&1
