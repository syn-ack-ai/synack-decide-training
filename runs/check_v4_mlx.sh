#!/bin/sh
# Wait for the v4 MLX download, then measure 8-bit vs bf16 on PRs and ordinal probes (one MLX model at a time).
cd ~/systemone
until grep -qiE "Download complete|^/|^path=/" runs/dl_v4_mlx.log 2>/dev/null && ! pgrep -f "hf download SkyPanther/synack-decide-26b-a4b-mlx-8bit" >/dev/null; do sleep 30; done
mlxenv/bin/python prcx/predict_mlx.py --model models/synack-decide-v4-mlx8 --records prcx/data/dataset/test_records.jsonl --limit 500 --out runs/v4_mlx_pr500.jsonl
mlxenv/bin/python v2/predict_records.py --backend mlx --model models/synack-decide-v4-mlx8 --records v2/data/v4/eval_ordinal_records.jsonl --out runs/v4_mlx_ordinal.jsonl
echo CHECK_DONE
