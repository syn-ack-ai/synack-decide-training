#!/bin/sh
# v7b MLX 8-bit vs bf16 parity (PRs, ordinal) once the MLX download has finished. GGUF parity runs on dreamer instead.
cd ~/systemone
while pgrep -f "hf download SkyPanther/synack-decide-26b-a4b-mlx-8bit" >/dev/null; do sleep 15; done
mlxenv/bin/hf download SkyPanther/synack-decide-26b-a4b-mlx-8bit --local-dir models/synack-decide-v7b-mlx8 > runs/dl_v7b_mlx.log 2>&1 || exit 1
mlxenv/bin/python prcx/predict_mlx.py --model models/synack-decide-v7b-mlx8 --records prcx/data/dataset/test_records.jsonl --limit 500 --out runs/v7b_mlx_pr500.jsonl
mlxenv/bin/python v2/predict_records.py --backend mlx --model models/synack-decide-v7b-mlx8 --records v2/data/v4/eval_ordinal_records.jsonl --out runs/v7b_mlx_ordinal.jsonl
echo MLX_CHECK_DONE
