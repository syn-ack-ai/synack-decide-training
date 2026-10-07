#!/bin/sh
# v5 parity on the Mac: MLX 8-bit vs bf16 (PRs, ordinal), then GGUF Q8_0 vs MLX 8-bit (60 questions). One model in memory at a time.
cd ~/systemone
mlxenv/bin/hf download SkyPanther/synack-decide-26b-a4b-mlx-8bit --local-dir models/synack-decide-v5-mlx8 > runs/dl_v5_mlx.log 2>&1 || exit 1
mlxenv/bin/python prcx/predict_mlx.py --model models/synack-decide-v5-mlx8 --records prcx/data/dataset/test_records.jsonl --limit 500 --out runs/v5_mlx_pr500.jsonl
mlxenv/bin/python v2/predict_records.py --backend mlx --model models/synack-decide-v5-mlx8 --records v2/data/v4/eval_ordinal_records.jsonl --out runs/v5_mlx_ordinal.jsonl
echo MLX_CHECK_DONE
mlxenv/bin/hf download SkyPanther/synack-decide-26b-a4b-GGUF synack-decide-26b-a4b-Q8_0.gguf --local-dir models/gguf-v5 > runs/dl_v5_gguf.log 2>&1 || exit 1
llama.cpp/build/bin/llama-server -m models/gguf-v5/synack-decide-26b-a4b-Q8_0.gguf --port 8080 > runs/llama_v5.log 2>&1 &
LP=$!
until curl -sf localhost:8080/health >/dev/null; do sleep 5; done
CMP_TAG=v5 mlxenv/bin/python gguf/compare_llamacpp.py llama http://localhost:8080/v1/systemone 40
kill $LP; wait $LP 2>/dev/null
CMP_TAG=v5 MLX_MODEL=models/synack-decide-v5-mlx8 mlxenv/bin/python gguf/compare_llamacpp.py mlx 40
CMP_TAG=v5 mlxenv/bin/python gguf/compare_llamacpp.py report | head -1
echo V5_CHECK_DONE
