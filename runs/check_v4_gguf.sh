#!/bin/sh
# After the v4 MLX check: download v4 Q8_0 GGUF, compare llama.cpp vs MLX 8-bit on 60 questions (one model in memory at a time).
cd ~/systemone
until grep -q CHECK_DONE runs/check_v4_mlx.log 2>/dev/null; do sleep 30; done
mlxenv/bin/hf download SkyPanther/synack-decide-26b-a4b-GGUF synack-decide-26b-a4b-Q8_0.gguf --local-dir models/gguf-v4 || exit 1
llama.cpp/build/bin/llama-server -m models/gguf-v4/synack-decide-26b-a4b-Q8_0.gguf --port 8080 > runs/llama_v4.log 2>&1 &
LP=$!
until curl -sf localhost:8080/health >/dev/null; do sleep 5; done
CMP_TAG=v4 mlxenv/bin/python gguf/compare_llamacpp.py llama http://localhost:8080/v1/systemone 40
kill $LP; wait $LP 2>/dev/null
CMP_TAG=v4 MLX_MODEL=models/synack-decide-v4-mlx8 mlxenv/bin/python gguf/compare_llamacpp.py mlx 40
CMP_TAG=v4 mlxenv/bin/python gguf/compare_llamacpp.py report
echo GGUF_DONE
