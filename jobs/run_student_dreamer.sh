#!/bin/bash
# Student: Gemma 4 E2B distilled from v7 (+ Kimi/Gemma) on dreamer's 4090, inside ai-lab. Pauses the on-demand server
# (GPU must be free), smoke-tests memory/speed, trains, then restarts the server. Log: ~/synack/student/train.log
set -e
cd ~/synack/student
source "$HOME/anaconda3/bin/activate" ai-lab
systemctl --user stop synack-decide-ondemand
trap 'systemctl --user start synack-decide-ondemand' EXIT
nvidia-smi --query-gpu=memory.used --format=csv,noheader
M=google/gemma-4-E2B-it
ARGS="--model $M --train train.jsonl --valid valid.jsonl --labels labels255.json --no-experts --rank 32 --alpha 64 --lr 1e-4 --token-budget 16000 --batch 32 --warmup 0.03 --spike 8"
echo "=== SMOKE ==="
python train_hf.py $ARGS --out /tmp/student-smoke --iters 20 --eval-every 100000 2>&1 | grep -aE "trainable|loaded|s/step|Error|Traceback|out of memory" | tail -6
rm -rf /tmp/student-smoke
echo "=== TRAIN ==="
python train_hf.py $ARGS --out adapters --eval-every 1000 --push-to SkyPanther/synack-decide-e2b-student-lora 2>&1 | grep -aE "trainable|loaded|val loss|best so far|SKIP|done|s/step|Error|Traceback|out of memory" | awk '!/s\/step/ || NR%10==0'
echo STUDENT_DONE
