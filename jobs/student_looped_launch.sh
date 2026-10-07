#!/bin/sh
# After the prep chain uploads the student data, launch the LOOPED student job (layers 9-12, K<=4 random, full v7 recipe).
until grep -q STUDENT_DATA_READY "$1" 2>/dev/null; do
  if grep -qE "LABEL JOB|Traceback|Error" "$1" 2>/dev/null; then echo "PREP FAILED"; cat "$1"; exit 1; fi
  sleep 60
done
ssh box 'hf jobs run --flavor rtx-pro-6000 --timeout 330m --name synack-decide-e2b-looped -s HF_TOKEN -e TRAIN_FILE=train.jsonl -e LOOP=9-12:4 -e TRAIN_ARGS="--loop 9-12:4 --loop-random --rank 32 --alpha 64 --lr 1e-4 --token-budget 24000 --batch 32 --warmup 0.03 --spike 8 --eval-every 1000" -d pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime bash -c "$(cat /tmp/job_student_looped.sh)" 2>&1 | grep -E "url:|Error"'
echo LOOPED_JOB_LAUNCHED
