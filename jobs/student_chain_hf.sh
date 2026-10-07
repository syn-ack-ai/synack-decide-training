#!/bin/sh
# Wait for the HF v7-labelling job, fetch its bf16 labels, build the student mix, start E2B training on dreamer.
cd ~/systemone
J=$1
while :; do
  s=$(ssh box "hf jobs inspect --format json $J" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d; print(d['status']['stage'])" 2>/dev/null)
  case "$s" in COMPLETED) break;; ERROR|CANCELED|TIMEOUT) echo "LABEL JOB $s"; exit 1;; esac
  sleep 300
done
mlxenv/bin/python -c "
from huggingface_hub import hf_hub_download
import shutil
p=hf_hub_download('SkyPanther/systemone-train-v7b','teacher_v7_bf16.jsonl',repo_type='dataset')
shutil.copy(p,'v2/data/student/teacher_v7.jsonl'); print('labels fetched')" || exit 1
echo "labels: $(wc -l < v2/data/student/teacher_v7.jsonl)"
PYTHONPATH=v2 python3 v2/build_student.py || exit 1
ssh dreamer 'mkdir -p ~/synack/student'
scp -q v2/data/student/train.jsonl v2/data/student/valid.jsonl v2/train_hf.py jobs/run_student_dreamer.sh bench/labels255.json dreamer:~/synack/student/
ssh dreamer 'cd ~/synack/student && chmod +x run_student_dreamer.sh && (setsid nohup ./run_student_dreamer.sh > train.log 2>&1 < /dev/null &) && echo started'
echo CHAIN_STARTED_TRAINING
