#!/bin/sh
# Wait for the HF v7-labelling job, fetch labels, build the student mixes, upload them (private). Does NOT launch training.
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
shutil.copy(hf_hub_download('SkyPanther/systemone-train-v7b','teacher_v7_bf16.jsonl',repo_type='dataset'),'v2/data/student/teacher_v7.jsonl'); print('labels fetched')" || exit 1
PYTHONPATH=v2 python3 v2/build_student.py || exit 1
S=$(mktemp -d) && mkdir -p $S/v2 && cp v2/data/student/train.jsonl v2/data/student/train_1pass.jsonl v2/data/student/valid.jsonl bench/labels255.json v2/train_hf.py $S/ \
  && cp v2/engine_v2.py v2/common.py v2/predict_records.py v2/predict_cuda.py v2/score_v4_probes.py $S/v2/
mlxenv/bin/python - <<PY
from huggingface_hub import HfApi
api=HfApi(); r="SkyPanther/systemone-student"
assert api.dataset_info(r).private
api.upload_folder(folder_path="$S", repo_id=r, repo_type="dataset", commit_message="student mixes (v7 teacher)")
print("STUDENT_DATA_READY", api.list_repo_files(r, repo_type="dataset"))
PY
rm -rf $S
