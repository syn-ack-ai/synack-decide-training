# LOOPED student (layers 9-12, K up to 4, random K per training step; v2/looping.py): Gemma 4 E2B distilled from v7 (0.5 v7 + 0.4 Kimi + 0.1 Gemma where available, else v7) on one RTX PRO 6000,
# then evaluated like v7. Env: TRAIN_FILE (train.jsonl = full v7 recipe | train_1pass.jsonl), TRAIN_ARGS.
set -e
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -q "transformers>=5.18" "peft>=0.21" accelerate "huggingface_hub>=1.0" "https://github.com/apolinario/decision-index/archive/refs/heads/main.zip"
python -c "import torch, transformers, peft, decision_index; print('deps ok: torch', torch.__version__, flush=True)"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("SkyPanther/systemone-student", repo_type="dataset", local_dir="/data")
snapshot_download("SkyPanther/systemone-eval-v3", repo_type="dataset", local_dir="/work", allow_patterns=["bench/*", "pr_test_records.jsonl", "sample-3000.jsonl.gz", "tev1/*"])
snapshot_download("SkyPanther/systemone-gen-v1", repo_type="dataset", local_dir="/gen1")
snapshot_download("SkyPanther/systemone-gen-v2", repo_type="dataset", local_dir="/gen2")
snapshot_download("SkyPanther/systemone-train-v7b", repo_type="dataset", local_dir="/probes", allow_patterns=["eval_*"])
print("data downloaded", flush=True)
PY
mkdir -p /work/v2 /work/train && cp /data/v2/*.py /work/v2/ && cp /data/train/train_cuda.py /work/train/
M=google/gemma-4-E2B-it
cd /data && python train_hf.py --model $M --train ${TRAIN_FILE:-train.jsonl} --valid valid.jsonl --labels labels255.json --out /data/adapters \
  --no-experts --push-to SkyPanther/synack-decide-e2b-looped-lora $TRAIN_ARGS
A=/data/adapters/best
export SYNACK_LOOP="$LOOP"
cd /work/v2
echo "=== ORDINAL ==="; python predict_records.py --backend cuda --model $M --adapter $A --records /probes/eval_ordinal_records.jsonl --out /work/e2b_looped_ordinal_preds.jsonl 2>&1 | tail -1
python score_v4_probes.py ordinal /work/e2b_looped_ordinal_preds.jsonl
echo "=== INJECTION ==="; python predict_records.py --backend cuda --model $M --adapter $A --records /probes/eval_injection_records.jsonl --out /work/e2b_looped_injection_preds.jsonl 2>&1 | tail -1
python score_v4_probes.py injection /work/e2b_looped_injection_preds.jsonl
echo "=== GEN1 ==="; python predict_records.py --backend cuda --model $M --adapter $A --records /gen1/gen_v1.jsonl --out /work/gen_v1_e2b_looped.jsonl 2>&1 | tail -1
echo "=== GEN2 ==="; python predict_records.py --backend cuda --model $M --adapter $A --records /gen2/gen_v2.jsonl --out /work/gen_v2_e2b_looped.jsonl 2>&1 | tail -1
echo "=== PR ==="; python predict_cuda.py --model $M --adapter $A --records ../pr_test_records.jsonl --out /work/pr_preds_e2b_looped.jsonl 2>&1 | grep -E "done" | tail -1
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi(); r = "SkyPanther/systemone-eval-v3"
for f in ["e2b_looped_ordinal_preds.jsonl", "e2b_looped_injection_preds.jsonl", "gen_v1_e2b_looped.jsonl", "gen_v2_e2b_looped.jsonl", "pr_preds_e2b_looped.jsonl"]:
    api.upload_file(path_or_fileobj=f"/work/{f}", path_in_repo=f"results/{f}", repo_id=r, repo_type="dataset")
print("results uploaded", flush=True)
PY
echo "=== LOOP SWEEP ==="
for K in 1 2 3 6; do
  for G in 1 2; do SYNACK_LOOP="9-12:$K" python predict_records.py --backend cuda --model $M --adapter $A --records /gen$G/gen_v$G.jsonl --out /work/gen_v${G}_e2b_looped_k$K.jsonl 2>&1 | tail -1; done
done
python -c "
from huggingface_hub import HfApi
import glob
for f in glob.glob('/work/gen_v*_e2b_looped_k*.jsonl'):
    HfApi().upload_file(path_or_fileobj=f, path_in_repo='results/'+f.split('/')[-1], repo_id='SkyPanther/systemone-eval-v3', repo_type='dataset')
print('sweep uploaded', flush=True)"
echo "=== LEADERBOARD SAMPLE ==="; PYTHONPATH=/work/v2 python -m decision_index run --engine engine_v2:V2Engine --option model=$M --option adapter=$A --rows ../sample-3000.jsonl.gz --out /work/lb_sample_e2b_looped 2>&1 | grep -aE "complete|failed|Traceback" | tail -2
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path='/work/lb_sample_e2b_looped', repo_id='SkyPanther/systemone-eval-v3', repo_type='dataset', path_in_repo='results/lb-sample-e2b-looped')
print('STUDENT_JOB_DONE', flush=True)"
