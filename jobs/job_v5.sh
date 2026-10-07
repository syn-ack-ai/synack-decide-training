# v5: continue from v4 (main of SkyPanther/synack-decide-26b-a4b) with v4 self-distillation on replay rows, then evaluate.
set -e
pip install -q "transformers>=5.18" "peft>=0.21" accelerate "huggingface_hub>=1.0" "https://github.com/apolinario/decision-index/archive/refs/heads/main.zip" 2>&1 | tail -1
python -c "import decision_index, peft; print('deps ok', peft.__version__)"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("SkyPanther/systemone-eval-v3", repo_type="dataset", local_dir="/work")
snapshot_download("SkyPanther/systemone-train-v5", repo_type="dataset", local_dir="/data")
print("data downloaded", flush=True)
PY
cp /data/v2/*.py /work/v2/
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
M=SkyPanther/synack-decide-26b-a4b
if [ ! -f /data/train_sd.jsonl ]; then
  cd /work/v2 && python self_teacher.py --model $M --inp /data/train.jsonl --out /data/train_sd.jsonl
  python -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='/data/train_sd.jsonl', path_in_repo='train_sd.jsonl', repo_id='SkyPanther/systemone-train-v5', repo_type='dataset'); print('self-teacher targets cached', flush=True)"
fi
cd /data && python train_hf.py --model $M --train train_sd.jsonl --valid valid.jsonl --labels labels255.json --out /data/adapters \
   --push-to SkyPanther/synack-decide-v5-lora $TRAIN_ARGS
A=/data/adapters/best
cd /work/v2
echo "=== ORDINAL PROBES ==="; python predict_records.py --backend cuda --model $M --adapter $A --records /data/eval_ordinal_records.jsonl --out /work/v5_ordinal_preds.jsonl 2>&1 | tail -1
python score_v4_probes.py ordinal /work/v5_ordinal_preds.jsonl
echo "=== INJECTION PROBES ==="; python predict_records.py --backend cuda --model $M --adapter $A --records /data/eval_injection_records.jsonl --out /work/v5_injection_preds.jsonl 2>&1 | tail -1
python score_v4_probes.py injection /work/v5_injection_preds.jsonl
echo "=== PR TEST ==="; python predict_cuda.py --model $M --adapter $A --records ../pr_test_records.jsonl --out /work/pr_preds_v5.jsonl 2>&1 | grep -E "done" | tail -1
echo "=== TEV1 1300 ==="; python bench_tev1_v2.py --model $M --adapter $A --name v5-best 2>&1 | grep -E "^\{|/1300" | tail -2
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi(); r = "SkyPanther/systemone-eval-v3"
for f in ["v5_ordinal_preds.jsonl", "v5_injection_preds.jsonl", "pr_preds_v5.jsonl"]:
    api.upload_file(path_or_fileobj=f"/work/{f}", path_in_repo=f"results/{f}", repo_id=r, repo_type="dataset")
api.upload_folder(folder_path="/work/bench/results/v5-best", repo_id=r, repo_type="dataset", path_in_repo="results/tev1-v5-best")
print("results uploaded (probes, PR, tev1)", flush=True)
PY
echo "=== LEADERBOARD SAMPLE ==="; PYTHONPATH=/work/v2 python -m decision_index run --engine engine_v2:V2Engine --option model=$M --option adapter=$A --rows ../sample-3000.jsonl.gz --out /work/lb_sample_v5 2>&1 | grep -aE "complete|failed|Traceback" | tail -2
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path='/work/lb_sample_v5', repo_id='SkyPanther/systemone-eval-v3', repo_type='dataset', path_in_repo='results/lb-sample-v5')
print('leaderboard sample uploaded', flush=True)"
