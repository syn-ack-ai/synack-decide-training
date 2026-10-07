# engine_v2 prefix-reuse A/B on the 3,000-row sample (rtx-pro-6000): merged v5, reuse off then on, same job.
set -e
pip install -q "transformers>=5.18" accelerate "huggingface_hub>=1.0" "https://github.com/apolinario/decision-index/archive/refs/heads/main.zip" 2>&1 | tail -1
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("SkyPanther/systemone-eval-v3", repo_type="dataset", local_dir="/work", allow_patterns=["v2/*", "bench/*", "sample-3000.jsonl.gz"])
snapshot_download("SkyPanther/synack-decide-26b-a4b", local_dir="/model")
print("data + model downloaded", flush=True)
PY
cd /work
for pr in 0 1; do
  T0=$(date +%s)
  PYTHONPATH=/work/v2 python -m decision_index run --engine engine_v2:V2Engine --option model=/model --option prefix_reuse=$pr \
    --rows sample-3000.jsonl.gz --out /work/reuse$pr 2>&1 | grep -aE "\"complete\"|Traceback|Error" | tail -2
  echo "REUSE$pr WALL $(( $(date +%s) - T0 )) s"
done
python -c "
from huggingface_hub import HfApi
for pr in (0, 1):
    HfApi().upload_folder(folder_path=f'/work/reuse{pr}', repo_id='SkyPanther/systemone-eval-v3', repo_type='dataset', path_in_repo=f'results/lb-sample-v5-reuse{pr}')
print('REUSE_CHECK_UPLOADED', flush=True)"
