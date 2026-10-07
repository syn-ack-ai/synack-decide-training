# v7 bf16 (exact, all options) teacher labels for the v8 ops slice: every question in SkyPanther/systemone-ops-v8
# records/ -> teachers/ops_v7_bf16.jsonl. One RTX PRO 6000. Partial uploads every 10 min (resumable).
set -e
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -q "transformers>=5.18" accelerate "huggingface_hub>=1.0" "https://github.com/apolinario/decision-index/archive/87d4650b42b377c0291a89c1f1a879f9b31082bf.zip"
python -c "import torch, transformers; print('deps ok: torch', torch.__version__, 'transformers', transformers.__version__, flush=True)"
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("SkyPanther/systemone-ops-v8", repo_type="dataset", local_dir="/work", allow_patterns=["records/*", "code/labeller/**"])
try:
    hf_hub_download("SkyPanther/systemone-ops-v8", "teachers/ops_v7_bf16.jsonl", repo_type="dataset", local_dir="/work/prev")
    import shutil; shutil.copy("/work/prev/teachers/ops_v7_bf16.jsonl", "/work/ops_v7_bf16.jsonl"); print("resuming", flush=True)
except Exception:
    print("fresh run", flush=True)
print("data downloaded", flush=True)
PY
( while true; do sleep 600; python -c "
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj='/work/ops_v7_bf16.jsonl', path_in_repo='teachers/ops_v7_bf16.jsonl', repo_id='SkyPanther/systemone-ops-v8', repo_type='dataset', commit_message='v7 bf16 ops labels (partial)')
print('partial uploaded', flush=True)" 2>&1 | tail -1; done ) &
cd /work/code/labeller/v2
python label_records_v7.py --model SkyPanther/synack-decide-26b-a4b --inp /work/records/*.jsonl --out /work/ops_v7_bf16.jsonl 2>&1 | grep -avE "Warning|warn"
python -c "
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj='/work/ops_v7_bf16.jsonl', path_in_repo='teachers/ops_v7_bf16.jsonl', repo_id='SkyPanther/systemone-ops-v8', repo_type='dataset', commit_message='v7 bf16 ops labels (complete)')
print('final upload done', flush=True)"
echo LABEL_JOB_DONE
