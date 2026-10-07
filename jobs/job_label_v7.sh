# v7 (bf16) teacher labels for the student mix: every unique example of systemone-train-v7b. RTX PRO 6000, PyTorch 2.14.1.
# Output (private): SkyPanther/systemone-train-v7b teacher_v7_bf16.jsonl
set -e
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -q "transformers>=5.18" "peft>=0.21" accelerate "huggingface_hub>=1.0" "https://github.com/apolinario/decision-index/archive/refs/heads/main.zip"
python -c "import torch, transformers, decision_index; print('deps ok: torch', torch.__version__, flush=True)"
python - <<'PY'
from huggingface_hub import snapshot_download, HfApi
snapshot_download("SkyPanther/systemone-train-v7b", repo_type="dataset", local_dir="/data", allow_patterns=["v2/*", "train.jsonl"])
snapshot_download("SkyPanther/systemone-eval-v3", repo_type="dataset", local_dir="/data", allow_patterns=["bench/*"])
print("teacher model commit", HfApi().model_info("SkyPanther/synack-decide-26b-a4b").sha, flush=True)
PY
cd /data/v2
python label_v7_hf.py --model SkyPanther/synack-decide-26b-a4b --inp /data/train.jsonl --out /data/teacher_v7_bf16.jsonl
python -c "
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj='/data/teacher_v7_bf16.jsonl', path_in_repo='teacher_v7_bf16.jsonl', repo_id='SkyPanther/systemone-train-v7b', repo_type='dataset')
print('LABELS_UPLOADED', flush=True)"
