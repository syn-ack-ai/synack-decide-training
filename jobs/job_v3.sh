set -e
pip install -q "transformers>=5.18" "peft>=0.21" accelerate "huggingface_hub>=1.0" 2>&1 | tail -1
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("SkyPanther/systemone-train-v3", repo_type="dataset", local_dir="/data")
print("data downloaded", flush=True)
PY
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /data && python train_hf.py --model google/gemma-4-26B-A4B-it --train train.jsonl --valid valid.jsonl --labels labels255.json $TRAIN_ARGS
