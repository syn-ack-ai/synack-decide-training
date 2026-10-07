# Generalization set (SkyPanther/systemone-gen-v1, 2,693 never-trained questions): v5 vs v5+v6 adapter, engine_v2.
set -eo pipefail
pip install -q "transformers>=5.18" "peft>=0.21" accelerate "huggingface_hub>=1.0" "https://github.com/apolinario/decision-index/archive/refs/heads/main.zip" 2>&1 | tail -1
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("SkyPanther/systemone-train-v6", repo_type="dataset", local_dir="/work", allow_patterns=["v2/*"])
snapshot_download("SkyPanther/systemone-eval-v3", repo_type="dataset", local_dir="/work", allow_patterns=["bench/*"])
snapshot_download("SkyPanther/systemone-gen-v1", repo_type="dataset", local_dir="/gen")
snapshot_download("SkyPanther/synack-decide-v6-lora", local_dir="/lora", allow_patterns=["best/*"])
print("downloaded", flush=True)
PY
cd /work/v2
M=SkyPanther/synack-decide-26b-a4b
python predict_records.py --backend cuda --model $M --records /gen/gen_v1.jsonl --out /work/gen_v1_v5.jsonl 2>&1 | tail -1
python predict_records.py --backend cuda --model $M --adapter /lora/best --records /gen/gen_v1.jsonl --out /work/gen_v1_v6.jsonl 2>&1 | tail -1
python -c "
from huggingface_hub import HfApi
for v in ('v5', 'v6'):
    HfApi().upload_file(path_or_fileobj=f'/work/gen_v1_{v}.jsonl', path_in_repo=f'results/gen_v1_{v}.jsonl', repo_id='SkyPanther/systemone-eval-v3', repo_type='dataset')
print('GEN_EVAL_UPLOADED', flush=True)"
