# Full Decision Index 0.2.1 run (155k rows) for SynACK Decide v7b: bf16 weights, engine_v2 with shared-prefix reuse,
# PyTorch 2.14.1 on one RTX PRO 6000. Uses the harness's own `pipeline` (run + official scoring). Results (private) ->
# SkyPanther/decision-index-results runs/synack-decide-v7b; partial results.jsonl every 30 min for resume.
set -e
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -q "transformers>=5.18" "peft>=0.21" accelerate "huggingface_hub>=1.0" "https://github.com/apolinario/decision-index/archive/refs/heads/main.zip"
python -c "import torch, transformers, decision_index; print('deps ok: torch', torch.__version__, 'transformers', transformers.__version__, flush=True)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download, HfApi
snapshot_download("SkyPanther/systemone-train-v7b", repo_type="dataset", local_dir="/work", allow_patterns=["v2/*"])
snapshot_download("SkyPanther/systemone-eval-v3", repo_type="dataset", local_dir="/work", allow_patterns=["bench/*"])
print("model commit", HfApi().model_info("SkyPanther/synack-decide-26b-a4b").sha, flush=True)
try:  # resume from a previous partial run if one exists
    p = hf_hub_download("SkyPanther/decision-index-results", "runs/synack-decide-v7b-partial/results.jsonl", repo_type="dataset", local_dir="/tmp/prev")
    import os, shutil; os.makedirs("/work/run-v7b", exist_ok=True); shutil.copy(p, "/work/run-v7b/results.jsonl"); print("resuming from partial results", flush=True)
except Exception:
    print("fresh run", flush=True)
PY
( while true; do sleep 1800; python -c "
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj='/work/run-v7b/results.jsonl', path_in_repo='runs/synack-decide-v7b-partial/results.jsonl', repo_id='SkyPanther/decision-index-results', repo_type='dataset')
print('partial results uploaded', flush=True)" 2>&1 | tail -1; done ) &
cd /work
PYTHONPATH=/work/v2 python -m decision_index pipeline --engine engine_v2:V2Engine --option model=SkyPanther/synack-decide-26b-a4b \
  --suite-dataset SkyPanther/decision-index-suite-0.2 --out /work/run-v7b \
  --upload SkyPanther/decision-index-results --upload-path runs/synack-decide-v7b 2>&1 | grep -aE "progress|complete|failed|index|Traceback|Error|upload" | awk 'NR%40==1 || !/progress/'
python -c "
import json; s=json.load(open('/work/run-v7b/scores.json')); print('FULL RUN: complete', s.get('complete'), '| index', s.get('decision_index') or s.get('index'), flush=True)" || true
echo FULL_EVAL_DONE
