# PyTorch 2.14.1 vs our 2.8 jobs on the RTX PRO 6000: is the fused MoE grouped_mm kernel used, how fast are inference
# (3,000-row sample, v5, engine_v2 + prefix reuse; 2.8 took 15.2 min) and training (30 steps; 2.8: ~9.3 s/step), and do
# answers match? Results -> systemone-eval-v3 results/lb-sample-v5-torch214.
set -e
export PIP_BREAK_SYSTEM_PACKAGES=1   # the 2.1x images mark system Python as externally managed (PEP 668)
pip install -q "transformers>=5.18" "peft>=0.21" accelerate "huggingface_hub>=1.0" "https://github.com/apolinario/decision-index/archive/refs/heads/main.zip"
python -c "import torch, transformers, peft, decision_index; print('installed: torch', torch.__version__, 'transformers', transformers.__version__, flush=True)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python - <<'PY'
import torch, transformers
from transformers.integrations.moe import _can_use_grouped_mm
w = torch.zeros(4, 8, 8, device="cuda", dtype=torch.bfloat16); x = torch.zeros(16, 8, device="cuda", dtype=torch.bfloat16)
print("torch", torch.__version__, "cuda", torch.version.cuda, "capability", torch.cuda.get_device_capability(),
      "transformers", transformers.__version__, "| fused grouped_mm used:", _can_use_grouped_mm(x, w, torch.tensor([4, 8, 12, 16], device="cuda")), flush=True)
from huggingface_hub import snapshot_download
snapshot_download("SkyPanther/systemone-eval-v3", repo_type="dataset", local_dir="/work", allow_patterns=["bench/*", "sample-3000.jsonl.gz"])
snapshot_download("SkyPanther/systemone-train-v6", repo_type="dataset", local_dir="/data",
                  allow_patterns=["v2/*", "train_hf.py", "train_sd.jsonl", "valid.jsonl", "labels255.json"])
snapshot_download("SkyPanther/synack-decide-26b-a4b", local_dir="/model")
print("downloaded", flush=True)
PY
mkdir -p /work/v2 && cp /data/v2/*.py /work/v2/
cd /work
T0=$(date +%s)
PYTHONPATH=/work/v2 python -m decision_index run --engine engine_v2:V2Engine --option model=/model --rows sample-3000.jsonl.gz \
  --out /work/lb_torch214 2>&1 | grep -aE "\"complete\"|Traceback|Error" | tail -2
echo "SAMPLE WALL $(( $(date +%s) - T0 )) s"
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path='/work/lb_torch214', repo_id='SkyPanther/systemone-eval-v3', repo_type='dataset', path_in_repo='results/lb-sample-v5-torch214')
print('SAMPLE_UPLOADED', flush=True)"
cd /data && python train_hf.py --model /model --train train_sd.jsonl --valid valid.jsonl --labels labels255.json --out /tmp/smoke \
  --iters 30 --eval-every 100000 --token-budget 24000 --batch 32 --lr 1e-5 2>&1 | grep -aE "s/step|Traceback|Error" | tail -6
echo TORCH_TEST_DONE
