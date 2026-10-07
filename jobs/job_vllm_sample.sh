# vLLM engine check on the 3,000-row leaderboard sample (rtx-pro-6000): same merged model as the engine_v2 run,
# to compare answers, score and wall time. Env: MODEL_REV (commit of SkyPanther/synack-decide-26b-a4b to test).
set -e
pip install -q "vllm>=0.19.1" "huggingface_hub>=1.0" "https://github.com/apolinario/decision-index/archive/refs/heads/main.zip" 2>&1 | tail -1
python -c "import vllm, transformers, decision_index; print('vllm', vllm.__version__, 'transformers', transformers.__version__, flush=True)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("SkyPanther/systemone-eval-v3", repo_type="dataset", local_dir="/work",
                  allow_patterns=["v2/*", "bench/*", "sample-3000.jsonl.gz"])
print("data downloaded", flush=True)
PY
cd /work
T0=$(date +%s)
PYTHONPATH=/work/v2 python -m decision_index run --engine engine_vllm:VLLMEngine \
  --option model=SkyPanther/synack-decide-26b-a4b --option revision=$MODEL_REV \
  --rows sample-3000.jsonl.gz --out /work/lb_sample_vllm 2>&1 | grep -avE "^\s*$|it/s\]" | grep -aE "complete|failed|Traceback|Error|error|vllm|INFO.*(model|KV|cache|memory)" | tail -25
echo "WALL $(( $(date +%s) - T0 )) s"
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path='/work/lb_sample_vllm', repo_id='SkyPanther/systemone-eval-v3', repo_type='dataset', path_in_repo='results/lb-sample-v5-vllm')
print('VLLM_SAMPLE_UPLOADED', flush=True)"
