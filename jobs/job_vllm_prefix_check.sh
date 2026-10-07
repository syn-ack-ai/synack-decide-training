# Does vLLM prefix caching cause the ACOS/BRIGHT disagreements with engine_v2? Same rows, caching off and on.
set -e
pip install -q "vllm>=0.19.1" "huggingface_hub>=1.0" "https://github.com/apolinario/decision-index/archive/refs/heads/main.zip" 2>&1 | tail -1
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("SkyPanther/systemone-eval-v3", repo_type="dataset", local_dir="/work", allow_patterns=["v2/*", "bench/*", "sample-vllmcheck.jsonl.gz"])
PY
cd /work
for pc in 0 1; do
  T0=$(date +%s)
  PYTHONPATH=/work/v2 python -m decision_index run --engine engine_vllm:VLLMEngine --option model=SkyPanther/synack-decide-26b-a4b \
    --option revision=$MODEL_REV --option prefix_caching=$pc --rows sample-vllmcheck.jsonl.gz --out /work/pc$pc 2>&1 | grep -aE "\"complete\"|Traceback|Error" | tail -2
  echo "PC$pc WALL $(( $(date +%s) - T0 )) s"
done
python -c "
from huggingface_hub import HfApi
for pc in (0, 1):
    HfApi().upload_folder(folder_path=f'/work/pc{pc}', repo_id='SkyPanther/systemone-eval-v3', repo_type='dataset', path_in_repo=f'results/vllm-prefixcheck-pc{pc}')
print('PREFIXCHECK_UPLOADED', flush=True)"
