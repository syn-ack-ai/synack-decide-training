set -e
pip install -q "transformers>=5.18" "peft>=0.21" accelerate "huggingface_hub>=1.0" "https://github.com/apolinario/decision-index/archive/refs/heads/main.zip"
python -c "import decision_index, peft, huggingface_hub; print('deps ok', peft.__version__)"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("SkyPanther/systemone-eval-v3", repo_type="dataset", local_dir="/work")
PY
python -c "from huggingface_hub import snapshot_download; snapshot_download('SkyPanther/systemone-moe-v3-lora', local_dir='/work/adapter', allow_patterns=['$CKPT/*'])"
M=google/gemma-4-26B-A4B-it; A=/work/adapter/$CKPT
cd /work/v2
echo "=== TEV1 1300 ==="; python bench_tev1_v2.py --model $M --adapter $A --name v3-$CKPT ${LIMIT:+--limit $LIMIT} 2>&1 | grep -E "^\{|/1300" | tail -2
[ -n "$LIMIT" ] && head -n $LIMIT ../pr_test_records.jsonl > ../pr_test_records_head.jsonl
echo "=== PR TEST ==="; python predict_cuda.py --model $M --adapter $A --records ../pr_test_records${LIMIT:+_head}.jsonl --out ../pr_preds_v3_$CKPT.jsonl 2>&1 | grep -E "done|/s" | tail -2
echo "=== LEADERBOARD SAMPLE ==="; PYTHONPATH=/work/v2 python -m decision_index run --engine engine_v2:V2Engine --option model=$M --option adapter=$A --rows ../sample-3000.jsonl.gz ${LBLIMIT:+--limit $LBLIMIT} --out /work/lb_sample_v3_$CKPT 2>&1 | grep -aE "complete|failed|Traceback" | tail -2
python - <<PY
from huggingface_hub import HfApi
api = HfApi()
for local, remote in [("/work/bench/results/v3-$CKPT", "results/tev1-v3-$CKPT"), ("/work/lb_sample_v3_$CKPT", "results/lb-sample-v3-$CKPT")]:
    api.upload_folder(folder_path=local, repo_id="SkyPanther/systemone-eval-v3", repo_type="dataset", path_in_repo=remote)
api.upload_file(path_or_fileobj="/work/pr_preds_v3_$CKPT.jsonl", path_in_repo="results/pr_preds_v3_$CKPT.jsonl", repo_id="SkyPanther/systemone-eval-v3", repo_type="dataset")
print("results uploaded", flush=True)
PY
