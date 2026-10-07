set -e
pip install -q "transformers>=5.18" "peft>=0.21" accelerate "huggingface_hub>=1.0"
python - <<'PY'
import torch, time
from huggingface_hub import snapshot_download, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from peft import PeftModel
t0 = time.time()
base = "google/gemma-4-26B-A4B-it"
ad = snapshot_download("SkyPanther/systemone-moe-v3-lora", allow_patterns=["best/*"])
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": 0})
m = PeftModel.from_pretrained(m, ad + "/best").merge_and_unload()
print(f"merged in {(time.time()-t0)/60:.1f} min; class {type(m).__name__}", flush=True)
m.save_pretrained("/out", safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained(base).save_pretrained("/out")
try:
    AutoProcessor.from_pretrained(base).save_pretrained("/out")
except Exception as e:
    print("processor not saved:", type(e).__name__, flush=True)
open("/out/README.md", "w").write("# systemone MoE v3 (merged)\n\nGemma-4-26B-A4B-it with the systemone v3 LoRA (step 1800) merged into the bf16 weights. Private. Subject to the Gemma terms of use. Training data includes CC-BY-NC sources (ANLI).\n")
api = HfApi(); api.create_repo("SkyPanther/systemone-moe-v3-merged", private=True, exist_ok=True)
api.upload_folder(folder_path="/out", repo_id="SkyPanther/systemone-moe-v3-merged", commit_message="merged bf16 weights (v3 best, step 1800)")
print(f"uploaded; total {(time.time()-t0)/60:.1f} min", flush=True)
PY
