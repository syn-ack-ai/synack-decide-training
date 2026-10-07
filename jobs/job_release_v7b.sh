# v7b release build on a CPU box (cpu-xl: 16 vCPU, 124 GB): merge the v7b LoRA into google/gemma-4-26B-A4B-it, then build the
# MLX 8-bit and GGUF (Q8_0, Q4_K_M) versions and upload each into the public release repos (older versions are removed from the repos afterwards).
# Model cards are not touched. Env: CKPT (best|final), LLAMA_REV (llama.cpp commit with /v1/systemone).
apt-get update -qq && apt-get install -y -qq git cmake build-essential > /dev/null
pip install -q "transformers>=5.18" "peft>=0.21" accelerate "huggingface_hub>=1.0" sentencepiece protobuf 2>&1 | tail -1
set -e
python - <<'PY'
import os, shutil, time, torch
from huggingface_hub import snapshot_download, hf_hub_download, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from peft import PeftModel
t0 = time.time()
base, ck = "SkyPanther/synack-decide-26b-a4b", os.environ.get("CKPT", "best")  # upload target
src = "google/gemma-4-26B-A4B-it"  # v7b is a fresh LoRA on the original Gemma base (no stacking)
ad = snapshot_download("SkyPanther/synack-decide-v7b-lora", allow_patterns=[ck + "/*"])
m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, ad + "/" + ck).merge_and_unload()
print(f"merged in {(time.time() - t0) / 60:.1f} min", flush=True)
m.save_pretrained("/out", safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained(src).save_pretrained("/out")
try:
    AutoProcessor.from_pretrained(src).save_pretrained("/out")
except Exception as e:
    print("processor not saved:", type(e).__name__, flush=True)
# Google's original config.json: transformers 5.18 writes fields that mlx-lm misreads; weights are unchanged.
shutil.copy(hf_hub_download("google/gemma-4-26B-A4B-it", "config.json"), "/out/config.json")
for f in os.listdir("/out"):
    if f.lower() == "readme.md":
        os.remove(os.path.join("/out", f))
print(f"saved in {(time.time() - t0) / 60:.1f} min: {sorted(os.listdir('/out'))}", flush=True)
HfApi().upload_folder(folder_path="/out", repo_id=base, commit_message="v7b weights (fresh LoRA on the Gemma base: full mix, 2nd pass over the reasoning core, injection fix)",
                      delete_patterns=["*.safetensors", "model.safetensors.index.json"])
print("BF16_UPLOADED", flush=True)
PY
set +e
# MLX 8-bit (mlx CPU backend on Linux); a failure here does not stop the GGUF build.
( pip install -q "mlx[cpu]" mlx-lm 2>&1 | tail -1 && python -m mlx_lm convert --hf-path /out --mlx-path /mlx -q --q-bits 8 --q-group-size 64 && \
  python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path='/mlx', repo_id='SkyPanther/synack-decide-26b-a4b-mlx-8bit', ignore_patterns=['README.md'],
                      commit_message='v7b MLX 8-bit', delete_patterns=['*.safetensors', 'model.safetensors.index.json'])
print('MLX_UPLOADED', flush=True)" ) || echo "MLX_FAILED"
rm -rf /mlx
set -e
git clone -q https://github.com/ggml-org/llama.cpp /llama.cpp && cd /llama.cpp && git checkout -q ${LLAMA_REV:-11fe021}
cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF > /dev/null && cmake --build build -j 16 --target llama-quantize > /dev/null
pip install -q -r requirements/requirements-convert_hf_to_gguf.txt -e gguf-py 2>&1 | tail -1
pip install -q "transformers>=5.18" 2>&1 | tail -1
mkdir -p /gguf/gguf /gguf/up
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil
for f in ["gguf/systemone.jinja", "gguf/add_metadata.py"]:
    shutil.copy(hf_hub_download("SkyPanther/systemone-train-v4", f, repo_type="dataset"), "/gguf/" + f)
PY
python convert_hf_to_gguf.py /out --outtype bf16 --outfile /gguf/raw-bf16.gguf
rm -rf /out
python /gguf/gguf/add_metadata.py /gguf/raw-bf16.gguf /gguf/decide-bf16.gguf --llama-cpp /llama.cpp && rm /gguf/raw-bf16.gguf
./build/bin/llama-quantize /gguf/decide-bf16.gguf /gguf/up/synack-decide-26b-a4b-Q8_0.gguf Q8_0 > /dev/null
./build/bin/llama-quantize /gguf/decide-bf16.gguf /gguf/up/synack-decide-26b-a4b-Q4_K_M.gguf Q4_K_M > /dev/null
cp /gguf/gguf/systemone.jinja /gguf/up/
python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path='/gguf/up', repo_id='SkyPanther/synack-decide-26b-a4b-GGUF', commit_message='v7b GGUF Q8_0 + Q4_K_M (decision type nimble)')
print('GGUF_UPLOADED', flush=True)"
echo RELEASE_DONE
