---
license: apache-2.0
base_model: SkyPanther/synack-decide-26b-a4b
library_name: gguf
tags: [gguf, llama.cpp, gemma4, moe, decision, system-one, jev, classification, routing]
---

# SynACK Decide 26B-A4B (GGUF)

GGUF builds of [SkyPanther/synack-decide-26b-a4b](https://huggingface.co/SkyPanther/synack-decide-26b-a4b) for llama.cpp's `/v1/systemone` decision endpoint (llama.cpp Oct 2026 or newer).

| File | Size | Notes |
|---|---:|---|
| `synack-decide-26b-a4b-Q8_0.gguf` | 27 GB | recommended |
| `synack-decide-26b-a4b-Q4_K_M.gguf` | 17 GB | smaller GPUs and Macs |

Each file carries decision type `nimble` (labels A–Z, AA, AB, …, up to 255 options) and a `systemone` chat template (also included here as `systemone.jinja`). The template reproduces the training prompt byte for byte; it was checked against the Python reference on 22,145 questions.

**Checked against the bf16 weights on 2,693 held-out questions** (the private generalization set: 9 public datasets that are neither in the Decision Index suite nor in training): Q8_0 gives the same answer on 97.6% of them, with equal accuracy (77.2% vs 76.8% for bf16) and a median probability difference of 0.007. Speed on an RTX 4090 (24 GB) with `--n-cpu-moe 5` (the expert weights of 5 layers in system RAM): about 220 ms for a short question, 1.7 s for a 2–8k-token prompt.

**Known difference:** llama.cpp's Jinja engine prints floats differently from Python. Whole numbers lose the `.0` (`18.0` becomes `18`), and long floats are shortened. States with floats therefore render slightly differently from training: a few tokens on PR records. The effect on answers was small in our checks.

