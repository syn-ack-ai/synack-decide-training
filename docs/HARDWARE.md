# Measured hardware performance (so we don't re-measure)

Every number here was measured in this project, October 2026. "Step" = one optimizer step; batch is
examples per step. Prompt lengths matter a lot: leaderboard-mix prompts average ~400 tokens (compact v2
format), PR-routing prompts ~1,900 tokens.

## Apple M5 Max, 64 GB unified memory (MLX 0.32, mlx-lm 0.32)

| Job | Setting | Speed | Peak memory | Notes |
|---|---|---|---|---|
| Train Gemma-4-E4B, bf16 LoRA r8 (tev1 recipe) | batch 8, grad checkpointing | 6.1–6.4 s/step | ~46 GB wired | OOM without checkpointing while LM Studio held a 9 GB model; unload other models first |
| Train Gemma-4-26B-A4B MoE, QAT 4-bit base, LoRA r16 incl. experts (667M trainable) | batch 16, ~400-token prompts | 10.4–13.7 s/step (3,861 steps in 12.4 h) | 33.5 GB | best val 82.25% |
| Same MoE, PR fine-tune | batch 16, ~1,900-token prompts | 32–52 s/step (156 steps in 95 min) | 32 GB | slows under sustained load |
| Inference Tev1-4B bf16 (tev1 1,300 records) | 1 forward pass, letter readout | 55 ms median | | same 1,179/1,300 as CUDA |
| Inference Gemma-4-E4B bf16 (tev1 records) | | 55 ms median | | |
| Inference MoE 26B-A4B 4-bit (tev1 records) | | 117 ms median | | 1,176/1,300 |
| Inference MoE 26B-A4B 4-bit (PR prompts) | | 0.63 s/PR (2,196 in 23 min) | | |
| Teacher Gemma-4-31B, 8-bit MLX | labelling | 0.9–2.4 questions/s | ~35 GB | API was ~25–50× faster; see below |

## RTX 4090 24 GB ("dreamer"), native Ubuntu 25.10, driver 595, torch 2.14 cu130

| Job | Setting | Speed | Peak memory | Notes |
|---|---|---|---|---|
| Train Gemma-4-E4B bf16 LoRA r8 | batch 8, checkpointing, per-layer embeddings on CPU | 1.8–2.0 s/step (~3.3× the Mac) | 11.4 GB | without the CPU offload / train-mode fix it OOMs at 22 GB |
| Train Gemma-4-12B QLoRA (nf4, double quant), LoRA r32 | batch 16, ~240-token prompts | 7.4 s/step | 13.9 GB | |
| Same, compact v2 leaderboard mix | batch 16, ~400-token prompts | 8.6–9.0 s/step (3,861 steps in 10 h) | 17.3 GB | verbose option JSON was 18.4 s/step |
| Inference Gemma-4-12B nf4 (tev1 records) | | 87 ms (base), 100 ms (adapter) median | | |
| Leaderboard 3,000-row sample, 12B nf4, no prefix cache | | ~3 h | ~13 GB | multi-question rows dominate; add prefix caching before a full run |
| Cannot train | Gemma-4-12B bf16 (23.9 GB weights); Gemma-4-26B-A4B 4-bit via bitsandbytes (fused experts are not quantized) | | | |

**Windows/WSL2 on the same card:** ~2 GB VRAM held by Windows, and the driver's sysmem fallback spilled
to RAM instead of OOM-ing (100% "utilization" at ~100 W). Use native Linux, or set NVIDIA Control
Panel → CUDA Sysmem Fallback Policy → Prefer No Sysmem Fallback.

## RTX 3080 Ti 12 GB ("box"), Omarchy

| Job | Speed | Notes |
|---|---|---|
| Inference Tev1-4B bf16 (tev1 records) | 67.5 ms median | 1,179/1,300 |
| Leaderboard sample with Tev1-4B bf16 | — | OOM on long HLE prompts (~2k proxy tokens); not suitable for full runs |
| CPU: PR complexity scoring (scorer + PyDriller + diff), 16 cores | ~880 PRs/min with 24 workers on full clones | partial (blobless) clones were ~60 PRs/min: lazy blob fetches per diff |

## Jetson Orin Nano Super 8 GB ("milo")
Too small for any training here (12B 4-bit alone is ~7 GB). Candidate for an edge demo only.

## Hugging Face Jobs
| Flavor | $/h | Measured |
|---|---|---|
| cpu-basic | 0.01 | billing check job ran in seconds |
| h200 (141 GB, Hopper) | 5.00 | Same train_hf.py smoke on Gemma-4-26B-A4B bf16: loaded in 1.0 min (53.6 GB) but **no optimizer step completed in ~15 min** (RTX PRO 6000: 10 steps in ~1.6 min). Likely a slow MoE/expert kernel path on sm_90 with torch 2.8 + transformers 5.18 + PEFT target_parameters. Cancelled after ~18 min (~$1.50). Don't use H200 for this stack without re-testing. |
| rtx-pro-6000 (96 GB, Blackwell) | 2.75 | train_hf.py, Gemma-4-26B-A4B **bf16**, LoRA r16 incl. experts (494M trainable): model load 0.6 min, 53.6 GB resident; token budget 24k → ~38.5 examples/step at **9.5–9.8 s/step** (~3.9 ex/s, ~2,400 tok/s), **peak 81.7 GB**. 30-step smoke job: 5 m 52 s wall, ~$0.27. Full v3 run (77,882 ex, ~2,020 steps) projects to ~5.5 h train / ~6 h wall ≈ $16–17. v4 continuation (base = merged v3 repo, 34.5k ex): token budget 16k → 23 ex/step at **6.4–6.5 s/step** (~3.6 ex/s), peak 74.3 GB, 1,503 steps ≈ 2.7 h |
| rtx-pro-6000, PyTorch 2.14.1 vs 2.8 (Oct 5) | — | With torch <= 2.8, transformers' fused MoE `grouped_mm` is Hopper-only, so on Blackwell (sm_120) it falls back to a per-expert loop; 2.14.1 (image `pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime`, needs `PIP_BREAK_SYSTEM_PACKAGES=1`) uses the fused kernel. Measured gain is small: training 9.3-9.55 -> 8.9-9.0 s/step (~4%), 3,000-row sample 15.2 -> 13.5 min (~10%), median latency 159 -> 136 ms; index 58.31 vs 58.28, 98.9% same answers. Use 2.14.1 for new jobs; not worth restarting a running one |
| dreamer RTX 4090 24 GB + 62 GB RAM, llama.cpp (CUDA 13.4) Q8_0 (26.9 GB), expert placement (Oct 5, over LAN) | — | `--n-cpu-moe 5` (default): 23.9 GB VRAM, median 216 ms (<500-token prompt), 459 ms (500-2k), 1.65 s (2k-8k), 15k-token prompt 4.4 s, 17-question doc 3.6 s. `--n-cpu-moe 6`: 23.4 GB, ~12-15% slower. `-ot exps=CPU` (all experts in RAM): 4.9 GB VRAM but ~4x slower (1.0 s / 2.1 s / 7.2 s / 16.6 s). `--n-cpu-moe 4` does not fit. Same answers in every setting (60/60). Prompt processing ~2,000 tok/s |
| cpu-xl (16 vCPU, 124 GB) | 1.00 | candidate for merge + MLX + GGUF release builds without a GPU (jobs/job_release_v4.sh) |

### Inference, Gemma-4-26B-A4B bf16 + v3 LoRA (incl. fused experts) on rtx-pro-6000, engine_v2
| | LoRA unmerged (PEFT target_parameters) | **LoRA merged on load** |
|---|---:|---:|
| tev1 record (median) | 419 ms | **142 ms** |
| PR-routing record (~1,900 tokens) | 504 ms | **229 ms** |
| leaderboard-sample row (multi-question) | 3.0 s | **0.79 s** |
Accuracy identical (same checkpoint, 43/50 vs 86/100 tev1). Always merge for inference: unmerged LoRA on fused
experts recomputes the expert deltas on every forward. 3,000-row leaderboard sample ≈ 40 min (~$2); a full
~150k-row suite run ≈ 33 GPU-h before prefix caching. HF's MoE forward loops over experts in Python; a fused-MoE
serving stack (vLLM/SGLang) should be faster still.
Batching all questions of a request into right-padded forwards (engine_v2 `token_budget`): on a 3080 Ti with
Gemma-4-E4B, 105/105 identical answers vs one-question-at-a-time, mean |Δp| 0.005, 1.7× faster at a 2k-token
budget. The batch path alone (one row, no padding) differs from the original path by mean |Δp| 0.0025: that is
bf16 numerics between code paths, not batching. Left padding was worse (sliding-window layers); use right padding.
Eval job notes: install decision-index from the GitHub zip (`.../archive/refs/heads/main.zip`); the pytorch
runtime image has no git, and piping pip through `tail` hides install failures.

## GGUF / llama.cpp (Mac M5 Max, 64 GB)
| Step | Measured |
|---|---|
| llama.cpp Metal build (llama-server + llama-quantize) | a few minutes |
| convert_hf_to_gguf.py, merged 26B-A4B → bf16 GGUF (50.5 GB) | **1 m 44 s** write (~500 MB/s); needs transformers ≥ 5.5 in the converter venv (4.57 fails on the tokenizer config) |
| add decision metadata (gguf_new_metadata-style copy of 50 GB) | ~3–4 min |
| llama-quantize → Q8_0 (26.9 GB) / Q4_K_M (16.8 GB) | about a minute each |
| llama-server `/v1/systemone`, Q8_0, one question | **329 ms/question** vs MLX 8-bit 301 ms (60-question mix) |
| v4 Q8_0 download from the Hub to the Mac (27 GB) | ~5 min (~90 MB/s); downloads are far faster than home uploads |
| v4 parity (Oct 4) | MLX 8-bit vs bf16: 500 PRs AUC 0.769 vs 0.776, 98.2% same decision at 0.38, ordinal 80.3/77.8% vs 79.3/76.3%, 744 probes in 1.1 min. GGUF Q8_0 vs MLX 8-bit: 59/60 same argmax, 330 vs 308 ms/q (runs/check_v4_mlx.sh, runs/check_v4_gguf.sh) |
| Home upload to the Hub | ~3 MB/s total: 25 GB MLX + 44 GB GGUF ≈ 6–7 h. Prefer building release artifacts on HF Jobs |
Memory: the Mac has 64 GB. llama-server Q8_0 (25 GB RSS) + one MLX 8-bit copy (25 GB) is the limit; a third copy
OOMs the machine. Run comparisons one model at a time (gguf/compare_llamacpp.py llama → stop server → mlx).
llama.cpp's Jinja `tojson` prints floats differently from Python (`18.0` → `18`, long floats shortened), so float-heavy
states render a few tokens differently from training; same argmax 54/60, median |Δp| 0.008 on identical prompts.
macOS paths are case-insensitive: writing `x-BF16.gguf` while reading `x-bf16.gguf` truncates the input.

## Teacher labelling via OpenRouter (top-20 logprobs, max_tokens 1)
| Teacher | Throughput | Cost | Val accuracy (1,020 q) |
|---|---|---|---|
| google/gemma-4-31b-it (exclude Venice/Novita providers) | ~50 q/s at concurrency 48 | $1.96 / 34k q | 77.1% |
| moonshotai/kimi-k3 | ~20 q/s at concurrency 32 | $24.80 / 34k q | 80.5% |
| deepseek/deepseek-v4-pro | ~13 q/s | $0.11 / 2.3k q | 66.4% |
| z-ai/glm-5.3, qwen/qwen3.8-max | unusable: reasoning cannot be disabled | | |

## GitHub API (PR collection)
GraphQL, 40 PRs/page with reviews, threads, commits, labels: ~27 points/page. 122 repos / 32,621 PRs
in ~1.5 h with 3 parallel workers; peak use well under the 5,000 points/h limit.

## PR-routing input length vs accuracy (Mac MLX, MoE 4-bit, 2,196 held-out PRs)
| Inputs | Mean prompt tokens | Time | AUC zero-shot | AUC fine-tuned | Miss rate fine-tuned |
|---|---:|---:|---:|---:|---:|
| full (diff 3,500 chars, top 12 files, reasoning) | 1,885 | 23.0 min | 0.743 | 0.758 | 30.3% |
| trimmed (diff 1,500 chars, top 6 files, no reasoning) | 1,193 | 15.4–15.7 min | 0.729 | 0.746 | 34.0% |
Trimming saves ~33% time but loses the gain over the script (script: 0.737 AUC, 32.7% miss). Keep full inputs;
get speed from a cascade (model only for script scores 0.15–0.50, ~47% of PRs), GPU serving, or distillation.

## Gotchas that cost time
- Hugging Face loads models in eval mode; gradient checkpointing only applies after `model.train()`.
- Gemma-4-12B/31B chat templates add an empty thought channel to the generation prompt but not to rendered
  conversations: build training sequences as `generation_prompt + label + <turn|>`.
- Gemma-4-E4B: `embed_tokens_per_layer` (2.8B params) is lookup-only; keep it on the CPU.
- Gemma-4-26B-A4B experts are fused `nn.Parameter`s: bitsandbytes skips them; PEFT needs
  `target_parameters=["experts.gate_up_proj","experts.down_proj"]` and `target_modules` restricted to
  `language_model` (vision tower uses `Gemma4ClippableLinear`, which PEFT rejects).
- Merged Gemma-4-26B-A4B saved with transformers 5.18 writes `per_layer_config` / `global_head_dim` /
  `num_global_key_value_heads` in a newer format that mlx-lm 0.32 misreads (shape error at layer 5 `k_proj`,
  a full-attention layer). The weights are unchanged: copy Google's original `config.json` before `mlx_lm.convert`.
  8-bit MLX conversion of the merged model: 8.5 bits/weight, 25 GB, a few minutes on the M5 Max.
- Merge job (rtx-pro-6000): load bf16 + PeftModel + merge_and_unload in 1.3 min; save + upload 51.6 GB to the Hub
  in ~1.5 min more (2.8 min total, ~$0.13).
- Toy configs of Gemma 4 must trim `per_layer_*` overrides to the reduced layer count.
- `pkill -f <pattern>` over ssh can match the ssh command itself; use `[p]attern` tricks or a script file.
- macOS has no `timeout`; long waits need Monitor/background jobs.
