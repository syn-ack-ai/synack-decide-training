# Handoff (2026-10-04)

Read this first in a new session. Also see docs/V4_BACKLOG.md, docs/V6_PLAN.md (v6 stacks on v5; v7 = final full round + full eval), docs/HARDWARE.md and README.md.

## Released (private): SynACK Decide 26B-A4B v5 — the only version on HF
- Oct 4: v5 built by jobs/job_release_v5.sh (v5 LoRA merged onto the v4 commit), cards + benchmark_card.png
  uploaded, then the v3 tag deleted and history squashed: each repo is one commit, v3/v4 are gone from HF.
  - HF SkyPanther/synack-decide-26b-a4b (bf16), -mlx-8bit, -GGUF (Q8_0, Q4_K_M). All still PRIVATE.
  - Rebuild sources if ever needed: models/systemone-v3-merged-hf (local), SkyPanther/synack-decide-v4-lora,
    SkyPanther/synack-decide-v5-lora (adapters kept).
- Cards: gguf/cards/header-{main,mlx,gguf}.md + body.md -> gguf/cards/build.py. Image: runs/board_card.py
  (`uv run --with pillow --with matplotlib python runs/board_card.py --ver v5 --pr-threshold 0.27`).
- v5 results (v4): leaderboard sample 58.2 (55.7; board leader Jev 57.9), Tev1 1,169 (1,171), PR AUC 0.776
  (0.778), PR mean p 0.31 (0.35; true 0.28, cut-off ~0.30), ordinal 78.6/77.2 (79.3/76.3), injection +0.000
  (-0.002), median latency 149 ms bf16. Regression gate v4->v5: PASS.
- Still to do before going public:
  1. ~~v5 MLX/GGUF parity~~ Done: MLX vs bf16 500 PRs AUC 0.762 vs 0.765, 98.0% same at 0.30; ordinal 78.1/76.9 vs
     78.6/77.2; GGUF Q8_0 vs MLX 59/60. Card headers updated and uploaded.
  2. Make the 3 repos public (user decides when).
  3. Update the syn-ack.ai post (PUT /api/posts/<slug>; admin key via `vercel env pull` into a temp file).

## OFFICIAL: v7 (built as "v7b") Decision Index 0.2.1 = 59.54, complete run (Oct 6, 2026)
- 150,759 rows (150,317 scored, all ok), 0 errors; 167 unsupported = unscored ToolRet prompts of 34-38k tokens over the 32,768 limit; raw index 69.25; areas knowledge 44.4,
  language 63.9, retrieval 62.4, tools 79.1, arts 45.9; median latency 132 ms (p95 360) on one RTX PRO 6000.
- Board snapshot 2026-09-28: would be #1 of 72 (Jev 57.91, Surogate Rune 57.44, Decider Gemma-4-31B 57.33, pplx 56.40).
- Results (private): SkyPanther/decision-index-results runs/synack-decide-v7b/ (scores.json complete=true,
  results.jsonl.gz, index.json, benchmark-summary.json, environment.json); local copy runs/eval-v3/lb-score-v7-full/.
- Cards + benchmark image updated with the official numbers (runs/board_card.py --scores ...). Repos still private.
- Leaderboard submission (README "Submitting"): PR to github.com/apolinario/decision-index adding a line to
  submissions/README.md with model name, results dataset link, engine/commit, hardware; maintainers re-measure latency
  and validate answers on a private sample, so they need access to the weights and results. USER DECIDES (public).

- v7 MLX 8-bit parity (Oct 6): 500 PRs 97.6% same decision at 0.27 (98.4% at 0.30), AUC 0.774 vs 0.777, Spearman 0.985;
  ordinal 82.5/77.8 vs 82.3/78.1. MLX card updated (no longer 'pending').

## PUBLIC + SUBMITTED (Oct 6, 2026)
- Model repos synack-decide-26b-a4b, -mlx-8bit, -GGUF are PUBLIC (user's go-ahead). Weights = run commit c20b9b19
  (history squashed into ec95c3f4; later commits are card-only).
- Public results: SkyPanther/synack-decide-decision-index @ 770cf945 (runs/synack-decide-v7/: kit outputs + compact
  results without payload/raw_output; re-scores to 59.54). The original private dataset decision-index-results
  (full rows incl. suite text + a 647 MB partial) must stay private: the suite may not be republished.
- Exact run engine published: github.com/syn-ack-ai/synack-decide leaderboard/ @ b60f018.
- Leaderboard PR: https://github.com/apolinario/decision-index/pull/71 (fork syn-ack-ai/decision-index).
  Pending PRs at submit time include Torchcast Decision 27B claiming 65.00 (#58), so #1 is not assured.

## Student: looped Gemma 4 E2B distilled from v7 (job 6ac47fd9, done Oct 6, 2026; ~5.3 h, ~$15)
- Recipe: layers 9-12 looped (v2/looping.py), random K 1..4 per step, LoRA r32/a64, lr 1e-4, full v7 recipe rows with
  teacher 0.5 v7 + 0.4 Kimi + 0.1 Gemma (v7 alone where no external teacher); best val 67.6% at step 3000 (v7 82.7%).
  Adapter: SkyPanther/synack-decide-e2b-looped-lora (private). Results: systemone-eval-v3 results/*e2b_looped*.
- vs v7 (K=4): gen-v1 62.5 vs 76.8; gen-v2 56.5 vs 78.0; PR AUC 0.752 vs 0.774 (bal acc @0.27 68.1 vs 71.2);
  ordinal 53.0/60.4 vs 82.3/78.1; injection shift +0.002 (clean); LB sample 32.4 vs 61.1 (median 51 ms vs 152).
  Sample overstates the full run by ~1.6 for v7, so ~31 full: about #33 of 71, above every <=2.3B entry (Decider 2B 28.97).
- LOOP SWEEP (same weights): gen-v1 K=1 64.4, K=2 63.4, K=3 63.1, K=4 62.5, K=6 60.8; gen-v2 59.7/58.9/56.8/56.5/54.5.
  More passes = worse, monotonically. Looping did not help here; K=1 is the best way to serve it. Open question: does a
  plain (no-loop) E2B trained the same way beat K=1? jobs/job_student.sh is that control (~$10, not launched).

## v6: NOT released (Oct 4/5) — v5 stays
- v6 (job 6ac2b96c, adapters SkyPanther/synack-decide-v6-lora) vs v5: sample 58.44 vs 58.31 (noise); trained-related
  benchmarks 71.03 vs 70.64, NEVER-TRAINED 60.56 vs 61.24, knowledge 44.9 vs 46.5; PR AUC 0.775 vs 0.776; Tev1 1,168
  vs 1,169; generalization set 76.31% vs 76.49% (26/31, p=0.60). Flat with a benchmaxing shape: not released.

## v7: NOT released (Oct 5) — v5 stays
- v7 = fresh LoRA on the Gemma base, one epoch on 96,534 examples (job 6ac2e9fb, ~8 h, ~$22; adapters
  SkyPanther/synack-decide-v7-lora; data SkyPanther/systemone-train-v7). Val 77.85% (v5 78.11%).
- vs v5: generalization set 77.16% vs 76.49% (+18, p=0.15); NEVER-TRAINED 58.28 vs 61.24 (SGD -14.5, SATA -13.8,
  HLE -5.2, MuSR -3.4); knowledge 41.0 vs 46.5; CRUXEval -12.1; injection shift -0.022 vs +0.000 (AUC 0.761 vs
  0.771); PR AUC 0.773 vs 0.776; Tev1 1,167 vs 1,169; ordinal 80.5/77.2 vs 78.6/77.2; sample 58.78 vs 58.31 (gains
  on trained benchmarks: POP909 +23.7, home appliances +18.6). Gate FAIL (SATA). Not better -> v5 stays.
- Likely cause: v5 saw the core data several times across stacked rounds; v7 one epoch. A 2-epoch v7 (~$20) might win.
- Next (pending user OK): firm up the full-run estimate from the full suite on box (free), then run the full 0.2.1
  eval on v5 (~$29; HF credit ~$37 left after v7).

## Running: v7b — job 6ac39a72fbc85ba6823a842d (launched 12:39 UTC Oct 5, timeout 780 min, est. ~12 h / ~$33)
- Data SkyPanther/systemone-train-v7b (v2/build_v7b.py, 163,511 examples): v7 mix with cached targets + v4 50/50
  PR-claim rows (injection fix) + 8.5k generator reasoning items + a second pass over all hf/* and synthetic/* rows.
  Fresh LoRA on google/gemma-4-26B-A4B-it, lr 3e-5, eval every 1000. Adapters -> SkyPanther/synack-decide-v7b-lora.
- Results -> systemone-eval-v3 results/: v7b_* probes, gen_v1_v7b, pr_preds_v7b, tev1-v7b-best, lb-sample-v7b.
- User plan (Oct 5): if v7b is better than v5 (generalization set + never-trained group, no meaningful PR/probe/Tev1
  drop; the injection shift must be back near 0), release v7b and run the full eval on it ("C"; may need more
  credit). If not, keep v5. HF balance $57.40 at 12:30 UTC Oct 5 after another user top-up (real, not a billing lag).
  After v7b (~$33) ~$24 left; the full eval (~$29) needs ~$5-10 more: firm up the estimate on box first, then tell the user.

## dreamer serves v5 (Oct 5)
- llama.cpp (rev = Mac's, built with CUDA sm_89 in ~/synack/llama.cpp) serving SynACK Decide v5 GGUF Q8_0
  (~/synack/models, sha256 matches HF) as systemd user service `synack-decide` (linger on: starts at boot):
  started by ~/synack/start-server.sh inside the `ai-lab` conda env, llama.cpp rebuilt for CUDA 13.4
  (build-cu134); expert placement in ~/synack/server-args (`--n-cpu-moe 5`, fastest that fits; see HARDWARE.md);
  `-ngl 99 -c 16384 --parallel 1 --host 0.0.0.0 --port 8080`, no API key (user's choice).
  GPU 23.7/24.5 GB used. URL the 4090's LAN address on port 8080 (also reachable over Tailscale).
- Checked: 60/60 same answers as the Mac llama.cpp on the same file, 59/60 vs MLX 8-bit; ~0.6 s/question over
  Tailscale; 8k-token prompts in ~4 s. Testers use public or made-up data only.

## Experiments Oct 4
- Training-free looping (arXiv 2605.23872) on v4 MLX: net loss (beta=0: -2.8 avg, gate FAIL; beta=0.5: -1.2,
  +23% latency). Not used in v6. Code: v2/engine_mlx.py `loop` option (off by default); results runs/loop-test/.
- GLM 5.3 as a teacher: reasoning can't be disabled on OpenRouter; with minimal reasoning the probabilities are
  near one-hot. Not useful as a soft-label teacher.
- vLLM engine (v2/engine_vllm.py): 4.5x faster than engine_v2 on the sample (median 43 ms vs 149), 98.8% same
  answers, but a systematic loss on ACOS (81 vs 10) and BRIGHT; NOT prefix caching (caching off: ACOS 63 vs 17 still). vLLM kernels
  shift near-ties against the model's training numerics (~-0.5 DI) and are not run-to-run deterministic (0.4%).
- **Shared-prefix reuse (adopted, default on in engine_v2 and engine_mlx):** each request's common prompt prefix
  (system + state) runs once and its KV cache is reused for every question. HF A/B on v5 bf16 (same job):
  21.0 -> 15.2 min on the sample (1.4x), Decision Index 58.23 -> 58.31, 99.3% same answers, ACOS 35/34;
  the gate flags BRIGHT 4/0 but its score is unchanged (43.21 -> 43.19; ranking metric, gate approximate).
  MLX: 1.55x, 55.72 -> 55.62. Results: results/lb-sample-v5-reuse{0,1}; jobs/job_prefix_reuse_check.sh.
  Full v7 run estimate with reuse: ~10.5 h, ~$29 on one RTX PRO 6000 (or shard rows over 4 GPUs, same cost).
- v2/engine_decide.py picks the backend: NVIDIA GPU -> engine_v2 (bf16 repo), Apple Silicon -> engine_mlx (MLX repo).
- dreamer refused SSH on Oct 4 evening; it was rebooted Oct 5 and works again.
- Architecture ideas to test next (training-free on the Mac first): prompt repetition / question-first layout,
  option-order averaging; later: weight blending, pause tokens.

## Datasets: all private (user decision, Oct 4)
Audit of the 5 HF datasets before any publishing (Oct 4):
- PR records are public GitHub PRs from 116 OSS repos; no private references beyond the scorer's `complexity_script` fields.
  One public diff (spring-cloud-alibaba) contains a third-party `sk-` key: redact it if PR data is ever published.
- Not redistributable: eval-v3 `sample-3000.jsonl.gz` and `results/lb-sample-*` (leaderboard test rows incl. HLE, GPQA, BBH canary),
  and RAGTruth rows in train-v3/v4/v5 and teacher-labels-v1 (Yelp/MS MARCO contexts, "never republished" per the suite manifest).
- If publishing later: only a fresh PR-only dataset (prcx/data/dataset/*_records.jsonl, labels per prcx/build_dataset.py), not the working repos.

## Budget and constraints
- HF credits: $31.30 on Oct 4 morning; since then v5 job (~$8.8, 3.2 h), v5 release (~$0.6), vLLM sample (~$0.7)
  and vLLM prefix check (~$0.7): roughly $20 left (estimate; ask the user for the real balance).
- No billing API exists; ask the user for the balance.
- OpenRouter key: ~/.config/openrouter/key (it was pasted in chat once; rotation is recommended).
- The Mac has 64 GB: never load more than one 25 GB model at a time.
- Excluded from Time Machine: models/, venvs, llama.cpp.
- On Oct 4 the Mac panicked: APFS hung under heavy I/O (configd watchdog). The disk verified healthy.
- Never commit prcx/vendor/ (the PR complexity scorer; not part of this repo).
- Do big transfers from the box, not the Mac.
