# v6 and v7 plan

Decided Oct 4 (user): **v6 stacks on v5** and is the testbed; **v7 is the final, full round** trained once from
the original Gemma base with everything that proved out. v7 is the model we run the full Decision Index 0.2.1
eval on and submit.

# v6 (stacked on v5)

## Teacher targets: re-weight what we already have
Teachers on the same 1,018 validation questions. NLL is the negative log-likelihood of the gold answer, i.e. the
quality of the soft targets the student learns from; lower is better.

| Teacher | Accuracy | NLL |
|---|---:|---:|
| Kimi-K3 (OpenRouter, thinking off) | 80.6% | 0.675 |
| Gemma-4-31B | 77.1% | 2.087 |
| Qwen3.5-397B-A17B (Alibaba/Parasail/StreamLake) | 74.4% | 0.917 |
| GLM-5.2 (thinking off; Morph/StreamLake/Alibaba/Wafer FP8) | 67.7% | 1.060 |
| DeepSeek-V4.1-Flash (thinking off; StreamLake FP8; Oct 5, $0.07) | 71.5% | 1.463 |
| Kimi + Gemma 50/50 (used for v3-v5) | 79.6% | 0.689 |
| **Kimi 0.8 + Gemma 0.2** | **81.3%** | **0.643** |

- Qwen3.8-Max on OpenRouter: "Reasoning is mandatory for this endpoint" (HTTP 400). Thinking can only be turned off
  through Alibaba's direct API (DashScope `enable_thinking: false`), which needs an Alibaba Cloud key.
- Mixing in GLM-5.2 or Qwen3.5 did not survive 2-fold cross-validation: the chosen weights flipped between folds.
- The validation runs cost $0.40 in total (`v2/data/teacher_valid_*.jsonl`).
- DeepSeek-V4.1-Flash (Oct 5): overconfident (>0.99 on 70% of questions); adding it at 0.2 to Kimi/Gemma lowers the
  blend (80.5%, NLL 0.648). Not used. Its own DeepSeek API is blocked by our OpenRouter privacy setting (may train on data).
- GLM-5.3 (Oct 4): thinking cannot be disabled on OpenRouter; with minimal thinking, near one-hot. Not used.

**Decision:** v6 uses Kimi 0.8 / Gemma 0.2 targets for every example that has both teachers. This costs nothing
because both label sets exist for the ~34k training questions. Only buy a new teacher's labels if it beats this on
the validation set.

## v6 recipe
- **Base:** merged v5, pinned to commit `d8d2319c` of SkyPanther/synack-decide-26b-a4b (not `main`). The release
  merges the v6 LoRA onto that same commit (adapt jobs/job_release_v5.sh with BASE_REV).
- **Targets:** Kimi 0.8 / Gemma 0.2 teacher probabilities where both exist (above); replay rows get v5's own
  probabilities (self-distillation), as in v5.
- **Drift guard:** more PR records in the replay (PR AUC went 0.780 -> 0.778 -> 0.776 over v3-v5).
- **Close v5's dips** where train data exists: CLINC150 (~3k unused pool rows), Amazon ESCI (~1.6k), synthetic
  CRUXEval (~1.4k, free; only 118 used so far), API-Bank-style tool selection if a clean train source exists.
- **Format experiments (only if the training-free Mac tests show a gain):** question-first / repeated question
  layout, option-order averaging. v6 is where a format change gets tried, so v7 only carries proven changes.
- **Gate:** runs/regression_check.py against v5 AND v4 (catches slow drift across rounds, not only round to round),
  plus PR AUC, ordinal and injection probes. Release v6 only if it passes and the sample index is >= v5's 58.2.
- **Cost:** ~$9-10 training+eval job, ~$0.6 release build.

## Other v6 inputs
- Lessons from v5's per-benchmark results and regression gate (`runs/regression_check.py`).
- Keep the v5 safeguards: continue from the best model, self-distillation on replay, spike guard 8x for new
  long-prompt families, and the gate before release.
- Possible stronger teachers later: Qwen3.8-Max through DashScope directly, or self-hosting GLM-5.2 or DeepSeek-V4-Pro
  on h200x8 for full distributions (about $30-50 per labelling run).

# No benchmaxing (user, Oct 4) — applies to v7 and later
- v3->v5 on the 3,000-row sample: +4.7 on the 26 benchmarks we trained related data for, +1.7 on the 18 never
  trained (ACOS, API-Bank, ARC, BFCL, BPoMP, BRIGHT, ContractNLI, ForecastBench, HLE, HoVer, Humicroedit, MuSR,
  PhishNChips, SATA, SGD, SimpleBench, ToolRet). Most of the gain is benchmark-specific.
- Decision metrics: the never-trained group, a private held-out set of novel decision tasks (8-10 public datasets
  not in the suite or training, ~2-3k questions, MLX on the Mac), the PR test and the probes. The full Decision
  Index score is reported, not optimized.
- No data added because a benchmark is low. Benchmark-targeted sets (board/*, CLINC/ESCI top-ups) go into v7 only
  if they also help the decision metrics.

# v7 (final consolidation round)

- **One fresh LoRA from google/gemma-4-26B-A4B-it** (no stacking): resets accumulated drift and bf16 merge
  rounding from v3-v6.
- **Data (user, Oct 4): original Gemma base + distilled logprobs + PRs + every added training set.** Concretely:
  - The v3 mix (77.9k examples: HF train splits, Tev1 data, code-verified synthetic tasks).
  - All 13,064 labelled public training PRs, plus the PR author-claim records at the natural 28% rate
    (not v4's 50/50 version, which pushed probabilities high).
  - v4's ordinal questions.
  - v5's added sets: iSarcasmEval, VAST, Habermas Machine, cfcolor, home-appliance generator, POP909-style
    chords, extra WinoGrande, financial sentiment, SmartBugs.
  - v6's additions (CLINC150, ESCI, synthetic CRUXEval, anything else that passes v6's gate).
  - Deduplicated and re-decontaminated against the 0.2.1 suite. Roughly 100k+ examples.
- **Distilled logprobs (soft targets), applied only when they agree with gold:** Kimi 0.8 / Gemma 0.2 teacher
  probabilities where they exist; v6's own probabilities for everything else (distil the best stacked model
  into the clean one).
- **Format:** whatever v6 proved (prompt layout, label format). Nothing untested goes into v7.
- **Checks before the full eval:** 3,000-row sample + regression gate vs v6 and v5, PR / ordinal / injection
  probes, Tev1, then MLX/GGUF parity after the release build.
- **Full Decision Index 0.2.1 run (~125k rows)** on the released v7, with the vLLM engine if the prefix-caching
  check clears it (est. ~3.5 h, ~$10 on one RTX PRO 6000), else engine_v2 (~14 h, ~$40). Then submit via PR to
  github.com/apolinario/decision-index and publish v7 as the only version in the repos.
- **Cost:** training ~7-8 h on one RTX PRO 6000 (~$20-25 incl. evals) + release ~$1 + full eval ~$10.

## v7 result and the recipe for the next full run (Oct 5)
v7 (fresh LoRA, 1 epoch, 96.5k examples) was NOT better than v5: generalization set +0.7 (n.s.), never-trained group
-3.0 (SGD -14.5, SATA -13.8, HLE -5.2, MuSR -3.4), CRUXEval -12.1, knowledge -5.5, injection shift -0.022 (v5 +0.000).
Its sample gains came from trained benchmarks (POP909 +23.7, home appliances +18.6). Likely underfit on general
reasoning: v5 saw the core mix several times across stacked rounds, v7 once.

Next full-run recipe (user, Oct 5: fix those losses and the v4 injection fix, without benchmaxing):
- **2 epochs** over the core mix (or replay the v3 core twice), same base and distillation.
- **Injection fix:** both PR-claim sets, v4's balanced 50/50 claims (1,902) and the natural-rate set (1,817),
  ~4k total. The probe uses held-out test PRs.
- **General reasoning from our own generators, not from benchmarks:** unused synthetic tracking, ordering,
  web-of-lies, navigate, boolean, temporal (~1.4k each in pools/synthetic.jsonl) and synthetic CRUXEval-style code
  execution (~1.3k).
- **Not allowed:** HLE, SATA-Bench, MuSR are test-only (training on them = contamination); SGD's train split would
  remove it from the never-trained yardstick. They stay held out so we can see whether the fix generalizes.
- Judge as before: generalization set + never-trained group + PR + probes, then the sample.
- Cost: ~2x v7's training, ~$35-40 (+ ~$2 evals).

# v8 (future planning)

## Ops / log slice (user, Oct 6)
Goal: make the model useful for AIOps (log triage, anomaly, root cause, command guarding). The Decision Index has
no log/ops benchmarks, so this cannot move the index; judge it only on held-out ops tests (no benchmaxing).

Training sources (~15-20k rows):
- **Loghub labelled sets** (github.com/logpai/loghub: HDFS, BGL, Thunderbird, OpenStack, Hadoop, Zookeeper):
  "is this window anomalous?" (noul), BGL alert category (choice), component (choice). Gold labels.
- **Loghub-2.0 / LogPub templates:** "same event template?" (noul) and "which template?" (choice). Gold labels.
- **RCAEval** (Zenodo 14504481, 735 cases, 3 microservice systems, 11 fault types): root-cause service and fault
  type (choice) from a summary of logs and metrics. Gold labels. Check the licence first.
- **Unlabelled real syslog** (Loghub Linux/Mac/Windows/Apache/OpenSSH): the triage questions from
  runs/usecases/log_triage.py (severity, subsystem, act?). No gold: the teacher IS the label (see below).
- Terms: Loghub is "freely available for research or academic work" (non-commercial, like ANLI): list it on the card.

Held out, never trained: OpenRCA (335 failures), OpsEval (7,334 MC questions), LogEval (4,000 entries), our own
machines' journals (dreamer, hand-checked) and runs/usecases/command_guard.py (50 labelled commands). Decontaminate
the slice against these and against the suite, as for v7.

Teachers:
- **Gold-labelled rows (Loghub, LogPub, RCAEval):** v7 is enough. Same rule as v7: soft targets only where the
  teacher agrees with gold, else gold alone. Cost: one HF labelling job (v2/label_v7_hf.py), about $3-4.
- **Unlabelled triage rows:** v7 must NOT be the only teacher. With no gold there, self-distillation copies v7's
  own mistakes (it rated the dreamer PostgreSQL start failure "minor" and gave a harmless Tailscale warning 0.89
  "act"). Use Kimi-K3 + Gemma-4-31B (0.8/0.2, top-20 logprobs via OpenRouter as before) and keep only rows where
  Kimi and Gemma agree on the argmax. v7 can be a third vote, never the deciding one.
  Cost at the measured rates ($24.80 Kimi + $1.96 Gemma per 34k questions): 10k lines x 3 questions = 30k
  questions, about $24.
- **Before labelling at scale:** validate the teachers on ~200 hand-checked log lines from our own machines, as
  with the 1,018-question teacher validation.

Judge: held-out ops tests up, and no regression on the gen sets, never-trained group, PR, probes and the sample.

## Evidence-augmented decisions (user idea, Oct 6)
When asked to verify, or when a question is factual, a thin wrapper searches first (SearXNG on box) and adds the top
results to the request as `state.evidence`; the model then decides in its usual single pass. For v8, add training
records with search snippets: helpful, irrelevant and misleading, so it learns to rely on good evidence and ignore
noise. Test on held-out knowledge sets with and without evidence, never on suite questions.

**Built Oct 6:** SkyPanther/systemone-ops-v8 (private), 20,739 records:
- 9,681 template, 2,063 same_event, 3,000 bgl_window, 2,000 hdfs_session, 165 hadoop_fault, 359 rca and 3,471 triage.
- Loghub, Loghub-2.0 and RCAEval are all CC-BY-4.0 on Zenodo.
- Triage labels from Kimi + Gemma cost $3.71. Kept only where the two agreed and the answer matched the operator
  alert labels.
- Decontaminated with 4 generic error strings allow-listed. LogEval is no longer a clean test.
- v7 baseline on the local smoke test:
  - template 92%, same_event 100%;
  - BGL alert 75%, HDFS 50% (chance), Hadoop fault 33-42%;
  - root cause 58-67%.
- Code: v2/build_ops.py, v2/assemble_ops.py, the teacher_api.py flags --no-gold and --url, and decontam.py --allow.
- teachers/ops_v7_bf16.jsonl: v7 bf16 exact labels for all 28,345 questions. HF job 6ac52f24, 21 min of labelling,
  about $1.60. It replaced the partial Q8 file; Q8 agreed with bf16 on 99.7% of answers.
- v7 baseline on the full slice: template 99.2%, same_event 85.0%, BGL alert 92.8% (which line 91.9%),
  HDFS 67.2%, Hadoop 34.5%, root cause 75.2%, RCA fault type 48.7%.

## If v6 is flat (user, Oct 4)
- Keep v5 released and go straight to v7. Let v6 finish anyway: its per-benchmark results decide which v6
  additions (CLINC, ESCI, CRUXEval, SuperGPQA, MMLU-aux, extra PRs) go into v7.
- v7's model teacher is then v5 (not v6) for rows without Kimi/Gemma labels.

## Budget (updated Oct 4, evening)
- After v6: ~$9 of HF credit left (estimate; the user checks the real balance).
- v7 training + evals ~$22, release ~$1, full Decision Index run with engine_v2 + prefix reuse ~$29
  (1 GPU ~10.5 h, or 4 GPUs ~2.7 h at the same cost): **~$52 -> add ~$45**.
- Cheaper: full run on the Mac Studio with MLX 8-bit (free, ~1.3 days; submitted numbers would be the 8-bit
  model's) -> add ~$15-20.
- User will add $50 of HF credit when v7 is ready to launch (Oct 4): ~$59 total covers the all-HF path
  (~$52) with ~$7 margin. Confirm the balance before launching.
