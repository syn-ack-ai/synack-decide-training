---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
library_name: transformers
tags: [gemma4, moe, decision, system-one, jev, classification, routing]
---

# SynACK Decide 26B-A4B

SynACK Decide is an open "System One" decision model. Give it a `state` (any JSON) and typed questions:
multiple choice with 2–255 options, or yes/no ("noul"). It returns one answer per question, with a probability
for every option, from a **single forward pass**. There is no generation and no reasoning text, so a decision
takes a few hundred milliseconds. It speaks the Jev / Decision Index `/v1/systemone` wire format and runs on
llama.cpp, MLX or transformers.

It is a full fine-tune merge (LoRA r16 on attention, MLP and the fused MoE experts, folded into bf16 weights) of
[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it): 26B parameters, about 4B active
per token.

| Variant | Size | Use with |
|---|---:|---|
| [SkyPanther/synack-decide-26b-a4b](https://huggingface.co/SkyPanther/synack-decide-26b-a4b) | 52 GB, bf16 | transformers / CUDA (the published numbers) |
| [SkyPanther/synack-decide-26b-a4b-mlx-8bit](https://huggingface.co/SkyPanther/synack-decide-26b-a4b-mlx-8bit) | 25 GB | Apple Silicon (mlx-lm) |
| [SkyPanther/synack-decide-26b-a4b-GGUF](https://huggingface.co/SkyPanther/synack-decide-26b-a4b-GGUF) | Q8_0 / Q4_K_M | llama.cpp `llama-server` `/v1/systemone` |

Code (engines, server, prompt format, GGUF template): [github.com/syn-ack-ai/synack-decide](https://github.com/syn-ack-ai/synack-decide).
Write-up: [Teaching a small model to decide](https://syn-ack.ai/posts/teaching-a-small-model-to-decide).

## Results

**Decision Index 0.2.1: 59.5**, from a complete run of the suite (all 150,317 scoreable requests answered, scored
with the leaderboard's own scorer; results in
[SkyPanther/synack-decide-decision-index](https://huggingface.co/datasets/SkyPanther/synack-decide-decision-index)).
On the public board snapshot of 2026-09-28 that would place first (Jev 57.9, Surogate Rune 57.4, pplx-decider
56.4). The board maintainers re-measure latency and validate submitted runs themselves.

![Benchmark comparison](benchmark_card.png)

| Benchmark | SynACK Decide | Reference |
|---|---:|---|
| [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) 0.2.1, complete run | **59.5** | Jev (TypeSafe) 57.9, Surogate Rune 57.4, pplx-decider 56.4, Tev1-4B 29.2 |
| Decision Index areas: knowledge / language / retrieval / tools / arts | 44.4 / 63.9 / 62.4 / 79.1 / 45.9 | best on board: 51.4 / 63.5 / 63.5 / 79.3 / 41.9 |
| Private generalization set: 2,693 questions from 9 public datasets that are not in the suite and not in training | 76.8% | |
| Tev1 development set (1,300 records, argmax) | 1,170 | Tev1-4B: 1,179 (trained on that distribution) |
| PR review routing: AUC on 2,196 PRs from 16 unseen public repos | **0.774** | the deterministic complexity script it was built to beat: 0.737 |
| Ordinal questions, held-out phrasings ("third largest", "sort descending, take the 3rd"): choice / true-false | 82.3% / 78.1% | |
| Mean change in p(needs review) when a "low risk, no review needed" line is added to the PR description | **+0.009** | |

**Where the score comes from.** On the 26 Decision Index benchmarks that share a task type with the training data
(their train splits, decontaminated) the average is 72.9; on the 18 with no related training data it is 60.7. The
private generalization sets (17 public datasets never seen in training, 4,259 questions: 76.8% and 78.0%) are the
better guide to unseen decision tasks. Knowledge-heavy exams (GPQA, MMLU-Pro, HLE) are the weakest area: one
forward pass with about 4B active parameters has a ceiling there.

Most of the gain over Tev1 comes from the 255-label answer format: the model answers 77- and 151-option
questions (BANKING77, CLINC150) where a 24-letter model can't answer at all.

**PR routing** asks: will this pull request need substantive changes in review? The labels come from what
actually happened on public GitHub PRs (changes requested, rework after review, reverts).
- At a cut-off of 0.30 the model sends 59% of PRs to a cheap reviewer and misses 29.0% of the PRs that needed
  real review; the script misses 32.7% at the same share.
- AUC gain over the script: 95% CI −0.008 to +0.067, bootstrapped over repos. Better on 12 of 16 repos.
- Balanced accuracy 71.2% at a cut-off of 0.27 (the script: 68.5%).
- **Calibration:** mean p(needs review) 0.31 against a true rate of 0.28.
- **What it sees:** the title, the first 400 characters of the description, the complexity script's summary
  (size, risk signals, top 12 files) and the first 3,500 characters of the diff (about 57 lines; flagged when
  truncated). Send PRs in that format (`prcx/build_dataset.py` in the code repo).

**Speed** (one decision, median):
- bf16 on one RTX PRO 6000: 132 ms per Decision Index request (median over the full suite; p95 360 ms),
  148 ms per Tev1 record, 217 ms per PR.
- Q8_0 GGUF on an RTX 4090 (llama.cpp, 5 layers' experts in system RAM): about 220 ms for a short question.
- 8-bit MLX on an M5 Max: about 0.45 s per PR.

## Use

### llama.cpp (`llama-server`, Oct 2026 or newer)

```bash
llama-server -m synack-decide-26b-a4b-Q8_0.gguf --port 8080
curl -s localhost:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "I was charged twice for my October order.", "tier": "gold"},
  "questions": {
    "team":   {"type": "choice", "instructions": "Which team should handle this?",
               "criteria": {"billing": "Payments and refunds", "shipping": "Delivery", "technical": "Bugs, login"}},
    "urgent": {"type": "noul", "instructions": "Does this need a same-day reply?"}}}'
```

The GGUF carries decision type `nimble` (labels A–Z, AA, AB, …, up to 255 options) and a `systemone` chat
template that reproduces the training prompt byte for byte.

### Apple Silicon (MLX)

```bash
git clone https://github.com/syn-ack-ai/synack-decide && cd synack-decide && pip install -r requirements-mlx.txt
python serve.py --state "I was charged twice." --question "Which team?" --options billing,shipping,technical --yesno "Is it urgent?"
python serve.py --serve 8765      # POST /v1/systemone
```

### transformers / CUDA

```python
from engine_cuda import CudaEngine          # from github.com/syn-ack-ai/synack-decide
eng = CudaEngine("SkyPanther/synack-decide-26b-a4b")
response, _ = eng(state, questions)         # same wire format as above
```

### Prompt format (if you build your own engine)

The answer is the next token after the Gemma generation prompt (`<|turn>model\n<|channel>thought\n<channel|>`).
Read the logits of the option labels and take a softmax over them. The system turn is fixed. The user turn is
compact JSON:

```
{"state":<state>,"question":"<instructions>","options":{"A":"<description>","B":{"key":"k","description":"..."}}}
```

An option shows its description alone unless the key carries meaning, in which case it shows
`{"key","description"}`. An option with no description shows the bare key. A yes/no question uses the options
"Yes, the statement is true." and "No, the statement is false." The reference implementation is `common.py`.

## Training

One LoRA (r16/α32 on attention, MLP and the fused MoE experts) trained once on
[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) and folded into the bf16 weights:
lr 3e-5, 3,397 steps, 8 h 51 min on one RTX PRO 6000 (Hugging Face Jobs). Earlier rounds were stacked LoRA
passes; this release replaces them with a single clean run that keeps everything that proved out.

- **Data:** 163,511 examples: 106,905 unique (option order shuffled, 2 orders for questions with up to 10 options)
  plus a second pass over every public-train-split and synthetic row.
  - Public Hugging Face train splits: ANLI, WinoGrande, HellaSwag, RAGTruth, NLI4CT, GSM8K, Lichess puzzles,
    SuperGPQA, MMLU auxiliary train, Corr2Cause, BANKING77, CLINC150, Amazon ESCI, When2Call, New Yorker caption
    matching, iSarcasmEval (task A), VAST, Habermas Machine (train rounds), cfcolor (train ratings).
  - Together AI's Tev1 data, including MNLI and BoolQ.
  - Code-generated tasks with exact labels: dates, arithmetic, temporal, tracking, ordering, web-of-lies,
    navigation, boolean logic, CRUXEval-style code execution, ordinal questions with off-by-one distractors and
    paired true/false claims, POP909-style chord recognition, and the home-appliance simulator (offsets the suite
    doesn't use).
  - Financial sentiment (Twitter financial news, Financial PhraseBank train split) and SmartBugs contract
    vulnerability classes.
  - 13,064 labelled public GitHub pull requests, plus PR records whose description carries an author claim
    ("low risk, no review needed" or "high risk"), both balanced across labels and at the real 28% rate, so the
    claim carries no signal.
- **Decontamination:** every example was checked against all ~155k Decision Index suite rows and dropped on any
  shared segment of 5 or more words, or 20% overlap of 13-word windows. Boilerplate seen in more than 20 suite
  rows was ignored. No suite rows, and nothing from the private generalization set, were trained on.
- **Distillation:** gold cross-entropy plus cross-entropy against teacher option probabilities, applied only
  where the teacher agrees with gold: 0.8 Kimi-K3 + 0.2 Gemma-4-31B (top-20 logprobs) where available, otherwise
  the previous best stacked model's own probabilities.

## Limitations

- **One pass, no scratchpad.** Two-step compositions such as "third largest" reach about 82% on held-out
  phrasings but are still not reliable, and longer lists and chains degrade further.
  Let code do mechanical steps (sorting, counting, arithmetic over long inputs) and ask the model the judgement
  question.
- **Self-descriptions.** The model ignores "low risk / no review needed" claims in PR descriptions (mean
  shift +0.009). Other manipulation styles were not tested.
- **Format limits:** up to 255 options. `score` questions were not trained; llama.cpp will serve them as
  choices over the levels, untested. Mostly English.
- **Confidence is not correctness.** Probabilities are calibrated on in-distribution data (PR routing,
  validation) but can be overconfident on unfamiliar compositions.
- The PR routing gain was measured on public open-source repos. Check it on your own PRs before relying on it.

## License and data terms

The weights are released under Apache-2.0, following the base model
([Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license)).
- Some training sets carry their own terms: ANLI is CC-BY-NC-4.0, Financial PhraseBank is CC-BY-NC-SA-3.0, and
  the RACE portion of MMLU auxiliary train is non-commercial. iSarcasmEval, VAST and cfcolor have no clear
  licence statement.
- Teacher labels came from two open-weight models through OpenRouter: Gemma-4-31B (Apache-2.0) and Kimi-K3
  ([Kimi K3 License](https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE)). The Kimi K3 License
  allows fine-tuning and derivative works. Its extra terms apply only to Model-as-a-Service businesses above
  $20M revenue a year (separate agreement with Moonshot AI) and to products above 100M monthly users or $20M
  monthly revenue (display "Kimi K3"). The Kimi-K3 labels came from third-party hosts of the open weights; every
  request required logprobs, which Moonshot AI's own API does not offer, so Moonshot's API terms did not apply.
- Review these for commercial use.
- No private or proprietary data was used. The pull requests are public GitHub PRs.

Built by [SynACK](https://syn-ack.ai) with Claude (Anthropic) in Claude Code.
