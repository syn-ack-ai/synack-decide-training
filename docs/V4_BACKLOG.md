# v4 backlog

Found while testing v3 (2026-10-03). Nothing here is trained yet.

## 1. Two-step ordinal questions ("third largest")

Each step works on its own: sorting is right (0.99), indexing an already-sorted list is right (0.99), and
"second smallest" is right (0.92–0.97). Combining the steps when counting from the top fails: "sort descending,
third element" and "third largest" pick the second largest at 0.70–0.85. The true/false version is a guess: the
correct claim gets 0.50 and an off-by-one claim gets 0.75.

Cause: the model answers in one forward pass and was never trained to chain steps. The training mix has almost
no ordinal questions. Of the 34,357 records, the 22 "Nth largest" hits are incidental ANLI phrases, and only
15 mention "descending".

Fix plan:
- **Data:** add `synthetic/ordinal` next to `synthetic/dates`, with every label computed in code.
  - Task types: Nth largest, Nth smallest, sort then index, last but one, position counted from either end.
  - Lists: 3–12 items, with duplicates, negatives and decimals. The list goes in the state or in the question.
  - Distractors always include the N−1 and N+1 neighbours. That is the actual error; random wrong options teach nothing.
- **Paraphrases:** write each item several ways with the same answer ("3rd largest", "descending position 3",
  and "2nd smallest" for a 4-item list). Optionally add a consistency loss between paraphrases; `train_v2.py`
  and `train_hf.py` don't have one yet.
- **True/false:** pair every true claim with an off-by-one false claim.
- **Soft targets:** the reasoning-only teachers (GLM-5.3, Qwen3.8) refused one-token logprobs. Instead, sample a
  reasoning teacher k times and use the vote shares, or use code-exact labels with mild label smoothing.
- **Mix share:** 1–3% so the routing and classification data isn't diluted.
- **Eval:** decontaminate against the tester's probe questions. Hold out the 16 probe variants as a regression
  eval, scored on accuracy and on agreement across phrasings.
- **No-retrain mitigation:** fit one temperature on validation to soften overconfidence (the engines accept
  `temperature`; llama.cpp reads `<arch>.decision.temperature.*`).
- **Limit:** fine-tuning sharpens short compositions but won't add depth. Longer lists (more than about 10
  items) and longer chains should still have code do the mechanical step first. In tests, the two-step version
  went from 0.70 wrong to 0.95 right with code preprocessing.

## 2. Self-description injection in PR routing

PR #729 dropped from p=0.50 to 0.165 because its description claimed it was "low risk, no review needed".
Fix: add routing examples where the description says it is low risk or needs no review but the label is still
`yes`, and examples where the description says it is risky but the label is `no`. The aim is for the model to
judge the change, not the author's claim about it.
