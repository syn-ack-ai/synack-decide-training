# How the training works, from the ground up

This page explains, without assuming any machine-learning background, what happens when we "train" or "distil"
SynACK Decide. It follows one made-up support ticket through the whole process: what the model computes, how we
measure its mistakes, how the teacher models come in, how the mistake is traced back through the network
(backpropagation), and how the weights are changed.

The small examples use made-up numbers on a 5-word vocabulary so the arithmetic can be followed by hand. Every
number in them is computed exactly. The settings at the end (learning rate, steps and so on) are the real values
from the v7 run. The code is [v2/train_hf.py](../v2/train_hf.py) and [v2/common.py](../v2/common.py).

## Contents

1. [What the model computes](#1-what-the-model-computes)
2. [Measuring a mistake: cross-entropy](#2-measuring-a-mistake-cross-entropy)
3. [Distillation: learning from a teacher's probabilities](#3-distillation-learning-from-a-teachers-probabilities)
4. [The full loss for one example](#4-the-full-loss-for-one-example)
5. [Gradients: which way to move](#5-gradients-which-way-to-move)
6. [Backpropagation: from the answer back to every weight](#6-backpropagation-from-the-answer-back-to-every-weight)
7. [LoRA: training a small add-on instead of the whole model](#7-lora-training-a-small-add-on-instead-of-the-whole-model)
8. [The training loop](#8-the-training-loop)
9. [How we know it worked](#9-how-we-know-it-worked)
10. [Glossary](#10-glossary)

## 1. What the model computes

**Text becomes tokens.** A language model does not read letters or words. It reads **tokens**: pieces of text
from a fixed list called the **vocabulary**. Gemma 4's vocabulary has 262,144 tokens. A token can be a whole word
("billing"), part of a word, a punctuation mark, or a single letter such as `A`. Each token has a number (an ID).

**The model predicts the next token.** Given all the tokens so far, the model outputs one score for every token in
the vocabulary: how likely that token is to come next. These raw scores are called **logits**. They can be any
number, positive or negative. Higher means "more likely".

**Softmax turns scores into probabilities.** To turn logits into probabilities that are all positive and add up
to 1, the model uses **softmax**: raise *e* (about 2.718) to the power of each score, then divide each result by
the total.

Take our support ticket. The prompt lists three options, labelled `A` (shipping), `B` (technical) and `C`
(billing), and the right answer is `C`. Imagine a tiny vocabulary of just five tokens. Before training, the model
might produce:

| Token | Logit | e^logit | Probability (÷ 46.86) |
|---|---:|---:|---:|
| `A` | 1.0 | 2.72 | 0.058 |
| `B` | 1.5 | 4.48 | 0.096 |
| `C` | 2.0 | 7.39 | **0.158** |
| `The` | 3.0 | 20.09 | 0.429 |
| `Billing` | 2.5 | 12.18 | 0.260 |
| **Total** | | **46.86** | **1.000** |

The model is a chat model, so it "wants" to start a sentence ("The billing team...") rather than give a label.
Only 15.8% of its probability is on the right label.

**Where do the logits come from?** The model is a long chain of arithmetic. Each token is turned into a list of
2,816 numbers (its "hidden state"). That list passes through 30 **layers**. Each layer mixes information between
tokens (attention) and transforms each token's numbers (a feed-forward block; in this mixture-of-experts model,
each token is routed to a few of 128 "expert" blocks). At the end, the last token's numbers are multiplied by an
output matrix to give one logit per vocabulary token. All of this arithmetic is set by **weights**: about 26
billion fixed numbers learned by Google during pre-training. Training means changing some of those numbers.

## 2. Measuring a mistake: cross-entropy

To improve the model we need one number that says how wrong it was. That number is the **loss**. For "predict the
right token", the standard loss is **cross-entropy**:

> loss = −log(probability the model gave to the right answer)

- If the model gives the right answer probability 1.0, the loss is −log(1) = **0**: no mistake.
- If it gives 0.5, the loss is 0.69. If it gives 0.158, as above, the loss is **1.847**.
- If it gives 0.001, the loss is 6.9. The loss grows quickly as the model becomes confidently wrong.

Think of it as "how surprised was the model by the right answer". Training tries to make this surprise small,
on average, across all training examples.

In our training, this cross-entropy is computed on two tokens: the label (`C`) and the end-of-turn token that
follows it. The second one teaches the model to stop after the label. Over the whole vocabulary, this pushes
probability away from `The`, `Billing` and every other non-label token, and onto `C`.

## 3. Distillation: learning from a teacher's probabilities

The gold answer says only "C is right". It says nothing about the other options: was `B` (technical) a reasonable
second guess, or nonsense? A strong model knows more than the gold label shows. **Distillation** means training
a model (the **student**) to copy the probabilities of a stronger model (the **teacher**), not just the single
right answer. These full probability lists are called **soft labels**. The gold answer alone is a **hard label**.

**Our teachers.** We asked two large models, Kimi-K3 and Gemma-4-31B, each training question through the
OpenRouter API ([v2/teacher_api.py](../v2/teacher_api.py)):

1. Send the exact training prompt, and ask for just one token of answer with the top 20 most likely first tokens
   and their log-probabilities (`max_tokens 1`, `temperature 0`, `top_logprobs 20`).
2. Keep only the option labels (`A`, `B`, `C`, ...) and turn their log-probabilities back into probabilities.
   Labels outside the top 20 share the small leftover probability.
3. Reject the answer if less than half of the teacher's probability was on option labels at all (for example if
   it started to explain instead of answering).
4. Blend the two teachers: 0.8 x Kimi-K3 + 0.2 x Gemma-4-31B. On 1,018 held-out questions this blend was more
   accurate (81.3%) than either teacher alone (Kimi-K3 80.6%, Gemma-4-31B 77.1%).

For rows that the external teachers did not label, the previous released model (v5) acted as the teacher
([v2/self_teacher.py](../v2/self_teacher.py)). This is called **self-distillation**: it keeps what earlier
versions had learned.

**One safety rule.** A teacher's probabilities are used only when the teacher's top choice matches the gold
answer. If the teacher was wrong, the example trains on the gold answer alone, so the student never learns a
teacher's mistakes.

**The distillation loss.** Suppose the teacher gave `A` 0.02, `B` 0.06, `C` 0.92. We compare it with the
student's probabilities **over the option labels only**. A softmax over just `A`, `B` and `C`, ignoring every
other token, gives:

| Label | Teacher (target) | Student (softmax over A, B, C only) |
|---|---:|---:|
| `A` | 0.02 | 0.186 |
| `B` | 0.06 | 0.307 |
| `C` | 0.92 | 0.506 |

The loss is again a cross-entropy, now weighted by the teacher's probabilities:

> distillation loss = −(0.02 x log 0.186 + 0.06 x log 0.307 + 0.92 x log 0.506) = **0.730**

It is smallest when the student's spread exactly matches the teacher's. This teaches the model *how confident* to
be, which is why the released model returns useful probabilities, not just a choice.

## 4. The full loss for one example

For each training example ([v2/train_hf.py](../v2/train_hf.py), `batch_loss`):

> **loss = cross-entropy on `C` and the end-of-turn token + 1.0 x distillation loss over the option labels**

- The first part teaches the **format**: always answer with one listed label, then stop.
- The second part teaches the **probabilities**: how the student should spread its confidence over the options.
  It is only there when a teacher label exists and agrees with gold.

In our example the label part is 1.847 and the distillation part is 0.730. (The real loss also includes the
end-of-turn token, left out here to keep the arithmetic short.)

## 5. Gradients: which way to move

We want to change the model so the loss goes down. The question is: for each number we could change, does making
it slightly bigger raise or lower the loss, and by how much? That sensitivity is the **gradient**.

For softmax followed by cross-entropy, the gradient with respect to each logit has a very simple form:

> gradient of a logit = (the model's probability) − (the target probability)

A **positive** gradient means "this score is too high, lower it". A **negative** gradient means "too low, raise
it".

| Token | Probability | Target | Gradient, label loss | Gradient, distillation (A, B, C only) | Total |
|---|---:|---:|---:|---:|---:|
| `A` | 0.058 | 0 | +0.058 | 0.186 − 0.02 = +0.166 | +0.224 |
| `B` | 0.096 | 0 | +0.096 | 0.307 − 0.06 = +0.247 | +0.343 |
| `C` | 0.158 | 1 | −0.842 | 0.506 − 0.92 = −0.414 | **−1.256** |
| `The` | 0.429 | 0 | +0.429 | | +0.429 |
| `Billing` | 0.260 | 0 | +0.260 | | +0.260 |

**What one step would do.** If we could move the logits directly, one step of size 1 against the gradient would
give new logits A 0.78, B 1.16, C 3.26, The 2.57, Billing 2.24. Then:

| | Before | After one step |
|---|---:|---:|
| Probability of `C` (whole vocabulary) | 0.158 | **0.482** |
| Label loss | 1.847 | 0.729 |
| Student over A, B, C | 0.186 / 0.307 / 0.506 | 0.069 / 0.102 / **0.829** |
| Distillation loss | 0.730 | 0.363 |

After many such steps on many examples, a trained model gives something like `C` 0.97 over the whole vocabulary,
with almost nothing left on `The` or `Billing`.

In reality we cannot set logits directly. They are outputs of the network, and the only things we can change are
the weights. That is what backpropagation is for.

## 6. Backpropagation: from the answer back to every weight

The logits come from the last layer, which takes its input from the layer before it, and so on back through all 30
layers. A weight deep inside layer 5 affects the logits only through everything that happens after it.

**Backpropagation** works out, for every weight being trained, how much the loss would change if that weight
changed slightly. It uses the chain rule from calculus: if A affects B and B affects C, then how much A affects C
is (how much A affects B) x (how much B affects C). Training runs in two passes:

1. **Forward pass.** Run the example through the model, layer by layer, to get the logits and the loss. Keep the
   intermediate results.
2. **Backward pass.** Start from the gradients of the logits (the table above) and go backwards, layer by layer.
   At each layer, use the saved intermediate results to compute (a) the gradient for each of that layer's
   trainable weights and (b) the gradient to pass to the layer before it.

One way to picture it is assigning blame: the final error is shared out backwards, and every weight learns how
much it contributed and in which direction. Libraries (PyTorch here) do this automatically. The code calls
`loss.backward()` and every trainable weight gets its gradient.

**Memory trick.** Keeping every intermediate result for a 26-billion-weight model on a long prompt needs a lot of
memory. **Gradient checkpointing** keeps only some of them and recomputes the rest during the backward pass. That
trades some extra compute for much less memory.

## 7. LoRA: training a small add-on instead of the whole model

Changing all 26 billion weights would need several times the model's memory just for the training bookkeeping,
and it risks erasing what the model already knows. Instead we use **LoRA** (low-rank adaptation):

- The original weights are **frozen**: they never change.
- Next to a frozen weight matrix *W*, LoRA adds two small matrices, *A* and *B*. The layer then uses
  *W* + (α / r) x *B* x *A*.
- *r*, the **rank**, sets how small they are. We used r = 16 and α = 32, so the add-on is scaled by 2.
- *B* starts at zero, so at the start the model behaves exactly like the original Gemma.

**Size.** The model's hidden size is 2,816. A 2,816 x 2,816 weight matrix has 7,929,856 numbers. Its rank-16
add-on has 2,816 x 16 + 16 x 2,816 = 90,112 numbers, about 1.1% of the original.

We put LoRA on the attention projections, the feed-forward projections and the 128 experts in every layer. That
is about 494 million trainable numbers, roughly 2% of the model. Backpropagation computes gradients only for
these. When training is done, *B* x *A* is added into *W* once ("merging"), so the released model has the same
size and speed as the original Gemma.

## 8. The training loop

One **step** of training:

1. Take a **batch**: at least 32 training examples. Examples of similar length are grouped into micro-batches of
   up to 24,000 tokens, so less memory is wasted on padding.
2. Forward pass, loss, backward pass for each micro-batch. Gradients add up across the micro-batches
   (**gradient accumulation**) until the batch is complete.
3. **Gradient clipping:** if the gradients are unusually large overall, scale them down to a fixed size (1.0).
   This stops one strange batch from making a huge, damaging change.
4. **Spike guard:** if the batch's loss is more than 8 times the recent average, skip the step entirely. In the
   v7 run, no steps were skipped.
5. **Update the weights** with the **AdamW** optimizer. Each trainable number moves a small amount against its
   gradient. AdamW keeps running averages of each number's past gradients and of their size, so each number gets
   its own sensible step size and the steps are smoother than with raw gradients.
6. Clear the gradients and go to the next batch.

**Learning rate.** The learning rate sets how big the steps are. Too big and the model is damaged: an earlier
experiment at 8e-5 collapsed after about 175 steps. Too small and nothing is learned. v7 used **3e-5**
(0.00003), with two adjustments:

- **Warmup:** the rate starts near zero and rises over the first 5% of steps (169 of 3,397), while the
  optimizer's running averages settle.
- **Cosine decay:** after warmup, the rate falls smoothly towards zero along a cosine curve, so the last steps
  make only fine adjustments.

**Numbers for v7:** 163,511 examples, one pass (**epoch**) over them, which is 3,397 steps. It ran on one 96 GB
GPU for 8 h 51 min, in bf16 (a 16-bit number format that halves memory compared with 32-bit).

## 9. How we know it worked

- **Validation:** every 1,000 steps, the model answers 2,011 examples it never trains on. We keep the checkpoint
  with the best validation accuracy. For v7 that was 79.39%, at the last step.
- **Overfitting:** a model can memorise its training data and get worse on anything new. To catch that, we judge
  each version on data it has never seen: private generalization sets built from datasets not used anywhere in
  training, a group of leaderboard benchmarks with no related training data, held-out GitHub repositories for PR
  routing, and targeted probes. We decide on these, not on the training benchmarks ("no benchmaxing").
- **Final test:** the complete Decision Index 0.2.1 run, 150,759 requests, scored 59.54.

## 10. Glossary

| Term | Meaning |
|---|---|
| Token | A piece of text from the model's fixed vocabulary (262,144 for Gemma 4) |
| Logit | The raw score the model gives each possible next token |
| Softmax | Turns logits into probabilities that are positive and add up to 1 |
| Loss | One number for how wrong the model was; training makes it smaller |
| Cross-entropy | −log(probability given to the right answer); the standard loss for predicting tokens |
| Hard label / soft label | The single right answer / a full probability list over all options |
| Teacher / student | The stronger model whose probabilities are copied / the model being trained |
| Distillation | Training a student to match a teacher's probabilities |
| Self-distillation | Using an earlier version of the same model as the teacher |
| Gradient | How much, and in which direction, the loss changes when one number changes slightly |
| Backpropagation | Computing every trainable number's gradient by going backwards through the layers |
| Weights | The model's learned numbers (about 26 billion here) |
| LoRA | Small trainable add-on matrices next to frozen weights; merged in at the end |
| Rank (r) | How small the LoRA matrices are (16 here) |
| Batch / step | A group of examples / one weight update from one batch |
| Epoch | One full pass over the training data |
| Learning rate | How big each update step is (3e-5 here) |
| Warmup / cosine decay | Ramping the learning rate up at the start / smoothly down to the end |
| Gradient clipping | Capping how large one update can be |
| AdamW | The optimizer: gives each number its own smoothed step size |
| Validation set | Examples held back to measure progress during training |
| Overfitting | Memorising the training data instead of learning something that transfers |
| Checkpoint | A saved copy of the trainable weights at one point in training |
| bf16 | A 16-bit number format used to save memory |
