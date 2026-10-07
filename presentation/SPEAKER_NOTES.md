# How we trained SynACK Decide v7: speaker notes

One section per slide, in order. The slides are in [SynACK-Decide-v7-how-we-trained-it.pdf](SynACK-Decide-v7-how-we-trained-it.pdf).

## 1. How we trained SynACK Decide v7

This talk explains how we built SynACK Decide v7 and how it reached 59.54 on the Jev Decision Index. It assumes no machine-learning background: the middle section explains how training works from the ground up. The #1 is against the board snapshot of 28 September 2026; our submission is pull request 71 on the harness repo and other submissions were pending at the time, so first place is not assured.

## 2. A decision model picks an option and says how sure it is

Our model is not a chatbot. It reads a state, here a support ticket, plus typed questions. A choice question has between 2 and 255 options; a yes/no question asks whether a statement is true. For every question it returns one answer and a probability for every option. The probabilities matter: they tell the caller how sure the model is, so a system can act automatically when the model is confident and ask a person when it is not. The example is made up and the numbers are illustrative.

## 3. The Jev Decision Index: 38 benchmarks, one number

The Jev Decision Index is a community leaderboard. Its harness sends 150,759 requests drawn from 38 benchmarks in five areas and turns the answers into one number. Scores are corrected for chance, so random guessing scores about zero, and a question a model cannot answer counts as wrong. Bars start at zero. Our first target was Tev1-4B from Together AI, which scores 29.2; the leaders on the 28 September board were around 57 to 58. v7 scored 59.54 on a complete run. Our entry is not on that snapshot; it was submitted afterwards as pull request 71.

## 4. We never taught the model to write JSON.

This is the single most important idea in the talk. Gemma 4 is a chat model that writes free text. We did not try to make it write JSON or explanations. Instead we trained it so that its very first answer token is always one of the option labels. Everything structured, the choice and the probabilities, is then read straight from the model's scores for those label tokens. That gives one forward pass per question, no text generation, no parsing and an output that is always valid.

## 5. Why every answer is a single token

A token is a piece of text from the model's fixed vocabulary of 262,144 pieces, each with an ID number. The model never sees letters or sentences directly, only a sequence of tokens. Common words are usually one token each: this question splits into 8. The tags in the prompt, like turn start and turn end, are special tokens, one each, that the chat format uses as structure. The answer is just the label, here C, which is one token. The labels work like exam choices and then like spreadsheet columns: A to Z, then AA, AB and so on, up to IV, for 255 labels in total. Every one of them is a single token in Gemma's vocabulary; AA is one token, not A plus A. We checked each one, and GZ is skipped because it is not a single token. Because the answer is always one token, the model's very first step after the prompt already gives a score for every label, so we can read the whole answer and its probabilities at once. If the answer were the option name, like technical support, it could take several tokens and several steps.

## 6. One example, as the model sees it

This is the whole training example as the model sees it, shortened in the middle: a few hundred tokens of prompt, then a one-token answer. The tags like turn and channel are special tokens from Gemma's chat format. The ticket and the question go in as compact JSON. The options are shuffled and labelled A, B, C; here billing, the right answer, landed on C. Everything up to the model's turn is the prompt and is not graded. The model is graded only on producing C and then the end-of-turn token, which teaches it to answer with one label and stop. The empty thought block is added by Gemma's own chat template when thinking is off, and we keep it so training and real use match exactly.

## 7. From label scores to an answer

When the model is used, the engine builds the same prompt without shuffling, runs the model once, and looks only at the scores of the option labels at the answer position. A softmax turns those scores into probabilities that add up to one, and the labels are mapped back to the option names. The choice is simply the most likely option; for a yes/no question we report the probability of yes. Because nothing is generated, the output can never be malformed. Questions that share the same state, like the two here, reuse the state's computation, which made inference about 1.4 times faster.

## 8. In plain words

A one-slide recap of how the model answers, in plain words. Each choice gets a label, A, B, C and so on, and each label is a single token. The model reads the question once and scores every label in one step. We turn those scores into probabilities that add up to 100 percent. Then we return the most likely option, mapped back to its name, plus the probability of every option, so the caller can see how sure the model is. In the ticket example the model picks billing with 91 percent; technical gets 8 percent and shipping 1 percent. A spread like 0.91 against 0.08 means the model is confident; something like 0.45 against 0.40 would mean it is torn, and a system could then ask a person. The option descriptions themselves, like Billing and refunds, are ordinary text in the prompt and take several tokens; only the labels are single tokens.

## 9. The model scores every possible next token

A language model reads tokens, which are pieces of text from a fixed vocabulary; Gemma 4 has 262,144 of them, and single letters like A, B and C are tokens too. Given the text so far, the model outputs one raw score, called a logit, for every token. Softmax turns the scores into probabilities: raise e, about 2.718, to the power of each score and divide by the total. Here, with a toy five-token vocabulary, the model before training puts most of its probability on starting a sentence with The or Billing. Only 15.8 percent is on C, the right label. Training is about moving probability onto C.

## 10. Measuring a mistake: cross-entropy

To improve the model we need one number that says how wrong it was: the loss. For predicting the right token, the standard loss is cross-entropy, minus the log of the probability the model gave to the right answer. If it gave the right answer probability 1, the loss is 0. At 0.5 it is 0.69. Our example's 0.158 gives 1.85. And if the model was confidently wrong, giving the right answer one in a thousand, the loss jumps to 6.9. So the loss punishes confident mistakes hardest. Across all training examples, training tries to make the average surprise small. In our setup the graded tokens are the label and the end-of-turn token after it.

## 11. Distillation: copy a stronger model's confidence

The gold answer is a hard label: it only says C is right. A strong model knows more, for example that B, technical, was a reasonable second guess because the app was also crashing. Distillation means training the student to match a teacher's full probability list, called soft labels. We compare the teacher with the student's probabilities over the option labels only, and the loss is again a cross-entropy, now weighted by the teacher's probabilities; here it is 0.730, and it is smallest when the student's spread matches the teacher's. This is what teaches the model how confident to be. One safety rule: if the teacher's top choice disagrees with the gold answer, we ignore the teacher for that example, so the student never learns a teacher's mistakes. The teacher numbers here are illustrative; the arithmetic is exact.

## 12. Gradients: which way to move

To lower the loss we need to know, for each score, whether making it slightly bigger would raise or lower the loss, and by how much. That is the gradient. For softmax with cross-entropy it has a very simple form: the model's probability minus the target probability. The right label C has target 1, so its gradient is negative: raise it. Every other token has target 0, so its gradient is positive: lower it, most of all for The, which was stealing the most probability. The teacher loss adds a similar push on A, B and C. If we could move the scores directly and took one step of size one against both gradients combined, the probability of C would jump from 0.158 to 0.482 and both losses would roughly halve. Real training repeats this in tiny steps over thousands of examples.

## 13. Backpropagation: share out the blame

We cannot set the scores directly: they are produced by about 26 billion weights spread over 30 layers. A weight deep in layer 5 affects the scores only through every layer after it. Backpropagation works out, for every weight we are training, how much the loss would change if that weight changed slightly. It runs in two passes. The forward pass runs the example through the model to get the scores and the loss, keeping the intermediate results. The backward pass starts from the score gradients we just saw and walks back through the layers, using the chain rule: if A affects B and B affects C, A's effect on C is the two effects multiplied. The picture to keep in mind is sharing out the blame. Libraries such as PyTorch do this automatically with one call. To save memory on long prompts we keep only some intermediate results and recompute the rest, which is called gradient checkpointing.

## 14. LoRA: train a small add-on

Changing all 26 billion weights would need several times the model's memory for training bookkeeping, and it risks erasing what the model already knows. LoRA, low-rank adaptation, freezes the original weights and adds a thin pair of matrices, B and A, next to each big matrix; the layer uses W plus B times A, scaled. The model's hidden size is 2,816, so a square 2,816 by 2,816 matrix has about 7.9 million numbers, while its rank-16 add-on has 90,112, about 1.1 percent. We put LoRA on the attention and feed-forward projections and on all 128 experts in every layer: about 494 million trainable numbers, roughly 2 percent of the model. Backpropagation only computes gradients for these. B starts at zero, so at step zero the model is exactly the original Gemma. After training, B times A is added into W once, so the released model has the same size and speed as the original.

## 15. The training loop, 3,397 times

One training step: take a batch of at least 32 examples, grouped by length so little memory is wasted on padding; run the forward pass and the loss; run the backward pass to get gradients for the LoRA numbers; then two safety checks. Gradient clipping caps an unusually large update, and the spike guard skips a step entirely if its loss is more than 8 times the recent average; in the v7 run nothing was skipped. Then the AdamW optimizer moves each trainable number a small step against its gradient, using running averages so each number gets its own smooth step size. The learning rate sets the step size. Too big damages the model: an early run at 8e-5 collapsed after about 175 steps. v7 used 3e-5, ramped up over the first 5 percent of steps and then lowered along a cosine curve so the last steps are fine adjustments. One pass over 163,511 examples was 3,397 steps, 8 hours 51 minutes on one 96 GB GPU.

## 16. Nine steps from Gemma to 59.54

Here is the whole path in nine steps; the next slides go through the important ones. We start from Google's open Gemma 4 26B-A4B, a mixture-of-experts model with 26 billion weights of which about 4 billion are active for each token. We collect training data from public dataset training splits, code-generated puzzles with computed answers, Tev1's records and public GitHub pull requests. We remove anything that overlaps the leaderboard's test rows, mix the sources in proportion to the leaderboard's areas, label them with two teacher models, build the final set of 163,511 examples, train one LoRA on a rented 96 GB GPU, check the result on data it never saw, then merge, publish in three formats and run the full leaderboard. Every step has a script in the private repository.

## 17. 163,511 examples from seven sources

The released v7 trained on 163,511 examples. The biggest share comes from the training splits of 15 public datasets, such as ANLI, HellaSwag, GSM8K and BANKING77, and from puzzles our own code generates with computed answers, like logic, dates and code output; both of these appear twice, a deliberate second pass. Then Tev1's training records, public GitHub pull requests labelled by whether they needed real review changes, plus balanced examples of PR descriptions that claim to be low or high risk, so the model learns to ignore such claims. The gap sets were added in v5 for weak areas. Some benchmarks were kept out of training completely, so we could tell whether the model was really getting better at reasoning or just at our training data.

## 18. Two big models, blended 80/20

For the soft labels we used two large models through the OpenRouter API. For each training question we sent the exact prompt the student sees, allowed one answer token, and asked for the top 20 most likely first tokens with their probabilities. We kept only the option labels, and rejected an answer if less than half of the probability was on labels, for example when the teacher started to explain instead of answering. On 1,018 held-out questions Kimi-K3 was right 80.6 percent of the time and Gemma-4-31B 77.1 percent, and blending them 80/20 gave 81.3 percent, better than either alone. Labelling 34,357 questions cost $24.80 with Kimi and $1.96 with Gemma. Rows without an external label used the previous release, v5, as the teacher, which is called self-distillation.

## 19. Judge the model on what it never saw

A leaderboard score is easy to inflate without making a model better, so we built three guards. First, decontamination: any training record that shares a five-word passage with the leaderboard's test rows, or 20 percent of its 13-word runs, is dropped. Second, a never-trained group: benchmarks for which we used no related training data at all. If a new version only improves where we trained, it learned our data rather than the skill. Third, private test sets built from 17 datasets that are used nowhere else, plus pull requests from 16 repositories the model never saw and targeted probes for known weaknesses. The rule we followed is no benchmaxing: we never add data just because a benchmark is low, and we release a version only if it improves on what it never saw.

## 20. Six versions in five days

We trained six versions in about five days, from 2 to 6 October. v2 established the format: the compact prompt and the 255 single-token labels. v3 moved the mixture-of-experts model to full precision on a rented GPU. v4 and v5 each continued training from the previous version with fixes for specific weaknesses, and v5 became the first version we were happy with. v6 stacked another round on v5; its sample score was flat and it lost ground on the never-trained group, which is the shape of benchmaxing, so we did not release it. Old v7 tried a clean, fresh training run and failed in an instructive way, which is the next slide. v7 fixed that and is the released model. The sample scores are quick 3,000-row estimates with an uncertainty of about three to four points; v7's complete run scored 1.57 below its sample.

## 21. A fresh run must match the dose

Old v7 is the most useful failure of the project. The idea was sound: instead of stacking yet another round on v5, train one clean LoRA from the Gemma base on all the proven data. But v5 had effectively seen its core data about three times across its stacked rounds, and the fresh run saw it once. The result looked fine on the overall sample, but on benchmarks we never trained on it lost multi-step reasoning: SGD down 14.5, SATA-Bench down 13.8, CRUXEval down 12.1 and HLE down 5.2. It had also dropped the balanced PR-claim data, so a pull request description saying low risk could sway it again. v7 fixed both: a second pass over the public-dataset and puzzle rows, the PR-claim rows put back and 8.5 thousand extra reasoning items. The lesson we wrote down: before any fresh run, count how many passes each source got in the model it replaces, and list every past fix together with the test that proves it.

## 22. The official run: 59.54

The complete run of the Decision Index 0.2.1 scored 59.54. On the 28 September board that would be first of 72, ahead of Jev at 57.91; our submission is pending as pull request 71, and other pending entries may change the order. 150,317 requests were scored with no errors. 167 prompts, all from one retrieval benchmark, were longer than the 32,768-token limit and count as wrong. Median latency was 132 milliseconds per request on one GPU. By area, v7 is close to the best in language, retrieval and tools, ahead in arts and human taste, and clearly behind in knowledge: exam-style questions like GPQA and MMLU-Pro reward a model's stored knowledge, and that is where a 26B model with about 4B active parameters is weakest. The best-on-board column takes the best model in each area separately.

## 23. One model, three formats, public

After training, the LoRA was merged into the Gemma weights and published in three formats on Hugging Face, all public: bf16 full precision for NVIDIA GPUs, which is the exact version that produced 59.54; an 8-bit MLX version for Apple Silicon Macs, 25 gigabytes, which makes the same routing decision as bf16 on 97.6 percent of 500 test pull requests; and GGUF files for llama.cpp, at 27 and 17 gigabytes, where Q8_0 gives the same answer on 97.6 percent of 2,693 held-out questions. Recorded costs for the final model: $26.76 of teacher labels, about $26.90 for the v7 training job including its evaluations, and about $26 estimated before launch for the complete leaderboard run; the final bill for that run was not recorded. Earlier versions and experiments cost extra and are not included here.

## 24. What we would tell the next team

Six lessons. One: the answer format was the ceiling; Tev1's 24-letter labels score zero on benchmarks with 77 or 151 options, and our 255 single-token labels removed that limit. Two: distil only where the teacher agrees with the right answer, and a blend of teachers beat either alone. Three: judge on what the model never saw; v6 improved the sample score and we still did not release it. Four: a fresh run must match the training dose of the model it replaces and carry every earlier fix, which is what old v7 taught us. Five: small steps with safety guards; a learning rate of 8e-5 collapsed after about 175 steps, while 3e-5 with a spike guard trained cleanly. Six: the inference engine changes the answers; vLLM was 4.5 times faster but cost about half an index point on near-ties, so the official run used the same numerics as training.

## 25. Questions?

Everything in this talk is in the private GitHub repository, with a README that walks through each step and links to every script and dataset, plus a beginner explainer of the training maths. The models and the official results are public on Hugging Face, and the leaderboard submission is pull request 71 on the harness repository. The repository is private, so people need to be given access before they can open it.
