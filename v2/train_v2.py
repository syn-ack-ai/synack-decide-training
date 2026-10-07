"""v2 trainer: Gemma 4 (12B QLoRA or E4B bf16) LoRA with optional teacher distillation.

Data: JSONL from common.expand (+ optional "teacher": {label: prob}). Loss per example:
  CE(gold answer tokens, full vocab)                                  -- format + gold label
  + lam * CE(teacher || softmax over the example's option labels)    -- only if teacher argmax == gold
One sequence at a time with gradient accumulation; logits only at answer positions.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from train_cuda import offload_per_layer_embeddings  # noqa: E402

TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def load_model(path, four_bit):
    if four_bit:
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               llm_int8_skip_modules=["vision_tower", "audio_tower", "embed_vision", "embed_audio",
                                                      "lm_head"])
        model = AutoModelForCausalLM.from_pretrained(path, quantization_config=q, dtype=torch.bfloat16,
                                                     device_map={"": 0})
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
        offload_per_layer_embeddings(model)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    return model


def encode(tok, ex, max_len, label_ids):
    # Train on exactly what inference sees: the generation prompt (Gemma 4 12B/31B append an empty
    # thought channel there when thinking is off) followed by the answer label and end of turn.
    m = ex["messages"]
    prompt = tok.apply_chat_template(m[:-1], add_generation_prompt=True, enable_thinking=False,
                                     tokenize=True, return_dict=False)
    full = prompt + tok.encode(m[-1]["content"] + "<turn|>", add_special_tokens=False)
    if len(full) > max_len:
        return None
    opt = [label_ids[l] for l in ex["labels"]]
    teacher = None
    t = ex.get("teacher")
    if t and max(t, key=t.get) == ex["gold"]:
        teacher = [t.get(l, 0.0) for l in ex["labels"]]
    return full, len(prompt), opt, teacher


def example_loss(model, item, lam, device):
    ids, off, opt, teacher = item
    x = torch.tensor([ids], device=device)
    keep = torch.arange(off - 1, len(ids) - 1, device=device)
    logits = model(input_ids=x, logits_to_keep=keep).logits[0].float()
    tgt = x[0, off:]
    loss = torch.nn.functional.cross_entropy(logits, tgt, reduction="sum")
    n = len(tgt)
    correct = int(logits[0, opt].argmax().item() == opt.index(int(tgt[0])))
    if teacher is not None and lam > 0:
        logp = torch.log_softmax(logits[0, opt], -1)
        p = torch.tensor(teacher, device=device)
        loss = loss + lam * -(p / p.sum() * logp).sum()
    return loss, n, correct


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    tot, n, acc = 0.0, 0, 0
    for item in data:
        l, k, c = example_loss(model, item, 0.0, device)
        tot, n, acc = tot + l.item(), n + k, acc + c
    model.train()
    return tot / max(n, 1), acc / max(len(data), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/gemma4-12b-it")
    p.add_argument("--train", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--four-bit", action="store_true")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--spike", type=float, default=4.0, help="skip a step whose loss exceeds this x the running mean")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--iters", type=int)
    p.add_argument("--val-n", type=int, default=400)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--save-every", type=int, default=500)
    a = p.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(a.model)
    labels = json.loads((Path(__file__).resolve().parents[1] / "bench/labels255.json").read_text())
    label_ids = {l: tok.encode(l, add_special_tokens=False)[0] for l in labels}
    model = load_model(a.model, a.four_bit)
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, bias="none",
                                             target_modules=TARGETS))
    model.train()
    model.print_trainable_parameters()
    device = next(model.parameters()).device

    def load(path, limit=None):
        rows = [json.loads(l) for l in open(path)]
        if limit:
            random.Random(7).shuffle(rows)
            rows = rows[:limit]
        enc = [encode(tok, r, a.max_len, label_ids) for r in rows]
        return [e for e in enc if e], sum(e is None for e in enc)

    train, skipped = load(a.train)
    valid, _ = load(a.valid, a.val_n)
    per_epoch = len(train) // a.batch
    total = int(per_epoch * a.epochs)
    steps = min(total, a.iters) if a.iters else total
    n_teacher = sum(t[3] is not None for t in train)
    print(f"train {len(train)} (skipped {skipped} over {a.max_len} tok), teacher-supervised {n_teacher}; "
          f"{per_epoch} steps/epoch, running {steps}; val {len(valid)}", flush=True)

    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(a.warmup * total), total)
    order = []
    while len(order) < steps * a.batch:
        ep = list(range(len(train)))
        random.Random(42 + len(order)).shuffle(ep)
        order += ep
    if not a.iters:
        vl, va = evaluate(model, valid, device)
        print(f"     0 val loss {vl:.4f} acc {va:.4f}", flush=True)
    t0 = window = time.time()
    run, run_acc, ema, skipped_steps = 0.0, 0, None, 0
    best = (-1.0, None)
    for step in range(1, steps + 1):
        batch = [train[i] for i in order[(step - 1) * a.batch: step * a.batch]]
        n_tok = sum(len(b[0]) - b[1] for b in batch)
        step_loss = 0.0
        for item in batch:
            l, _, c = example_loss(model, item, a.lam, device)
            (l / n_tok).backward()
            step_loss += l.item() / n_tok
            run_acc += c
        run += step_loss
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if ema is not None and (step_loss > a.spike * ema or not torch.isfinite(torch.tensor(gnorm))):
            skipped_steps += 1
            print(f"{step:6d} SKIP spike: loss {step_loss:.3f} vs mean {ema:.3f}, grad norm {gnorm:.2f}", flush=True)
        else:
            opt.step()
            ema = step_loss if ema is None else 0.95 * ema + 0.05 * step_loss
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            dt = (time.time() - window) / 10
            print(f"{step:6d} loss {run / 10:.4f} acc {run_acc / (10 * a.batch):.3f}  {dt:.2f} s/step  "
                  f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB  eta {dt * (steps - step) / 3600:.2f} h",
                  flush=True)
            window, run, run_acc = time.time(), 0.0, 0
        if not a.iters and (step % a.eval_every == 0 or step == steps):
            vl, va = evaluate(model, valid, device)
            print(f"{step:6d} val loss {vl:.4f} acc {va:.4f}", flush=True)
            model.save_pretrained(Path(a.out) / f"step-{step:05d}")
            if va > best[0]:
                best = (va, step)
                model.save_pretrained(Path(a.out) / "best")
            print(f"{step:6d} Saved adapter (best so far: step {best[1]}, acc {best[0]:.4f})", flush=True)
    print(f"done {steps} steps in {(time.time() - t0) / 60:.1f} min; skipped {skipped_steps} spike steps; "
          f"best val acc {best[0]:.4f} at step {best[1]}", flush=True)
    if not a.iters:
        model.save_pretrained(Path(a.out) / "final")


if __name__ == "__main__":
    main()
