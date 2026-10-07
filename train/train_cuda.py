"""tev1 recipe on CUDA (transformers + peft), matching train/lora_gemma.yaml.

LoRA r8 / alpha 16 / dropout 0 on every linear layer of the language model, completion-only
loss, effective batch 8, lr 5e-5 cosine with 3% warmup, 1 epoch, max 2,048 tokens, seed 42.
Runs one sequence at a time with 8-step accumulation (no padding) and computes logits only at
the answer positions, which keeps Gemma's 262k-token vocabulary out of memory.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def offload_per_layer_embeddings(model, device="cuda"):
    """Keep Gemma's per-layer embedding table (2.8B params, lookup-only, frozen) on the CPU.

    Everything else goes to the GPU; token ids are sent to the CPU for the lookup and
    the looked-up rows come back to the GPU. Saves ~5.6 GB of VRAM on Gemma 4 E4B.
    """
    lm = model.model.language_model
    ple = getattr(lm, "embed_tokens_per_layer", None)
    if ple is None:
        model.to(device)
        return
    lm.embed_tokens_per_layer = torch.nn.Identity()
    model.to(device)
    ple.requires_grad_(False)
    ple.register_forward_pre_hook(lambda mod, args: tuple(a.cpu() if torch.is_tensor(a) else a for a in args))
    ple.register_forward_hook(lambda mod, args, out: out.to(device, non_blocking=True))
    lm.embed_tokens_per_layer = ple


def encode(tok, messages, max_len):
    prompt = tok.apply_chat_template(messages[:-1], add_generation_prompt=True, enable_thinking=False,
                                     tokenize=True, return_dict=False)
    full = tok.apply_chat_template(messages, enable_thinking=False, tokenize=True, return_dict=False)
    assert full[:len(prompt)] == prompt, "chat template prefix mismatch"
    return (full, len(prompt)) if len(full) <= max_len else None


def loss_sum(model, ids, offset, device):
    x = torch.tensor([ids], device=device)
    keep = torch.arange(offset - 1, len(ids) - 1, device=device)  # positions that predict the answer
    logits = model(input_ids=x, logits_to_keep=keep).logits[0].float()
    return torch.nn.functional.cross_entropy(logits, x[0, offset:], reduction="sum"), len(ids) - offset


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    tot, n = 0.0, 0
    for ids, off in data:
        l, k = loss_sum(model, ids, off, device)
        tot, n = tot + l.item(), n + k
    model.train()
    return tot / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/gemma4-e4b-it")
    p.add_argument("--data", default="train/data")
    p.add_argument("--out", default="train/adapters/gemma4-e4b-tev-cuda")
    p.add_argument("--iters", type=int, default=None, help="stop early (speed test)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--val-batches", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=1000)
    args = p.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    offload_per_layer_embeddings(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, lora)
    model.train()  # from_pretrained returns eval mode, where gradient checkpointing is skipped
    model.print_trainable_parameters()
    device = model.device

    def load(name):
        rows = [json.loads(l)["messages"] for l in Path(args.data, name).read_text().splitlines()]
        enc = [encode(tok, m, args.max_len) for m in rows]
        return [e for e in enc if e]

    train = load("train.jsonl")
    valid = load("valid.jsonl")
    random.Random(42).shuffle(valid)
    valid = valid[:args.val_batches * args.batch]
    total = len(train) // args.batch
    steps = min(total, args.iters) if args.iters else total
    print(f"train {len(train)} examples -> {total} steps/epoch; running {steps}; val {len(valid)}", flush=True)

    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total), total)
    order = list(range(len(train)))
    random.Random(42).shuffle(order)

    if not args.iters:
        print(f"     0 val {evaluate(model, valid, device):.3f}", flush=True)
    t0, window, run_loss = time.time(), time.time(), 0.0
    for step in range(1, steps + 1):
        batch = [train[i] for i in order[(step - 1) * args.batch: step * args.batch]]
        n_tok = sum(len(ids) - off for ids, off in batch)
        for ids, off in batch:
            l, _ = loss_sum(model, ids, off, device)
            (l / n_tok).backward()
            run_loss += l.item() / n_tok
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            dt = (time.time() - window) / 10
            eta = dt * (steps - step) / 3600
            print(f"{step:6d} loss {run_loss / 10:.3f}  {dt:.2f} s/step  peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
                  f"  eta {eta:.2f} h", flush=True)
            window, run_loss = time.time(), 0.0
        if not args.iters and step % args.eval_every == 0:
            print(f"{step:6d} val {evaluate(model, valid, device):.3f}", flush=True)
        if not args.iters and (step % args.save_every == 0 or step == steps):
            model.save_pretrained(Path(args.out) / f"step-{step:05d}")
            print(f"{step:6d} Saved adapter", flush=True)
    print(f"done {steps} steps in {(time.time() - t0) / 60:.1f} min", flush=True)
    if not args.iters:
        model.save_pretrained(args.out)


if __name__ == "__main__":
    main()
