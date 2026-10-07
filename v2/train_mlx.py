"""MLX twin of train_v2.py for Apple Silicon (e.g. Gemma-4-26B-A4B MoE on a quantized base).

LoRA on attention, dense MLP and MoE expert projections (LoRASwitchLinear on quantized experts).
Loss = CE(answer tokens) + lam * CE(teacher || softmax over option labels) when the teacher agrees
with gold. Logits are computed only at answer positions. One sequence at a time, gradient
accumulation over --batch, AdamW + cosine schedule, spike guard, best-checkpoint saving.
Adapters load with mlx_lm.load(model, adapter_path=...).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.tuner.utils import linear_to_lora_layers

LORA_KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
             "experts.switch_glu.gate_proj", "experts.switch_glu.up_proj", "experts.switch_glu.down_proj"]


def text_model(model):
    return getattr(model, "language_model", model)


def answer_logits(model, ids, positions):
    lm = text_model(model)
    h = lm.model(ids[None])[0]           # (seq, hidden)
    h = h[positions]
    out = lm.model.embed_tokens.as_linear(h) if lm.tie_word_embeddings else lm.lm_head(h)
    if lm.final_logit_softcapping is not None:
        out = mx.tanh(out / lm.final_logit_softcapping) * lm.final_logit_softcapping
    return out.astype(mx.float32)


def encode(tok, ex, max_len, label_ids):
    m = ex["messages"]
    prompt = tok.apply_chat_template(m[:-1], add_generation_prompt=True, enable_thinking=False, tokenize=True)
    full = list(prompt) + tok.encode(m[-1]["content"] + "<turn|>", add_special_tokens=False)
    if len(full) > max_len:
        return None
    opt = [label_ids[l] for l in ex["labels"]]
    t = ex.get("teacher")
    teacher = [t.get(l, 0.0) for l in ex["labels"]] if t and max(t, key=t.get) == ex["gold"] else None
    return full, len(prompt), opt, teacher


def example_loss(model, item, lam):
    ids, off, opt, teacher = item
    x = mx.array(ids)
    logits = answer_logits(model, x, mx.arange(off - 1, len(ids) - 1))
    tgt = x[off:]
    loss = nn.losses.cross_entropy(logits, tgt, reduction="sum")
    if teacher is not None and lam > 0:
        logp = logits[0, mx.array(opt)]
        logp = logp - mx.logsumexp(logp)
        p = mx.array(teacher)
        loss = loss + lam * -(p / p.sum() * logp).sum()
    return loss


def evaluate(model, data):
    tot, n, acc = 0.0, 0, 0
    for ids, off, opt, _ in data:
        x = mx.array(ids)
        logits = answer_logits(model, x, mx.arange(off - 1, len(ids) - 1))
        l = nn.losses.cross_entropy(logits, x[off:], reduction="sum")
        pred = int(mx.argmax(logits[0, mx.array(opt)]).item())
        mx.eval(l)
        tot, n, acc = tot + l.item(), n + len(ids) - off, acc + int(opt[pred] == ids[off])
    return tot / max(n, 1), acc / max(len(data), 1)


def save(model, path, cfg):
    path.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    (path / "adapter_config.json").write_text(json.dumps(cfg, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--scale", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--spike", type=float, default=4.0)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--iters", type=int)
    p.add_argument("--val-n", type=int, default=400)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--init-adapter", help="adapters.safetensors to start from (continue training)")
    a = p.parse_args()

    random.seed(42)
    mx.random.seed(42)
    model, tok = load(a.model)
    labels = json.loads((Path(__file__).resolve().parents[1] / "bench/labels255.json").read_text())
    label_ids = {l: tok.encode(l, add_special_tokens=False)[0] for l in labels}
    model.freeze()
    lora_cfg = {"rank": a.rank, "scale": a.scale, "dropout": 0.0, "keys": LORA_KEYS}
    linear_to_lora_layers(model, len(model.layers), lora_cfg)
    if a.init_adapter:
        model.load_weights(a.init_adapter, strict=False)
        print(f"initialised LoRA weights from {a.init_adapter}", flush=True)
    grad_checkpoint(model.layers[0])
    n_tr = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"trainable params: {n_tr / 1e6:.1f}M", flush=True)
    adapter_cfg = {"fine_tune_type": "lora", "num_layers": len(model.layers), "lora_parameters": lora_cfg}

    def load_data(path, limit=None):
        rows = [json.loads(l) for l in open(path)]
        if limit:
            random.Random(7).shuffle(rows)
            rows = rows[:limit]
        enc = [encode(tok, r, a.max_len, label_ids) for r in rows]
        return [e for e in enc if e], sum(e is None for e in enc)

    train, skipped = load_data(a.train)
    valid, _ = load_data(a.valid, a.val_n)
    per_epoch = len(train) // a.batch
    total = int(per_epoch * a.epochs)
    steps = min(total, a.iters) if a.iters else total
    print(f"train {len(train)} (skipped {skipped}), teacher-supervised {sum(t[3] is not None for t in train)}; "
          f"{per_epoch} steps/epoch, running {steps}; val {len(valid)}", flush=True)

    warm = max(1, int(a.warmup * total))
    sched = optim.join_schedules([optim.linear_schedule(0.0, a.lr, warm),
                                  optim.cosine_decay(a.lr, max(1, total - warm))], [warm])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.0)
    loss_and_grad = nn.value_and_grad(model, lambda m, item: example_loss(m, item, a.lam))
    order = []
    while len(order) < steps * a.batch:
        ep = list(range(len(train)))
        random.Random(42 + len(order)).shuffle(ep)
        order += ep
    if not a.iters:
        vl, va = evaluate(model, valid)
        print(f"     0 val loss {vl:.4f} acc {va:.4f}", flush=True)
    best, ema, skipped_steps = (-1.0, None), None, 0
    t0 = window = time.time()
    run = 0.0
    for step in range(1, steps + 1):
        batch = [train[i] for i in order[(step - 1) * a.batch: step * a.batch]]
        n_tok = sum(len(b[0]) - b[1] for b in batch)
        acc_grads, step_loss = None, 0.0
        for item in batch:
            l, g = loss_and_grad(model, item)
            g = tree_map(lambda x: x / n_tok, g)
            acc_grads = g if acc_grads is None else tree_map(mx.add, acc_grads, g)
            mx.eval(l, acc_grads)
            step_loss += l.item() / n_tok
        run += step_loss
        acc_grads, gnorm = optim.clip_grad_norm(acc_grads, 1.0)
        gnorm = float(gnorm)
        if ema is not None and (step_loss > a.spike * ema or not math.isfinite(gnorm)):
            skipped_steps += 1
            opt.step = opt.step + 1  # keep the schedule moving
            print(f"{step:6d} SKIP spike: loss {step_loss:.3f} vs mean {ema:.3f}", flush=True)
        else:
            opt.update(model, acc_grads)
            mx.eval(model.trainable_parameters(), opt.state)
            ema = step_loss if ema is None else 0.95 * ema + 0.05 * step_loss
        if step % 10 == 0:
            dt = (time.time() - window) / 10
            print(f"{step:6d} loss {run / 10:.4f}  {dt:.2f} s/step  peak {mx.get_peak_memory() / 1e9:.1f} GB  "
                  f"eta {dt * (steps - step) / 3600:.2f} h", flush=True)
            window, run = time.time(), 0.0
        if not a.iters and (step % a.eval_every == 0 or step == steps):
            vl, va = evaluate(model, valid)
            print(f"{step:6d} val loss {vl:.4f} acc {va:.4f}", flush=True)
            save(model, Path(a.out) / f"step-{step:05d}", adapter_cfg)
            if va > best[0]:
                best = (va, step)
                save(model, Path(a.out) / "best", adapter_cfg)
            print(f"{step:6d} Saved adapter (best so far: step {best[1]}, acc {best[0]:.4f})", flush=True)
    print(f"done {steps} steps in {(time.time() - t0) / 60:.1f} min; skipped {skipped_steps}; "
          f"best val acc {best[0]:.4f} at step {best[1]}", flush=True)


if __name__ == "__main__":
    main()
