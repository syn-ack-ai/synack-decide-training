"""bf16 LoRA trainer for big GPUs (HF Jobs): Gemma-4-26B-A4B MoE, length-bucketed batches.

Same objective as train_v2.py: CE on the answer tokens (label + end of turn) plus teacher CE over the
option labels when the teacher agrees with gold. LoRA on attention and the shared MLP (target_modules)
and on the fused MoE experts (PEFT target_parameters). Sequences of similar length are padded into
micro-batches up to --token-budget; optimizer steps accumulate >= --batch examples. Logits are taken
only at answer positions (hidden states gathered per row, then lm_head + final softcap).
Saves best/final adapters and uploads them to --push-to (private model repo) as training goes.
"""
import argparse
import json
import sys
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
EXPERTS = ["experts.gate_up_proj", "experts.down_proj"]


def encode(tok, ex, max_len, label_ids):
    m = ex["messages"]
    prompt = tok.apply_chat_template(m[:-1], add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=False)
    full = list(prompt) + tok.encode(m[-1]["content"] + "<turn|>", add_special_tokens=False)
    if len(full) > max_len:
        return None
    t = ex.get("teacher")
    teacher = [t.get(l, 0.0) for l in ex["labels"]] if t and max(t, key=t.get) == ex["gold"] else None
    return {"ids": full, "off": len(prompt), "opt": [label_ids[l] for l in ex["labels"]], "teacher": teacher}


def micro_batches(items, budget, seed):
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = len(items[i]["ids"])
        if cur and max(cur_max, L) * (len(cur) + 1) > budget:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    random.Random(seed).shuffle(batches)
    return batches


class Head:
    """Answer-position logits: gather hidden states per row, then lm_head and Gemma's final softcap."""

    def __init__(self, peft_model):
        top = peft_model.base_model.model
        self.backbone = top.model
        self.lm_head = top.lm_head
        cfg = getattr(top.config, "text_config", top.config)
        self.softcap = getattr(cfg, "final_logit_softcapping", None)

    def __call__(self, batch, items, device, pad_id):
        L = max(len(items[i]["ids"]) for i in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        rows, cols, tgts, owners = [], [], [], []
        for r, i in enumerate(batch):
            it = items[i]
            n = len(it["ids"])
            ids[r, :n] = torch.tensor(it["ids"])
            mask[r, :n] = 1
            for pos in range(it["off"] - 1, n - 1):
                rows.append(r)
                cols.append(pos)
                tgts.append(it["ids"][pos + 1])
                owners.append(r)
        h = self.backbone(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state
        sel = h[torch.tensor(rows, device=device), torch.tensor(cols, device=device)]
        logits = self.lm_head(sel).float()
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits, torch.tensor(tgts, device=device), owners


def batch_loss(head, batch, items, device, pad_id, lam):
    logits, tgts, owners = head(batch, items, device, pad_id)
    loss = torch.nn.functional.cross_entropy(logits, tgts, reduction="sum")
    correct, first = 0, {}
    for k, r in enumerate(owners):
        first.setdefault(r, k)
    for r, k in first.items():
        it = items[batch[r]]
        opt = torch.tensor(it["opt"], device=device)
        lo = logits[k, opt]
        correct += int(int(lo.argmax()) == it["opt"].index(it["ids"][it["off"]]))
        if it["teacher"] is not None and lam > 0:
            p = torch.tensor(it["teacher"], device=device)
            loss = loss + lam * -(p / p.sum() * torch.log_softmax(lo, -1)).sum()
    return loss, len(tgts), correct


@torch.no_grad()
def evaluate(model, head, items, budget, device, pad_id):
    model.eval()
    tot, n, acc = 0.0, 0, 0
    for b in micro_batches(items, budget, 0):
        l, k, c = batch_loss(head, b, items, device, pad_id, 0.0)
        tot, n, acc = tot + l.item(), n + k, acc + c
    model.train()
    return tot / max(n, 1), acc / max(len(items), 1)


def push(path, repo, msg):
    if not repo:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(path), repo_id=repo, path_in_repo=path.name, commit_message=msg)
    except Exception as e:  # noqa: BLE001 - keep training if the upload fails
        print(f"push failed: {type(e).__name__}: {e}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    p.add_argument("--train", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", default="adapters/v3")
    p.add_argument("--push-to")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=32, help="examples per optimizer step (approximate)")
    p.add_argument("--token-budget", type=int, default=24000, help="padded tokens per micro-batch")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--spike", type=float, default=4.0)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--iters", type=int, help="stop after N optimizer steps (smoke test)")
    p.add_argument("--eval-every", type=int, default=300)
    p.add_argument("--limit", type=int, help="use only N training examples")
    p.add_argument("--no-experts", action="store_true", help="dense models (e.g. Gemma 4 E2B/E4B): no MoE expert LoRA")
    p.add_argument("--offload-ple", action="store_true", help="Gemma 4 E-models on small GPUs: per-layer embeddings on the CPU")
    p.add_argument("--loop", help='looped layers "A-B:K" (v2/looping.py), e.g. "9-12:4"; trained with the loops in place')
    p.add_argument("--loop-random", action="store_true", help="draw K uniformly from 1..K for each optimizer step (validation at K)")
    a = p.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    labels = json.loads(Path(a.labels).read_text())
    label_ids = {l: tok.encode(l, add_special_tokens=False)[0] for l in labels}
    t0 = time.time()
    if a.offload_ple:  # small GPUs: keep Gemma's per-layer embedding table (lookup-only, frozen) in CPU RAM
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
        from train_cuda import offload_per_layer_embeddings
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
        offload_per_layer_embeddings(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    if a.loop:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "v2"))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from looping import apply_loop, set_loop_k
        print("looped layers (a, b, k):", apply_loop(model, a.loop), "random K per step:", a.loop_random, flush=True)
    loop_model = model if a.loop else None
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, bias="none",
                                             target_modules=TARGETS, target_parameters=None if a.no_experts else EXPERTS))
    model.train()
    model.print_trainable_parameters()
    print(f"model loaded in {(time.time() - t0) / 60:.1f} min; GPU mem {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    device = next(model.parameters()).device
    head = Head(model)

    def load(path, limit=None):
        rows = [json.loads(l) for l in open(path)]
        if limit:
            rows = rows[:limit]
        enc = [encode(tok, r, a.max_len, label_ids) for r in rows]
        return [e for e in enc if e]

    train, valid = load(a.train, a.limit), load(a.valid)
    tokens = sum(len(x["ids"]) for x in train)
    print(f"train {len(train)} examples / {tokens / 1e6:.1f}M tokens; valid {len(valid)}; "
          f"teacher-supervised {sum(x['teacher'] is not None for x in train)}", flush=True)
    mbs = micro_batches(train, a.token_budget, 42)
    avg = len(train) / len(mbs)
    accum = max(1, round(a.batch / avg))
    total = int(len(mbs) * a.epochs / accum)
    steps = min(total, a.iters) if a.iters else total
    print(f"{len(mbs)} micro-batches (avg {avg:.1f} ex), accumulate {accum} -> {total} optimizer steps; running {steps}", flush=True)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(a.warmup * total), total)
    if not a.iters:
        vl, va = evaluate(model, head, valid, a.token_budget, device, pad_id)
        print(f"     0 val loss {vl:.4f} acc {va:.4f}", flush=True)
    best, ema, skipped = (-1.0, None), None, 0
    t0 = window = time.time()
    run_loss, run_acc, run_n, mb_i = 0.0, 0, 0, 0
    for step in range(1, steps + 1):
        if loop_model is not None and a.loop_random:
            set_loop_k(loop_model, random.randint(1, loop_model._synack_loop_state["k_max"]))
        group = [mbs[(mb_i + j) % len(mbs)] for j in range(accum)]
        mb_i += accum
        n_tok = sum(sum(len(train[i]["ids"]) - train[i]["off"] for i in b) for b in group)
        step_loss = 0.0
        for b in group:
            l, _, c = batch_loss(head, b, train, device, pad_id, a.lam)
            (l / n_tok).backward()
            step_loss += l.item() / n_tok
            run_acc += c
            run_n += len(b)
        run_loss += step_loss
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if ema is not None and (step_loss > a.spike * ema or not math.isfinite(gnorm)):
            skipped += 1
            print(f"{step:6d} SKIP spike: loss {step_loss:.3f} vs mean {ema:.3f}", flush=True)
        else:
            opt.step()
            ema = step_loss if ema is None else 0.95 * ema + 0.05 * step_loss
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            dt = (time.time() - window) / 10
            print(f"{step:6d} loss {run_loss / 10:.4f} acc {run_acc / max(run_n, 1):.3f}  {dt:.2f} s/step  "
                  f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB  eta {dt * (steps - step) / 3600:.2f} h", flush=True)
            window, run_loss, run_acc, run_n = time.time(), 0.0, 0, 0
        if not a.iters and (step % a.eval_every == 0 or step == steps):
            if loop_model is not None:
                set_loop_k(loop_model, loop_model._synack_loop_state["k_max"])
            vl, va = evaluate(model, head, valid, a.token_budget, device, pad_id)
            print(f"{step:6d} val loss {vl:.4f} acc {va:.4f}", flush=True)
            if va > best[0]:
                best = (va, step)
                path = Path(a.out) / "best"
                model.save_pretrained(path)
                push(path, a.push_to, f"best at step {step}: val acc {va:.4f}")
            print(f"{step:6d} best so far: step {best[1]}, acc {best[0]:.4f}", flush=True)
    print(f"done {steps} steps in {(time.time() - t0) / 60:.1f} min; skipped {skipped}; best val acc {best[0]:.4f} at step {best[1]}", flush=True)
    path = Path(a.out) / "final"
    model.save_pretrained(path)
    if not a.iters:
        push(path, a.push_to, "final adapter")


if __name__ == "__main__":
    main()
