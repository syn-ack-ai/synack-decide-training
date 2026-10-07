"""Chat distillation for a Gemma 4 student (E4B) from a same-tokenizer Gemma 4 teacher (31B): LoRA on one GPU.

Loss per response token: cross-entropy on the teacher's text, plus (where the teacher's tokens line up with the
student's tokenization of the same text) lam x cross-entropy against the teacher's renormalised top-k distribution.
Only the assistant tokens are trained. Gemma's per-layer embedding table stays on the CPU (--offload-ple) so E4B fits
a 24 GB card. One sequence per micro-batch; logits only at response positions.

  python train_sft.py --model google/gemma-4-E4B-it --teacher teacher.jsonl --prompts train_prompts.jsonl --out adapters
"""
import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def build(tok, prompts, teacher, max_len, top_k):
    """-> list of (ids, offset, kd) with kd = [(pos_in_response, [ids], [probs])] or None when tokens don't line up."""
    msgs = {r["id"]: r["messages"] for r in map(json.loads, open(prompts))}
    out, aligned, skipped = [], 0, 0
    cache = {}

    def one_id(s):  # token string -> id if it is a single token (cached: the same strings repeat millions of times)
        if s not in cache:
            e = tok.encode(s, add_special_tokens=False)
            cache[s] = e[0] if len(e) == 1 else None
        return cache[s]
    for line in open(teacher):
        t = json.loads(line)
        if t.get("finish") != "stop" or t["id"] not in msgs:
            skipped += 1
            continue
        m = msgs[t["id"]] + [{"role": "assistant", "content": t["response"]}]
        prompt = tok.apply_chat_template(m[:-1], add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=False)
        full = tok.apply_chat_template(m, enable_thinking=False, tokenize=True, return_dict=False)
        if full[:len(prompt)] != prompt or len(full) > max_len:
            skipped += 1
            continue
        resp = full[len(prompt):]
        kd = None
        teach = t.get("tokens") or []
        mine = tok.encode(t["response"], add_special_tokens=False)
        if teach and resp[:len(mine)] == mine and [tok.decode([i]) for i in mine] == [x[0] for x in teach[:len(mine)]]:
            kd = []
            for j, (_, _, alts) in enumerate(teach[:len(mine)]):
                ids, ps = [], []
                for s, lp in alts[:top_k]:
                    e = one_id(s)
                    if e is not None and e not in ids:
                        ids.append(e)
                        ps.append(math.exp(lp))
                if ids:
                    z = sum(ps)
                    kd.append((j, ids, [p / z for p in ps]))
            aligned += 1
            if kd:  # pack as fixed-width arrays (pad id 0 / prob 0) so the loss is one gather, not a Python loop
                w = max(len(x[1]) for x in kd)
                kd = ([x[0] for x in kd], [x[1] + [0] * (w - len(x[1])) for x in kd], [x[2] + [0.0] * (w - len(x[2])) for x in kd])
            else:
                kd = None
        out.append((full, len(prompt), kd))
    print(f"{len(out)} examples ({aligned} with teacher token distributions), {skipped} skipped", flush=True)
    return out


def build_opd(path, max_len):
    """On-policy distillation examples (opd_score.py): the student's own ids with the teacher's top-k at every response
    position. Trained on the teacher distribution only (no cross-entropy on the student's sampled tokens)."""
    out = []
    for line in open(path):
        r = json.loads(line)
        P, R = r["prompt_ids"], r["response_ids"]
        if len(P) + len(R) > max_len or not R:
            continue
        out.append((P + R, len(P), (list(range(len(R))), r["kd_ids"], r["kd_probs"]), "opd"))
    print(f"{len(out)} on-policy distillation examples from {path}", flush=True)
    return out


def build_messages(tok, path, max_len, tools):
    """Tool-use conversations ({"messages": [...]}) -> (ids, mask, None): train only on the model's own turns (tool
    calls and answers), never on the system prompt, the user or the tool responses."""
    out, skipped = [], 0
    for line in open(path):
        r = json.loads(line)
        if not r.get("correct", True):
            continue
        text = tok.apply_chat_template(r["messages"], tools=r.get("tools") or tools, tokenize=False, enable_thinking=False)  # per-row tools (e.g. run_shell)
        spans = []
        for m in re.finditer(r"<\|turn>model\n(.*?)<turn\|>", text, re.S):
            s, e = m.start(1), m.end(0)
            pos = s
            for x in re.finditer(r"<\|tool_response>.*?<tool_response\|>", text[s:e], re.S):
                spans.append((pos, s + x.start()))
                pos = s + x.end()
            spans.append((pos, e))
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        mask = [any(a <= o[0] < b for a, b in spans) for o in enc["offset_mapping"]]
        if len(ids) > max_len or not any(mask):
            skipped += 1
            continue
        out.append((ids, mask, None))
    print(f"{len(out)} tool-use conversations from {path} ({skipped} skipped: too long or nothing to train)", flush=True)
    return out


def seq_loss(model, ids, off, kd, lam, device, opd=False):
    if isinstance(off, list):  # tool-use example: off is a per-token mask, trained positions predict token t from t-1
        x = torch.tensor([ids], device=device)
        pos = torch.tensor([i - 1 for i, m in enumerate(off) if m and i > 0], device=device)
        logits = model(input_ids=x, logits_to_keep=pos).logits[0].float()
        tgt = x[0, pos + 1]
        ce = torch.nn.functional.cross_entropy(logits, tgt, reduction="sum")
        return ce, len(pos), ce.item(), 0.0, 0
    x = torch.tensor([ids], device=device)
    keep = torch.arange(off - 1, len(ids) - 1, device=device)
    logits = model(input_ids=x, logits_to_keep=keep).logits[0].float()
    logp = torch.log_softmax(logits, -1)
    tgt = x[0, off:]
    ce = -logp.gather(1, tgt[:, None]).squeeze(1)
    loss, n = (ce.sum() * 0.0 if opd else ce.sum()), len(tgt)  # on-policy: learn only from the teacher distribution
    kd_sum = torch.zeros((), device=device)
    if kd and lam > 0:
        pos = torch.tensor(kd[0], device=device)
        ids = torch.tensor(kd[1], device=device)
        ps = torch.tensor(kd[2], device=device)
        kd_sum = -(ps * logp[pos].gather(1, ids)).sum()
        loss = loss + lam * kd_sum
    return loss, n, ce.sum().item(), kd_sum.item(), (len(kd[0]) if kd else 0)


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    tot = n = 0.0
    for ex in data:
        ids, off = ex[0], ex[1]
        if len(ex) > 3:  # on-policy examples: validate on the teacher-distribution loss
            _, k, _, kdv, kn = seq_loss(model, ids, off, ex[2], 1.0, device, opd=True)
            tot, n = tot + kdv, n + kn
            continue
        _, k, ce, _, _ = seq_loss(model, ids, off, None, 0.0, device)
        tot, n = tot + ce, n + k
    model.train()
    return tot / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--teacher", help="prompt+teacher-answer files (stage 1 format)")
    p.add_argument("--prompts")
    p.add_argument("--messages", help="tool-use conversations (gen_agent.py format), trained with span masking")
    p.add_argument("--replay", type=int, default=0, help="with --messages: also mix in N stage-1 examples from --teacher/--prompts")
    p.add_argument("--init-adapter", action="append", default=[], help="merge these LoRAs into the base first (in order), then train a new one")
    p.add_argument("--opd", help="on-policy distillation file from opd_score.py (teacher top-k on the student's own answers)")
    p.add_argument("--out", required=True)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=32, help="sequences per optimizer step")
    p.add_argument("--lam", type=float, default=1.0, help="weight of the teacher-distribution loss")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--valid", type=int, default=150, help="held-back teacher responses for validation loss")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--iters", type=int, help="stop after N optimizer steps (smoke test)")
    p.add_argument("--offload-ple", action="store_true")
    a = p.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.opd:
        data = build_opd(a.opd, a.max_len)
    elif a.messages:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from gen_agent import TOOLS
        data = build_messages(tok, a.messages, a.max_len, TOOLS)
        if a.replay and a.teacher:
            rep = build(tok, a.prompts, a.teacher, a.max_len, a.top_k)
            random.Random(1).shuffle(rep)
            data += rep[:a.replay]
            print(f"+ {min(a.replay, len(rep))} stage-1 replay examples", flush=True)
    else:
        data = build(tok, a.prompts, a.teacher, a.max_len, a.top_k)
    random.Random(0).shuffle(data)
    valid, train = data[:a.valid], data[a.valid:]
    t0 = time.time()
    if a.offload_ple:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
        from train_cuda import offload_per_layer_embeddings
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
        offload_per_layer_embeddings(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    for ad in a.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ad).merge_and_unload()
        print(f"merged {ad} into the base", flush=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, bias="none", target_modules=TARGETS))
    model.train()
    model.print_trainable_parameters()
    print(f"loaded in {(time.time() - t0) / 60:.1f} min; GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    steps = int(len(train) * a.epochs) // a.batch
    if a.iters:
        steps = min(steps, a.iters)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, steps // 20), steps)
    print(f"{len(train)} train / {len(valid)} valid sequences; {steps} steps of {a.batch}", flush=True)
    print(f"step 0 valid CE {evaluate(model, valid, dev):.4f}", flush=True)
    best, t0, i = 9e9, time.time(), 0
    for step in range(1, steps + 1):
        tot_tok = ce_sum = kd_sum = kd_pos = 0
        for _ in range(a.batch):
            ex = train[i % len(train)]
            i += 1
            ids, off, kd = ex[:3]
            loss, n, ce, kdv, kdn = seq_loss(model, ids, off, kd, a.lam, dev, opd=len(ex) > 3)
            (loss / (a.batch * 400)).backward()  # ~400 response tokens per sequence: keeps the step size stable
            tot_tok, ce_sum, kd_sum, kd_pos = tot_tok + n, ce_sum + ce, kd_sum + kdv, kd_pos + kdn
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            el = time.time() - t0
            print(f"step {step}/{steps} CE {ce_sum / tot_tok:.4f} KD {kd_sum / max(kd_pos, 1):.4f} "
                  f"{el / step:.1f} s/step eta {(steps - step) * el / step / 60:.0f} min peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        if step % a.eval_every == 0 or step == steps:
            v = evaluate(model, valid, dev)
            print(f"step {step} valid CE {v:.4f}", flush=True)
            if v < best:
                best = v
                model.save_pretrained(Path(a.out) / "best")
    model.save_pretrained(Path(a.out) / "last")
    print(f"SFT_DONE {steps} steps in {(time.time() - t0) / 60:.1f} min; best valid CE {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
