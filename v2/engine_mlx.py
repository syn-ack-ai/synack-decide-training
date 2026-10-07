"""MLX twin of engine_v2.py: Decision Index engine for v2 models on Apple Silicon.

Same prompt and label readout as training (v2/common.py). One forward pass per question;
softmax over the option labels; noul -> p(true). No truncation; > 255 options -> Unsupported.
Run: PYTHONPATH=v2 python -m decision_index run --engine engine_mlx:MLXEngine \
       --model models/gemma4-26b-a4b-qat-mlx4 --option adapter=v2/adapters/run1-moe/best ...
"""
import copy
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from decision_index.engines.base import Engine, Unsupported

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, SYSTEM, decision_user, question_criteria  # noqa: E402


def shared_prefix_len(seqs, min_prefix=64):
    """Longest common token prefix of several prompts, leaving at least one token of each; 0 if shorter than min_prefix."""
    if len(seqs) < 2:
        return 0
    first, n = seqs[0], min(len(s) for s in seqs) - 1
    P = 0
    while P < n and all(s[P] == first[P] for s in seqs):
        P += 1
    return P if P >= min_prefix else 0


class LoopedLayer(nn.Module):
    """Training-free looping, layer mode (arXiv 2605.23872, Alg. 3): K damped Euler sub-steps
    x <- (1 - 1/K) x + (1/K) L(x), blended as beta * L(x0) + (1 - beta) * x_K. K passes instead of one."""

    def __init__(self, layer, k, beta):
        super().__init__()
        self.layer, self.k, self.beta = layer, k, beta
        self.layer_type = layer.layer_type  # read by the model to build attention masks

    def __call__(self, x, *args, **kwargs):
        anchor = None
        for _ in range(self.k):
            y, kvs, offset = self.layer(x, *args, **kwargs)
            anchor = y if anchor is None else anchor
            x = (1 - 1 / self.k) * x + (1 / self.k) * y
        return self.beta * anchor + (1 - self.beta) * x, kvs, offset


def apply_loop(m, spec):
    """spec "A-B:K:BETA", e.g. "13-16:3:0" loops decoder layers 13..16 (0-based, inclusive) with K=3, beta=0."""
    window, k, beta = spec.split(":")
    a, b = map(int, window.split("-"))
    layers = m.language_model.model.layers
    for i in range(a, b + 1):
        layers[i] = LoopedLayer(layers[i], int(k), float(beta))


def load_mlx(model, adapter=None, loop=None):
    m, tok = load(model, adapter_path=adapter)
    if loop:
        apply_loop(m, loop)
    ids = []
    for l in LABELS:
        t = tok.encode(l, add_special_tokens=False)
        assert len(t) == 1, (l, t)
        ids.append(t[0])
    return tok, m, ids


def prompt_ids(tok, state, q, limit=32768):
    crit = question_criteria(q)
    keys = list(crit)
    if not 2 <= len(keys) <= len(LABELS):
        raise Unsupported(f"supports 2-{len(LABELS)} options per question; question has {len(keys)}")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": decision_user(state, q, keys, crit)}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)
    if len(ids) + 1 > limit:
        raise Unsupported(f"prompt of {len(ids)} tokens exceeds the {limit}-token context window")
    return keys, ids


def probs_from_ids(model, label_ids, keys, ids, temperature=1.0, cache=None, P=0):
    """One forward; with a prefix cache (shared prompt prefix of length P) only ids[P:] are run, on a copy."""
    if cache is None:
        logits = model(mx.array(ids)[None])[0, -1]
    else:
        logits = model(mx.array(ids[P:])[None], cache=copy.deepcopy(cache))[0, -1]
    p = mx.softmax(logits[mx.array(label_ids[:len(keys)])].astype(mx.float32) / temperature)
    mx.eval(p)
    return dict(zip(keys, p.tolist()))


def prefix_cache(model, prefix):
    cache = make_prompt_cache(model)
    model(mx.array(prefix)[None], cache=cache)
    mx.eval([c.state for c in cache])
    return cache


def label_probs(tok, model, label_ids, state, q, limit=32768, temperature=1.0):
    keys, ids = prompt_ids(tok, state, q, limit)
    return probs_from_ids(model, label_ids, keys, ids, temperature), len(ids)


class MLXEngine(Engine):
    name = "v2-mlx"
    latency = "In-process MLX wall time per request (one forward pass per question, evaluated); excludes loading."

    def __init__(self, model, adapter=None, max_tokens=32768, temperature=1.0, loop=None, prefix_reuse=True, **options):
        super().__init__(**options)
        self.tok, self.model, self.label_ids = load_mlx(model, adapter, loop)
        self.limit, self.temperature = int(max_tokens), float(temperature)
        self.prefix_reuse = str(prefix_reuse).lower() in ("1", "true", "yes")  # shared prompt prefix run once
        self.provenance = {"kind": "v2-mlx", "model": model, "adapter": adapter, "temperature": self.temperature, "loop": loop,
                           "prefix_reuse": self.prefix_reuse}

    def __call__(self, state, questions):
        items = [(qkey, q, *prompt_ids(self.tok, state, q, self.limit)) for qkey, q in questions.items()]
        P = shared_prefix_len([ids for *_, ids in items]) if self.prefix_reuse else 0
        cache = prefix_cache(self.model, items[0][3][:P]) if P else None
        answers, n_in = {}, sum(len(ids) for *_, ids in items)
        for qkey, q, keys, ids in items:
            p = probs_from_ids(self.model, self.label_ids, keys, ids, self.temperature, cache, P)
            answers[qkey] = ({"type": "noul", "noul": p["true"]} if q["type"] == "noul"
                             else {"type": "choice", "choice": max(p, key=p.get), "probabilities": p})
        return {"model": self.provenance["model"], "answers": answers, "usage": {"input_tokens": n_in}}, None
