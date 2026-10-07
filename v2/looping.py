"""Looped (recurrent-depth) layers for Gemma 4 in transformers, used by train_hf.py and engine_v2.py.

spec "A-B:K" repeats each decoder layer A..B (0-based, inclusive) K times per forward pass (layer mode, as in
"Training-Free Looped Transformers", arXiv 2605.23872) with a damped step on the layer's UPDATE only:
    g(x) = L(x) / s          (s = Gemma's per-layer output scalar, which multiplies the whole hidden state)
    x <- x + (g(x) - x) / K  (K times),   output = s * x
K=1 is exactly the normal layer, and for any K the hidden-state scale matches one pass (damping the scaled output
instead would compound s to about e^-(1-s)). The model then learns to use the extra passes; set_loop_k() changes K at
run time (random K per training step gives a model that works for any K at inference).

The layer's forward is patched in place (no wrapper module), so parameter and LoRA names are unchanged and adapters
load into a plain model. Constraints: the looped layers must not be KV-sharing sources or users (Gemma 4 E-models share
KV from the last non-shared layers into the later ones), and no KV cache may be used (each pass would append its keys
again): callers run looped models with use_cache=False and without prefix reuse.
"""


def parse_spec(spec):
    parts = spec.split(":")
    a, b = map(int, parts[0].split("-"))
    return a, b, int(parts[1])


def text_layers(model):
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    lm = getattr(getattr(m, "model", m), "language_model", None) or getattr(m, "model", m)
    return lm.layers, getattr(lm, "config", None)


def apply_loop(model, spec):
    a, b, k = parse_spec(spec)
    layers, cfg = text_layers(model)
    shared = getattr(cfg, "num_kv_shared_layers", 0) or 0
    first_shared = len(layers) - shared
    if shared and b >= first_shared - 2:
        raise ValueError(f"loop window {a}-{b} touches KV-sharing layers (first shared layer {first_shared}); pick a lower window")
    state = {"k": k, "k_max": k}
    for i in range(a, b + 1):
        layer = layers[i]
        orig = layer.forward

        def forward(hidden_states, *args, _orig=orig, _layer=layer, **kwargs):
            K = state["k"]
            s = _layer.layer_scalar.to(hidden_states.dtype)
            x = hidden_states
            for _ in range(K):
                g = _orig(x, *args, **kwargs) / s
                x = x + (g - x) / K
            return s * x

        layer.forward = forward
    model._synack_loop_state = state
    return a, b, k


def set_loop_k(model, k):
    model._synack_loop_state["k"] = k
