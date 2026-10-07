"""Decision Index engine that picks its backend: NVIDIA GPU -> engine_v2 (transformers, bf16 SynACK Decide),
Apple Silicon -> engine_mlx (8-bit MLX SynACK Decide). Both reuse each request's shared prompt prefix.

Run: PYTHONPATH=v2 python -m decision_index run --engine engine_decide:DecideEngine [--option backend=cuda|mlx] ...
Options: backend (auto), model (default per backend), plus the chosen engine's own options.
"""
import importlib.util
import platform
import sys
from pathlib import Path

from decision_index.engines.base import Engine

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_MODEL = {"cuda": "SkyPanther/synack-decide-26b-a4b", "mlx": "SkyPanther/synack-decide-26b-a4b-mlx-8bit"}


def detect_backend():
    if importlib.util.find_spec("torch") is not None:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    if sys.platform == "darwin" and platform.machine() == "arm64" and importlib.util.find_spec("mlx") is not None:
        return "mlx"
    raise RuntimeError("no supported backend: needs an NVIDIA GPU (torch + CUDA) or Apple Silicon with mlx installed")


class DecideEngine(Engine):
    name = "synack-decide"

    def __init__(self, backend="auto", model=None, **options):
        self.backend = detect_backend() if backend == "auto" else backend
        if self.backend == "cuda":
            from engine_v2 import V2Engine as Inner
        elif self.backend == "mlx":
            from engine_mlx import MLXEngine as Inner
        else:
            raise ValueError(f"unknown backend {backend!r} (auto, cuda or mlx)")
        self.inner = Inner(model=model or DEFAULT_MODEL[self.backend], **options)
        super().__init__(**options)
        self.latency = self.inner.latency
        self.provenance = {**self.inner.provenance, "backend": self.backend}

    def __call__(self, state, questions):
        return self.inner(state, questions)

    def synchronize(self):
        if hasattr(self.inner, "synchronize"):
            self.inner.synchronize()

    def runtime(self):
        return {**self.inner.runtime(), "backend": self.backend}
