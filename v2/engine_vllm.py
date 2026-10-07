"""vLLM twin of engine_v2.py: Decision Index engine for merged v2 models (fused MoE kernels + prefix caching).

Same prompt and readout as training (v2/common.py): one next-token step per question, restricted to the option
labels with allowed_token_ids; logprobs_mode="processed_logprobs" makes the returned top-k exactly those labels,
renormalized (= softmax over the label logits, as in engine_v2). All questions of a request go to vLLM as one
batch. No truncation; > 255 options raises Unsupported.

Run: PYTHONPATH=v2 python -m decision_index run --engine engine_vllm:VLLMEngine \
       --option model=SkyPanther/synack-decide-26b-a4b ...
"""
import math
import sys
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from decision_index.engines.base import Engine, Unsupported

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, SYSTEM, decision_user, question_criteria  # noqa: E402


class VLLMEngine(Engine):
    name = "v2-vllm"
    latency = "In-process vLLM wall time per request (all questions of the request in one batch); excludes loading."

    def __init__(self, model="SkyPanther/synack-decide-26b-a4b", revision=None, max_tokens=32768, temperature=1.0,
                 gpu_memory_utilization=0.9, prefix_caching=True, **options):
        super().__init__(**options)
        self.tok = AutoTokenizer.from_pretrained(model, revision=revision)
        self.label_ids = []
        for l in LABELS:
            ids = self.tok.encode(l, add_special_tokens=False)
            assert len(ids) == 1, (l, ids)
            self.label_ids.append(ids[0])
        self.limit, self.temperature = int(max_tokens), float(temperature)
        self.llm = LLM(model=model, revision=revision, dtype="bfloat16", max_model_len=self.limit,
                       gpu_memory_utilization=float(gpu_memory_utilization),
                       enable_prefix_caching=str(prefix_caching).lower() in ("1", "true", "yes"),
                       max_logprobs=len(LABELS), logprobs_mode="processed_logprobs",
                       limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0})
        self.provenance = {"kind": "v2-vllm", "model": model, "revision": revision, "temperature": self.temperature,
                           "policy": "v2 prompt, labels A..IV (single tokens), softmax over option labels; "
                                     "no truncation; >255 options raises Unsupported."}

    def _prompt(self, state, q):
        crit = question_criteria(q)
        keys = list(crit)
        if not 2 <= len(keys) <= len(LABELS):
            raise Unsupported(f"supports 2-{len(LABELS)} options per question; question has {len(keys)}")
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": decision_user(state, q, keys, crit)}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                           tokenize=True, return_dict=False)
        if len(ids) + 1 > self.limit:
            raise Unsupported(f"prompt of {len(ids)} tokens exceeds the {self.limit}-token context window")
        return keys, ids

    def __call__(self, state, questions):
        items = [(qkey, q, *self._prompt(state, q)) for qkey, q in questions.items()]
        params = [SamplingParams(max_tokens=1, temperature=self.temperature, logprobs=len(keys),
                                 allowed_token_ids=self.label_ids[:len(keys)]) for _, _, keys, _ in items]
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=ids) for *_, ids in items], params, use_tqdm=False)
        answers = {}
        for (qkey, q, keys, _), out in zip(items, outs):
            lp = out.outputs[0].logprobs[0]
            raw = [lp[t].logprob if t in lp else -math.inf for t in self.label_ids[:len(keys)]]
            if all(v == -math.inf for v in raw):
                raise RuntimeError("vLLM returned no label logprobs")
            m = max(raw)
            e = [math.exp(v - m) for v in raw]
            z = sum(e)
            p = {k: v / z for k, v in zip(keys, e)}
            answers[qkey] = ({"type": "noul", "noul": p["true"]} if q["type"] == "noul"
                             else {"type": "choice", "choice": max(p, key=p.get), "probabilities": p})
        return {"model": self.provenance["model"], "answers": answers,
                "usage": {"input_tokens": sum(len(ids) for *_, ids in items)}}, None
