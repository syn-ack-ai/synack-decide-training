"""Package decision records + teacher labels into a reusable, self-describing dataset folder.

Layout:
  records/{train,valid}.jsonl      decision records (state, typed questions, gold) -- leaderboard row shape
  pools/*.jsonl                    the full decontaminated pools the mix was drawn from
  labels/<teacher>/{train,valid}.jsonl   {"qid": "<record id>#<question key>", "probs": {option_key: p}}
  manifest.json, README.md
Labels are keyed by option KEY (not label letter), so any prompt format or label scheme can reuse them.
"""
import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

V2 = Path(__file__).resolve().parent
TEACHERS = {  # name -> (train file, valid file, description)
    "gemma-4-31b-it": ("teacher_api_gemma31.jsonl", "teacher_valid_google_gemma-4-31b-it.jsonl",
                       "google/gemma-4-31b-it via OpenRouter (bf16 providers; Venice/Novita excluded)"),
    "kimi-k3": ("teacher_api_kimi_k3.jsonl", "teacher_valid_moonshotai_kimi-k3.jsonl", "moonshotai/kimi-k3 via OpenRouter"),
    "deepseek-v4-pro": ("teacher_api_dsv4.jsonl", "teacher_valid_deepseek_deepseek-v4-pro.jsonl",
                        "deepseek/deepseek-v4-pro via OpenRouter (train: corr2cause + supergpqa only)"),
    "gemma-4-31b-it-mlx8bit": ("teacher_mix.jsonl", None, "local Gemma-4-31B 8-bit MLX on M5 Max (partial; mixed prompt formats)"),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def count(p):
    return sum(1 for _ in open(p))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--out", default=str(V2.parent / "datasets/systemone-teacher-labels-v1"))
    out = Path(a.parse_args().out)
    for d in ("records", "pools", "labels"):
        (out / d).mkdir(parents=True, exist_ok=True)
    data = V2 / "data"
    shutil.copy(data / "mix_train_records.jsonl", out / "records/train.jsonl")
    shutil.copy(data / "mix_valid_records.jsonl", out / "records/valid.jsonl")
    for pool in ("hf.clean.jsonl", "hf_ragtruth.clean.jsonl", "synthetic.clean.jsonl", "tev1_train.clean.jsonl"):
        if (data / pool).exists():
            shutil.copy(data / pool, out / "pools" / pool.replace(".clean", ""))
    teachers = {}
    for name, (tr, va, desc) in TEACHERS.items():
        files = {}
        for split, fn in (("train", tr), ("valid", va)):
            if fn and (data / fn).exists():
                dst = out / "labels" / name / f"{split}.jsonl"
                dst.parent.mkdir(parents=True, exist_ok=True)
                # normalise to {"qid","probs"} (drop per-row teacher field)
                with open(data / fn) as f, open(dst, "w") as g:
                    for line in f:
                        x = json.loads(line)
                        g.write(json.dumps({"qid": x["qid"], "probs": x["probs"]}, ensure_ascii=False) + "\n")
                files[split] = {"path": str(dst.relative_to(out)), "rows": count(dst), "sha256": sha(dst)}
        if files:
            teachers[name] = {"description": desc, "files": files}
    manifest = {
        "name": "systemone-teacher-labels-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "records": {s: {"rows": count(out / f"records/{s}.jsonl"), "sha256": sha(out / f"records/{s}.jsonl")}
                    for s in ("train", "valid")},
        "pools": {p.name: count(p) for p in sorted((out / "pools").glob("*.jsonl"))},
        "teachers": teachers,
        "label_method": "one forward pass; top-20 logprobs of the first generated token restricted to the option "
                        "labels A..; options outside the top 20 share the leftover mass; mapped back to option keys",
        "prompt": "v2/common.py decision_user(): JSON {state, question, options:{label: description}} with the v2 "
                  "SYSTEM prompt; temperature 0, max_tokens 1, reasoning disabled",
        "decontamination": "v2/decontam.py against Decision Index 0.2.1 suite (b2b56d6f...): >=5-word segment exact "
                           "match or >=20% 13-gram overlap, boilerplate (>20 suite rows) excluded",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "README.md").write_text(README.format(**{"n_train": manifest["records"]["train"]["rows"],
                                                    "n_valid": manifest["records"]["valid"]["rows"],
                                                    "teachers": "\n".join(f"- **{k}**: {v['description']} "
                                                                          f"({', '.join(f'{s} {f['rows']}' for s, f in v['files'].items())})"
                                                                          for k, v in teachers.items())}))
    print(json.dumps({"out": str(out), "teachers": {k: {s: f["rows"] for s, f in v["files"].items()}
                                                    for k, v in teachers.items()}}, indent=1))


README = """# systemone-teacher-labels-v1

Decision records plus teacher probability distributions for training small "System One" decision models
(state + typed multiple-choice / yes-no questions -> one answer with a probability per option).

- `records/train.jsonl` ({n_train} records) and `records/valid.jsonl` ({n_valid}): the training mix,
  in the Decision Index row shape: `{{id, source, state, questions: {{qkey: {{type, instructions, criteria}}}}, expected}}`.
- `pools/`: the full decontaminated pools the mix was sampled from (for building different mixes later).
- `labels/<teacher>/{{train,valid}}.jsonl`: `{{"qid": "<record id>#<question key>", "probs": {{option_key: p}}}}`.
  Keyed by option key, so they can be re-mapped onto any label scheme or option order.

## Teachers
{teachers}

Validation accuracy (argmax vs gold, 1,020 questions): Kimi-K3 80.5%, Gemma-4-31B 77.1%, DeepSeek-V4-Pro 66.4%.
Per-source numbers are in the training notes; Gemma is stronger on CLINC150/NLI4CT, Kimi on math/knowledge/BANKING77.

## Reuse
Average the teachers you want per question, then map `probs[key]` to whatever label each option gets in your
prompt. `v2/attach_teacher.py` does this for the v2 prompt format.

## Provenance and licences
Sources are Hugging Face train splits (ANLI is CC-BY-NC; ACOS/VAST-style sources without stated licences were not
used), code-generated synthetic items, and Together's tev1 recipe data. Every record was decontaminated against
the Decision Index 0.2.1 suite. Teacher outputs come from open-weight models: Gemma (Gemma terms), DeepSeek-V4
(MIT) and Kimi-K3 (custom "Kimi K3 License", not MIT: fine-tuning and derivatives allowed; extra terms only above
100M MAU / $20M monthly revenue, or for Model-as-a-Service businesses above $20M/yr). None prohibits training on
outputs. Kimi-K3 labels came from third-party hosts (logprobs required; Moonshot's own API, whose terms forbid
training competing models, offers no logprobs). Re-check before commercial use.
"""

if __name__ == "__main__":
    main()
