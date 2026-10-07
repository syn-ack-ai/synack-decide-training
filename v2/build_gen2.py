"""Second generalization set (gen-v2): 200 questions from each of 8 public datasets that are neither in the Decision
Index suite nor in our training data. Same row shape and checks as build_gen.py (decontam.py vs the suite, then a
13-gram overlap check vs all training text).

  python build_gen2.py --out gen2_records.jsonl
"""
import argparse
import ast
import json
import random

from datasets import load_dataset

N = 200


def parquet(repo, cfg, split):
    return load_dataset("parquet", data_files=f"hf://datasets/{repo}@refs/convert/parquet/{cfg}/{split}/*.parquet", split="train")


def row(name, i, state, q, gold):
    if q["type"] == "choice":
        assert gold in q["criteria"], (name, gold)
    return {"id": f"gen2/{name}:{i}", "source": f"gen2/{name}", "state": state, "questions": {"q1": q}, "expected": {"q1": gold}}


def take(ds, rng, n=N):
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    return [(i, ds[i]) for i in idx[:n]]


def choice(instr, opts):
    return {"type": "choice", "instructions": instr, "criteria": opts}


def truthfulqa(rng):
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    out = []
    for i, x in take(ds, rng):
        ch, lab = x["mc1_targets"]["choices"], x["mc1_targets"]["labels"]
        opts = {str(j + 1): c for j, c in enumerate(ch)}
        out.append(row("truthfulqa", i, {}, choice(x["question"], opts), str(lab.index(1) + 1)))
    return out


def strategyqa(rng):
    ds = load_dataset("ChilleD/StrategyQA", split="test")
    return [row("strategyqa", i, {}, {"type": "noul", "instructions": x["question"]}, "true" if x["answer"] else "false")
            for i, x in take(ds, rng)]


def sciq(rng):
    ds = load_dataset("allenai/sciq", split="test")
    out = []
    for i, x in take(ds, rng):
        opts = [x["correct_answer"], x["distractor1"], x["distractor2"], x["distractor3"]]
        rng.shuffle(opts)
        crit = {str(j + 1): o for j, o in enumerate(opts)}
        out.append(row("sciq", i, {}, choice(x["question"], crit), str(opts.index(x["correct_answer"]) + 1)))
    return out


def qasc(rng):
    ds = load_dataset("allenai/qasc", split="validation")
    return [row("qasc", i, {}, choice(x["question"], dict(zip(x["choices"]["label"], x["choices"]["text"]))), x["answerKey"])
            for i, x in take(ds, rng)]


def piqa(rng):
    ds = parquet("ybisk/piqa", "plain_text", "validation")
    return [row("piqa", i, {"goal": x["goal"]}, choice("Which solution achieves the goal?", {"1": x["sol1"], "2": x["sol2"]}),
                str(int(x["label"]) + 1)) for i, x in take(ds, rng)]


def aqua(rng):
    ds = load_dataset("deepmind/aqua_rat", "raw", split="test")
    out = []
    for i, x in take(ds, rng):
        opts = x["options"] if isinstance(x["options"], list) else ast.literal_eval(x["options"])
        crit = {o.split(")", 1)[0].strip(): o.split(")", 1)[1].strip() for o in opts}
        out.append(row("aqua", i, {}, choice(x["question"], crit), x["correct"].strip()))
    return out


def casehold(rng):
    ds = load_dataset("coastalcph/lex_glue", "case_hold", split="test")
    out = []
    for i, x in take(ds, rng):
        ends = x["endings"] if isinstance(x["endings"], list) else ast.literal_eval(x["endings"])
        out.append(row("casehold", i, {"citing_context": x["context"][:2500]},
                       choice("Which holding fits the citation marked <HOLDING>?", {str(j + 1): e for j, e in enumerate(ends)}),
                       str(int(x["label"]) + 1)))
    return out


def ethics(rng):
    ds = parquet("hendrycks/ethics", "commonsense", "test")
    ds = ds.filter(lambda x: len(x["input"]) < 1200)
    return [row("ethics", i, {"scenario": x["input"]},
                {"type": "noul", "instructions": "Is what the narrator did clearly morally wrong by ordinary standards?"},
                "true" if int(x["label"]) == 1 else "false") for i, x in take(ds, rng)]


BUILDERS = [truthfulqa, strategyqa, sciq, qasc, piqa, aqua, casehold, ethics]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20261006)
    a = p.parse_args()
    rows = []
    for b in BUILDERS:
        try:
            got = b(random.Random(f"{a.seed}:{b.__name__}"))
            rows += got
            print(f"{b.__name__}: {len(got)}", flush=True)
        except Exception as e:
            print(f"{b.__name__}: FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"total {len(rows)} -> {a.out}")


if __name__ == "__main__":
    main()
