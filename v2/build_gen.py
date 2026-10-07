"""Private generalization set: decision questions from public datasets that are neither in the Decision Index
suite nor in any of our training data (docs/V6_PLAN.md, "No benchmaxing"). Fixed seed, 300 per dataset.

  python build_gen.py --out gen_records.jsonl        (then decontam.py against the suite, and a training-overlap check)

Rows use the Decision Index shape: {id, source, state, questions: {q1: {type, instructions, criteria}}, expected}.
"""
import argparse
import json
import random

from datasets import load_dataset

N = 300


def parquet(repo, split):
    """Datasets that still ship a loading script: read the Hub's auto-converted Parquet copy instead."""
    return load_dataset("parquet", data_files=f"hf://datasets/{repo}@refs/convert/parquet/default/{split}/*.parquet", split="train")


def row(name, i, state, instructions, criteria, gold):
    assert gold in criteria, (name, gold)
    return {"id": f"gen/{name}:{i}", "source": f"gen/{name}", "state": state,
            "questions": {"q1": {"type": "choice", "instructions": instructions, "criteria": criteria}},
            "expected": {"q1": gold}}


def take(ds, rng, n=N):
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    return [(i, ds[i]) for i in idx[:n]]


def trec(rng):
    ds = parquet("CogComp/trec", "test")
    names = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]  # coarse_label order of CogComp/trec
    desc = {"ABBR": "an abbreviation or its expansion", "ENTY": "an entity (thing, object, animal, event, ...)",
            "DESC": "a description, definition or reason", "HUM": "a person or group of people",
            "LOC": "a location", "NUM": "a number, date, quantity or amount"}
    return [row("trec", i, {"question": x["text"]}, "What kind of answer is this question asking for?",
                {k: f"It asks for {desc[k]}." for k in names}, names[x["coarse_label"]]) for i, x in take(ds, rng)]


def emotion(rng):
    ds = load_dataset("dair-ai/emotion", split="test")
    names = ds.features["label"].names
    return [row("emotion", i, {"text": x["text"]}, "Which emotion does the writer mainly express?",
                {k: f"{k}" for k in names}, names[x["label"]]) for i, x in take(ds, rng)]


def siqa(rng):
    ds = parquet("allenai/social_i_qa", "validation")
    return [row("siqa", i, {"context": x["context"]}, x["question"],
                {"1": x["answerA"], "2": x["answerB"], "3": x["answerC"]}, x["label"].strip()) for i, x in take(ds, rng)]


def csqa(rng):
    ds = load_dataset("tau/commonsense_qa", split="validation")
    return [row("csqa", i, {}, x["question"], dict(zip(x["choices"]["label"], x["choices"]["text"])), x["answerKey"])
            for i, x in take(ds, rng)]


def bitext(rng):
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    intents = sorted(set(ds["intent"]))
    return [row("support_intent", i, {"customer_message": x["instruction"]}, "Which support intent should this message be routed to?",
                {k: k.replace("_", " ") for k in intents}, x["intent"]) for i, x in take(ds, rng)]


def ledgar(rng):
    ds = load_dataset("coastalcph/lex_glue", "ledgar", split="test")
    names = ds.features["label"].names
    return [row("ledgar", i, {"contract_clause": x["text"][:3000]}, "What type of contract provision is this clause?",
                {k: k for k in names}, names[x["label"]]) for i, x in take(ds, rng)]


def logiqa(rng):
    ds = parquet("lucasmccabe/logiqa", "test")
    return [row("logiqa", i, {"passage": x["context"]}, x["query"], {str(j + 1): o for j, o in enumerate(x["options"])},
                str(x["correct_option"] + 1)) for i, x in take(ds, rng)]


def medmcqa(rng):
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    return [row("medmcqa", i, {}, x["question"], {"1": x["opa"], "2": x["opb"], "3": x["opc"], "4": x["opd"]},
                str(x["cop"] + 1)) for i, x in take(ds, rng)]


def offensive(rng):
    ds = load_dataset("cardiffnlp/tweet_eval", "offensive", split="test")
    return [row("offensive", i, {"tweet": x["text"]}, "Is this tweet offensive?",
                {"not_offensive": "No, it is not offensive.", "offensive": "Yes, it is offensive (insults, threats, profanity aimed at someone)."},
                ["not_offensive", "offensive"][x["label"]]) for i, x in take(ds, rng)]


BUILDERS = [trec, emotion, siqa, csqa, bitext, ledgar, logiqa, medmcqa, offensive]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20261004)
    a = p.parse_args()
    rows = []
    for b in BUILDERS:
        try:
            got = b(random.Random(f"{a.seed}:{b.__name__}"))
            rows += got
            print(f"{b.__name__}: {len(got)}", flush=True)
        except Exception as e:  # report and continue: a missing dataset should not block the rest
            print(f"{b.__name__}: FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"total {len(rows)} -> {a.out}")


if __name__ == "__main__":
    main()
