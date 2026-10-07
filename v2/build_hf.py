"""Build v2 decision records from Hugging Face TRAIN splits, framed like the Decision Index suite.

Only train splits (or configs the suite does not read) are used; decontam.py then removes any row
that still overlaps the suite. See runs/data-manifest.json for sources, licences and traps.
Usage: python build_hf.py --out data/hf.jsonl [--only anli,gsm8k] [--scale 1.0]
"""
import argparse
import ast
import json
import random
import re

import chess
from datasets import load_dataset

LETTERS = "ABCDEFGHIJ"
OUT = []


def rec(src, rid, state, qkey, q, gold):
    OUT.append({"id": f"{src}:{rid}", "source": src, "state": state, "questions": {qkey: q}, "expected": {qkey: gold}})


def take(ds, n, seed):
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return [ds[i] for i in idx[:n]]


def anli(n):
    rows = []
    for r in ("train_r1", "train_r2", "train_r3"):
        rows += take(load_dataset("facebook/anli", split=r), n // 3, r)
    for i, r in enumerate(rows):
        rec("hf/anli", r["uid"], {}, "q1", {"type": "choice", "instructions":
            f"Classify the relationship between the premise and hypothesis.\nPremise: {r['premise']}\nHypothesis: {r['hypothesis']}",
            "criteria": {"A": "entailment", "B": "neutral", "C": "contradiction"}}, "ABC"[int(r["label"])])


def winogrande(n):
    for i, r in enumerate(take(load_dataset("allenai/winogrande", "winogrande_xl", split="train"), n, "wg")):
        rec("hf/winogrande", i, {}, "q1", {"type": "choice", "instructions": f"Which option correctly fills the blank?\n{r['sentence']}",
            "criteria": {"A": r["option1"], "B": r["option2"]}}, "AB"[int(r["answer"]) - 1])


def hellaswag(n):
    for r in take(load_dataset("Rowan/hellaswag", split="train"), n, "hs"):
        rec("hf/hellaswag", r["ind"], {}, "q1", {"type": "choice", "instructions": f"Which continuation is most plausible?\n{r['ctx']}",
            "criteria": dict(zip("ABCD", r["endings"]))}, "ABCD"[int(r["label"])])


def ragtruth(n):
    for r in take(load_dataset("wandb/RAGTruth-processed", split="train"), n, "rt"):
        lab = r["hallucination_labels"]
        hall = len(json.loads(lab) if isinstance(lab, str) else lab) > 0
        rec("hf/ragtruth", r["id"], {"prompt": f"{r['query']}\n{r['context']}", "response": r["output"]}, "q",
            {"type": "noul", "instructions": "The response contains content that is not supported by the context in the prompt."},
            hall)


def nli4ct(n):
    for i, r in enumerate(take(load_dataset("tasksource/nli4ct", split="train"), n, "nli4ct")):
        state = {"primary_trial": {"id": r["Primary_id"], "section": r["Section_id"], "text": r["Primary_ct"]}}
        if r.get("Secondary_id"):
            state["secondary_trial"] = {"id": r["Secondary_id"], "section": r["Section_id"], "text": r["Secondary_ct"]}
        rec("hf/nli4ct", i, state, "answer", {"type": "choice",
            "instructions": f"Classify the statement against the supplied clinical trial evidence:\n{r['Statement']}",
            "criteria": {"Entailment": "The clinical trial evidence entails the statement.",
                         "Contradiction": "The clinical trial evidence contradicts the statement."}}, r["Label"])


def num(s):
    s = s.replace(",", "").strip()
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return None


def gsm8k(n):
    for i, r in enumerate(take(load_dataset("openai/gsm8k", "main", split="train"), n, "gsm")):
        gold = num(r["answer"].split("####")[-1])
        if gold is None:
            continue
        rng = random.Random(f"gsm:{i}")
        k = 4 if i % 2 == 0 else 10
        pool = [num(x) for x in re.findall(r"=\s*([-\d.,]+)>>", r["answer"])] + \
               [num(x) for x in re.findall(r"\d[\d,]*\.?\d*", r["question"])]
        g = float(gold)
        pool += [num(str(v)) for v in (g + 1, g - 1, g * 2, g / 2, g + 10, g - 10, g * 10, g + 2, g - 2, g * 3)]
        opts = [gold] + [p for p in dict.fromkeys(pool) if p and p != gold and not p.startswith("-")]
        if len(opts) < k:
            continue
        opts = [gold] + rng.sample(opts[1:], k - 1)
        rng.shuffle(opts)
        crit = {f"option_{j}": o for j, o in enumerate(opts)}
        rec("hf/gsm8k", i, {"question": r["question"], "task": f"GSM8K deterministic {k}-choice numeric selection"}, "answer",
            {"type": "choice", "instructions": "Choose the numeric answer to the problem in state. Do not provide reasoning. "
             "This is a named multiple-choice adaptation of GSM8K.", "criteria": crit},
            next(c for c, o in crit.items() if o == gold))


def chess_puzzles(n):
    ds = load_dataset("Lichess/chess-puzzles", split="train", streaming=True).shuffle(seed=7, buffer_size=200_000)
    got = 0
    for r in ds:
        if got >= n:
            break
        if not 1000 <= int(r["Rating"]) <= 2600:
            continue
        board = chess.Board(r["FEN"])
        moves = r["Moves"].split()
        board.push_uci(moves[0])
        legal = list(board.legal_moves)
        if not 3 <= len(legal) <= 255:
            continue
        crit = {m.uci(): {"uci": m.uci(), "san": board.san(m)} for m in legal}
        if moves[1] not in crit:
            continue
        rec("hf/chess", r["PuzzleId"], {"fen": board.fen(), "board": str(board),
                                        "side_to_move": "white" if board.turn else "black"}, "move",
            {"type": "choice", "instructions": "Choose the strongest legal move for the side to move. Board rows run from rank 8 "
             "to rank 1; columns a to h. Uppercase pieces are white. All legal moves are supplied.", "criteria": crit}, moves[1])
        got += 1


def supergpqa(n):
    for r in take(load_dataset("m-a-p/SuperGPQA", split="train"), n, "sgpqa"):
        opts = r["options"]
        if not 2 <= len(opts) <= 10:
            continue
        rec("hf/supergpqa", r["uuid"], r["question"], "q", {"type": "choice", "instructions": "Which option is the correct answer?",
            "criteria": dict(zip(LETTERS, opts))}, r["answer_letter"])


def mmlu_aux(n):
    for i, r in enumerate(take(load_dataset("cais/mmlu", "auxiliary_train", split="train"), n, "mmlu")):
        r = r["train"]
        rec("hf/mmlu_aux", i, {}, "q1", {"type": "choice", "instructions": r["question"],
            "criteria": dict(zip("ABCD", r["choices"]))}, "ABCD"[int(r["answer"])])


def corr2cause(n):
    for i, r in enumerate(take(load_dataset("causal-nlp/corr2cause", split="train"), n, "c2c")):
        rec("hf/corr2cause", i, {}, "q1", {"type": "choice", "instructions": r["input"] + "\n\nIs the hypothesis valid?",
            "criteria": {"A": "yes", "B": "no"}}, "A" if int(r["label"]) == 1 else "B")


def banking77(n):
    ds = load_dataset("legacy-datasets/banking77", split="train")
    names = ds.features["label"].names
    crit = {f"option_{j}": nm for j, nm in enumerate(names)}
    for i, r in enumerate(take(ds, n, "b77")):
        rec("hf/banking77", i, {}, "q1", {"type": "choice", "instructions": f"Classify the banking intent of this user request:\n{r['text']}",
            "criteria": crit}, f"option_{int(r['label'])}")


def clinc(n):
    ds = load_dataset("clinc/clinc_oos", "plus", split="train")
    names = [("out of scope" if nm == "oos" else nm.replace("_", " ")) for nm in ds.features["intent"].names]
    crit = {f"option_{j}": nm for j, nm in enumerate(names)}
    for i, r in enumerate(take(ds, n, "clinc")):
        rec("hf/clinc", i, {}, "q1", {"type": "choice", "instructions":
            f"Classify the intent of this user request, or choose out of scope if none applies:\n{r['text']}",
            "criteria": crit}, f"option_{int(r['intent'])}")


ESCI = {"E": "Exact: the product satisfies the search query.",
        "S": "Substitute: a product that could substitute for the requested product.",
        "C": "Complement: a product that complements the requested product.",
        "I": "Irrelevant: the product does not address the requested product need."}


def esci(n):
    ds = load_dataset("tasksource/esci", split="train", streaming=True).shuffle(seed=11, buffer_size=100_000)
    per = {k: 0 for k in ESCI}
    for r in ds:
        if sum(per.values()) >= n:
            break
        lab = (r["esci_label"] or "")[:1].upper()
        if r["product_locale"] != "us" or lab not in ESCI or per[lab] >= n // 4:
            continue
        prod = {"title": r["product_title"], "description": r["product_description"],
                "bullet_point": r["product_bullet_point"], "brand": r["product_brand"], "color": r["product_color"]}
        rec("hf/esci", r["example_id"], {"search_query": r["query"].strip(), "product": {k: v for k, v in prod.items() if v}},
            "answer", {"type": "choice", "instructions": "Classify the relevance of this product to the search query using "
                       "the ESCI categories.", "criteria": ESCI}, lab)
        per[lab] += 1


def when2call(n):
    for i, r in enumerate(take(load_dataset("nvidia/When2Call", "train_pref", split="train"), n, "w2c")):
        msgs = r["messages"] if isinstance(r["messages"], list) else ast.literal_eval(r["messages"])
        user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), None)
        tools = [json.loads(t) if isinstance(t, str) else t for t in
                 (r["tools"] if isinstance(r["tools"], list) else ast.literal_eval(r["tools"]))]
        ch = r["chosen_response"] if isinstance(r["chosen_response"], dict) else ast.literal_eval(r["chosen_response"])
        rj = r["rejected_response"] if isinstance(r["rejected_response"], dict) else ast.literal_eval(r["rejected_response"])
        if not user:
            continue
        clean = lambda c: re.sub(r"^<TOOLCALL>\[(.*)\]</TOOLCALL>$", r"\1", c["content"].strip(), flags=re.S)
        rec("hf/when2call", i, {"tools": tools, "question": user}, "q", {"type": "choice",
            "instructions": "Which response should the assistant give to the user's question, given the available tools?",
            "criteria": {"A": clean(ch), "B": clean(rj)}}, "A")


def newyorker(n):
    ds = load_dataset("jmhessel/newyorker_caption_contest", "matching", split="train")  # NOT matching_1..4 (other folds)
    ds = ds.remove_columns(["image"])
    for r in take(ds, n, "ny"):
        rec("hf/newyorker", r["instance_id"], {"scene": r["image_location"], "description": r["image_description"],
            "uncanny_description": r["image_uncanny_description"], "entities": r["entities"]}, "q",
            {"type": "choice", "instructions": "Which caption was written for this cartoon?",
             "criteria": dict(zip("ABCDE", r["caption_choices"]))}, r["label"])


BUILDERS = {"anli": (anli, 6000), "winogrande": (winogrande, 5000), "hellaswag": (hellaswag, 6000),
            "ragtruth": (ragtruth, 4000), "nli4ct": (nli4ct, 1700), "gsm8k": (gsm8k, 6000),
            "chess": (chess_puzzles, 4000), "supergpqa": (supergpqa, 5000), "mmlu_aux": (mmlu_aux, 3000),
            "corr2cause": (corr2cause, 3000), "banking77": (banking77, 4000), "clinc": (clinc, 5000),
            "esci": (esci, 4000), "when2call": (when2call, 4000), "newyorker": (newyorker, 3000)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--only")
    p.add_argument("--scale", type=float, default=1.0)
    a = p.parse_args()
    names = a.only.split(",") if a.only else list(BUILDERS)
    for name in names:
        fn, n = BUILDERS[name]
        before = len(OUT)
        try:
            fn(int(n * a.scale))
            print(f"{name}: {len(OUT) - before}", flush=True)
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
            del OUT[before:]
    with open(a.out, "w") as f:
        for r in OUT:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"total {len(OUT)}")


if __name__ == "__main__":
    main()
