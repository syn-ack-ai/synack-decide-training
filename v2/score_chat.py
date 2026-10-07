"""Score chat_generate.py outputs: IFEval (official checks, prompt- and instruction-level, strict and loose) and GSM8K
(final '#### n' answer, else the last number). Needs the three IFEval checker files from lm-evaluation-harness
(instructions.py, instructions_registry.py, instructions_util.py) in --ifeval-lib.

  uv run --with langdetect --with nltk --with immutabledict --with packaging python score_chat.py --gen DIR --data DIR --ifeval-lib DIR
"""
import argparse
import json
import re
import sys
from pathlib import Path

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def gsm_answer(text):
    m = re.search(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    s = m.group(1) if m else (NUM.findall(text) or [""])[-1]
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def ifeval(gen, data, lib):
    sys.path.insert(0, str(lib))
    import instructions_registry
    inputs = {r["key"]: r for r in map(json.loads, open(Path(data) / "eval_ifeval.jsonl"))}
    res = {"prompt_strict": 0, "prompt_loose": 0, "inst_strict": 0, "inst_loose": 0, "n_inst": 0, "n": 0}
    for row in map(json.loads, open(Path(gen) / "ifeval.jsonl")):
        q, resp = inputs[row["id"]], row["answers"][0]
        lines = resp.split("\n")
        variants = [resp, resp.replace("*", ""), "\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(),
                    "\n".join(lines[1:-1]).strip()]
        variants += [v.replace("*", "") for v in variants[2:]]
        strict, loose = [], []
        for iid, kw in zip(q["instruction_id_list"], q["kwargs"]):
            inst = instructions_registry.INSTRUCTION_DICT[iid](iid)
            inst.build_description(**{k: v for k, v in kw.items() if v is not None})
            args = inst.get_instruction_args()
            if args and "prompt" in args:
                inst.build_description(prompt=q["prompt"])
            strict.append(bool(resp.strip()) and inst.check_following(resp))
            loose.append(any(v.strip() and inst.check_following(v) for v in variants))
        res["n"] += 1
        res["n_inst"] += len(strict)
        res["prompt_strict"] += all(strict)
        res["prompt_loose"] += all(loose)
        res["inst_strict"] += sum(strict)
        res["inst_loose"] += sum(loose)
    return {"prompt_strict": res["prompt_strict"] / res["n"], "prompt_loose": res["prompt_loose"] / res["n"],
            "inst_strict": res["inst_strict"] / res["n_inst"], "inst_loose": res["inst_loose"] / res["n_inst"], "n": res["n"]}


def norm(t):
    t = re.sub(r"[^a-z0-9 ]", " ", t.lower())
    return " ".join(w for w in t.split() if w not in ("a", "an", "the"))


def mmlu_pro(gen):
    rows = [json.loads(l) for l in open(Path(gen) / "mmlu_pro.jsonl")]
    pick = lambda t: (re.findall(r"Answer:\s*\(?([A-J])\b", t) or [None])[-1]
    ok = sum(pick(r["answers"][0]) == r["gold"] for r in rows)
    return {"acc": ok / len(rows), "n": len(rows), "unparsed": sum(pick(r["answers"][0]) is None for r in rows)}


def triviaqa(gen):
    rows = [json.loads(l) for l in open(Path(gen) / "triviaqa.jsonl")]
    em = cont = 0
    for r in rows:
        pred = norm(r["answers"][0].split("\n")[0])
        al = {norm(x) for x in r["aliases"] if x}
        em += pred in al
        cont += any(a and f" {a} " in f" {pred} " for a in al)
    return {"em": em / len(rows), "contains": cont / len(rows), "n": len(rows)}


def gsm8k(gen):
    rows = [json.loads(l) for l in open(Path(gen) / "gsm8k.jsonl")]
    ok = sum(gsm_answer(r["answers"][0]) == gsm_answer(r["gold"]) for r in rows)
    return {"acc": ok / len(rows), "n": len(rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", required=True, nargs="+")
    p.add_argument("--data", required=True)
    p.add_argument("--ifeval-lib", required=True)
    a = p.parse_args()
    import nltk
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    for g in a.gen:
        out = {"ifeval": ifeval(g, a.data, a.ifeval_lib), "gsm8k": gsm8k(g)}
        for name, fn in (("mmlu_pro", mmlu_pro), ("triviaqa", triviaqa)):
            if (Path(g) / f"{name}.jsonl").exists():
                out[name] = fn(g)
        print(g, json.dumps({k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in out.items()}))


if __name__ == "__main__":
    main()
